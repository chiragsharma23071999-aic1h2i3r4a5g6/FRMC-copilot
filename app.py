import json
import os
from pathlib import Path
from threading import RLock
from tempfile import TemporaryDirectory
import shutil

from flask import Flask, request, jsonify, render_template
from openai import APIError
from werkzeug.exceptions import BadRequest, HTTPException, NotFound, RequestEntityTooLarge
from werkzeug.utils import secure_filename

from config import UPLOAD_DIR
from models import init_db
from rag_pipeline import (load_documents, build_vectorstore, get_retriever,
                          extract_text_from_file, SUPPORTED_EXTENSIONS, KnowledgeBaseError,
                          remove_document_from_index)
from audit_controller import mcp
from backend import report_service
from agents import build_workflow, validate_evidence_input
from knowledge import answer_from_documents
from document_policy import require_finance_content
from chat_service import conversational_answer, validate_history
from uploaded_knowledge import UploadedKnowledge

app = Flask(__name__, template_folder="template")
app.config.update(UPLOAD_FOLDER=str(UPLOAD_DIR), MAX_CONTENT_LENGTH=1024 * 1024 * 1024,
                  MAX_FILE_SIZE=500 * 1024 * 1024, MAX_UPLOAD_FILES=1000,
                  MAX_FORM_PARTS=1010)
if os.getenv("FLASK_SECRET_KEY"):
    app.secret_key = os.environ["FLASK_SECRET_KEY"]
init_db()
index_lock = RLock()


def payload():
    data = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    if not isinstance(data, dict):
        raise BadRequest("Provide a JSON object or form data.")
    return data


def required_text(data, name, default=None):
    value = data.get(name, default)
    if not isinstance(value, str) or not value.strip():
        raise BadRequest(f"{name} must be a non-empty string.")
    return value.strip()


@app.errorhandler(HTTPException)
def http_error(exc):
    return jsonify(error=exc.description), exc.code


@app.errorhandler(ValueError)
def validation_error(exc):
    return jsonify(error=str(exc)), 400


@app.errorhandler(KnowledgeBaseError)
def knowledge_error(exc):
    return jsonify(error=str(exc)), 409


@app.errorhandler(APIError)
def provider_error(exc):
    app.logger.warning("Embedding provider failed: %s", type(exc).__name__)
    return jsonify(error="Embedding service unavailable. Check API credentials, model, and account balance."), 502


@app.errorhandler(Exception)
def unexpected_error(exc):
    app.logger.exception("Request failed")
    return jsonify(error="The request could not be completed. Check the server log for details."), 500


@app.route("/")
def home():
    return render_template("index.html", upload_limits={
        "fileBytes": app.config["MAX_FILE_SIZE"],
        "batchBytes": app.config["MAX_CONTENT_LENGTH"],
        "fileCount": app.config["MAX_UPLOAD_FILES"],
    })


@app.route("/documents")
def list_documents():
    with index_lock:
        folder = Path(app.config["UPLOAD_FOLDER"])
        files = [{"filename": path.name, "size": path.stat().st_size}
                 for path in sorted(folder.iterdir())
                 if path.is_file() and not path.is_symlink()
                 and path.suffix.lower() in SUPPORTED_EXTENSIONS] if folder.exists() else []
    return jsonify(files=files)


@app.route("/documents", methods=["DELETE"])
def remove_document():
    filename = required_text(payload(), "filename")
    if filename != Path(filename).name or "/" in filename or "\\" in filename:
        raise BadRequest("Invalid filename.")
    folder = Path(app.config["UPLOAD_FOLDER"])
    with index_lock:
        target = folder / filename
        if target.is_symlink() or not target.is_file() or target.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise NotFound("Uploaded document not found.")
        # Keep a recoverable copy until removal from the searchable index succeeds.
        with TemporaryDirectory(prefix=".remove-", dir=folder) as staging:
            backup = Path(staging) / filename
            target.rename(backup)
            try:
                remove_document_from_index(filename)
            except Exception:
                backup.rename(target)
                raise
    return jsonify(message=f"{filename} removed from the knowledge base.")


@app.route("/upload", methods=["POST"])
def upload_file():
    files = request.files.getlist("file") + request.files.getlist("files")
    if not files or any(not file.filename for file in files):
        raise BadRequest("No file uploaded.")
    if len(files) > app.config["MAX_UPLOAD_FILES"]:
        raise BadRequest(f"Upload at most {app.config['MAX_UPLOAD_FILES']} files per batch.")
    folder = Path(app.config["UPLOAD_FOLDER"])
    folder.mkdir(parents=True, exist_ok=True)
    with index_lock:
        with TemporaryDirectory(prefix=".upload-", dir=folder) as staging:
            accepted, rejected, names = [], [], set()
            for file in files:
                filename = secure_filename(file.filename)
                if not filename or Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
                    rejected.append({"filename": filename, "reason": "Supported formats: TXT, CSV, XLSX, XLS, PDF, DOCX.", "status": 400})
                    continue
                if filename.casefold() in names or (folder / filename).exists():
                    rejected.append({"filename": filename, "reason": "Duplicate filename. Rename the file before uploading.", "status": 409})
                    continue
                names.add(filename.casefold())
                staged = Path(staging) / filename
                try:
                    with staged.open("wb") as destination:
                        size = 0
                        while chunk := file.stream.read(1024 * 1024):
                            size += len(chunk)
                            if size > app.config["MAX_FILE_SIZE"]:
                                raise RequestEntityTooLarge("A file exceeds the 500 MB per-file limit. No files were added.")
                            destination.write(chunk)
                    assessment = require_finance_content(extract_text_from_file(staged), filename)
                    accepted.append({"filename": filename, **assessment})
                except ValueError as exc:
                    rejected.append({"filename": filename, "reason": str(exc), "status": 400})
            if rejected:
                status = 409 if all(item["status"] == 409 for item in rejected) else 400
                return jsonify(error="Batch rejected. No files were added.", rejected=rejected), status
            created = []
            try:
                for item in accepted:
                    target = folder / item["filename"]
                    with target.open("xb") as destination:
                        created.append(target)
                        with (Path(staging) / item["filename"]).open("rb") as source:
                            shutil.copyfileobj(source, destination)
                build_vectorstore(load_documents(folder))
            except Exception:
                for target in created:
                    target.unlink(missing_ok=True)
                raise
    return jsonify(message=f"{len(accepted)} finance document(s) uploaded and processed.", files=accepted)


@app.route("/build_rag", methods=["POST"])
def build_rag():
    with index_lock:
        build_vectorstore(load_documents(app.config["UPLOAD_FOLDER"]))
    return jsonify(message="Knowledge base rebuilt.")


@app.route("/ask", methods=["POST"])
@app.route("/query", methods=["POST"])
def query_rag():
    question = required_text(payload(), "question")
    with index_lock:
        results = UploadedKnowledge(app.config["UPLOAD_FOLDER"]).invoke(question)
    return jsonify(results=[doc.page_content for doc in results], **answer_from_documents(results))


@app.route("/chat", methods=["POST"])
def chat():
    data = payload()
    question = required_text(data, "question")
    if len(question) > 8000:
        raise BadRequest("question must be at most 8000 characters.")
    history = validate_history(data.get("history", []))
    # Include the preceding user turn to retrieve context for follow-ups such as 'give an example'.
    with index_lock:
        try:
            documents = UploadedKnowledge(app.config["UPLOAD_FOLDER"]).invoke(question)
            if not documents:
                previous = next((item["content"] for item in reversed(history) if item["role"] == "user"), None)
                if previous:
                    documents = UploadedKnowledge(app.config["UPLOAD_FOLDER"]).invoke(previous)
        except KnowledgeBaseError:
            documents = []
    return jsonify(conversational_answer(question, documents, history))


def report_json(report):
    return {
        "id": report.id, "title": report.title, "author": report.author,
        "timestamp": report.timestamp.isoformat(),
    }


@app.route("/report", methods=["POST"])
def create_report():
    data = payload()
    report = report_service.create_report(
        required_text(data, "title"), required_text(data, "content"),
        required_text(data, "author", "System"),
    )
    return jsonify(report_json(report)), 201


@app.route("/reports")
def get_reports():
    return jsonify([report_json(report) for report in report_service.get_reports()])


@app.route("/logs")
def get_logs():
    return jsonify([{
        "id": log.id, "agent": log.agent, "action": log.action,
        "details": log.details, "timestamp": log.timestamp.isoformat(),
    } for log in mcp.get_logs()])


def workflow_result(data):
    state = {"question": required_text(data, "question")}
    for name in ("evidence", "risk_factors"):
        value = data.get(name, {})
        if not isinstance(value, dict):
            raise BadRequest(f"{name} must be an object.")
        state[name] = value
    criticality = state["risk_factors"].get("criticality")
    if criticality is not None and (not isinstance(criticality, str) or criticality.lower() not in {"low", "medium", "high"}):
        raise BadRequest("risk_factors.criticality must be Low, Medium, or High.")
    for obj, key in ((state["risk_factors"], "exception"),):
        if key in obj and not isinstance(obj[key], bool):
            raise BadRequest(f"{key} must be a boolean.")
    if "control_requirement" in data:
        state["control_requirement"] = required_text(data, "control_requirement")
    if "control_id" in data:
        state["control_id"] = required_text(data, "control_id")
    validate_evidence_input(state["evidence"], state.get("control_id"), state.get("control_requirement"))
    with index_lock:
        uploaded = UploadedKnowledge(app.config["UPLOAD_FOLDER"])
        result = build_workflow(uploaded, uploaded_knowledge=uploaded).invoke(state)
    mcp.log_event("AgentWorkflow", "Run", f"Question: {state['question']}")
    return result


@app.route("/run_agents", methods=["POST"])
def run_agents():
    return jsonify(workflow_result(payload()))


@app.route("/generate_report", methods=["POST"])
def generate_report():
    data = payload()
    question = required_text(data, "question")
    author = required_text(data, "author", "System")
    result = workflow_result(data)
    report = report_service.create_report(question, json.dumps(result["final_report"]), author)
    return jsonify(**report_json(report), final_report=result["final_report"]), 201


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")
