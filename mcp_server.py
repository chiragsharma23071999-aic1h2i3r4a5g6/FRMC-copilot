"""FRMC MCP server. Start with the project's Python interpreter (stdio transport)."""
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from app import app
from knowledge import KNOWLEDGE_DIR

server = FastMCP(
    "FRMC Copilot",
    instructions="Answer SOX/COSO questions with cited reference material. Assessments use illustrative project rules. Obtain transaction evidence from the user; never invent it.",
)


def application_request(path: str, data: dict | None = None):
    """Reuse the application's validation and error handling without a web server."""
    with app.test_client() as client:
        response = client.get(path) if data is None else client.post(path, json=data)
        result = response.get_json()
        if response.status_code >= 400:
            raise ToolError(result.get("error", "FRMC request failed."))
        return result


@server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
def ask_policy(question: str) -> dict[str, Any]:
    """Answer a SOX/COSO question with sources, or retrieve uploaded policy excerpts.

    Built-in fundamentals and uploaded-document retrieval work locally.
    """
    return application_request("/query", {"question": question})


@server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
def chat(question: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Converse using free OpenRouter models; optionally pass up to 12 user/assistant messages.

    Relevant reference excerpts and the supplied history are sent to OpenRouter.
    Free-service failures return clearly labeled local reference answers.
    """
    return application_request("/chat", {"question": question, "history": history or []})


@server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def assess_transaction(
    question: str,
    evidence: dict[str, Any] | None = None,
    risk_factors: dict[str, Any] | None = None,
    control_requirement: str | None = None,
    control_id: str | None = None,
) -> dict[str, Any]:
    """Run RAG, Evidence, Risk and Review agents and save an audit event.

    evidence accepts amount, transaction_id, legacy boolean approval, approval_status
    (Approved/Rejected/Pending), and records keyed by type. An approval record has
    record_id, transaction_id, amount, status, reviewer, recorded_at (ISO timestamp
    with timezone), and source. A stated approval alone is incomplete evidence.
    control_id selects uploaded requirements; the latest uploaded definition wins.
    evidence.transaction_id locates uploaded transaction and approval rows.
    Unmapped controls are not assessed. Ratings require an uploaded risk-rule table;
    documented finding severities are returned separately from transaction ratings.
    Returns evidence_assessment with status, findings, missing_evidence,
    contradictions, sources and limitations, alongside the legacy output fields.
    risk_factors accepts criticality (Low/Medium/High) and boolean exception.
    Omitted evidence yields an unassessed result, not proof of compliance.
    """
    data = {"question": question, "evidence": evidence or {}, "risk_factors": risk_factors or {}}
    if control_requirement is not None:
        data["control_requirement"] = control_requirement
    if control_id is not None:
        data["control_id"] = control_id
    return application_request("/run_agents", data)


@server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def create_report(title: str, content: str, author: str = "System") -> dict[str, Any]:
    """Save a report in the project's database. Creates a new record on every call."""
    return application_request("/report", {"title": title, "content": content, "author": author})


@server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
def list_reports() -> dict[str, Any]:
    """List metadata for saved reports."""
    return {"reports": application_request("/reports")}


@server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
def list_logs() -> dict[str, Any]:
    """Read saved agent workflow audit events."""
    return {"logs": application_request("/logs")}


@server.resource("frmc://knowledge/fundamentals", mime_type="application/json")
def fundamentals() -> str:
    """Read the project's sourced SOX/COSO educational reference memory."""
    return (KNOWLEDGE_DIR / "fundamentals.json").read_text(encoding="utf-8")


@server.resource("frmc://knowledge/control-rules", mime_type="application/json")
def rules() -> str:
    """Read the illustrative approval threshold and its scope."""
    return (KNOWLEDGE_DIR / "control_rules.json").read_text(encoding="utf-8")


if __name__ == "__main__":
    server.run(transport="stdio")
