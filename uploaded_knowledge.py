"""Document-backed assessment context. No generated evidence or implicit risk matrix."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import math
import re

import pandas as pd
from langchain_core.documents import Document
from knowledge import tokens, reference_documents
from rag_pipeline import SUPPORTED_EXTENSIONS, extract_text_from_file


def key(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def number(value):
    value = float(str(value).replace(",", ""))
    if not math.isfinite(value) or value < 0:
        raise ValueError("Amount must be finite and non-negative.")
    return value


@lru_cache(maxsize=4)
def _read_snapshot(signature):
    rows, documents, errors = [], [], []
    for filename, modified, size in signature:
        path = Path(filename)
        try:
            if path.suffix.lower() == ".csv":
                sheets = {"CSV": pd.read_csv(path, dtype=str, keep_default_na=False)}
            elif path.suffix.lower() in {".xls", ".xlsx"}:
                sheets = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False,
                                       engine="xlrd" if path.suffix.lower() == ".xls" else "openpyxl")
            else:
                documents.append(Document(page_content=extract_text_from_file(path), metadata={
                    "source": path.name, "title": path.name, "uploaded_at": modified}))
                continue
            for sheet, frame in sheets.items():
                for index, values in enumerate(frame.to_dict("records"), start=2):
                    data = {key(name): str(value).strip() for name, value in values.items() if str(value).strip()}
                    reference = f"{path.name} / {sheet} / row {index}"
                    rows.append({"data": data, "source": path.name, "reference": reference, "uploaded_at": modified})
                    documents.append(Document(page_content="\n".join(f"{name}: {value}" for name, value in values.items()),
                                              metadata={"source": path.name, "title": reference, "reference": reference,
                                                        "uploaded_at": modified, "control_id": data.get("control_id", data.get("related_control_id", ""))}))
        except (ValueError, OSError) as exc:
            errors.append(f"{path.name}: unable to read uploaded content ({type(exc).__name__}).")
    return rows, documents, errors


class UploadedKnowledge:
    def __init__(self, folder):
        root = Path(folder)
        signature = tuple(sorted((str(p.resolve()), p.stat().st_mtime_ns, p.stat().st_size)
                                 for p in root.iterdir() if p.is_file() and not p.is_symlink()
                                 and p.suffix.lower() in SUPPORTED_EXTENSIONS)) if root.exists() else ()
        self.rows, self.documents, self.errors = _read_snapshot(signature)
        # Exclude superseded definitions/records from answers as well as assessments.
        def identity(row):
            data = row["data"]
            control = data.get("control_id", "").casefold()
            kind = data.get("record_type", "").casefold()
            if data.get("control_description") and control:
                return ("control", control)
            if kind in {"transaction", "approval"}:
                return (kind, control, data.get("transaction_id"))
            if kind == "risk_rule":
                return (kind, control)
            if data.get("finding_id"):
                return ("finding", data["finding_id"])
            return ("row", row["reference"])
        versions = {}
        for row in self.rows:
            name = identity(row)
            versions[name] = max(versions.get(name, 0), row["uploaded_at"])
        superseded = {row["reference"] for row in self.rows if row["uploaded_at"] < versions[identity(row)]}
        self.rows = [row for row in self.rows if row["reference"] not in superseded]
        self.documents = [doc for doc in self.documents if doc.metadata.get("reference") not in superseded]

    def invoke(self, question):
        """Uploaded matches first; fundamentals remain available for general questions."""
        query = tokens(question)
        ranked = []
        for doc in self.documents:
            terms = tokens(doc.page_content)
            overlap = len(query & terms)
            if overlap and overlap / max(len(query), 1) >= 0.35:
                ranked.append((overlap, doc.metadata["uploaded_at"], doc))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if ranked:
            return [item[2] for item in ranked[:5]]
        return reference_documents(question)

    def latest(self, rows):
        if not rows:
            return []
        newest = max(row["uploaded_at"] for row in rows)
        return [row for row in rows if row["uploaded_at"] == newest]

    @staticmethod
    def source(row, kind):
        return {"kind": kind, "source": row["source"], "reference": row["reference"]}

    def resolve(self, state):
        result = {"status": "NOT_ASSESSED", "control_id": state.get("control_id"), "rules": {},
                  "evidence": deepcopy(state.get("evidence", {})), "sources": [], "findings": [],
                  "missing_evidence": [], "contradictions": [], "timing_event": None,
                  "limitations": ["Latest upload wins for the same control ID; upload recency is not a verified policy effective date."]}
        def stop(reason, missing="applicable_uploaded_control"):
            result["findings"].append(reason)
            if missing:
                result["missing_evidence"].append(missing)
            return result
        if self.errors:
            return stop(" ".join(self.errors), "readable_uploaded_documents")
        controls = [r for r in self.rows if r["data"].get("control_id") and r["data"].get("control_description")]
        control_id = state.get("control_id")
        if not control_id:
            mentioned = {r["data"]["control_id"] for r in controls
                         if re.search(r"(?<!\w)" + re.escape(r["data"]["control_id"]) + r"(?!\w)", state["question"], re.I)}
            if len(mentioned) == 1:
                control_id = mentioned.pop()
            elif len({r["data"]["control_id"] for r in controls}) == 1:
                control_id = controls[0]["data"]["control_id"]
            else:
                return stop("Select a Control ID from an uploaded control register; no unique control could be selected.", "control_id")
        result["control_id"] = control_id
        matches = self.latest([r for r in controls if r["data"]["control_id"].casefold() == control_id.casefold()])
        if not matches:
            return stop(f"No uploaded control requirements found for {control_id}. The illustrative default rule was not used.")
        result["sources"] = [self.source(row, "uploaded_control") for row in matches]
        if len(matches) != 1:
            return stop("Conflicting or duplicate control definitions in the latest upload need review.")
        row = matches[0]
        data = row["data"]
        newer_text = [doc for doc in self.documents if not doc.metadata.get("reference")
                      and doc.metadata["uploaded_at"] > row["uploaded_at"]
                      and re.search(r"(?<!\w)" + re.escape(control_id) + r"(?!\w)", doc.page_content, re.I)]
        if newer_text:
            result["sources"].extend({"kind": "uploaded_policy", "source": doc.metadata["source"],
                                      "reference": doc.metadata["source"]} for doc in newer_text)
            return stop("Newer uploaded text refers to this control. Map its requirements before relying on the older structured definition.", "mapped_newer_policy")
        result["control_id"] = data["control_id"]
        result["description"] = data["control_description"]
        result["findings"].append(f"Selected {data['control_id']} from {row['reference']}: {data['control_description']}")
        if data.get("status", "").lower() != "active":
            return stop("The uploaded control must explicitly be Active before it can be evaluated.")
        if state.get("control_requirement") and state["control_requirement"].strip() != data["control_description"]:
            return stop("The supplied requirement differs from the selected uploaded control. Resolve the mismatch before assessment.", "mapped_control_requirement")
        # A deliberately narrow, full-sentence grammar avoids discarding extra policy conditions.
        description = data["control_description"].replace("\u2019", "'").rstrip(".")
        pattern = (r"All (?P<type>.+?) above (?P<currency>[A-Z]{3}) (?P<amount>[\d,]+(?:\.\d+)?) "
                   r"require approval from (?P<reviewer>.+?) before (?P<event>posting|payment|execution)")
        policy = re.fullmatch(pattern, description)
        if not policy:
            return stop("The uploaded requirement is available for review, but its full conditions cannot yet be evaluated deterministically. No approval or risk rule was invented.", "supported_executable_requirement")
        attributes = policy.groupdict()
        # This role requires a transaction-specific designation; arbitrary role prose is not executable.
        if attributes["reviewer"].lower() != "the preparer's designated finance manager":
            return stop("The required approval authority needs an explicit, supported mapping.", "mapped_approval_authority")
        try:
            threshold = number(attributes["amount"])
        except ValueError:
            return stop("The uploaded threshold is invalid.")
        result["threshold"] = threshold
        result["currency"] = attributes["currency"]
        result["transaction_type"] = attributes["type"]
        result["timing_event"] = attributes["event"] + "_at"
        evidence = result["evidence"]
        transaction_id = evidence.get("transaction_id")
        if not transaction_id:
            return stop("Supply the transaction ID to locate uploaded transaction and approval evidence.", "evidence.transaction_id")
        related = [r for r in self.rows if r["data"].get("transaction_id") == transaction_id
                   and r["data"].get("control_id", "").casefold() == control_id.casefold()]
        transaction_rows = self.latest([r for r in related if r["data"].get("record_type", "").lower() == "transaction"])
        approval_rows = self.latest([r for r in related if r["data"].get("record_type", "").lower() == "approval"])
        for label, selected in (("transaction", transaction_rows), ("approval", approval_rows)):
            result["sources"].extend(self.source(r, "uploaded_evidence") for r in selected)
            if len(selected) > 1:
                result["contradictions"].append(f"Multiple {label} records for the transaction in the latest evidence upload require review.")
        if result["contradictions"]:
            result["status"] = "CONTRADICTORY"
            return result
        if not transaction_rows:
            result["status"] = "INCOMPLETE"
            return stop("No matching uploaded transaction record was found.", "uploaded_transaction_record")
        transaction = transaction_rows[0]["data"]
        for name in ("entity_id", "criticality"):
            if name in transaction:
                if name in evidence and evidence[name] != transaction[name]:
                    result["contradictions"].append(f"Supplied {name} differs from the uploaded transaction.")
                evidence[name] = transaction[name]
        for name in ("amount", "currency", "transaction_type", "designated_finance_manager", result["timing_event"]):
            value = transaction.get(name)
            if value is None:
                result["missing_evidence"].append("uploaded_transaction." + name)
                continue
            try:
                value = number(value) if name == "amount" else value
            except ValueError:
                result["missing_evidence"].append("valid_uploaded_transaction.amount")
                continue
            if name in evidence and evidence[name] != value:
                result["contradictions"].append(f"Supplied {name} differs from {transaction_rows[0]['reference']}.")
            evidence[name] = value
        if result["contradictions"]:
            result["status"] = "CONTRADICTORY"
            return result
        if result["missing_evidence"]:
            result["status"] = "INCOMPLETE"
            return stop("Required transaction fields are missing from uploaded evidence.", None)
        if evidence["currency"].upper() != attributes["currency"] or evidence["transaction_type"].casefold() != attributes["type"].casefold():
            return stop("Transaction currency or type does not match this uploaded control's scope.", None)
        if evidence["amount"] <= threshold:
            return stop("The transaction does not exceed this uploaded control's threshold.", None)
        if not approval_rows:
            result["status"] = "INCOMPLETE"
            return stop("No matching uploaded approval record was found. A stated approval is not supporting evidence.", "uploaded_approval_record")
        approval = approval_rows[0]["data"]
        record = {name: approval[name] for name in ("record_id", "transaction_id", "status", "reviewer", "recorded_at", "currency") if name in approval}
        if "amount" in approval:
            try:
                record["amount"] = number(approval["amount"])
            except ValueError:
                result["status"] = "INCOMPLETE"
                return stop("Uploaded approval amount is invalid.", "valid_uploaded_approval.amount")
        record["source"] = approval_rows[0]["reference"]
        for name, value in evidence.get("records", {}).get("approval", {}).items():
            matches_value = value == record.get(name)
            if name == "status" and isinstance(value, str) and isinstance(record.get(name), str):
                matches_value = value.strip().casefold() == record[name].strip().casefold()
            if name != "source" and name in record and not matches_value:
                result["contradictions"].append(f"Supplied approval {name} differs from the uploaded approval record.")
        evidence["records"] = {"approval": record}
        control = {"title": data.get("control_name", control_id), "applicability": "amount_above_approval_threshold",
                   "required_fields": ["amount", "transaction_id", "currency", "designated_finance_manager"],
                   "required_records": {"approval": {
                       "required_fields": ["record_id", "transaction_id", "amount", "status", "reviewer", "recorded_at", "source", "currency"],
                       "match_fields": {"transaction_id": "transaction_id", "amount": "amount", "currency": "currency", "reviewer": "designated_finance_manager"},
                       "expected_values": {"status": "Approved"}}}}
        result["rules"] = {"approval_threshold": threshold, "default_control_id": result["control_id"],
                           "description": data["control_description"], "controls": {result["control_id"]: control}}
        result["status"] = "CONTRADICTORY" if result["contradictions"] else "APPLICABLE"
        return result

    def risk(self, state):
        control_id = state.get("policy_control_result", {}).get("control_id")
        evidence = state.get("policy_control_result", {}).get("evidence", {})
        findings = [r for r in self.rows if r["data"].get("related_control_id") == control_id
                    and r["data"].get("finding_id") and r["data"].get("severity")]
        if evidence.get("entity_id"):
            findings = [r for r in findings if r["data"].get("entity_id") == evidence["entity_id"]]
        sources = [self.source(r, "uploaded_finding") for r in findings]
        rating = "NOT_ASSESSED"
        reason = "No matching uploaded risk rule supports a transaction rating. Recorded finding severities are historical document facts, not a new transaction rating."
        rules = self.latest([r for r in self.rows if r["data"].get("record_type", "").lower() == "risk_rule"
                             and r["data"].get("control_id") == control_id])
        status = state.get("evidence_assessment", {}).get("status")
        criticality = evidence.get("criticality", "").lower()
        matching = [r for r in rules if r["data"].get("evidence_status") == status
                    and r["data"].get("criticality", "").lower() == criticality and criticality]
        if status not in {"NOT_ASSESSED", "CONTRADICTORY"} and len(matching) == 1:
            row = matching[0]
            if (set(row["data"]) <= {"record_type", "control_id", "evidence_status", "criticality", "risk_rating"}
                    and row["data"].get("risk_rating", "").upper() in {"LOW", "MEDIUM", "HIGH"}):
                rating = row["data"]["risk_rating"].upper()
                sources.append(self.source(row, "uploaded_risk_rule"))
                reason = f"Applied {row['reference']} to evidence status {status} and uploaded transaction criticality {criticality}."
        supplied = state.get("risk_factors", {})
        if supplied.get("exception"):
            rating = "NOT_ASSESSED"
            reason = "A user-reported exception needs documentary reconciliation before a transaction rating can be assigned."
        elif supplied.get("criticality") and criticality and supplied["criticality"].lower() != criticality:
            rating = "NOT_ASSESSED"
            reason = "Entered criticality conflicts with the uploaded transaction. Resolve the discrepancy before assigning risk."
        return {"risk_result": rating, "risk_reason": reason,
                "documented_risks": [{"finding_id": r["data"]["finding_id"], "severity": r["data"]["severity"],
                                      "status": r["data"].get("status"), "description": r["data"].get("finding_description"),
                                      "entity_id": r["data"].get("entity_id"), "source": r["reference"]} for r in findings],
                "risk_sources": sources, "agent_trace": state.get("agent_trace", []) + ["Risk"]}
