from copy import deepcopy
from datetime import datetime
import math
from typing import Any, Literal, TypedDict

from langgraph.graph import StateGraph, END
from knowledge import answer_from_documents, control_rules


class EvidenceAssessment(TypedDict):
    status: Literal["SUFFICIENT", "INSUFFICIENT", "INCOMPLETE", "CONTRADICTORY", "NOT_ASSESSED"]
    completeness: Literal["COMPLETE", "INCOMPLETE", "NOT_ASSESSED"]
    control_id: str | None
    requirement: dict[str, Any]
    findings: list[str]
    missing_evidence: list[str]
    contradictions: list[str]
    evaluated_fields: dict[str, Any]
    sources: list[dict[str, str]]
    limitations: list[str]
    exception_detected: bool
    human_review_required: bool


class AgentState(TypedDict, total=False):
    question: str
    evidence: dict[str, Any]
    control_requirement: str
    control_id: str
    risk_factors: dict[str, Any]
    rag_result: list[dict[str, Any]]
    evidence_result: list[str]
    risk_result: str
    final_report: dict[str, Any]
    evidence_status: str
    evidence_assessment: EvidenceAssessment
    risk_reason: str
    agent_trace: list[str]
    answer: str
    sources: list[dict[str, Any]]
    answer_mode: str
    policy_control_result: dict[str, Any]
    documented_risks: list[dict[str, Any]]
    risk_sources: list[dict[str, str]]


def validate_evidence_input(evidence, control_id=None, control_requirement=None):
    """Validate at both the HTTP boundary and the directly callable graph node."""
    if not isinstance(evidence, dict):
        raise ValueError("evidence must be an object.")
    for name, value in (("control_id", control_id), ("control_requirement", control_requirement)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a non-empty string.")

    def fields(values, prefix):
        for name, value in values.items():
            path = f"{prefix}.{name}"
            if name == "amount":
                try:
                    finite = isinstance(value, (int, float)) and math.isfinite(value)
                except OverflowError:
                    finite = False
                if isinstance(value, bool) or not finite or value < 0:
                    raise ValueError(f"{path} must be a non-negative finite number.")
            elif name == "approval":
                if not isinstance(value, bool):
                    raise ValueError(f"{path} must be a boolean.")
            elif name in {"status", "approval_status"}:
                if not isinstance(value, str) or value.strip().lower() not in {"approved", "rejected", "pending"}:
                    raise ValueError(f"{path} must be Approved, Rejected, or Pending.")
            elif name in {"transaction_id", "record_id", "reviewer", "source", "recorded_at", "control_id",
                          "currency", "transaction_type", "designated_finance_manager", "entity_id", "criticality"}:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{path} must be a non-empty string; omit unavailable fields.")
                if name == "recorded_at":
                    try:
                        date = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
                        if date.tzinfo is None:
                            raise ValueError()
                    except ValueError:
                        raise ValueError(f"{path} must be an ISO 8601 timestamp with timezone, e.g. 2026-09-01T10:00:00Z.") from None

    fields(evidence, "evidence")
    records = evidence.get("records", {})
    if not isinstance(records, dict):
        raise ValueError("evidence.records must be an object keyed by record type, e.g. approval.")
    for name, record in records.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(record, dict):
            raise ValueError("Each evidence.records entry must have a non-empty name and an object value.")
        fields(record, f"evidence.records.{name}")


def _configured_control(rules, control_id):
    """Fail closed when rules are missing, incomplete, or use unsupported checks."""
    controls = rules.get("controls", {})
    control = controls.get(control_id) if isinstance(controls, dict) and isinstance(control_id, str) else None
    if not isinstance(control, dict):
        return None, "No configured evidence requirements were found for the selected control. Map an approved control before assessing it."

    def field_list(value):
        return isinstance(value, list) and bool(value) and all(isinstance(item, str) and item.strip() for item in value)

    if (set(control) - {"title", "applicability", "required_fields", "required_records"}
            or not isinstance(control.get("title"), str) or not control["title"].strip()
            or not isinstance(control.get("applicability"), str)
            or control["applicability"] not in {"always", "amount_above_approval_threshold"}
            or not field_list(control.get("required_fields"))
            or not isinstance(control.get("required_records"), dict) or not control["required_records"]):
        return None, "Control configuration is incomplete or unsupported. Check title, applicability, required_fields, and required_records in knowledge/control_rules.json."
    if control["applicability"] == "amount_above_approval_threshold":
        threshold = rules.get("approval_threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
            return None, "The configured approval_threshold must be a non-negative finite number."
    for name, record in control["required_records"].items():
        if (not isinstance(name, str) or not name.strip() or not isinstance(record, dict)
                or set(record) - {"required_fields", "match_fields", "expected_values"}
                or not field_list(record.get("required_fields"))
                or not {"record_id", "source"} <= set(record["required_fields"])
                or not isinstance(record.get("match_fields"), dict) or not record["match_fields"]
                or not isinstance(record.get("expected_values"), dict) or not record["expected_values"]):
            return None, "Each configured record needs required_fields (including record_id and source), match_fields, and expected_values."
        for key, target in record["match_fields"].items():
            if not isinstance(target, str) or key not in record["required_fields"] or target not in control["required_fields"]:
                return None, "Record match_fields must link required record fields to required transaction/control fields."
        for key, value in record["expected_values"].items():
            if key not in record["required_fields"] or not isinstance(value, (str, bool, int, float)):
                return None, "Record expected_values must specify scalar values for required fields."
            if (isinstance(value, str) and not value.strip()) or (isinstance(value, float) and not math.isfinite(value)):
                return None, "Record expected_values cannot contain blank strings or non-finite numbers."
    return control, None


def evidence_agent(state):
    document_policy = state.get("policy_control_result")
    evidence = document_policy["evidence"] if document_policy is not None else state.get("evidence", {})
    validation_issue = None
    try:
        validate_evidence_input(evidence, state.get("control_id"), state.get("control_requirement"))
    except ValueError as exc:
        if document_policy is None:
            raise
        validation_issue = str(exc)
    if document_policy is not None:
        rules = document_policy["rules"]
    else:
        try:
            rules = control_rules()
        except (OSError, ValueError):
            rules = {}
    if not isinstance(rules, dict):
        rules = {}
    control_id = document_policy["control_id"] if document_policy is not None else state.get("control_id", rules.get("default_control_id"))
    control_id = control_id.strip() if isinstance(control_id, str) else None
    control, unavailable = _configured_control(rules, control_id)
    assessment: EvidenceAssessment = {
        "status": "NOT_ASSESSED", "completeness": "NOT_ASSESSED", "control_id": control_id,
        "requirement": {}, "findings": [], "missing_evidence": [], "contradictions": [],
        "evaluated_fields": {}, "sources": [], "limitations": [
            "Checks cover supplied structured fields only. Source references and reviewer identity are not independently authenticated; human verification is required.",
            "Retrieved passages are reference context, not verified transaction evidence or automatically executable policy."
            , "Reviewer authority, transaction timing, population coverage, and overall control effectiveness are not checked."
        ], "exception_detected": False, "human_review_required": True,
    }
    findings, missing = assessment["findings"], assessment["missing_evidence"]

    def finish(status):
        assessment["status"] = status
        # Retain the public v1 string/list fields; richer status lives in the additive object.
        legacy = "EXCEPTION" if assessment["exception_detected"] else (
            "PASS" if status == "SUFFICIENT" else "INSUFFICIENT_EVIDENCE")
        return {"evidence_result": findings, "evidence_status": legacy, "evidence_assessment": assessment,
            "agent_trace": state.get("agent_trace", []) + ["Evidence"]}

    if document_policy is not None:
        assessment["sources"] = deepcopy(document_policy["sources"])
        assessment["limitations"].extend(document_policy["limitations"])
        findings.extend(document_policy["findings"])
        missing.extend(document_policy["missing_evidence"])
        assessment["contradictions"].extend(document_policy["contradictions"])
        if validation_issue:
            findings.append("Uploaded evidence is invalid: " + validation_issue)
            missing.append("valid_uploaded_evidence")
            assessment["completeness"] = "INCOMPLETE"
            return finish("INCOMPLETE")
        if document_policy["status"] != "APPLICABLE":
            findings.extend(assessment["contradictions"])
            assessment["completeness"] = "INCOMPLETE" if missing else "NOT_ASSESSED"
            return finish(document_policy["status"])
    if unavailable:
        findings.append(unavailable)
        missing.append("applicable_control_requirements")
        return finish("NOT_ASSESSED")
    assessment["requirement"] = {**deepcopy(control), "approval_threshold": rules.get("approval_threshold"),
                                 "basis": rules.get("description", "Configured evidence requirements")}
    if document_policy is None:
        assessment["sources"].append({"kind": "control_configuration", "source": "knowledge/control_rules.json",
                                      "reference": f"controls.{control_id}"})
    if state.get("control_requirement") and document_policy is None:
        findings.append("The supplied free-text control requirement has no verified mapping to executable checks. Human policy/control mapping is required.")
        missing.append("mapped_control_requirement")
        return finish("NOT_ASSESSED")

    def present(value):
        return value is not None and (not isinstance(value, (str, list, dict)) or bool(value)) and (
            not isinstance(value, str) or bool(value.strip()))

    def read(values, key, prefix, required=True):
        path = f"{prefix}.{key}"
        value = values.get(key)
        if present(value):
            if not isinstance(value, (str, bool, int, float)) or (isinstance(value, float) and not math.isfinite(value)):
                raise ValueError(f"{path} must be a finite scalar value for configured evidence checks.")
            assessment["evaluated_fields"][path] = value
        elif required and path not in missing:
            missing.append(path)
        return value

    if control["applicability"] == "amount_above_approval_threshold":
        amount = read(evidence, "amount", "evidence")
        if amount is None:
            assessment["completeness"] = "INCOMPLETE"
            findings.append("Transaction amount is missing; the configured approval rule's applicability cannot be determined.")
            return finish("INCOMPLETE")
        if amount <= rules["approval_threshold"]:
            findings.append("The amount does not exceed the configured approval threshold. No applicable evidence check is configured for this transaction.")
            return finish("NOT_ASSESSED")

    for key in control["required_fields"]:
        read(evidence, key, "evidence")
    records = evidence.get("records", {})
    unmet = []
    if document_policy is not None and document_policy.get("timing_event"):
        event = document_policy["timing_event"]
        try:
            event_time = datetime.fromisoformat(evidence[event].replace("Z", "+00:00"))
            approval_time = datetime.fromisoformat(records["approval"]["recorded_at"].replace("Z", "+00:00"))
            if event_time.tzinfo is None or approval_time.tzinfo is None:
                raise ValueError("Timezone required")
            if approval_time >= event_time:
                unmet.append(f"The uploaded approval record is not before {event}; it does not satisfy the uploaded requirement.")
        except (KeyError, ValueError, TypeError):
            missing.append("valid_uploaded_approval_and_" + event)
    if "control_id" in evidence:
        read(evidence, "control_id", "evidence")
        if evidence["control_id"].strip() != control_id:
            assessment["contradictions"].append("evidence.control_id contradicts the selected control_id.")

    def equal(left, right):
        if isinstance(left, str) and isinstance(right, str):
            return left.strip() == right.strip()
        if isinstance(left, bool) != isinstance(right, bool):
            return False
        return left == right

    for name, spec in control["required_records"].items():
        record = records.get(name, {})
        prefix = f"evidence.records.{name}"
        for key in spec["required_fields"]:
            read(record, key, prefix)
        if record.get("source"):
            assessment["sources"].append({"kind": "submitted_record", "source": record["source"],
                                          "reference": f"{prefix}.source"})
        if "control_id" in record:
            read(record, "control_id", prefix)
            if record["control_id"].strip() != control_id:
                assessment["contradictions"].append(f"{prefix}.control_id contradicts the selected control_id.")
        for key, target in spec["match_fields"].items():
            if present(record.get(key)) and present(evidence.get(target)) and not equal(record[key], evidence[target]):
                assessment["contradictions"].append(f"{prefix}.{key} contradicts evidence.{target}; the record does not match the supplied evidence.")
        for key, expected in spec["expected_values"].items():
            value = record.get(key)
            matches = (isinstance(value, str) and isinstance(expected, str) and
                       value.strip().lower() == expected.strip().lower()) if key == "status" else equal(value, expected)
            if present(value) and not matches:
                unmet.append(f"{prefix}.{key} does not satisfy the configured expected value {expected!r}.")

    # Compare all supplied approval representations; never infer a record from a claim.
    approval_spec = control["required_records"].get("approval", {})
    expected_approval = approval_spec.get("expected_values", {}).get("status")
    if isinstance(expected_approval, str) and expected_approval.strip().lower() == "approved":
        claims = {}
        if "approval" in evidence:
            read(evidence, "approval", "evidence")
            claims["evidence.approval"] = "approved" if evidence["approval"] else "rejected"
        if "approval_status" in evidence:
            read(evidence, "approval_status", "evidence")
            claims["evidence.approval_status"] = evidence["approval_status"].strip().lower()
        record_status = records.get("approval", {}).get("status")
        if record_status:
            claims["evidence.records.approval.status"] = record_status.strip().lower()
        if len(set(claims.values())) > 1:
            assessment["contradictions"].append("Approval representations disagree: " + ", ".join(f"{key}={value}" for key, value in claims.items()) + ".")
        elif "rejected" in claims.values():
            assessment["exception_detected"] = True
            unmet.append("Approval is explicitly reported as rejected/not approved. This is a reported exception, not independently verified control failure.")
        elif "pending" in claims.values():
            unmet.append("Approval is pending; the approval requirement is not yet satisfied.")

    assessment["completeness"] = "INCOMPLETE" if missing else "COMPLETE"
    if missing:
        findings.append("Missing required evidence: " + ", ".join(missing) + ".")
    findings.extend(unmet)
    findings.extend(assessment["contradictions"])
    if assessment["contradictions"]:
        assessment["exception_detected"] = False
        return finish("CONTRADICTORY")
    if unmet:
        return finish("INSUFFICIENT")
    if missing:
        return finish("INCOMPLETE")
    findings.append("Supplied evidence is complete, consistent, and satisfies the configured field checks. Sufficiency is limited to these checks and remains subject to source verification and human review.")
    return finish("SUFFICIENT")


def risk_agent(state):
    factors = state.get("risk_factors", {})
    criticality = str(factors.get("criticality", "")).lower()
    exception = factors.get("exception", False) or state.get("evidence_status") == "EXCEPTION"
    if exception:
        score = "HIGH" if criticality == "high" else "MEDIUM"
        reason = "An exception was reported or detected by the evidence agent."
    elif (state.get("evidence_status") == "INSUFFICIENT_EVIDENCE" or not criticality
          or ("evidence_assessment" in state and state["evidence_assessment"].get("status") != "SUFFICIENT")):
        score = "NOT_ASSESSED"
        reason = "Provide sufficient evidence and risk criticality before assigning a risk rating."
    else:
        score = "LOW"
        reason = "No exception detected by the configured rule for the supplied inputs."
    return {"risk_result": score, "risk_reason": reason,
            "agent_trace": state.get("agent_trace", []) + ["Risk"]}


def review_agent(state):
    return {"agent_trace": state.get("agent_trace", []) + ["Review"], "final_report": {
        "question": state.get("question"),
        "rag": state.get("rag_result", []),
        "evidence": state.get("evidence_result", []),
        "risk": state.get("risk_result"),
        "evidence_status": state.get("evidence_status"),
        "evidence_assessment": state.get("evidence_assessment", {}),
        "policy_control": state.get("policy_control_result", {}),
        "documented_risks": state.get("documented_risks", []),
        "risk_sources": state.get("risk_sources", []),
        "supplied_risk_factors": state.get("risk_factors", {}),
        "risk_reason": state.get("risk_reason"),
        "answer": state.get("answer"),
        "sources": state.get("sources", []),
        "assessment_basis": state.get("evidence_assessment", {}).get("requirement", {}).get("basis", "No applicable mapped control requirements."),
        "control_requirement": state.get("control_requirement", ""),
        "assessment_scope": "Only explicitly configured evidence fields and record checks are evaluated. Source authenticity, broader control effectiveness, and free-text requirements need human review.",
    }}


def build_workflow(retriever, uploaded_knowledge=None):
    def rag_agent(state):
        query = state["question"] + (" " + state["control_id"] if state.get("control_id") else "")
        docs = retriever.invoke(query)
        return {**answer_from_documents(docs), "agent_trace": ["RAG"], "rag_result": [
            {"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs
        ]}

    workflow = StateGraph(AgentState)
    workflow.add_node("RAG", rag_agent)
    workflow.add_node("Evidence", evidence_agent)
    workflow.add_node("Risk", uploaded_knowledge.risk if uploaded_knowledge is not None else risk_agent)
    workflow.add_node("Review", review_agent)
    workflow.set_entry_point("RAG")
    if uploaded_knowledge is not None:
        def policy_control_agent(state):
            return {"policy_control_result": uploaded_knowledge.resolve(state),
                    "agent_trace": state.get("agent_trace", []) + ["Policy & Control"]}
        workflow.add_node("Policy & Control", policy_control_agent)
        workflow.add_edge("RAG", "Policy & Control")
        workflow.add_edge("Policy & Control", "Evidence")
    else:
        workflow.add_edge("RAG", "Evidence")
    workflow.add_edge("Evidence", "Risk")
    workflow.add_edge("Risk", "Review")
    workflow.add_edge("Review", END)
    return workflow.compile()
