"""Real uploaded rows drive policy, evidence, risk and reports without a model call."""
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import app
from agents import build_workflow
from models import Base
from uploaded_knowledge import UploadedKnowledge


class UploadedAgentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.control = {"Control ID": "FRMC-001", "Control Name": "Journal Entry Approval", "Status": "Active",
                        "Control Description": "All manual journal entries above USD 10000 require approval from the preparer's designated finance manager before posting"}
        self.transaction = {"Control ID": "FRMC-001", "Record Type": "transaction", "Transaction ID": "TX-1",
                            "Amount": "15000", "Currency": "USD", "Transaction Type": "manual journal entries",
                            "Designated Finance Manager": "Reviewer A", "Posting At": "2026-09-02T12:00:00Z",
                            "Criticality": "High", "Entity ID": "ENT-1"}
        self.approval = {"Control ID": "FRMC-001", "Record Type": "approval", "Transaction ID": "TX-1",
                         "Amount": "15000", "Currency": "USD", "Record ID": "APR-1", "Status": "Approved",
                         "Reviewer": "Reviewer A", "Recorded At": "2026-09-01T12:00:00Z"}
        self.write("controls.csv", [self.control])
        self.write("evidence.csv", [self.transaction, self.approval])

    def write(self, name, rows):
        with (self.folder / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)

    def run_agents(self, **extra):
        uploaded = UploadedKnowledge(self.folder)
        with patch("agents.control_rules", side_effect=AssertionError("Must not use illustrative rules")):
            result = build_workflow(uploaded, uploaded_knowledge=uploaded).invoke({
                "question": "Assess FRMC-001", "evidence": {"transaction_id": "TX-1"}, **extra})
        self.assertEqual(result["agent_trace"], ["RAG", "Policy & Control", "Evidence", "Risk", "Review"])
        self.assertEqual(result["final_report"]["evidence_assessment"], result["evidence_assessment"])
        self.assertEqual(result["final_report"]["policy_control"], result["policy_control_result"])
        json.dumps(result)
        return result

    def test_actual_control_grammar_and_uploaded_evidence(self):
        result = self.run_agents()
        self.assertEqual(result["evidence_assessment"]["status"], "SUFFICIENT", result)
        self.assertEqual(result["evidence_assessment"]["requirement"]["approval_threshold"], 10000)
        self.assertEqual(result["evidence_assessment"]["missing_evidence"], [])
        self.assertEqual(result["risk_result"], "NOT_ASSESSED")
        self.assertTrue(any(s["source"] == "controls.csv" for s in result["evidence_assessment"]["sources"]))
        self.assertTrue(any(s["source"] == "evidence.csv" for s in result["evidence_assessment"]["sources"]))

    def test_latest_definition_and_removal(self):
        self.write("new-controls.csv", [{**self.control, "Control Description": self.control["Control Description"].replace("10000", "20000")}])
        old_time = (self.folder / "controls.csv").stat().st_mtime_ns
        os.utime(self.folder / "new-controls.csv", ns=(old_time + 1000000000, old_time + 1000000000))
        result = self.run_agents()
        self.assertEqual(result["evidence_assessment"]["status"], "NOT_ASSESSED")
        self.assertEqual(result["policy_control_result"]["threshold"], 20000)
        refs = UploadedKnowledge(self.folder).invoke("FRMC-001 journal entry approval")
        self.assertFalse(any(d.metadata["source"] == "controls.csv" for d in refs))
        (self.folder / "new-controls.csv").unlink()
        self.assertEqual(self.run_agents()["evidence_assessment"]["status"], "SUFFICIENT")
        (self.folder / "evidence.csv").unlink()
        result = self.run_agents()
        self.assertEqual(result["evidence_assessment"]["status"], "INCOMPLETE")
        self.assertIn("uploaded_transaction_record", result["evidence_assessment"]["missing_evidence"])

    def test_missing_approval_cannot_be_replaced_by_a_claim(self):
        self.write("evidence.csv", [self.transaction])
        result = self.run_agents(evidence={"transaction_id": "TX-1", "approval": True})
        self.assertEqual(result["evidence_assessment"]["status"], "INCOMPLETE")
        self.assertIn("uploaded_approval_record", result["evidence_assessment"]["missing_evidence"])

    def test_threshold_boundaries(self):
        for amount, status in ((9999, "NOT_ASSESSED"), (10000, "NOT_ASSESSED"), (10001, "SUFFICIENT")):
            with self.subTest(amount=amount):
                self.write("evidence.csv", [{**self.transaction, "Amount": str(amount)}, {**self.approval, "Amount": str(amount)}])
                self.assertEqual(self.run_agents()["evidence_assessment"]["status"], status)

    def test_rejection_and_pending_not_automatic_risk_ratings(self):
        for status in ("Rejected", "Pending"):
            with self.subTest(status=status):
                self.write("evidence.csv", [self.transaction, {**self.approval, "Status": status}])
                result = self.run_agents(risk_factors={"criticality": "High", "exception": True})
                self.assertEqual(result["evidence_assessment"]["status"], "INSUFFICIENT")
                self.assertEqual(result["risk_result"], "NOT_ASSESSED")

    def test_mismatched_claim_and_duplicate_records(self):
        result = self.run_agents(evidence={"transaction_id": "TX-1", "amount": 99999})
        self.assertEqual(result["evidence_assessment"]["status"], "CONTRADICTORY")
        self.assertTrue(result["evidence_assessment"]["contradictions"])
        self.write("evidence.csv", [self.transaction, self.approval, {**self.approval, "Status": "Rejected"}])
        self.assertEqual(self.run_agents()["evidence_assessment"]["status"], "CONTRADICTORY")

    def test_unsupported_extra_policy_condition_is_not_ignored(self):
        self.write("controls.csv", [{**self.control, "Control Description": self.control["Control Description"] + " and independent verification"}])
        result = self.run_agents()
        self.assertEqual(result["evidence_assessment"]["status"], "NOT_ASSESSED")
        self.assertIn("supported_executable_requirement", result["evidence_assessment"]["missing_evidence"])

    def test_scope_and_approval_timing(self):
        self.write("evidence.csv", [{**self.transaction, "Currency": "EUR"}, self.approval])
        self.assertEqual(self.run_agents()["evidence_assessment"]["status"], "NOT_ASSESSED")
        self.write("evidence.csv", [self.transaction, {**self.approval, "Recorded At": "2026-09-03T12:00:00Z"}])
        result = self.run_agents()
        self.assertEqual(result["evidence_assessment"]["status"], "INSUFFICIENT")
        self.assertTrue(any("not before" in f for f in result["evidence_result"]))

    def test_risk_rule_and_historical_findings_are_distinct(self):
        self.write("risk.csv", [{"Control ID": "FRMC-001", "Record Type": "risk_rule", "Evidence Status": "SUFFICIENT",
                                 "Criticality": "High", "Risk Rating": "Low"}])
        self.write("findings.csv", [{"Related Control ID": "FRMC-001", "Finding ID": "AUD-1", "Severity": "High",
                                      "Entity ID": "ENT-1", "Status": "Closed", "Finding Description": "Historical approval gap"}])
        result = self.run_agents()
        self.assertEqual(result["risk_result"], "LOW")
        self.assertEqual(result["documented_risks"][0]["severity"], "High")
        self.assertTrue(any(s["source"] == "risk.csv" for s in result["risk_sources"]))

    def test_no_cross_transaction_evidence_and_builtin_fundamentals(self):
        result = self.run_agents(evidence={"transaction_id": "OTHER"})
        self.assertEqual(result["evidence_assessment"]["status"], "INCOMPLETE")
        docs = UploadedKnowledge(self.folder).invoke("What is SOX?")
        self.assertTrue(docs[0].metadata["source"].startswith("builtin:"))

    def test_api_and_report_use_document_graph(self):
        engine = create_engine("sqlite://")
        self.addCleanup(engine.dispose)
        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with patch.dict(app.app.config, UPLOAD_FOLDER=str(self.folder)), patch("backend.SessionLocal", factory), patch("audit_controller.SessionLocal", factory):
            client = app.app.test_client()
            body = {"question": "Assess FRMC-001", "evidence": {"transaction_id": "TX-1"}}
            result = client.post("/run_agents", json=body)
            self.assertEqual(result.status_code, 200, result.json)
            self.assertEqual(result.json["evidence_assessment"]["status"], "SUFFICIENT")
            report = client.post("/generate_report", json=body)
            self.assertEqual(report.status_code, 201)
            saved = app.report_service.export_report(report.json["id"])
            self.assertEqual(json.loads(saved.content), report.json["final_report"])
            self.assertEqual(len(client.get("/logs").json), 2)

    def test_unknown_control_never_uses_configured_default(self):
        result = self.run_agents(control_id="configured-approval")
        self.assertEqual(result["evidence_assessment"]["status"], "NOT_ASSESSED")
        self.assertEqual(result["risk_result"], "NOT_ASSESSED")

    def test_newer_unstructured_policy_needs_mapping(self):
        text = self.folder / "updated-policy.txt"
        text.write_text("FRMC-001 now requires two independent finance approvals.", encoding="utf-8")
        later = (self.folder / "controls.csv").stat().st_mtime_ns + 1000000000
        os.utime(text, ns=(later, later))
        result = self.run_agents()
        self.assertEqual(result["evidence_assessment"]["status"], "NOT_ASSESSED")
        self.assertIn("mapped_newer_policy", result["evidence_assessment"]["missing_evidence"])
        self.assertTrue(any(s["source"] == text.name for s in result["evidence_assessment"]["sources"]))

    def test_entered_risk_claim_cannot_silently_conflict_with_documents(self):
        self.write("risk.csv", [{"Control ID": "FRMC-001", "Record Type": "risk_rule", "Evidence Status": "SUFFICIENT",
                                 "Criticality": "High", "Risk Rating": "Low"}])
        for factors in ({"exception": True}, {"criticality": "Low"}):
            with self.subTest(factors=factors):
                result = self.run_agents(risk_factors=factors)
                self.assertEqual(result["risk_result"], "NOT_ASSESSED")
                self.assertEqual(result["final_report"]["supplied_risk_factors"], factors)
