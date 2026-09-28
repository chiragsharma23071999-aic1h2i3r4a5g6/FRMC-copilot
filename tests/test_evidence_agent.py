"""Evidence scenarios exercise the real graph with isolated synthetic records."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from agents import build_workflow, evidence_agent, risk_agent
from knowledge import control_rules


def complete_evidence(amount=None):
    amount = control_rules()["approval_threshold"] + 1 if amount is None else amount
    return {"transaction_id": "TX-1", "amount": amount, "approval": True,
            "records": {"approval": {
                "record_id": "APP-1", "transaction_id": "TX-1", "amount": amount,
                "status": "Approved", "reviewer": "Synthetic reviewer",
                "recorded_at": "2026-09-01T10:00:00Z", "source": "fixture.csv#APP-1",
            }}}


class ReferenceRetriever:
    def invoke(self, question):
        return [Document(page_content="Approval needs supporting evidence.",
                         metadata={"source": "fixture-policy.txt"})]


class EvidenceAgentTests(unittest.TestCase):
    def run_case(self, evidence, status, risk, **extra):
        result = build_workflow(ReferenceRetriever()).invoke({
            "question": "Assess this transaction", "evidence": evidence,
            "risk_factors": {"criticality": "High"}, **extra,
        })
        assessment = result["evidence_assessment"]
        self.assertEqual(assessment["status"], status, assessment)
        self.assertEqual(result["risk_result"], risk)
        self.assertEqual(result["agent_trace"], ["RAG", "Evidence", "Risk", "Review"])
        self.assertEqual(result["final_report"]["evidence_assessment"], assessment)
        self.assertEqual(result["evidence_result"], assessment["findings"])
        self.assertEqual(result["final_report"]["evidence"], assessment["findings"])
        self.assertTrue(assessment["findings"])
        self.assertIsInstance(assessment["missing_evidence"], list)
        self.assertTrue(assessment["human_review_required"])
        self.assertTrue(assessment["limitations"])
        self.assertEqual(result["sources"][0]["source"], "fixture-policy.txt")
        self.assertNotIn("fixture-policy.txt", [source["source"] for source in assessment["sources"]])
        json.dumps(result)
        return assessment

    def test_complete_matching_approval_evidence(self):
        evidence = complete_evidence()
        original = deepcopy(evidence)
        result = self.run_case(evidence, "SUFFICIENT", "LOW")
        self.assertEqual(result["completeness"], "COMPLETE")
        self.assertEqual(result["missing_evidence"], [])
        self.assertEqual(result["contradictions"], [])
        self.assertFalse(result["exception_detected"])
        self.assertIn("human review", result["findings"][0])
        self.assertEqual(result["sources"][1]["source"], "fixture.csv#APP-1")
        self.assertEqual(result["evaluated_fields"]["evidence.records.approval.transaction_id"], "TX-1")
        self.assertEqual(evidence, original)

    def test_approved_status_without_record(self):
        evidence = complete_evidence()
        del evidence["records"]
        del evidence["approval"]
        evidence["approval_status"] = "Approved"
        result = self.run_case(evidence, "INCOMPLETE", "NOT_ASSESSED")
        self.assertIn("evidence.records.approval.record_id", result["missing_evidence"])
        self.assertIn("evidence.records.approval.source", result["missing_evidence"])
        self.assertFalse(result["exception_detected"])
        self.assertEqual(result["contradictions"], [])

    def test_high_value_missing_approval_evidence(self):
        result = self.run_case({"transaction_id": "TX-1", "amount": control_rules()["approval_threshold"] + 1},
                               "INCOMPLETE", "NOT_ASSESSED")
        self.assertIn("evidence.records.approval.status", result["missing_evidence"])
        self.assertTrue(any("Missing required evidence" in item for item in result["findings"]))
        self.assertFalse(result["exception_detected"])

    def test_explicitly_rejected_approval(self):
        evidence = complete_evidence()
        evidence["approval"] = False
        evidence["records"]["approval"]["status"] = "Rejected"
        result = self.run_case(evidence, "INSUFFICIENT", "HIGH")
        self.assertEqual(result["completeness"], "COMPLETE")
        self.assertEqual(result["missing_evidence"], [])
        self.assertTrue(result["exception_detected"])
        self.assertTrue(any("reported exception" in item for item in result["findings"]))

    def test_missing_amount_is_not_inferred_from_record(self):
        evidence = complete_evidence()
        del evidence["amount"]
        result = self.run_case(evidence, "INCOMPLETE", "NOT_ASSESSED")
        self.assertEqual(result["missing_evidence"], ["evidence.amount"])
        self.assertIn("applicability cannot be determined", result["findings"][0])
        self.assertFalse(result["exception_detected"])

    def test_missing_critical_record_fields(self):
        for field in ("record_id", "transaction_id", "amount", "status", "reviewer", "recorded_at", "source"):
            with self.subTest(field=field):
                evidence = complete_evidence()
                del evidence["records"]["approval"][field]
                result = self.run_case(evidence, "INCOMPLETE", "NOT_ASSESSED")
                self.assertEqual(result["missing_evidence"], [f"evidence.records.approval.{field}"])
                self.assertEqual(result["contradictions"], [])

    def test_contradictory_record_fields(self):
        for field, value in (("status", "Rejected"), ("transaction_id", "TX-OTHER"),
                             ("amount", complete_evidence()["amount"] + 1), ("control_id", "another-control")):
            with self.subTest(field=field):
                evidence = complete_evidence()
                evidence["records"]["approval"][field] = value
                result = self.run_case(evidence, "CONTRADICTORY", "NOT_ASSESSED")
                self.assertTrue(result["contradictions"])
                self.assertEqual(result["missing_evidence"], [])
                self.assertFalse(result["exception_detected"])

    def test_unknown_control_and_free_text_requirement(self):
        for extra, missing in (({"control_id": "FRMC-001"}, "applicable_control_requirements"),
                               ({"control_requirement": "Approval must precede posting."}, "mapped_control_requirement")):
            with self.subTest(extra=extra):
                result = self.run_case(complete_evidence(), "NOT_ASSESSED", "NOT_ASSESSED", **extra)
                self.assertEqual(result["missing_evidence"], [missing])
                self.assertFalse(result["exception_detected"])

    def test_legacy_approval_only_inputs(self):
        for approval, status, risk in ((True, "INCOMPLETE", "NOT_ASSESSED"), (False, "INSUFFICIENT", "HIGH")):
            with self.subTest(approval=approval):
                result = self.run_case({"amount": complete_evidence()["amount"], "approval": approval}, status, risk)
                self.assertIn("evidence.transaction_id", result["missing_evidence"])
                self.assertIn("evidence.records.approval.record_id", result["missing_evidence"])
                self.assertEqual(result["exception_detected"], not approval)

    def test_threshold_and_approval_matrix(self):
        threshold = control_rules()["approval_threshold"]
        for amount in (threshold - 1, threshold, threshold + 1):
            for mode in ("missing", "false", "complete", "record_only"):
                with self.subTest(amount=amount, mode=mode):
                    evidence = {"transaction_id": "TX-1", "amount": amount}
                    if mode == "false":
                        evidence["approval"] = False
                    elif mode in {"complete", "record_only"}:
                        evidence = complete_evidence(amount)
                        if mode == "record_only":
                            del evidence["approval"]
                    if amount <= threshold:
                        status, risk = "NOT_ASSESSED", "NOT_ASSESSED"
                    else:
                        status, risk = {"missing": ("INCOMPLETE", "NOT_ASSESSED"),
                                        "false": ("INSUFFICIENT", "HIGH"),
                                        "complete": ("SUFFICIENT", "LOW"),
                                        "record_only": ("SUFFICIENT", "LOW")}[mode]
                    result = self.run_case(evidence, status, risk)
                    self.assertEqual(result["exception_detected"], amount > threshold and mode == "false")
                    self.assertEqual(bool(result["missing_evidence"]), amount > threshold and mode in {"missing", "false"})
                    if amount <= threshold:
                        self.assertIn("does not exceed", result["findings"][0])

    def test_threshold_is_configured_not_hardcoded(self):
        rules = control_rules()
        rules["approval_threshold"] = 10
        with patch("agents.control_rules", return_value=rules):
            result = self.run_case(complete_evidence(11), "SUFFICIENT", "LOW")
            self.assertEqual(result["requirement"]["approval_threshold"], 10)
            self.run_case(complete_evidence(10), "NOT_ASSESSED", "NOT_ASSESSED")

    def test_always_applicable_rule_ignores_threshold(self):
        rules = control_rules()
        rules["controls"][rules["default_control_id"]]["applicability"] = "always"
        with patch("agents.control_rules", return_value=rules):
            self.run_case(complete_evidence(1), "SUFFICIENT", "LOW")
            result = self.run_case({"transaction_id": "TX-1", "amount": 1}, "INCOMPLETE", "NOT_ASSESSED")
            self.assertTrue(result["missing_evidence"])
            evidence = complete_evidence(1)
            evidence["approval"] = False
            evidence["records"]["approval"]["status"] = "Rejected"
            self.run_case(evidence, "INSUFFICIENT", "HIGH")

    def test_generic_record_requirement_without_approval_rule(self):
        rules = control_rules()
        # Test-only configuration; does not add a control to the application.
        rules["controls"]["fixture-callback"] = {
            "title": "Synthetic callback evidence", "applicability": "always",
            "required_fields": ["transaction_id"], "required_records": {"callback": {
                "required_fields": ["record_id", "source", "transaction_id", "completed"],
                "match_fields": {"transaction_id": "transaction_id"}, "expected_values": {"completed": True},
            }},
        }
        evidence = {"transaction_id": "CHANGE-1", "records": {"callback": {
            "record_id": "CALL-1", "source": "fixture-log", "transaction_id": "CHANGE-1", "completed": True,
        }}}
        with patch("agents.control_rules", return_value=rules):
            self.run_case(evidence, "SUFFICIENT", "LOW", control_id="fixture-callback")
            evidence["records"]["callback"]["completed"] = False
            result = self.run_case(evidence, "INSUFFICIENT", "NOT_ASSESSED", control_id="fixture-callback")
            self.assertEqual(result["missing_evidence"], [])
            self.assertFalse(result["exception_detected"])

    def test_pending_is_insufficient_not_an_exception(self):
        evidence = complete_evidence()
        del evidence["approval"]
        evidence["records"]["approval"]["status"] = "Pending"
        result = self.run_case(evidence, "INSUFFICIENT", "NOT_ASSESSED")
        self.assertEqual(result["missing_evidence"], [])
        self.assertFalse(result["exception_detected"])

    def test_conflicting_claims_without_record(self):
        evidence = {"amount": complete_evidence()["amount"], "approval": True, "approval_status": "Rejected"}
        result = self.run_case(evidence, "CONTRADICTORY", "NOT_ASSESSED")
        self.assertTrue(result["missing_evidence"])
        self.assertTrue(result["contradictions"])
        self.assertFalse(result["exception_detected"])

    def test_case_insensitive_configured_approval_decision(self):
        rules = control_rules()
        rules["controls"][rules["default_control_id"]]["required_records"]["approval"]["expected_values"]["status"] = " approved "
        evidence = complete_evidence()
        evidence["approval"] = False
        with patch("agents.control_rules", return_value=rules):
            result = self.run_case(evidence, "CONTRADICTORY", "NOT_ASSESSED")
        self.assertTrue(result["contradictions"])
        self.assertEqual(result["missing_evidence"], [])

    def test_configuration_errors_cannot_produce_positive_result(self):
        unsupported = control_rules()
        unsupported["controls"][unsupported["default_control_id"]]["unimplemented_check"] = True
        for configuration in ({}, [], {"approval_threshold": 500000}, unsupported):
            with self.subTest(configuration=configuration), patch("agents.control_rules", return_value=configuration):
                result = self.run_case(complete_evidence(), "NOT_ASSESSED", "NOT_ASSESSED")
                self.assertEqual(result["missing_evidence"], ["applicable_control_requirements"])
        with patch("agents.control_rules", side_effect=OSError("unavailable")):
            self.run_case(complete_evidence(), "NOT_ASSESSED", "NOT_ASSESSED")

    def test_risk_guard_and_explicit_exception_compatibility(self):
        for status in ("INCOMPLETE", "INSUFFICIENT", "CONTRADICTORY", "NOT_ASSESSED"):
            result = risk_agent({"evidence_status": "PASS", "evidence_assessment": {"status": status},
                                 "risk_factors": {"criticality": "High"}})
            self.assertEqual(result["risk_result"], "NOT_ASSESSED")
        self.run_case({}, "INCOMPLETE", "HIGH", risk_factors={"criticality": "High", "exception": True})

    def test_invalid_direct_inputs(self):
        for evidence in (None, [], {"amount": True}, {"amount": -1}, {"amount": float("nan")},
                         {"amount": float("inf")}, {"amount": "100"}, {"records": []},
                         {"records": {"approval": None}}, {"approval": "true"},
                         {"records": {"approval": {"recorded_at": "2026-09-01T10:00:00"}}}):
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                evidence_agent({"evidence": evidence})

    def test_extremely_large_numeric_amount_has_actionable_validation(self):
        with self.assertRaisesRegex(ValueError, "amount"):
            evidence_agent({"evidence": {"amount": 10 ** 400}})
