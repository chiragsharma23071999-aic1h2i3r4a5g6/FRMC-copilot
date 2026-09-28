import unittest
from unittest.mock import patch

from agents import build_workflow
from rag_pipeline import ProjectRetriever
from knowledge import answer_from_documents


class KnowledgeAgentTests(unittest.TestCase):
    def test_fundamental_answers_offline(self):
        cases = {
            "What is SOX?": "Sarbanes-Oxley",
            "What is section 302?": "certifications",
            "What is SOX Section 404?": "404(a)",
            "What is COSO?": "Treadway",
            "What are the five COSO components?": "Monitoring Activities",
            "Explain the 17 COSO principles": "accountability",
            "What is ICFR?": "financial reporting",
            "Difference between SOX and COSO?": "legal",
            "What is a material weakness?": "reasonable possibility",
            "Does SOX 404(b) apply to every company?": "Exemptions",
        }
        with patch("rag_pipeline.get_embeddings", side_effect=AssertionError("Must work offline")):
            for question, expected in cases.items():
                with self.subTest(question=question):
                    answer = answer_from_documents(ProjectRetriever().invoke(question))
                    self.assertIn(expected, answer["answer"])
                    self.assertTrue(answer["sources"][0]["url"].startswith("https://"))

    def test_unsupported_question(self):
        with patch("rag_pipeline.CHROMA_DIR", "nonexistent-test-index"):
            result = answer_from_documents(ProjectRetriever().invoke("Explain SOX section 999"))
        self.assertEqual(result["answer_mode"], "insufficient_knowledge")

    def run_workflow(self, evidence, factors):
        return build_workflow(ProjectRetriever()).invoke({
            "question": "What is SOX?", "evidence": evidence, "risk_factors": factors,
        })

    def test_evidence_exception_drives_risk(self):
        result = self.run_workflow({"amount": 600000, "approval": False}, {"criticality": "High"})
        self.assertEqual(result["evidence_status"], "EXCEPTION")
        self.assertEqual(result["risk_result"], "HIGH")
        self.assertEqual(result["agent_trace"], ["RAG", "Evidence", "Risk", "Review"])
        self.assertIn("Sarbanes-Oxley", result["final_report"]["answer"])

    def test_approved_claim_requires_supporting_record(self):
        result = self.run_workflow({"amount": 600000, "approval": True}, {"criticality": "High"})
        self.assertEqual(result["evidence_status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["evidence_assessment"]["status"], "INCOMPLETE")
        self.assertEqual(result["risk_result"], "NOT_ASSESSED")

    def test_missing_evidence_is_not_low_risk(self):
        for evidence in ({}, {"amount": 600000}):
            result = self.run_workflow(evidence, {"criticality": "High"})
            self.assertEqual(result["evidence_status"], "INSUFFICIENT_EVIDENCE")
            self.assertEqual(result["risk_result"], "NOT_ASSESSED")

    def test_threshold_boundary_and_medium_risk(self):
        result = self.run_workflow({"amount": 500000}, {"criticality": "Low"})
        self.assertEqual(result["evidence_assessment"]["status"], "NOT_ASSESSED")
        self.assertEqual(result["risk_result"], "NOT_ASSESSED")
        result = self.run_workflow({"amount": 500001, "approval": False}, {"criticality": "Low"})
        self.assertEqual(result["risk_result"], "MEDIUM")

    def test_explicit_exception_is_preserved(self):
        result = self.run_workflow({"amount": 100}, {"criticality": "High", "exception": True})
        self.assertEqual(result["risk_result"], "HIGH")


if __name__ == "__main__":
    unittest.main()
