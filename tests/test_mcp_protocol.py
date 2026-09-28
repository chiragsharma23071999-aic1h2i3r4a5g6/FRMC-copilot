"""Exercise a real MCP subprocess over stdio, using a temporary database."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import anyio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


class MCPProtocolTests(unittest.TestCase):
    def test_protocol_and_application_integration(self):
        asyncio.run(self.check_protocol())

    async def check_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            params = StdioServerParameters(
                command=sys.executable,
                args=[str(ROOT / "mcp_server.py")],
                cwd=folder,
                env={**os.environ, "DATABASE_URL": "sqlite:///" + str(Path(folder) / "test.db"),
                     "OPENROUTER_API_KEY": "", "PYTHONIOENCODING": "utf-8"},
            )
            with anyio.fail_after(60):
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        initialized = await session.initialize()
                        self.assertEqual(initialized.serverInfo.name, "FRMC Copilot")
                        tools = await session.list_tools()
                        self.assertEqual({tool.name for tool in tools.tools}, {
                            "ask_policy", "assess_transaction", "create_report", "list_reports", "list_logs", "chat",
                        })
                        resources = await session.list_resources()
                        self.assertEqual(len(resources.resources), 2)
                        reference = await session.read_resource("frmc://knowledge/fundamentals")
                        self.assertGreater(len(json.loads(reference.contents[0].text)), 0)
                        rules = await session.read_resource("frmc://knowledge/control-rules")
                        self.assertIn("approval_threshold", json.loads(rules.contents[0].text))

                        def result_data(result):
                            self.assertFalse(result.isError, result)
                            return json.loads(result.content[0].text)

                        answer = result_data(await session.call_tool("ask_policy", {"question": "What is SOX?"}))
                        self.assertIn("Sarbanes-Oxley", answer["answer"])
                        self.assertTrue(answer["sources"])
                        chat = result_data(await session.call_tool("chat", {"question": "What is SOX?"}))
                        self.assertEqual(chat["answer_mode"], "offline_reference")
                        assessment = result_data(await session.call_tool("assess_transaction", {
                            "question": "What is SOX?",
                            "evidence": {"amount": 600000, "approval": False},
                            "risk_factors": {"criticality": "High"},
                        }))
                        self.assertEqual(assessment["risk_result"], "NOT_ASSESSED")
                        self.assertEqual(assessment["evidence_assessment"]["status"], "NOT_ASSESSED")
                        self.assertEqual(assessment["agent_trace"], ["RAG", "Policy & Control", "Evidence", "Risk", "Review"])
                        logs = result_data(await session.call_tool("list_logs", {}))
                        self.assertEqual(len(logs["logs"]), 1)
                        self.assertEqual(logs["logs"][0]["agent"], "AgentWorkflow")
                        created = result_data(await session.call_tool("create_report", {"title": "MCP test", "content": "Verified"}))
                        reports = result_data(await session.call_tool("list_reports", {}))
                        self.assertEqual(reports["reports"][0]["id"], created["id"])
                        for tool, args in [
                            ("ask_policy", {"question": ""}),
                            ("assess_transaction", {"question": "SOX", "evidence": {"amount": -1}}),
                            ("create_report", {"title": "", "content": "x"}),
                        ]:
                            invalid = await session.call_tool(tool, args)
                            self.assertTrue(invalid.isError)
                        # A failed tool invocation must not terminate the protocol session.
                        result_data(await session.call_tool("ask_policy", {"question": "What is COSO?"}))


if __name__ == "__main__":
    unittest.main()
