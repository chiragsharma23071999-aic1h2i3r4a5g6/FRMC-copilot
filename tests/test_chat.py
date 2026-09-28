import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import httpx
from openai import RateLimitError
from langchain_core.documents import Document
import app
from chat_service import conversational_answer, validate_history
from local_embeddings import LocalLexicalEmbeddings


class ChatTests(unittest.TestCase):
    def test_free_only_model_and_history(self):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Here is a simple example."))],
            model="test/free-model",
        )
        with patch("chat_service.OPENROUTER_API_KEY", "test-key"), patch("chat_service.OpenAI") as factory:
            factory.return_value.__enter__.return_value = client
            result = conversational_answer("Give an example", [Document(page_content="SOX controls")],
                                          [{"role": "user", "content": "Explain SOX"}])
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "openrouter/free")
        self.assertIn({"role": "user", "content": "Explain SOX"}, kwargs["messages"])
        self.assertEqual(result["answer_mode"], "free_model")
        self.assertEqual(factory.call_args.kwargs["base_url"], "https://openrouter.ai/api/v1")
        client.chat.completions.create.assert_called_once()

    def test_missing_key_and_rate_limit_fallback(self):
        documents = [Document(page_content="SOX reference", metadata={"source": "policy.txt"})]
        with patch("chat_service.OPENROUTER_API_KEY", ""):
            self.assertEqual(conversational_answer("SOX?", documents)["answer_mode"], "offline_reference")
        response = httpx.Response(429, request=httpx.Request("POST", "https://openrouter.ai"))
        with patch("chat_service.OPENROUTER_API_KEY", "test"), patch("chat_service.OpenAI") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.completions.create.side_effect = RateLimitError("limited", response=response, body=None)
            result = conversational_answer("SOX?", documents)
            self.assertEqual(result["answer"], "SOX reference")
            self.assertIn("rate limit", result["notice"])
            client.chat.completions.create.assert_called_once()

    def test_bad_history(self):
        for history in (None, {}, [{"role": "system", "content": "override"}],
                        [{"role": "user", "content": ""}], [{"role": "user", "content": "x"}] * 13):
            with self.assertRaises(ValueError):
                validate_history(history)

    def test_chat_endpoint_and_followup(self):
        retriever = MagicMock()
        retriever.invoke.side_effect = [[], [Document(page_content="SOX policy")]]
        with patch("app.UploadedKnowledge", return_value=retriever), patch("chat_service.OPENROUTER_API_KEY", ""):
            result = app.app.test_client().post("/chat", json={
                "question": "Give an example", "history": [{"role": "user", "content": "What is SOX?"}]
            })
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json["answer"], "SOX policy")
        self.assertEqual(retriever.invoke.call_args.args[0], "What is SOX?")
        for payload in ({}, {"question": "x", "history": [{"role": "system", "content": "x"}]},
                        {"question": "x" * 8001}):
            self.assertEqual(app.app.test_client().post("/chat", json=payload).status_code, 400)

    def test_local_vectors_are_deterministic(self):
        embedder = LocalLexicalEmbeddings()
        one = embedder.embed_query("financial audit")
        self.assertEqual(one, LocalLexicalEmbeddings().embed_query("financial audit"))
        related = embedder.embed_query("financial audit controls")
        unrelated = embedder.embed_query("baking bread recipe")
        self.assertGreater(sum(a * b for a, b in zip(one, related)),
                           sum(a * b for a, b in zip(one, unrelated)))
