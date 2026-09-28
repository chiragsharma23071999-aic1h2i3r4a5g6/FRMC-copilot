"""Persistent, sourced reference answers available without a remote model."""
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
STOP_WORDS = set("what is are the a an of for to in and does do how can me explain tell about please section define could would should you we our my i it us your this that with on at from as be have has help give understand simple terms words mean means meaning actually briefly exactly really more some into when".split())


def normalize_question(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def tokens(text):
    text = re.sub(r"sarbanes[\s-]+oxley", "sox", text.lower())
    text = re.sub(r"\bseparation of duties\b", "segregation of duties", text)
    text = re.sub(r"\bseventeen\b", "17", text)
    text = re.sub(r"\bfive\b", "5", text)
    words = set(re.findall(r"[a-z0-9]+", text)) - STOP_WORDS
    singular = {"controls": "control", "risks": "risk", "companies": "company", "deficiencies": "deficiency",
                "weaknesses": "weakness", "auditors": "auditor", "reports": "report", "itgc": "itgc", "itgcs": "itgc"}
    return {singular.get(word, word) for word in words}


@lru_cache(maxsize=2)
def _reference_index(path, modified_ns, size):
    """Cache parsed data, exact aliases and lexical features; refresh on file edits."""
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    aliases, records, frequency = {}, [], Counter()
    for entry in entries:
        title = entry.get("question", entry.get("title", ""))
        forms = [title, *entry.get("alternate_questions", [])]
        for form in forms:
            key = normalize_question(form)
            if key in aliases and aliases[key]["id"] != entry["id"]:
                raise ValueError(f"Conflicting knowledge aliases: {form}")
            aliases[key] = entry
        # Related topics are deliberately excluded from matching: mentioning a
        # related concept must not silently route to a different subject.
        features = [tokens(form) for form in forms]
        union = set().union(*features)
        frequency.update(union)
        records.append((entry, features, union))
    weights = {word: 1 + math.log((len(records) + 1) / (count + 1)) for word, count in frequency.items()}
    return aliases, records, weights


def _document(entry):
    return Document(page_content=entry["answer"], metadata={
        "source": "builtin:" + entry["id"], "title": entry.get("question", entry.get("title", "")),
        "sources": entry.get("sources", []), "reviewed_on": entry.get("reviewed_on", ""),
        "category": entry.get("category", ""), "difficulty": entry.get("difficulty", ""),
        "follow_up_questions": entry.get("follow_up_questions", []),
    })


def reference_documents(question):
    path = KNOWLEDGE_DIR / "fundamentals.json"
    stat = path.stat()
    aliases, records, weights = _reference_index(str(path), stat.st_mtime_ns, stat.st_size)
    exact = aliases.get(normalize_question(question))
    if exact:
        return [_document(exact)]
    query = tokens(question)
    if not query:
        return []
    # Unknown substantive words cannot be ignored simply because SOX or audit
    # appears in the same question. The uploaded-document retriever may help.
    unknown = query - weights.keys()
    if unknown:
        return []
    query_weight = sum(weights[word] for word in query)
    numbers = {word for word in query if word.isdigit()}
    ranked = []
    for entry, forms, terms in records:
        if not numbers <= terms:
            continue
        best = 0.0
        for form in forms:
            overlap_weight = sum(weights[word] for word in query & form)
            coverage = overlap_weight / query_weight
            if coverage < 0.85:
                continue
            form_weight = sum(weights[word] for word in form)
            score = 2 * overlap_weight / (query_weight + form_weight)
            best = max(best, score)
        if best >= 0.72:
            ranked.append((best, entry))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    if not ranked:
        return []
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.025:
        # Do not choose a different topic or intent merely by file order.
        return []
    return [_document(ranked[0][1])]


def answer_from_documents(documents):
    if not documents:
        return {"answer": "I do not have enough relevant reference material to answer this question. Add a relevant policy document or ask a more specific GRC, SOX, COSO, financial risk, or internal audit question.",
                "sources": [], "answer_mode": "insufficient_knowledge"}
    sources = []
    for doc in documents:
        citations = doc.metadata.get("sources") or [{
            "title": doc.metadata.get("title", doc.metadata.get("source", "Document")),
            "source": doc.metadata.get("source", "Document"),
        }]
        for citation in citations:
            if citation not in sources:
                sources.append(citation)
    builtin = all(str(doc.metadata.get("source", "")).startswith("builtin:") for doc in documents)
    return {"answer": "\n\n".join(doc.page_content for doc in documents),
            "sources": sources, "answer_mode": "reference_answer" if builtin else "document_excerpts"}


def control_rules():
    return json.loads((KNOWLEDGE_DIR / "control_rules.json").read_text(encoding="utf-8"))
