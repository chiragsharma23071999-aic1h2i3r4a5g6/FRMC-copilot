"""Deterministic local lexical vectors: no API, model download, or usage fee."""
import hashlib
import math
import re
from collections import Counter

from langchain_core.embeddings import Embeddings


class LocalLexicalEmbeddings(Embeddings):
    dimensions = 1024

    def embed_query(self, text):
        words = re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)
        counts = Counter(word for word in words if len(word) > 2)
        vector = [0.0] * self.dimensions
        for word, count in counts.items():
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            vector[index] += (1 if digest[4] % 2 else -1) * (1 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]
