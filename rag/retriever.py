"""Retrieval: similarity search, keyword search (BM25) and the hybrid of both.

``BM25Retriever`` is complete: it is the keyword baseline you compare against.
``DenseRetriever`` and ``HybridRetriever`` are yours. All three return the same ``Hit``
objects so that ``rag/generate.py`` and ``rag/evaluate.py`` do not care which one is used.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.chunking import Chunk
from rag.embed import DEFAULT_QUERY_INSTRUCTION


@dataclass
class Hit:
    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """Keyword ranking: term frequency, inverse document frequency, length normalisation."""

    name = "bm25"

    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.index = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        scores = self.index.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]


class DenseRetriever:
    """Embed the chunks once, keep them as a NumPy matrix, rank by cosine similarity.

    The matrix is persisted in ``persist_dir`` (``vectors.npy`` and ``meta.json``) together
    with a fingerprint of the model and of every chunk id and text: the API loads it at start
    instead of embedding the corpus again, and any change of chunking or model rebuilds it.
    The query is embedded with ``instruction`` (instruction-aware embeddings, ``embed.py``).
    """

    name = "dense"

    def __init__(
        self,
        chunks: list[Chunk],
        embedder,
        persist_dir: str | Path = "rag/index",
        instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ):
        self.chunks, self.embedder, self.instruction = chunks, embedder, instruction
        persist_dir = Path(persist_dir)
        digest = hashlib.sha1(embedder.model_id.encode())
        for chunk in chunks:
            digest.update(f"\0{chunk.id}\0{chunk.text}".encode())
        fingerprint = digest.hexdigest()
        meta_path, vectors_path = persist_dir / "meta.json", persist_dir / "vectors.npy"
        if meta_path.exists() and vectors_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == fingerprint:
                self.vectors = np.load(vectors_path)
                return
        self.vectors = np.asarray(embedder.embed_passages([c.text for c in chunks]), np.float32)
        persist_dir.mkdir(parents=True, exist_ok=True)
        np.save(vectors_path, self.vectors)
        meta = {"fingerprint": fingerprint, "model": embedder.model_id, "n_chunks": len(chunks)}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def scores(self, query: str) -> np.ndarray:
        query_vector = self.embedder.embed_queries([query], instruction=self.instruction)[0]
        return self.vectors @ np.asarray(query_vector, np.float32)

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        scores = self.scores(query)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]


def _min_max(hits: list[Hit]) -> dict[str, float]:
    """Scores rescaled to [0, 1] within the list; all 1.0 when they are all equal."""
    if not hits:
        return {}
    low, high = min(h.score for h in hits), max(h.score for h in hits)
    span = high - low
    return {h.chunk.id: (h.score - low) / span if span else 1.0 for h in hits}


class HybridRetriever:
    """s_hybrid = λ · s_bm25 + (1 - λ) · s_dense, on scores rescaled to [0, 1].

    Take the top ``pool`` hits from each retriever, min-max normalise each score list so
    they are comparable, merge by chunk id (a chunk missing from one list gets 0 there: it
    scored below that list's minimum), and return the top ``top_k`` by the weighted sum.
    λ (``weight_bm25``) is chosen on the gold set by ``rag/evaluate.py``.
    """

    name = "hybrid"

    def __init__(
        self, bm25: BM25Retriever, dense: DenseRetriever, weight_bm25: float = 0.5, pool: int = 20
    ):
        self.bm25, self.dense, self.weight_bm25, self.pool = bm25, dense, weight_bm25, pool

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        pool = max(self.pool, top_k)
        bm25_hits, dense_hits = self.bm25.search(query, pool), self.dense.search(query, pool)
        return self.fuse(bm25_hits, dense_hits, self.weight_bm25, top_k)

    @staticmethod
    def fuse(bm25_hits: list[Hit], dense_hits: list[Hit], weight_bm25: float, top_k: int):
        """The weighted sum of both lists; separate so a λ sweep can reuse the same hits."""
        bm25, dense = _min_max(bm25_hits), _min_max(dense_hits)
        chunks = {h.chunk.id: h.chunk for h in [*bm25_hits, *dense_hits]}
        fused = {
            cid: weight_bm25 * bm25.get(cid, 0.0) + (1 - weight_bm25) * dense.get(cid, 0.0)
            for cid in chunks
        }
        order = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:top_k]
        return [Hit(chunks[cid], fused[cid]) for cid in order]
