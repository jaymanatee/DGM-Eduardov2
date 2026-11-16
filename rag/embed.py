"""Data indexing, step 3: embeddings.

We use Qwen3-Embedding through Sentence Transformers. Remember the point of the lecture:
relevance depends on the task, so the *query* is embedded together with an instruction
that says what "relevant" means, while passages are embedded as they are. Change the
instruction and you change which chunks come back; that is a lever you should play with.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_QUERY_INSTRUCTION = (
    "Given a user question, retrieve the passages from the knowledge base that answer it."
)


class Embedder:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.model = SentenceTransformer(model_id, device=device)

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Passages are embedded without instruction. Rows are L2-normalised."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        )

    def embed_queries(
        self, texts: list[str], instruction: str = DEFAULT_QUERY_INSTRUCTION
    ) -> np.ndarray:
        """Queries are prefixed with the task instruction (instruction-aware embeddings).

        An empty ``instruction`` embeds the bare query, with no prefix at all: the baseline
        of the instruction experiment (``rag/ablations.py``)."""
        prompt = f"Instruct: {instruction}\nQuery: " if instruction else None
        return self.model.encode(texts, prompt=prompt, normalize_embeddings=True)


def cosine_similarity(queries: np.ndarray, passages: np.ndarray) -> np.ndarray:
    """(n_queries, n_passages) similarities for normalised embeddings: just a dot product."""
    return queries @ passages.T


CACHE_DIR = Path(__file__).parent / "index" / "cache"


class CachedEmbedder:
    """An ``Embedder`` that remembers every passage it has embedded, on disk.

    Chunking with three strategies, sweeping the semantic threshold and rebuilding the dense
    index embed the same texts again and again. The cache maps the SHA-1 of a text to its
    vector, in ``rag/index/cache/<model>.npz``; the model is only loaded when a text is
    missing, so a rerun over a cached corpus needs no GPU. Queries are not cached: they are
    few, and they depend on the instruction.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, cache_dir: Path = CACHE_DIR, device=None):
        self.model_id, self.device = model_id, device
        self.path = Path(cache_dir) / (model_id.replace("/", "__") + ".npz")
        self._embedder: Embedder | None = None
        self._vectors: dict[str, np.ndarray] = {}
        if self.path.exists():
            with np.load(self.path) as stored:
                self._vectors = dict(zip(stored["keys"].tolist(), stored["vectors"], strict=True))

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.model_id, device=self.device)
        return self._embedder

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        missing = list(
            {k: t for k, t in zip(keys, texts, strict=True) if k not in self._vectors}.items()
        )
        if missing:
            vectors = self.embedder.embed_passages([t for _, t in missing], batch_size=batch_size)
            self._vectors.update(zip([k for k, _ in missing], vectors, strict=True))
            self.save()
        return np.stack([self._vectors[k] for k in keys]) if keys else np.zeros((0, 0))

    def embed_queries(
        self, texts: list[str], instruction: str = DEFAULT_QUERY_INSTRUCTION
    ) -> np.ndarray:
        return self.embedder.embed_queries(texts, instruction=instruction)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._vectors)
        partial = self.path.with_name(self.path.stem + ".part.npz")
        np.savez(partial, keys=np.array(keys), vectors=np.stack([self._vectors[k] for k in keys]))
        partial.replace(self.path)
