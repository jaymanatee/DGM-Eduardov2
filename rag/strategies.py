"""The chunking strategies we compare, with the parameters we chose, in one place.

``rag/evaluate.py`` builds every strategy and ``rag/generate.py`` the one that won, both
through ``build_chunks``, so the API serves exactly what was measured. ``recursive_1500``
is the recursive splitter at the size limit of the other two: it tells the effect of where
a strategy cuts from the effect of how long its chunks are.
"""

from __future__ import annotations

from rag.chunking import (
    Chunk,
    add_context,
    recursive_chunks,
    semantic_chunks,
    structural_chunks,
)
from rag.ingest import Document

# The distance threshold was chosen on the gold set (reports/phase3_semantic_sweep.json):
# 0.7 has the best hybrid MRR but one (0.75, within noise), where already 37 % of the chunks
# are closed by MAX_CHUNK_CHARS rather than by the distance. rag/semantic_analysis.py draws
# the distances.
SEMANTIC_THRESHOLD = 0.7
SEMANTIC_MIN_CHARS = 200
MAX_CHUNK_CHARS = 1500


def _recursive(docs, embedder):
    return recursive_chunks(docs, chunk_size=800, chunk_overlap=100)


def _recursive_1500(docs, embedder):
    return recursive_chunks(docs, chunk_size=MAX_CHUNK_CHARS, chunk_overlap=150)


def _semantic(docs, embedder, threshold: float = SEMANTIC_THRESHOLD):
    return semantic_chunks(
        docs,
        embedder.embed_passages,
        distance_threshold=threshold,
        max_chunk_chars=MAX_CHUNK_CHARS,
        min_chunk_chars=SEMANTIC_MIN_CHARS,
    )


def _structural(docs, embedder):
    return structural_chunks(docs, max_chunk_chars=MAX_CHUNK_CHARS, min_chunk_chars=200)


STRATEGIES = {
    "recursive": _recursive,
    "recursive_1500": _recursive_1500,
    "semantic": _semantic,
    "structural": _structural,
}


def split_strategy(name: str) -> tuple[str, float | None]:
    """``"semantic@0.6"`` -> ``("semantic", 0.6)``; any other name -> ``(name, None)``."""
    base, at, threshold = name.partition("@")
    if not at:
        return name, None
    if base != "semantic":
        raise ValueError(f"only the semantic strategy takes a threshold, not {base!r}")
    try:
        return base, float(threshold)
    except ValueError:
        raise ValueError(f"not a threshold: {threshold!r}") from None


def build_chunks(
    strategy: str, documents: list[Document], embedder=None, context: bool = True
) -> list[Chunk]:
    """The chunks of ``strategy``, with the document and section header unless ``context``
    is off. ``embedder`` is only needed by the semantic strategy, which also takes another
    threshold as ``semantic@<threshold>``."""
    base, threshold = split_strategy(strategy)
    if threshold is None:
        chunks = STRATEGIES[base](documents, embedder)
    else:
        chunks = _semantic(documents, embedder, threshold=threshold)
    return add_context(chunks, documents) if context else chunks
