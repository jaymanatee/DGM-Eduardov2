"""Phase 3 retrieval: dense and hybrid retrievers, and the evidence-based gold set."""

import numpy as np
import pytest

from rag.chunking import Chunk
from rag.evaluate import (
    GoldError,
    choose_lambda,
    evidence_recall_at_k,
    find_quote,
    lambda_stability,
    relevant_chunks,
    resolve_gold,
    score_rankings,
    scores_by_kind,
)
from rag.ingest import Document
from rag.retriever import DenseRetriever, Hit, HybridRetriever
from rag.strategies import build_chunks, split_strategy

pytest.importorskip("rank_bm25")


def _chunk(cid, start=0, end=0, text="", source="d.md"):
    return Chunk(id=cid, doc_id="d", source=source, text=text, index=0, start=start, end=end)


class FakeEmbedder:
    """Two-dimensional embeddings: "origin" points one way, anything else the other."""

    model_id = "fake"

    def __init__(self):
        self.passages_embedded = 0

    def _vector(self, text):
        return [1.0, 0.0] if "origin" in text else [0.0, 1.0]

    def embed_passages(self, texts):
        self.passages_embedded += len(texts)
        return np.array([self._vector(t) for t in texts])

    def embed_queries(self, texts, instruction=""):
        return np.array([self._vector(t) for t in texts])


def test_dense_retriever_ranks_by_cosine_and_persists(tmp_path):
    chunks = [_chunk("a", text="tariff rates"), _chunk("b", text="rules of origin")]
    embedder = FakeEmbedder()
    dense = DenseRetriever(chunks, embedder, persist_dir=tmp_path)
    hits = dense.search("proof of origin", top_k=2)
    assert [h.chunk.id for h in hits] == ["b", "a"]
    assert hits[0].score == pytest.approx(1.0)
    # Same chunks and model: loaded from disk, nothing embedded again.
    DenseRetriever(chunks, embedder, persist_dir=tmp_path)
    assert embedder.passages_embedded == 2
    # Different chunks: the index is rebuilt.
    DenseRetriever([*chunks, _chunk("c", text="other")], embedder, persist_dir=tmp_path)
    assert embedder.passages_embedded == 5


def test_hybrid_fuse_min_max_and_weights():
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    bm25 = [Hit(a, 12.0), Hit(b, 6.0), Hit(c, 2.0)]  # rescaled: a 1, b 0.4, c 0
    dense = [Hit(c, 0.9), Hit(b, 0.8), Hit(a, 0.4)]  # rescaled: c 1, b 0.8, a 0
    ids = lambda hits: [h.chunk.id for h in hits]  # noqa: E731
    assert ids(HybridRetriever.fuse(bm25, dense, 1.0, 3)) == ["a", "b", "c"]
    assert ids(HybridRetriever.fuse(bm25, dense, 0.0, 3)) == ["c", "b", "a"]
    fused = HybridRetriever.fuse(bm25, dense, 0.5, 3)
    assert ids(fused) == ["b", "a", "c"] and fused[0].score == pytest.approx(0.6)
    # A chunk found by one retriever only gets 0 from the other.
    only_dense = HybridRetriever.fuse([Hit(a, 1.0)], [Hit(b, 0.5)], 0.5, 2)
    assert [h.score for h in only_dense] == [0.5, 0.5]


TEXT = "Article 1\n\nThe statement on origin shall be valid for 12 months.\n\nArticle 2\n\nOther."


def test_find_quote_ignores_white_space():
    assert find_quote(TEXT, "valid  for\n12 months") == (TEXT.index("valid"), TEXT.index(".\n"))
    assert find_quote(TEXT, "valid for 24 months") is None


def test_relevant_chunks_hold_most_of_the_quote():
    span = (100, 200)
    chunks = [_chunk("a", 0, 120), _chunk("b", 110, 210), _chunk("c", 200, 300)]
    assert relevant_chunks(span, chunks) == {"b"}  # a holds 20 %, c nothing
    # Cut in three pieces, none holds half: the biggest piece wins.
    thirds = [_chunk("a", 0, 140), _chunk("b", 140, 170), _chunk("c", 170, 300)]
    assert relevant_chunks(span, thirds) == {"a"}


def test_resolve_gold_per_chunking_and_metrics():
    doc = Document(id="d", source="d.md", text=TEXT)
    start = TEXT.index("The statement")
    chunks = [_chunk("d-0", 0, start), _chunk("d-1", start, len(TEXT))]
    gold = [
        {"id": "q1", "question": "?", "evidence": [{"source": "d.md", "quote": "12 months"}]},
        {"id": "q2", "question": "not in the corpus", "evidence": []},
    ]
    (resolved,) = resolve_gold(gold, [doc], chunks)
    assert resolved["relevant_chunk_ids"] == ["d-1"]
    assert score_rankings([["d-0", "d-1"]], [resolved], ks=(1, 2)) == {
        "recall@1": 0.0,
        "recall@2": 1.0,
        "mrr": 0.5,
    }
    with pytest.raises(GoldError, match="24 months"):
        bad = [
            {"id": "q3", "question": "?", "evidence": [{"source": "d.md", "quote": "24 months"}]}
        ]
        resolve_gold(bad, [doc], chunks)


def test_evidence_recall_counts_each_piece_of_evidence_once():
    # Two pieces of evidence; the first is equally well held by two chunks.
    groups = [{"a", "b"}, {"c"}]
    assert evidence_recall_at_k(["a", "b", "x"], groups, k=3) == 0.5
    assert evidence_recall_at_k(["b", "c"], groups, k=2) == 1.0


# ------------------------------------------------------------------------------- choosing λ


def test_choose_lambda_by_mrr_then_recall_then_smaller_lambda():
    sweep = {
        "0.0": {"mrr": 0.60, "recall@5": 0.90},
        "0.1": {"mrr": 0.70, "recall@5": 0.80},
        "0.3": {"mrr": 0.70, "recall@5": 0.85},
        "0.4": {"mrr": 0.70, "recall@5": 0.85},
    }
    assert choose_lambda(sweep) == "0.3"  # best MRR; 0.3 and 0.4 tie, the smaller wins


def test_lambda_stability_on_held_out_halves():
    # Eight questions whose only relevant chunk is "r": λ = 0.5 always ranks it first,
    # λ = 0.0 second, so the choice is the same on every half.
    gold = [{"relevant_groups": [{"r"}], "relevant_chunk_ids": ["r"]} for _ in range(8)]
    rankings = {"0.0": [["x", "r"]] * 8, "0.5": [["r", "x"]] * 8}
    report = lambda_stability(rankings, gold, n_splits=10)
    assert report["chosen"] == {"0.5": 10}
    assert report["held_out_mrr"] == report["held_out_mrr_best_lambda"] == 1.0
    assert report["held_out_mrr_dense"] == 0.5


def test_semantic_strategy_with_another_threshold():
    assert split_strategy("structural") == ("structural", None)
    assert split_strategy("semantic@0.6") == ("semantic", 0.6)
    with pytest.raises(ValueError):
        split_strategy("recursive@0.6")
    doc = Document(
        id="d", source="d.md", text="The origin rules. More on origin. The tariff is low.",
        metadata={"suffix": ".md", "title": "D"},
    )  # fmt: skip
    chunks = build_chunks("semantic@0.5", [doc], FakeEmbedder(), context=False)
    assert [c.text for c in chunks] == ["The origin rules. More on origin. The tariff is low."]


def test_scores_by_kind_splits_the_metrics_by_kind_of_question():
    gold = [
        {"kind": "cross", "relevant_groups": [{"a"}], "relevant_chunk_ids": ["a"]},
        {"kind": "fact", "relevant_groups": [{"b"}], "relevant_chunk_ids": ["b"]},
        {"kind": "cross", "relevant_groups": [{"c"}], "relevant_chunk_ids": ["c"]},
    ]
    rankings = [["a"], ["x", "b"], ["x"]]
    report = scores_by_kind(rankings, gold)
    assert report["cross"]["n"] == 2 and report["cross"]["mrr"] == 0.5
    assert report["fact"]["mrr"] == 0.5
