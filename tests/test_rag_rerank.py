"""Phase 3 reranking: the score, the reordering and its evaluation, with a fake model."""

import pytest

from rag.chunking import Chunk
from rag.rerank import (
    RerankingRetriever,
    changes_in_top,
    evaluate,
    format_pair,
    rerank_ids,
    yes_probability,
)
from rag.retriever import Hit


def _hit(cid, score=1.0):
    return Hit(Chunk(id=cid, doc_id="d", source="d.md", text=f"text {cid}", index=0), score)


def test_yes_probability_is_a_softmax_over_yes_and_no():
    assert yes_probability(0.0, 0.0) == 0.5
    assert yes_probability(3.0, 1.0) == pytest.approx(1 / (1 + 2.718281828**-2))
    assert yes_probability(-5.0, 5.0) < 0.001


def test_format_pair_follows_the_model_card():
    assert format_pair("I", "Q", "D") == "<Instruct>: I\n<Query>: Q\n<Document>: D"


class FakeReranker:
    def __init__(self, scores):
        self.scores, self.seen = scores, []

    def score(self, query, documents):
        self.seen.append(len(documents))
        return [self.scores[d.removeprefix("text ")] for d in documents]


class FakeBase:
    name = "hybrid"

    def search(self, query, top_k=5):
        return [_hit(c) for c in ["a", "b", "c", "d"][:top_k]]


def test_reranking_retriever_reorders_the_pool_by_p_yes():
    reranker = FakeReranker({"a": 0.2, "b": 0.9, "c": 0.9, "d": 0.95})
    retriever = RerankingRetriever(FakeBase(), reranker, pool=3)
    hits = retriever.search("q", top_k=2)
    # "d" scores best but is outside the pool; "b" and "c" tie and keep the base order.
    assert [h.chunk.id for h in hits] == ["b", "c"] and hits[0].score == 0.9
    assert reranker.seen == [3] and retriever.name == "hybrid+rerank"


def test_rerank_ids_only_reorders_the_head():
    scores = {"a": 0.1, "b": 0.8, "c": 0.9}
    assert rerank_ids(["a", "b", "c"], scores, pool=2) == ["b", "a"]


def test_evaluate_before_and_after_and_the_changes_in_the_top():
    resolved = [
        {"id": "q1", "kind": "fact", "relevant_groups": [{"r"}], "relevant_chunk_ids": ["r"]},
        {"id": "q2", "kind": "fact", "relevant_groups": [{"s"}], "relevant_chunk_ids": ["s"]},
    ]
    rankings = {"dense": [["x", "y", "r"], ["s", "z", "w"]]}
    scores = [{"x": 0.1, "y": 0.2, "r": 0.9}, {"s": 0.1, "z": 0.8, "w": 0.7}]
    report = evaluate(resolved, rankings, scores, pools=(3,))["dense"]
    assert report["base"]["mrr"] == pytest.approx((1 / 3 + 1) / 2, abs=1e-4)
    after = report["rerank@3"]
    assert after["recall_of_pool"] == 1.0 and after["mrr"] == pytest.approx(
        (1 + 1 / 3) / 2, abs=1e-4
    )
    ranks = [(q["rank_before"], q["rank_after"]) for q in report["per_question"]]
    assert ranks == [(3, 1), (1, 3)]
    assert changes_in_top(report["per_question"], k=2) == {"fixed": ["q1"], "broken": ["q2"]}
