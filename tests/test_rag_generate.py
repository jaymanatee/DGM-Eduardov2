"""Phase 3 generation: prompt, citations, abstentions and scoring, with a fake model."""

import json
from datetime import date

from rag.chunking import Chunk
from rag.evaluate_generation import answer_correct, comparison_table, rescore
from rag.generate import (
    CLOSED_BOOK_NO_ANSWER,
    CLOSED_BOOK_NO_ANSWER_ES,
    NO_ANSWER,
    RagPipeline,
    build_closed_book_messages,
    build_messages,
    drop_stray_no_answer,
    is_abstention,
    restore_citations,
    strip_thinking,
)
from rag.retriever import Hit


def _hits(*ids):
    return [
        Hit(Chunk(id=cid, doc_id="d", source="pem/rules.md", text=f"text of {cid}", index=0), 1.0)
        for cid in ids
    ]


def test_the_prompt_numbers_the_passages_and_carries_the_rules():
    system, user = build_messages("How long?", _hits("aaa-0001", "bbb-0002"), date(2026, 11, 16))
    assert system["role"] == "system" and "Today is 2026-11-16" in system["content"]
    assert NO_ANSWER in system["content"]
    assert "[1] (from pem/rules.md)\ntext of aaa-0001" in user["content"]
    assert "[2] (from pem/rules.md)\ntext of bbb-0002" in user["content"]
    assert "aaa-0001]" not in user["content"]  # ids are not shown: the model miscopies them
    assert user["content"].rstrip().endswith("Question: How long?")


def test_restore_citations_turns_numbers_into_chunk_ids():
    hits = _hits("aaa-0001", "bbb-0002", "ccc-0003")
    answer = "Ten months [1]. Twelve in Korea [2, 3]. See also [ 3 ] and [7] and [2023]."
    assert restore_citations(answer, hits) == (
        "Ten months [aaa-0001]. Twelve in Korea [bbb-0002][ccc-0003]. "
        "See also [ccc-0003] and [7] and [2023]."
    )


def test_is_abstention_only_when_the_answer_opens_saying_so_and_cites_nothing():
    assert is_abstention(NO_ANSWER)
    assert is_abstention("No lo sé: los pasajes recuperados no responden a esta pregunta.")
    assert is_abstention("The MFN rate is not directly specified in the provided passages.")
    # It answers and then says what is left open: an answer, not an abstention.
    long_answer = "A proof of origin is valid for ten months. " * 6 + NO_ANSWER
    assert not is_abstention(long_answer)
    assert not is_abstention(NO_ANSWER, citations=["aaa-0001"])


def test_drop_stray_no_answer_only_after_an_answer_that_cites():
    answer = f"Ten months [1].\n\nTherefore, {NO_ANSWER}"
    assert drop_stray_no_answer(answer) == "Ten months [1]."
    assert drop_stray_no_answer(f'Ten months [1]. ["{NO_ANSWER}"]') == "Ten months [1]."
    assert drop_stray_no_answer(NO_ANSWER) == NO_ANSWER  # a real abstention stays
    uncited = f"Maybe ten months.\n\n{NO_ANSWER}"
    assert drop_stray_no_answer(uncited) == uncited


def test_strip_thinking():
    assert strip_thinking("<think>\nhmm\n</think>\n\nTen months [1].") == "Ten months [1]."
    assert strip_thinking("<think>never closed. Ten months [1].") == "never closed. Ten months [1]."


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, question, top_k=5):
        return self.hits[:top_k]


def test_answer_returns_the_chunks_and_only_valid_citations():
    pipeline = RagPipeline()
    hits = _hits("aaa-0001", "bbb-0002")
    pipeline.retrievers = {"hybrid": FakeRetriever(hits)}
    seen = {}

    def generate(messages):
        seen["messages"] = messages
        return "Ten months [1], see also [5]."

    pipeline.generate = generate
    response = pipeline.answer("How long?", top_k=2, retriever="hybrid")
    assert response.answer == "Ten months [aaa-0001], see also [5]."
    assert response.citations == ["aaa-0001"]
    assert [c.id for c in response.chunks] == ["aaa-0001", "bbb-0002"]
    assert response.retriever == "hybrid" and "Qwen3-1.7B" in response.model
    assert seen["messages"][1]["content"].endswith("Question: How long?\n")


# --------------------------------------------------------------------- right or wrong


def test_closed_book_baseline_has_the_rules_but_no_passages():
    system, user = build_closed_book_messages("How long?", date(2026, 11, 16))
    assert "Today is 2026-11-16" in system["content"] and "passage" not in system["content"]
    assert user == {"role": "user", "content": "How long?"}
    pipeline = RagPipeline()
    pipeline.generate = lambda messages: "Ten months."
    assert pipeline.answer_closed_book("How long?") == "Ten months."
    assert is_abstention(CLOSED_BOOK_NO_ANSWER) and is_abstention(CLOSED_BOOK_NO_ANSWER_ES)


ITEM = {
    "evidence": [{"source": "d.md", "quote": "q"}],
    "answer_contains": [["12 months", "12 meses"], "1200", "10%"],
    "verdict_excludes": ["yes"],
}


def test_answer_correct_normalises_figures_and_ignores_citations():
    answer = "No, it is valid for 12 months [abc-1200], up to EUR 1,200 and 10 per cent."
    assert answer_correct(answer, ITEM, abstained=False)
    assert answer_correct("No. Válida 12 meses, 1 200 €, un 10 %.", ITEM, abstained=False)
    # The figure only in a citation does not count; "2 years" is not found in "12 years".
    assert not answer_correct("No, 12 months and 10% [x-1200].", ITEM, abstained=False)
    two_years = {"evidence": [{}], "answer_contains": ["2 years"]}
    assert not answer_correct("For 12 years.", two_years, abstained=False)
    assert answer_correct("For 2 years.", two_years, abstained=False)


def test_answer_correct_verdict_and_abstentions():
    right = "12 months, 1200, 10%."
    assert not answer_correct(f"Yes. {right}", ITEM, abstained=False)
    assert answer_correct(f"{right} Yes, also later.", ITEM, abstained=False)  # not the verdict
    assert not answer_correct(right, ITEM, abstained=True)
    unanswerable = {"evidence": []}
    assert answer_correct(NO_ANSWER, unanswerable, abstained=True)
    assert not answer_correct("Twelve months.", unanswerable, abstained=False)
    assert answer_correct("anything", {"evidence": [{}]}, abstained=False) is None


def _row(qid, answer, answerable=True, abstained=False, kind="fact"):
    return {
        "id": qid, "kind": kind, "answer": answer, "answerable": answerable,
        "abstained": abstained, "retrieval_hit": None, "cited": None,
        "cites_evidence": None, "seconds": 1.0, "correct": None,
    }  # fmt: skip


def test_rescore_scores_saved_answers_with_the_current_checks(tmp_path):
    answers = tmp_path / "phase3_generation_none.jsonl"
    rows = [_row("q1", "Ten months."), _row("q2", NO_ANSWER, False, True, "none")]
    answers.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summary_file = tmp_path / "phase3_generation.json"
    summary_file.write_text(json.dumps({"variants": {"none": {"answers": str(answers)}}}))
    gold = [
        {"id": "q1", "evidence": [{}], "answer_contains": [["ten months"]]},
        {"id": "q2", "evidence": []},
    ]
    report = rescore(gold, summary_file)
    summary = report["variants"]["none"]["summary"]
    assert summary["accuracy"] == 1.0 and summary["answerable_retrieval_hit"] is None
    assert summary["by_kind"] == {
        "fact": {"n": 1, "correct": 1.0},
        "none": {"n": 1, "correct": 1.0},
    }
    assert "none" in comparison_table(report)
