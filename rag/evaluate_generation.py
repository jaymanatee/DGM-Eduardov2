"""Phase 3, the answers: the RAG pipeline on every question of the gold set, scored right or wrong.

Retrieval is measured by ``rag/evaluate.py``; this measures what the generator does with
what it gets, as phases 1 and 2 measured their models: one test set, one answer per
question, right or wrong, and the same set with and without the capability of the phase.

**Right or wrong.** Each answerable question of the gold set carries the phase-2 checks:

* ``answer_contains``: strings the answer must contain; an item that is a list means any of
  its alternatives. Only the key fact (the period, the threshold, the code), not the whole
  expected answer: "12 months" is right, whether or not it adds the 24-month ceiling;
* ``verdict_excludes``: strings the first sentence, where a yes/no is, must not contain.

The rule is phase 2's (``tool_use/evaluate.py``), with three changes for passages and
figures: citations are removed first (a chunk id holds digits), figures are written one
way (``10 %`` and ``10 per cent`` are ``10%``; ``1,200`` and ``1 200`` are ``1200``), and
a string must start a word (``2 years`` is not found in ``12 years``). An answer that says
it does not know is wrong. On the questions the corpus does not answer, right is saying it
does not know.

**Variants.** ``--retriever`` is one of the three of ``POST /rag``, or ``none``: the
baseline, the same model and rules without passages (``RagPipeline.answer_closed_book``),
the counterpart of phase 2's ``--no-tools``. ``ARCA_RAG_MODEL``, ``ARCA_RAG_ADAPTER`` and
``ARCA_RAG_THINKING`` change the generator, and ``--label`` names the variant.

Besides ``correct``, each answer records:

* ``retrieval_hit``: a chunk holding the evidence was among the ``top_k`` retrieved, i.e.
  the generator could answer;
* ``abstained``: it said it does not know (a false abstention when ``retrieval_hit``);
* ``cited``: it cites at least one retrieved chunk; ``cites_evidence``: one of the cited
  chunks holds the evidence. Every citation is a retrieved chunk by construction
  (``extract_citations``), so a wrong citation is a real chunk that does not hold the answer.

Each variant writes its answers to ``reports/phase3_generation_<label>.jsonl``, to read next
to the expected ones, and its summary into ``reports/phase3_generation.json``, which keeps
every variant evaluated (as phase 1 does) and prints them side by side.

Run::

    uv run python -m rag.evaluate_generation --retriever none     # baseline, no RAG
    uv run python -m rag.evaluate_generation --retriever hybrid   # also bm25, dense
    uv run python -m rag.evaluate_generation --rescore            # after editing the checks
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

REPORTS = Path("reports")
SUMMARY_FILE = REPORTS / "phase3_generation.json"

CITATION = re.compile(r"\[[^\]]*\]")
PERCENT = re.compile(r"(\d)\s*(?:%|per ?cent\b|percent\b|por ciento\b)")
THOUSANDS = re.compile(r"(?<=\d)[\s,.](?=\d{3}(?!\d))")


def normalise(text: str) -> str:
    """Lower case, no citations or markdown, one spacing, figures written one way."""
    text = CITATION.sub(" ", text.lower())
    text = re.sub(r"[*_`]", "", text)
    text = " ".join(text.replace(" ", " ").replace(" ", " ").split())
    return THOUSANDS.sub("", PERCENT.sub(r"\1%", text))


def _found(alternative: str, text: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(normalise(alternative)), text) is not None


def answer_correct(answer: str, item: dict, abstained: bool) -> bool | None:
    """Right or wrong by the gold set's checks; ``None`` if the question has none."""
    if not item.get("evidence"):
        return abstained  # the corpus does not answer it: right is saying so
    if "answer_contains" not in item and "verdict_excludes" not in item:
        return None
    if abstained:
        return False
    text = normalise(answer)
    verdict = next((s for s in re.split(r"(?<=[.!?])\s+|\n", text) if s), "")
    return all(
        any(_found(alt, text) for alt in ([entry] if isinstance(entry, str) else entry))
        for entry in item.get("answer_contains", [])
    ) and not any(_found(bad, verdict) for bad in item.get("verdict_excludes", []))


def _rate(items: list[dict], key: str) -> float | None:
    values = [r[key] for r in items if r.get(key) is not None]
    return round(sum(values) / len(values), 4) if values else None


def summarise(rows: list[dict]) -> dict:
    """Accuracy overall and by kind, and the rates over answerable and unanswerable questions."""
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    hit = [r for r in answerable if r["retrieval_hit"]]
    answered = [r for r in answerable if not r["abstained"]]
    by_kind = defaultdict(list)
    for row in rows:
        by_kind[row["kind"]].append(row)
    return {
        "accuracy": _rate(rows, "correct"),
        "answerable_correct": _rate(answerable, "correct"),
        "unanswerable_abstained": _rate(unanswerable, "abstained"),
        "by_kind": {
            kind: {"n": len(items), "correct": _rate(items, "correct")}
            for kind, items in sorted(by_kind.items())
        },
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "answerable_retrieval_hit": _rate(answerable, "retrieval_hit"),
        "answerable_abstained": _rate(answerable, "abstained"),
        "abstained_when_evidence_retrieved": _rate(hit, "abstained"),
        "correct_when_evidence_retrieved": _rate(hit, "correct"),
        "answered_cited": _rate(answered, "cited"),
        "answered_cites_evidence": _rate(answered, "cites_evidence"),
        "cites_evidence_when_evidence_retrieved": _rate(hit, "cites_evidence"),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / max(len(rows), 1), 2),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def _load_summaries(path: Path) -> dict:
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        if "variants" in report:
            return report
    return {"variants": {}}


def _save_summaries(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def comparison_table(report: dict) -> str:
    """One line per variant: accuracy overall, on answerable and unanswerable, and by kind."""
    variants = report["variants"]
    kinds = sorted({k for v in variants.values() for k in v["summary"]["by_kind"]})

    def pct(value):
        return "  -  " if value is None else f"{100 * value:4.0f}%"

    width = max([len("variant"), *map(len, variants)]) + 2
    header = f"{'variant':<{width}}{'all':>6}{'answ.':>7}{'none':>6}" + "".join(
        f"{k:>7}" for k in kinds
    )
    lines = [header, "-" * len(header)]
    for label, variant in variants.items():
        s = variant["summary"]
        lines.append(
            f"{label:<{width}}{pct(s['accuracy']):>6}{pct(s['answerable_correct']):>7}"
            f"{pct(s['unanswerable_abstained']):>6}"
            + "".join(f"{pct(s['by_kind'].get(k, {}).get('correct')):>7}" for k in kinds)
        )
    return "\n".join(lines)


def rescore(gold: list[dict], summary_file: Path = SUMMARY_FILE) -> dict:
    """Score again every variant's saved answers with the gold set's current checks."""
    items = {q["id"]: q for q in gold}
    report = _load_summaries(summary_file)
    for label, variant in report["variants"].items():
        path = Path(variant["answers"])
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        for row in rows:
            row["correct"] = answer_correct(row["answer"], items[row["id"]], row["abstained"])
        _write_jsonl(path, rows)
        variant["summary"] = summarise(rows)
        print(f"rescored {label}: {len(rows)} answers")
    _save_summaries(summary_file, report)
    return report


def run(pipeline, gold: list[dict], relevant: dict, retriever: str, top_k: int) -> list[dict]:
    """Answer every question of the gold set and score each answer."""
    from rag.generate import is_abstention

    rows = []
    for item in gold:
        started = time.perf_counter()
        if retriever == "none":
            answer, citations, retrieved = pipeline.answer_closed_book(item["question"]), [], []
        else:
            response = pipeline.answer(item["question"], top_k=top_k, retriever=retriever)
            answer, citations = response.answer, response.citations
            retrieved = [c.id for c in response.chunks]
        seconds = time.perf_counter() - started
        evidence = (
            set(relevant[item["id"]]["relevant_chunk_ids"]) if item["id"] in relevant else set()
        )
        abstained = is_abstention(answer, citations)
        with_rag = retriever != "none"
        rows.append(
            {
                "id": item["id"],
                "kind": item.get("kind", ""),
                "question": item["question"],
                "expected": item.get("answer", ""),
                "answer": answer,
                "correct": answer_correct(answer, item, abstained),
                "citations": citations,
                "retrieved": retrieved,
                "evidence": sorted(evidence),
                "answerable": bool(item.get("evidence")),
                "retrieval_hit": bool(evidence & set(retrieved)) if with_rag else None,
                "abstained": abstained,
                "cited": bool(citations) if with_rag else None,
                "cites_evidence": bool(evidence & set(citations)) if with_rag else None,
                "seconds": round(seconds, 2),
            }
        )
        row = rows[-1]
        mark = {True: "right", False: "WRONG", None: "  -  "}[row["correct"]]
        how = "abstains" if abstained else f"cites {len(citations)}"
        print(f"{item['id']} [{row['kind']:<5}] {seconds:5.1f}s {mark} {how}")
    return rows


def main(argv: list[str] | None = None) -> None:
    from rag.evaluate import GOLD_FILE, load_gold

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gold", default=str(GOLD_FILE))
    parser.add_argument(
        "--retriever", choices=("none", "dense", "bm25", "hybrid"), default="hybrid"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--label", help="name of the variant (default: the retriever)")
    parser.add_argument("--rescore", action="store_true", help="score the saved answers again")
    args = parser.parse_args(argv)
    gold = load_gold(args.gold)

    if args.rescore:
        print(comparison_table(rescore(gold)))
        return

    from rag.evaluate import resolve_gold
    from rag.generate import RagPipeline
    from rag.ingest import load_corpus

    pipeline = RagPipeline.from_env()
    pipeline.load()
    chunks = pipeline.retrievers["bm25"].chunks
    relevant = {q["id"]: q for q in resolve_gold(gold, load_corpus(), chunks)}
    rows = run(pipeline, gold, relevant, args.retriever, args.top_k)

    label = args.label or args.retriever
    answers = REPORTS / f"phase3_generation_{label}.jsonl"
    answers.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(answers, rows)
    report = _load_summaries(SUMMARY_FILE)
    report["variants"][label] = {
        "settings": {
            "retriever": args.retriever,
            "top_k": args.top_k if args.retriever != "none" else None,
            "model": pipeline.name,
        },
        "answers": str(answers),
        "summary": summarise(rows),
    }
    _save_summaries(SUMMARY_FILE, report)
    print(json.dumps(report["variants"][label]["summary"], indent=1))
    print(comparison_table(report))
    print(f"Answers in {answers}, summaries in {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
