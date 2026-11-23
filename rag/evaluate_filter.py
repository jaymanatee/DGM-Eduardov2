"""Phase 3: the retrievers with and without the agreement filter, on the gold set.

The filter (``rag/agreements.py``) keeps only the passages of the agreements a question
names, plus general documents. This script measures what it changes in retrieval, with the
serving configuration (structural chunking with header, generic instruction, pool 20) and
the same metrics as ``rag/evaluate.py``:

* Recall@k and MRR of BM25, dense and the hybrid at the served λ = 0.3, without and with
  the filter, overall and by kind of question;
* the λ sweep with the filter, to see whether the filter would move the choice of λ (the
  served λ is not changed: it was chosen without the filter);
* how many of the top 5 passages come from an agreement the question does not name;
* for each question the filter touches, the rank of the first relevant chunk before and
  after, so the gains and losses can be read one by one;
* a check that the filter never removes the gold evidence of a question.

The patterns of ``rag/agreements.py`` were written knowing the gold questions (they name
agreements and countries, nothing question-specific), so these numbers are an upper bound
of what an unseen question gains; ``tests/test_rag_agreements.py`` checks the patterns on
the agent's held-out knowledge tasks too.

Run::

    uv run --extra rag python -m rag.evaluate_filter     # reports/phase3_agreement_filter.json

The generation with the filter is measured by ``rag/evaluate_generation.py`` with
``ARCA_RAG_AGREEMENT_FILTER=1`` and ``--label hybrid_agreement``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.evaluate import (
    GOLD_FILE,
    KS,
    LAMBDAS,
    choose_lambda,
    load_gold,
    resolve_gold,
    score_rankings,
    scores_by_kind,
)

TOP = 5


def first_rank(ranking: list[str], relevant: set[str]) -> int | None:
    return next((i for i, cid in enumerate(ranking, start=1) if cid in relevant), None)


def foreign_in_top(hits, keep, top: int = TOP) -> int:
    """Passages in the top ``top`` the filter would have removed."""
    if keep is None:
        return 0
    return sum(not keep(h.chunk.source) for h in hits[:top])


def evidence_removed(item: dict, keep) -> list[str]:
    """Sources of the gold evidence of ``item`` that the filter would reject."""
    if keep is None:
        return []
    return [e["source"] for e in item.get("evidence", []) if not keep(e["source"])]


def main(argv: list[str] | None = None) -> None:
    from rag.agreements import agreements_named, keep_for
    from rag.embed import DEFAULT_QUERY_INSTRUCTION, CachedEmbedder
    from rag.generate import HYBRID_LAMBDA, HYBRID_POOL, SERVING_STRATEGY
    from rag.ingest import load_corpus
    from rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever
    from rag.strategies import build_chunks

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gold", default=str(GOLD_FILE))
    parser.add_argument("--index", default="rag/index")
    parser.add_argument("--out", default="reports/phase3_agreement_filter.json")
    args = parser.parse_args(argv)

    gold = load_gold(args.gold)
    removed = {
        item["id"]: lost
        for item in gold
        if (lost := evidence_removed(item, keep_for(item["question"])))
    }
    documents = load_corpus()
    embedder = CachedEmbedder()
    chunks = build_chunks(SERVING_STRATEGY, documents, embedder)
    resolved = resolve_gold(gold, documents, chunks)
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(
        chunks, embedder, persist_dir=Path(args.index) / SERVING_STRATEGY,
        instruction=DEFAULT_QUERY_INSTRUCTION,
    )  # fmt: skip
    pool = max(max(KS), HYBRID_POOL)

    hits = {"off": {"bm25": [], "dense": []}, "on": {"bm25": [], "dense": []}}
    keeps = [keep_for(item["question"]) for item in resolved]
    for item, keep in zip(resolved, keeps, strict=True):
        q = item["question"]
        hits["off"]["bm25"].append(bm25.search(q, pool))
        hits["off"]["dense"].append(dense.search(q, pool))
        hits["on"]["bm25"].append(bm25.search(q, pool, keep=keep))
        hits["on"]["dense"].append(dense.search(q, pool, keep=keep))

    def ids(question_hits):
        return [[h.chunk.id for h in q] for q in question_hits]

    results, rankings = {}, {}
    for mode in ("off", "on"):
        b, d = hits[mode]["bm25"], hits[mode]["dense"]
        sweep_hits = {
            str(lam): [HybridRetriever.fuse(x, y, lam, max(KS)) for x, y in zip(b, d, strict=True)]
            for lam in LAMBDAS
        }
        sweep = {lam: score_rankings(ids(h), resolved) for lam, h in sweep_hits.items()}
        hybrid_hits = sweep_hits[str(HYBRID_LAMBDA)]
        final = {"bm25": ids(b), "dense": ids(d), "hybrid": ids(hybrid_hits)}
        rankings[mode] = final["hybrid"]
        results[mode] = {
            **{name: score_rankings(r, resolved) for name, r in final.items()},
            "best_lambda_with_this_setting": float(choose_lambda(sweep)),
            "hybrid_sweep": sweep,
            "by_kind": {name: scores_by_kind(r, resolved) for name, r in final.items()},
            "foreign_in_top5": sum(
                foreign_in_top(h, keep) for h, keep in zip(hybrid_hits, keeps, strict=True)
            ),
        }

    per_question = []
    for i, (item, keep) in enumerate(zip(resolved, keeps, strict=True)):
        if keep is None:
            continue
        relevant = set(item["relevant_chunk_ids"])
        before = first_rank(rankings["off"][i], relevant)
        after = first_rank(rankings["on"][i], relevant)
        per_question.append(
            {
                "id": item["id"],
                "agreements": sorted(agreements_named(item["question"])),
                "rank_before": before,
                "rank_after": after,
                "foreign_before": foreign_in_top(
                    HybridRetriever.fuse(
                        hits["off"]["bm25"][i], hits["off"]["dense"][i], HYBRID_LAMBDA, TOP
                    ),
                    keep,
                ),
            }
        )

    def ranks(row):  # a question with no relevant chunk in the top 10 counts as rank 99
        return row["rank_before"] or 99, row["rank_after"] or 99

    summary = {
        "n_questions": len(resolved),
        "n_filtered": len(per_question),
        "evidence_removed": removed,
        "improved": [r["id"] for r in per_question if ranks(r)[1] < ranks(r)[0]],
        "worse": [r["id"] for r in per_question if ranks(r)[1] > ranks(r)[0]],
    }
    settings = {
        "strategy": SERVING_STRATEGY,
        "lambda": HYBRID_LAMBDA,
        "pool": HYBRID_POOL,
        "instruction": DEFAULT_QUERY_INSTRUCTION,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"settings": settings, "summary": summary, "results": results}
    report["per_question"] = per_question
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{len(resolved)} questions, {len(per_question)} name an agreement")
    columns = "".join(f"{f'R@{k}':>8}" for k in KS)
    print(f"{'retriever':<10}{'filter':>7}{columns}{'MRR':>8}")
    for name in ("bm25", "dense", "hybrid"):
        for mode in ("off", "on"):
            r = results[mode][name]
            row = f"{name:<10}{mode:>7}" + "".join(f"{r[f'recall@{k}']:>8.3f}" for k in KS)
            print(row + f"{r['mrr']:>8.3f}")
    for mode in ("off", "on"):
        print(
            f"filter {mode}: {results[mode]['foreign_in_top5']} passages of another agreement "
            f"in the hybrid top 5; best λ with this setting "
            f"{results[mode]['best_lambda_with_this_setting']}"
        )
    print(f"better rank: {', '.join(summary['improved']) or '-'}")
    print(f"worse rank:  {', '.join(summary['worse']) or '-'}")
    if removed:
        print(f"WARNING: the filter removes gold evidence of {', '.join(removed)}")
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()