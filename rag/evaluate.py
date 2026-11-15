"""Phase 3 evaluation: Recall@k and MRR of each retriever and each chunking on the gold set.

The gold set (``rag/gold/gold.jsonl``) gives, for each question, the evidence that answers
it as the document and a short literal quote, not as chunk ids: chunk ids change with the
chunking, and we compare four of them. For each chunking, the relevant chunks of a piece of
evidence are the ones that hold most of the quote (``resolve_gold``)::

    {"id": "q01", "question": "...", "answer": "...",
     "evidence": [{"source": "tca/rules_of_origin.md", "quote": "..."}], "notes": "..."}

A question whose answer is not in the corpus has ``"evidence": []``: it does not count for
retrieval, and the generator should say it does not know. Recall@k counts a piece of evidence
as found when one of its chunks is in the top k (two chunks equally valid count once); the
reciprocal rank is 1 / the position of the first chunk of any piece of evidence.

Run::

    uv run python -m rag.evaluate                       # every strategy, every retriever
    uv run python -m rag.evaluate --strategies structural --no-context
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

GOLD_FILE = Path(__file__).parent / "gold" / "gold.jsonl"
KS = (1, 3, 5, 10)
LAMBDAS = tuple(round(0.1 * i, 1) for i in range(11))


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant chunks that appear in the top k."""
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / position of the first relevant chunk (0 when none is retrieved)."""
    for position, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / position
    return 0.0


def evaluate_retriever(retriever, gold: list[dict], k: int) -> dict[str, float]:
    recalls, rrs = [], []
    for item in gold:
        ranked = [hit.chunk.id for hit in retriever.search(item["question"], top_k=max(k, 10))]
        relevant = set(item["relevant_chunk_ids"])
        recalls.append(recall_at_k(ranked, relevant, k))
        rrs.append(reciprocal_rank(ranked, relevant))
    n = max(len(gold), 1)
    return {f"recall@{k}": sum(recalls) / n, "mrr": sum(rrs) / n, "n_questions": len(gold)}


def load_gold(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------- gold set and chunks


class GoldError(ValueError):
    """A quote of the gold set is not in its document."""


def find_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Offsets of ``quote`` in ``text``, matching any run of white space between words."""
    words = quote.split()
    if not words:
        return None
    match = re.search(r"\s+".join(re.escape(w) for w in words), text)
    return (match.start(), match.end()) if match else None


def relevant_chunks(span: tuple[int, int], chunks) -> set[str]:
    """The chunks that hold at least half of the quote; if none does (a quote cut in three),
    the ones that hold the most of it."""
    start, end = span
    overlaps = {c.id: min(end, c.end) - max(start, c.start) for c in chunks}
    overlaps = {cid: o for cid, o in overlaps.items() if o > 0}
    if not overlaps:
        return set()
    enough = {cid for cid, o in overlaps.items() if o >= (end - start) / 2}
    best = max(overlaps.values())
    return enough or {cid for cid, o in overlaps.items() if o == best}


def resolve_gold(gold: list[dict], documents, chunks) -> list[dict]:
    """Each answerable question with ``relevant_groups`` (one set of chunk ids per piece of
    evidence) and ``relevant_chunk_ids`` (their union), for this chunking."""
    docs = {d.source: d for d in documents}
    by_doc: dict[str, list] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.source, []).append(chunk)
    resolved, errors = [], []
    for item in gold:
        groups = []
        for evidence in item.get("evidence", []):
            doc = docs.get(evidence["source"])
            span = find_quote(doc.text, evidence["quote"]) if doc else None
            if span is None:
                errors.append(
                    f"{item.get('id', item['question'][:40])}: {evidence['quote'][:60]!r}"
                )
                continue
            groups.append(relevant_chunks(span, by_doc.get(doc.source, [])))
        if groups:
            ids = sorted(set().union(*groups))
            resolved.append({**item, "relevant_groups": groups, "relevant_chunk_ids": ids})
    if errors:
        raise GoldError("quotes not found in their document:\n  " + "\n  ".join(errors))
    return resolved


def evidence_recall_at_k(ranked_ids: list[str], groups: list[set[str]], k: int) -> float:
    """Fraction of the pieces of evidence with at least one of their chunks in the top k."""
    top = set(ranked_ids[:k])
    return sum(bool(group & top) for group in groups) / len(groups) if groups else 0.0


def score_rankings(rankings: list[list[str]], gold: list[dict], ks=KS) -> dict[str, float]:
    """Recall@k for every k and MRR over the resolved gold questions."""
    n = max(len(gold), 1)
    result = {
        f"recall@{k}": sum(
            evidence_recall_at_k(r, item["relevant_groups"], k)
            for r, item in zip(rankings, gold, strict=True)
        )
        / n
        for k in ks
    }
    result["mrr"] = (
        sum(
            reciprocal_rank(r, set(item["relevant_chunk_ids"]))
            for r, item in zip(rankings, gold, strict=True)
        )
        / n
    )
    return {key: round(value, 4) for key, value in result.items()}


def evaluate_strategy(strategy, documents, embedder, gold, args) -> dict:
    """BM25, dense and the hybrid at every λ for one chunking strategy."""
    from rag.chunking import describe
    from rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever
    from rag.strategies import build_chunks

    chunks = build_chunks(strategy, documents, embedder, context=not args.no_context)
    resolved = resolve_gold(gold, documents, chunks)
    suffix = "" if not args.no_context else "_nocontext"
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(
        chunks, embedder, persist_dir=Path(args.index) / f"{strategy}{suffix}",
        instruction=args.instruction,
    )  # fmt: skip
    pool = max(max(KS), args.pool)
    bm25_hits = [bm25.search(item["question"], pool) for item in resolved]
    dense_hits = [dense.search(item["question"], pool) for item in resolved]

    def ids(hits):
        return [[h.chunk.id for h in question] for question in hits]

    hybrid = {
        str(lam): score_rankings(
            ids(
                HybridRetriever.fuse(b, d, lam, max(KS))
                for b, d in zip(bm25_hits, dense_hits, strict=True)
            ),
            resolved,
        )
        for lam in LAMBDAS
    }
    best = max(hybrid, key=lambda lam: (hybrid[lam]["recall@5"], hybrid[lam]["mrr"]))
    return {
        "chunks": {k: round(v, 1) for k, v in describe(chunks).items()},
        "n_questions": len(resolved),
        "bm25": score_rankings(ids(bm25_hits), resolved),
        "dense": score_rankings(ids(dense_hits), resolved),
        "hybrid": {"best_lambda": float(best), **hybrid[best]},
        "hybrid_sweep": hybrid,
        "per_question": [
            {
                "id": item.get("id"),
                "relevant": item["relevant_chunk_ids"],
                "bm25": [h.chunk.id for h in b[:5]],
                "dense": [h.chunk.id for h in d[:5]],
            }
            for item, b, d in zip(resolved, bm25_hits, dense_hits, strict=True)
        ],
    }


def print_table(results: dict) -> None:
    head = f"{'strategy':<16}{'chunks':>7}{'mean':>6}  {'retriever':<14}" + "".join(
        f"{'R@' + str(k):>7}" for k in KS
    )
    print(head + f"{'MRR':>7}")
    for strategy, r in results.items():
        rows = [("bm25", r["bm25"]), ("dense", r["dense"])]
        rows.append((f"hybrid λ={r['hybrid']['best_lambda']}", r["hybrid"]))
        for i, (name, m) in enumerate(rows):
            first = (
                f"{strategy:<16}{r['chunks']['n_chunks']:>7}{r['chunks']['mean_chars']:>6.0f}"
                if i == 0
                else " " * 29
            )
            print(
                f"{first}  {name:<14}"
                + "".join(f"{m[f'recall@{k}']:>7.2f}" for k in KS)
                + f"{m['mrr']:>7.2f}"
            )


def main(argv: list[str] | None = None) -> None:
    from rag.embed import DEFAULT_QUERY_INSTRUCTION, CachedEmbedder
    from rag.ingest import load_corpus
    from rag.strategies import STRATEGIES

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gold", default=str(GOLD_FILE))
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    parser.add_argument("--no-context", action="store_true", help="chunks without the header")
    parser.add_argument("--instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--pool", type=int, default=20, help="hits per retriever in the hybrid")
    parser.add_argument("--index", default="rag/index")
    parser.add_argument("--out", default="reports/phase3_eval.json")
    args = parser.parse_args(argv)

    gold = load_gold(args.gold)
    documents = load_corpus()
    embedder = CachedEmbedder()
    print(f"{len(gold)} gold questions, {len(documents)} documents")
    results = {s: evaluate_strategy(s, documents, embedder, gold, args) for s in args.strategies}
    print_table(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    settings = {"no_context": args.no_context, "instruction": args.instruction, "pool": args.pool}
    out.write_text(
        json.dumps({"settings": settings, "results": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
