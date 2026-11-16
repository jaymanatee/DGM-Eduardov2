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

The hybrid's λ is chosen by MRR (the generator reads the first chunks, so where the right
one lands matters more than whether it is somewhere in the top 5), ties broken by Recall@5.
As λ is chosen on the same questions it is scored on, ``lambda_stability`` repeats the choice
on random halves of the gold set and scores it on the other half.

Run::

    uv run python -m rag.evaluate                       # every strategy, every retriever
    uv run python -m rag.evaluate --strategies structural --no-context
    uv run python -m rag.evaluate --strategies semantic@0.6 semantic@0.7 \\
        --out reports/phase3_semantic_sweep.json        # the semantic threshold
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

GOLD_FILE = Path(__file__).parent / "gold" / "gold.jsonl"
KS = (1, 3, 5, 10)
LAMBDAS = tuple(round(0.1 * i, 1) for i in range(11))
SPLITS = 200


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


def choose_lambda(sweep: dict[str, dict[str, float]]) -> str:
    """The λ with the best MRR; ties broken by Recall@5, then by the smaller λ."""
    return max(sweep, key=lambda lam: (sweep[lam]["mrr"], sweep[lam]["recall@5"], -float(lam)))


def lambda_stability(
    rankings: dict[str, list[list[str]]], gold: list[dict], n_splits: int = SPLITS, seed: int = 0
) -> dict:
    """Choose λ on a random half of the questions and score it on the other half.

    ``rankings`` maps each λ to the ranking of every question. Returns how often each λ is
    chosen, and the mean MRR on the held-out half with the chosen λ, with the best λ of
    that half (what a perfect choice would give) and with λ = 0 (the dense retriever alone).
    """
    rng, order = random.Random(seed), list(range(len(gold)))
    chosen, held_out, best, dense = Counter(), [], [], []
    for _ in range(n_splits):
        rng.shuffle(order)
        halves = order[: len(order) // 2], order[len(order) // 2 :]

        def sweep(idx):
            sub = [gold[i] for i in idx]
            return {lam: score_rankings([r[i] for i in idx], sub) for lam, r in rankings.items()}

        tune, test = sweep(halves[0]), sweep(halves[1])
        lam = choose_lambda(tune)
        chosen[lam] += 1
        held_out.append(test[lam]["mrr"])
        best.append(max(s["mrr"] for s in test.values()))
        dense.append(test[min(test, key=float)]["mrr"])

    def mean(xs):
        return round(sum(xs) / len(xs), 4)

    return {
        "n_splits": n_splits,
        "chosen": dict(sorted(chosen.items(), key=lambda kv: float(kv[0]))),
        "held_out_mrr": mean(held_out),
        "held_out_mrr_best_lambda": mean(best),
        "held_out_mrr_dense": mean(dense),
    }


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

    rankings = {
        str(lam): ids(
            HybridRetriever.fuse(b, d, lam, max(KS))
            for b, d in zip(bm25_hits, dense_hits, strict=True)
        )
        for lam in LAMBDAS
    }
    hybrid = {lam: score_rankings(r, resolved) for lam, r in rankings.items()}
    best = choose_lambda(hybrid)
    return {
        "chunks": {k: round(v, 1) for k, v in describe(chunks).items()},
        "n_questions": len(resolved),
        "bm25": score_rankings(ids(bm25_hits), resolved),
        "dense": score_rankings(ids(dense_hits), resolved),
        "hybrid": {"best_lambda": float(best), **hybrid[best]},
        "hybrid_sweep": hybrid,
        "lambda_stability": lambda_stability(rankings, resolved),
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


SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # categorical slots 1-4, fixed order
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot_sweep(results: dict, path: Path) -> None:
    """MRR and Recall@5 of the hybrid against λ, one line per strategy, chosen λ marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    ends: list[list] = []
    for ax, metric, title in zip(axes, ("mrr", "recall@5"), ("MRR", "Recall@5"), strict=True):
        for color, (strategy, r) in zip(SERIES, results.items(), strict=False):
            lams = sorted(r["hybrid_sweep"], key=float)
            ys = [r["hybrid_sweep"][lam][metric] for lam in lams]
            xs = [float(lam) for lam in lams]
            ax.plot(xs, ys, color=color, linewidth=2, label=strategy)
            best = r["hybrid"]["best_lambda"]
            ax.scatter([best], [r["hybrid_sweep"][str(best)][metric]], s=48, color=color,
                       edgecolor=SURFACE, linewidth=2, zorder=3)  # fmt: skip
            ends.append([ys[-1], strategy])
        # Direct labels at the right end, pushed apart so that close lines do not overlap.
        low, high = ax.get_ylim()
        gap, previous = 0.045 * (high - low), -float("inf")
        for end in sorted(ends):
            end[0] = previous = max(end[0], previous + gap)
            ax.annotate(end[1], (1.0, end[0]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=MUTED)  # fmt: skip
        ends.clear()
        ax.set_title(f"Hybrid {title} by λ (dot: chosen λ)", loc="left", color=INK, fontsize=10)
        ax.set_xlabel("λ  (0 = dense only, 1 = BM25 only)", color=MUTED)
        ax.set_xlim(-0.03, 1.22)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED)
    axes[0].legend(loc="lower left", frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def _strategy(name: str) -> str:
    from rag.strategies import STRATEGIES, split_strategy

    try:
        base, _ = split_strategy(name)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if base not in STRATEGIES:
        raise argparse.ArgumentTypeError(f"unknown strategy {name!r}: {', '.join(STRATEGIES)}")
    return name


def main(argv: list[str] | None = None) -> None:
    from rag.embed import DEFAULT_QUERY_INSTRUCTION, CachedEmbedder
    from rag.ingest import load_corpus
    from rag.strategies import STRATEGIES

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gold", default=str(GOLD_FILE))
    parser.add_argument(
        "--strategies", nargs="+", default=list(STRATEGIES), type=_strategy,
        help=f"{', '.join(STRATEGIES)}; semantic@<threshold> for another threshold",
    )  # fmt: skip
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
    if len(results) <= len(SERIES):  # one fixed colour per strategy, never cycled
        png = out.with_name(out.stem + "_lambda.png")
        plot_sweep(results, png)
        print(f"λ sweep drawn in {png}")


if __name__ == "__main__":
    main()
