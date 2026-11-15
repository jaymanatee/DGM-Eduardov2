"""Why the semantic threshold is what it is: distances between consecutive sentences.

Run::

    uv run python -m rag.semantic_analysis

Writes ``reports/phase3_semantic_distances.png``, two panels:

* the distance between each pair of consecutive sentences of one document (the TCA origin
  chapter by default), with the threshold and a mark where each article starts: if the
  semantic chunker found the structure, the peaks would sit on the marks;
* the distribution of the distances over the whole corpus, with the threshold.

and ``reports/phase3_semantic_thresholds.json``: the percentiles of the distances, the mean
distance just before an article heading and elsewhere, and the chunks each threshold gives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rag.chunking import (
    consecutive_distances,
    describe,
    headings_of,
    semantic_chunks,
    sentence_spans,
)
from rag.embed import CachedEmbedder
from rag.ingest import load_corpus
from rag.strategies import MAX_CHUNK_CHARS, SEMANTIC_MIN_CHARS, SEMANTIC_THRESHOLD

THRESHOLDS = (0.25, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES, THRESHOLD_COLOR = "#2a78d6", "#eb6834"


def distances_by_document(documents, embedder) -> dict[str, tuple[list, np.ndarray]]:
    out = {}
    for doc in documents:
        spans = sentence_spans(doc.text)
        vectors = embedder.embed_passages([doc.text[s:e] for s, e in spans])
        out[doc.source] = (spans, np.array(consecutive_distances(vectors)))
    return out


def before_heading(doc, spans, distances) -> np.ndarray:
    """Mask of the distances between a sentence and the start of an article or note."""
    starts = {h.start for h in headings_of(doc) if h.level == "unit"}
    mask = np.zeros(len(distances), bool)
    for i, (start, _) in enumerate(spans[1:]):
        mask[i] = start in starts
    return mask


def plot(doc, spans, distances, all_distances, threshold, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 2], "hspace": 0.45}
    )
    fig.patch.set_facecolor(SURFACE)
    mask = before_heading(doc, spans, distances)
    x = np.arange(1, len(distances) + 1)
    for i in np.flatnonzero(mask):
        top.axvline(x[i], color=GRID, linewidth=2, zorder=0)
    top.plot(x, distances, color=SERIES, linewidth=1.2, zorder=2)
    top.scatter(x[mask], distances[mask], s=22, color=SERIES, edgecolor=SURFACE, zorder=3,
                label="pair where an article starts")  # fmt: skip
    top.axhline(threshold, color=THRESHOLD_COLOR, linewidth=2, zorder=1)
    top.text(x[-1], threshold + 0.01, f"threshold {threshold}", color=INK, ha="right", va="bottom")
    above = (distances > threshold).sum()
    top.set_title(
        f"{doc.metadata['title']}\n{len(distances) + 1} sentences; {above} pairs above the "
        f"threshold, {mask.sum()} article starts (grey lines)",
        loc="left", color=INK, fontsize=10,
    )  # fmt: skip
    top.set_xlabel("sentence pair (i, i+1)", color=MUTED)
    top.set_ylabel("cosine distance", color=MUTED)
    top.legend(loc="lower right", frameon=False, fontsize=9)

    bottom.hist(all_distances, bins=60, color=SERIES, edgecolor=SURFACE, linewidth=1)
    bottom.axvline(threshold, color=THRESHOLD_COLOR, linewidth=2)
    share = (all_distances > threshold).mean()
    bottom.set_title(
        f"Whole corpus: {len(all_distances)} pairs, median {np.median(all_distances):.2f}; "
        f"{share:.0%} above the threshold",
        loc="left", color=INK, fontsize=10,
    )  # fmt: skip
    bottom.set_xlabel("cosine distance between consecutive sentences", color=MUTED)
    bottom.set_ylabel("pairs", color=MUTED)
    for ax in (top, bottom):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--document", default="tca/rules_of_origin.md")
    parser.add_argument("--threshold", type=float, default=SEMANTIC_THRESHOLD)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)

    documents = load_corpus()
    embedder = CachedEmbedder()
    by_doc = distances_by_document(documents, embedder)
    all_distances = np.concatenate([d for _, d in by_doc.values()])

    structured = {}
    for doc in documents:
        spans, distances = by_doc[doc.source]
        mask = before_heading(doc, spans, distances)
        if mask.sum() >= 5:
            structured[doc.source] = {
                "before_article": round(float(distances[mask].mean()), 3),
                "elsewhere": round(float(distances[~mask].mean()), 3),
                "n_articles": int(mask.sum()),
            }
    thresholds = {}
    for threshold in THRESHOLDS:
        for min_chars in (0, SEMANTIC_MIN_CHARS):
            chunks = semantic_chunks(
                documents, embedder.embed_passages, threshold, MAX_CHUNK_CHARS, min_chars
            )
            stats = {k: round(v, 1) for k, v in describe(chunks).items()}
            stats["under_100_chars"] = sum(len(c.text) < 100 for c in chunks)
            stats["at_the_size_cap"] = sum(len(c.text) > MAX_CHUNK_CHARS - 100 for c in chunks)
            thresholds[f"{threshold}/min{min_chars}"] = stats

    report = {
        "pairs": len(all_distances),
        "percentiles": {
            str(p): round(float(np.percentile(all_distances, p)), 3) for p in (10, 25, 50, 75, 90)
        },
        "distance_before_an_article": structured,
        "chunks_by_threshold": thresholds,
    }
    args.reports.mkdir(parents=True, exist_ok=True)
    out = args.reports / "phase3_semantic_thresholds.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    doc = next(d for d in documents if d.source == args.document)
    png = args.reports / "phase3_semantic_distances.png"
    plot(doc, *by_doc[doc.source], all_distances, args.threshold, png)
    print(
        json.dumps({k: report[k] for k in ("percentiles", "distance_before_an_article")}, indent=1)
    )
    print(f"Written {out} and {png}")


if __name__ == "__main__":
    main()
