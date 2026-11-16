"""A second stage: Qwen3-Reranker reorders the first candidates of a retriever.

The dense retriever embeds the question and every chunk *apart* (a bi-encoder): the chunk's
vector is computed once, without knowing the question. A reranker (a cross-encoder) reads
the question and one chunk *together* and judges whether the chunk answers it. That is far
more expensive, one forward pass per pair, so it only reorders the ``pool`` first hits of a
retriever: the retriever makes sure the right chunk is among them, the reranker brings it
to the top.

Qwen3-Reranker is a Qwen3 LLM trained to answer "yes" or "no" to: does this document meet
the requirements of this query and instruction? It does not generate: the score is the
probability of "yes" against "no" at the first position of the answer,
``P(yes) / (P(yes) + P(no))``, read from the logits of those two tokens. The prompt is the
one of the model card.

``python -m rag.rerank`` measures it on the gold set, with the serving chunking (structural
with headers): for each retriever and pool size, Recall@k and MRR before and after
reranking, which questions it fixes and which it breaks. Every candidate is scored once:
the union of the first ``max(pools)`` hits of the three retrievers, so each retriever and
pool reuses the same scores. Writes ``reports/phase3_rerank_<instruction>.json``.

Run::

    HF_HOME=./hf_cache uv run python -m rag.rerank                          # generic
    HF_HOME=./hf_cache uv run python -m rag.rerank --instruction agreement
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from rag.embed import DEFAULT_QUERY_INSTRUCTION
from rag.retriever import Hit

DEFAULT_RERANKER = "Qwen/Qwen3-Reranker-0.6B"
# What "relevant" means for the reranker. ``generic`` is the embedding's instruction;
# ``agreement`` adds what the generic one misses on the gold set: a passage about another
# agreement does not answer, however close its wording.
INSTRUCTIONS = {
    "generic": DEFAULT_QUERY_INSTRUCTION,
    "agreement": (
        "Given a trader's question about EU preferential rules of origin, judge whether the "
        "passage answers it for the trade agreement, country or product named in the "
        "question. A passage about a different agreement does not answer it."
    ),
}
POOL = 20
POOLS = (10, 20, 30)

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_pair(instruction: str, query: str, document: str) -> str:
    """The user turn of the model card: instruction, query and document, labelled."""
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def yes_probability(yes_logit: float, no_logit: float) -> float:
    """P(yes) normalised against P(no) only: a softmax over the two logits."""
    return 1.0 / (1.0 + math.exp(no_logit - yes_logit))


class Reranker:
    """Scores (query, document) pairs with Qwen3-Reranker; loaded on first use.

    ``batch_size`` pairs go through the model at a time: 20 candidates in batches of 8 keep
    the peak memory low next to the generator. Documents longer than ``max_length`` tokens
    are cut (the structural chunks are under 1,500 characters, some 400 tokens)."""

    def __init__(
        self,
        model_id: str = DEFAULT_RERANKER,
        instruction: str = DEFAULT_QUERY_INSTRUCTION,
        batch_size: int = 8,
        max_length: int = 1024,
    ):
        self.model_id, self.instruction = model_id, instruction
        self.batch_size, self.max_length = batch_size, max_length
        self.model = self.tokenizer = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from rlm import disable_cudnn_attention

        disable_cudnn_attention()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        # Left padding: the score is read at the last position of every row.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype, device_map=device)
        self.model = model.eval()
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_ids = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(SUFFIX, add_special_tokens=False)

    def score(self, query: str, documents: list[str]) -> list[float]:
        """P(yes) for each document, in the order given."""
        import torch

        if self.model is None:
            self.load()
        room = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            pairs = [
                format_pair(self.instruction, query, d)
                for d in documents[start : start + self.batch_size]
            ]
            encoded = self.tokenizer(
                pairs, padding=False, truncation=True, max_length=room, add_special_tokens=False
            )
            encoded["input_ids"] = [
                self.prefix_ids + ids + self.suffix_ids for ids in encoded["input_ids"]
            ]
            encoded.pop("attention_mask", None)
            batch = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
            with torch.no_grad():  # only the last position: the full logits do not fit
                logits = self.model(**batch.to(self.model.device), logits_to_keep=1).logits[:, -1]
            pair_logits = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1)
            scores += torch.softmax(pair_logits.float(), dim=1)[:, 1].tolist()
        return scores


class RerankingRetriever:
    """The ``pool`` first hits of ``base``, reordered by the reranker's P(yes).

    The scores returned are P(yes), in [0, 1]: they mean the same whatever the retriever
    below. Ties keep the base retriever's order."""

    def __init__(self, base, reranker: Reranker, pool: int = POOL):
        self.base, self.reranker, self.pool = base, reranker, pool
        self.name = f"{getattr(base, 'name', 'base')}+rerank"
        self.chunks = getattr(base, "chunks", None)

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        candidates = self.base.search(query, top_k=max(self.pool, top_k))
        scores = self.reranker.score(query, [h.chunk.text for h in candidates])
        order = sorted(range(len(candidates)), key=lambda i: -scores[i])[:top_k]
        return [Hit(candidates[i].chunk, scores[i]) for i in order]


# ----------------------------------------------------------------------------- evaluation


def rerank_ids(base_ids: list[str], scores: dict[str, float], pool: int) -> list[str]:
    """The first ``pool`` ids of a ranking, reordered by score (ties keep the base order)."""
    head = base_ids[:pool]
    return sorted(head, key=lambda cid: -scores[cid])


def first_relevant_rank(ranking: list[str], relevant: set[str]) -> int | None:
    return next((i for i, cid in enumerate(ranking, start=1) if cid in relevant), None)


def evaluate(resolved: list[dict], base_rankings: dict, scores: list[dict], pools=POOLS) -> dict:
    """Recall@k and MRR of every retriever, before and after reranking its ``pool`` first."""
    from rag.evaluate import evidence_recall_at_k, score_rankings, scores_by_kind

    results: dict = {}
    for name, rankings in base_rankings.items():
        entry = {"base": score_rankings(rankings, resolved)}
        for pool in pools:
            reranked = [rerank_ids(r, s, pool) for r, s in zip(rankings, scores, strict=True)]
            entry[f"rerank@{pool}"] = {
                # The ceiling: the reranker can only reorder what the retriever brought.
                "recall_of_pool": round(
                    sum(
                        evidence_recall_at_k(r, q["relevant_groups"], pool)
                        for r, q in zip(rankings, resolved, strict=True)
                    )
                    / len(resolved),
                    4,
                ),
                **score_rankings(reranked, resolved),
            }
        reranked = [rerank_ids(r, s, POOL) for r, s in zip(rankings, scores, strict=True)]
        entry["by_kind"] = {
            "base": scores_by_kind(rankings, resolved),
            f"rerank@{POOL}": scores_by_kind(reranked, resolved),
        }
        entry["per_question"] = []
        for q, before, after in zip(resolved, rankings, reranked, strict=True):
            relevant = set(q["relevant_chunk_ids"])
            entry["per_question"].append(
                {
                    "id": q.get("id"),
                    "kind": q.get("kind"),
                    "rank_before": first_relevant_rank(before, relevant),
                    "rank_after": first_relevant_rank(after, relevant),
                }
            )
        results[name] = entry
    return results


def changes_in_top(per_question: list[dict], k: int = 5) -> dict[str, list[str]]:
    """Questions whose evidence enters (fixed) or leaves (broken) the top ``k``."""

    def inside(rank):
        return rank is not None and rank <= k

    moves = [(q["id"], inside(q["rank_before"]), inside(q["rank_after"])) for q in per_question]
    return {
        "fixed": [qid for qid, before, after in moves if after and not before],
        "broken": [qid for qid, before, after in moves if before and not after],
    }


def print_table(results: dict) -> None:
    cols = ("recall@1", "recall@3", "recall@5", "mrr")
    print(f"{'retriever':<10}{'ranking':<12}{'pool R':>8}" + "".join(f"{c:>10}" for c in cols))
    for name, entry in results.items():
        for label, row in entry.items():
            if label in ("by_kind", "per_question"):
                continue
            pool = f"{row['recall_of_pool']:.2f}" if "recall_of_pool" in row else "-"
            print(f"{name:<10}{label:<12}{pool:>8}" + "".join(f"{row[c]:>10.3f}" for c in cols))


def main(argv: list[str] | None = None) -> None:
    import torch

    from rag.embed import CachedEmbedder
    from rag.evaluate import GOLD_FILE, load_gold, resolve_gold
    from rag.generate import HYBRID_LAMBDA, HYBRID_POOL, SERVING_STRATEGY
    from rag.ingest import load_corpus
    from rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever
    from rag.strategies import build_chunks

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gold", default=str(GOLD_FILE))
    parser.add_argument("--model", default=DEFAULT_RERANKER)
    parser.add_argument("--instruction", choices=INSTRUCTIONS, default="generic")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", help="default: reports/phase3_rerank_<instruction>.json")
    args = parser.parse_args(argv)

    documents = load_corpus()
    embedder = CachedEmbedder()
    chunks = build_chunks(SERVING_STRATEGY, documents, embedder)
    resolved = resolve_gold(load_gold(args.gold), documents, chunks)
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, embedder, persist_dir=Path("rag/index") / SERVING_STRATEGY)
    depth = max(POOLS)
    hits = {
        "bm25": [bm25.search(q["question"], top_k=depth) for q in resolved],
        "dense": [dense.search(q["question"], top_k=depth) for q in resolved],
    }
    # The hybrid as served: fused from the first HYBRID_POOL of each list (asking it for
    # ``depth`` hits would fuse deeper lists, and min-max would rescale them differently).
    hits["hybrid"] = [
        HybridRetriever.fuse(b[:HYBRID_POOL], d[:HYBRID_POOL], HYBRID_LAMBDA, depth)
        for b, d in zip(hits["bm25"], hits["dense"], strict=True)
    ]
    base_rankings = {
        name: [[h.chunk.id for h in qh] for qh in per_q] for name, per_q in hits.items()
    }

    reranker = Reranker(args.model, INSTRUCTIONS[args.instruction], batch_size=args.batch_size)
    reranker.load()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    texts = {c.id: c.text for c in chunks}
    scores, seconds = [], []
    for i, q in enumerate(resolved):
        candidates = list(dict.fromkeys(cid for name in hits for cid in base_rankings[name][i]))
        started = time.perf_counter()
        values = reranker.score(q["question"], [texts[cid] for cid in candidates])
        seconds.append((time.perf_counter() - started) / len(candidates))
        scores.append(dict(zip(candidates, values, strict=True)))
        print(f"{q['id']}: {len(candidates)} candidates scored")

    results = evaluate(resolved, base_rankings, scores)
    report = {
        "settings": {
            "strategy": SERVING_STRATEGY,
            "hybrid_lambda": HYBRID_LAMBDA,
            "hybrid_pool": HYBRID_POOL,
            "model": args.model,
            "instruction": args.instruction,
            "instruction_text": INSTRUCTIONS[args.instruction],
            "pools": list(POOLS),
            "n_questions": len(resolved),
        },
        "cost": {
            "seconds_per_pair": round(sum(seconds) / len(seconds), 4),
            f"seconds_per_question_pool_{POOL}": round(POOL * sum(seconds) / len(seconds), 2),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if torch.cuda.is_available()
            else None,
        },
        "changes_in_top5": {
            name: changes_in_top(entry["per_question"]) for name, entry in results.items()
        },
        "results": results,
    }
    out = Path(args.out or f"reports/phase3_rerank_{args.instruction}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_table(results)
    print(
        json.dumps({"cost": report["cost"], "changes_in_top5": report["changes_in_top5"]}, indent=1)
    )
    print(f"Written {out}")


if __name__ == "__main__":
    main()
