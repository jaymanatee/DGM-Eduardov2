"""What the header and the query instruction are worth: two experiments on the gold set.

* Header ablation: the chunks without the ``[document › section]`` line that ``add_context``
  puts on top of every chunk. The same articles exist, almost word for word, in several
  agreements; without the header a chunk does not say which agreement it belongs to.
* Query instruction: Qwen3-Embedding embeds the question after an instruction that says
  what a relevant passage is. We compare no instruction at all, the generic one, two about
  this domain, and a wrong one (paraphrases) as a control: if the wrong one does no harm,
  the model is not listening to the instruction.

Only the query changes with the instruction, so the chunk index is reused; the header
ablation embeds the chunks once more without it (``rag/index/<strategy>_nocontext``).

Run::

    uv run python -m rag.ablations                      # structural and recursive
    uv run python -m rag.ablations --strategies structural

Writes ``reports/phase3_ablations.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.embed import DEFAULT_QUERY_INSTRUCTION

DOMAIN_INSTRUCTION = (
    "Given a question from a trader or customs professional about EU preferential rules "
    "of origin, retrieve the passage of a trade agreement, customs guidance or explanatory "
    "note that answers it."
)
INSTRUCTIONS = {
    "generic": DEFAULT_QUERY_INSTRUCTION,
    "none": "",
    "domain": DOMAIN_INSTRUCTION,
    "domain_agreement": DOMAIN_INSTRUCTION
    + " The passage must come from the agreement or guidance the question refers to.",
    "paraphrase": "Retrieve passages that are paraphrases of the question.",
}
# (name, header on, instruction): the first one is the configuration of rag/evaluate.py.
CONFIGS = [("header + generic", True, "generic"), ("no header + generic", False, "generic")] + [
    (f"header + {name}", True, name) for name in INSTRUCTIONS if name != "generic"
]


def run(strategies, documents, embedder, gold, index: str, pool: int) -> dict:
    from rag.evaluate import evaluate_strategy

    results: dict[str, dict] = {}
    for strategy in strategies:
        results[strategy] = {}
        for name, header, instruction in CONFIGS:
            args = argparse.Namespace(
                no_context=not header, index=index, pool=pool,
                instruction=INSTRUCTIONS[instruction],
            )  # fmt: skip
            r = evaluate_strategy(strategy, documents, embedder, gold, args)
            results[strategy][name] = {
                "header": header,
                "instruction": INSTRUCTIONS[instruction],
                **{k: r[k] for k in ("bm25", "dense", "hybrid", "by_kind", "lambda_stability")},
            }
    return results


def print_table(results: dict) -> None:
    print(
        f"{'strategy':<12}{'configuration':<26}{'dense R@5':>10}{'MRR':>6}{'cross':>7}{'es':>6}"
        f"{'hybrid λ':>10}{'R@5':>6}{'MRR':>6}{'cross':>7}"
    )
    for strategy, configs in results.items():
        for i, (name, r) in enumerate(configs.items()):
            d, h, kinds = r["dense"], r["hybrid"], r["by_kind"]
            print(
                f"{strategy if i == 0 else '':<12}{name:<26}{d['recall@5']:>10.2f}{d['mrr']:>6.2f}"
                f"{kinds['dense']['cross']['mrr']:>7.2f}{kinds['dense']['es']['mrr']:>6.2f}"
                f"{h['best_lambda']:>10}{h['recall@5']:>6.2f}{h['mrr']:>6.2f}"
                f"{kinds['hybrid']['cross']['mrr']:>7.2f}"
            )


def main(argv: list[str] | None = None) -> None:
    from rag.embed import CachedEmbedder
    from rag.evaluate import GOLD_FILE, load_gold
    from rag.ingest import load_corpus

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--strategies", nargs="+", default=["structural", "recursive"])
    parser.add_argument("--gold", default=str(GOLD_FILE))
    parser.add_argument("--pool", type=int, default=20)
    parser.add_argument("--index", default="rag/index")
    parser.add_argument("--out", default="reports/phase3_ablations.json")
    args = parser.parse_args(argv)

    results = run(
        args.strategies, load_corpus(), CachedEmbedder(), load_gold(args.gold),
        args.index, args.pool,
    )  # fmt: skip
    print("Dense columns: Recall@5 and MRR; cross and es: MRR on those kinds of question.")
    print_table(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
