"""Build the synthetic tool-use dataset: train, validation and test trajectories.

Splits are by product, so the synthetic test measures products the model has never seen.
Three test products have a sibling in training with the other subdivision of the same rule
(a diesel van against a hybrid one, plain trousers against embroidered ones, toothbrushes),
which is how we see whether the model reads the subdivision or remembers it. Concept topics
and the agreements we do not cover are split the same way. The phase 2 benchmark stays
apart: none of its products shares a heading with the catalogue.

Within each kind, outcomes have quotas: at random most products come out originating, and a
model trained on that would learn to expect "yes".

Run::

    uv run python -m tool_use.sft.build --out outputs/tool_sft
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from tool_use import tools
from tool_use.sft import lookups, no_tools
from tool_use.sft.catalog import PRODUCTS, Product
from tool_use.sft.generate import (
    Trajectory,
    eligible_pairs,
    evaluation_trajectory,
    pick,
    random_today,
)

TEST_PRODUCTS = frozenset(
    {
        "hair dryer",
        "forklift truck",
        "electric bicycle",
        "leather shoes",
        "porcelain tableware",
        "diesel van",
        "women's plain cotton trousers",
        "toothbrushes",
        "smartphone",
        "acrylic paint",
    }
)
VAL_PRODUCTS = frozenset({"gas cooker", "plastic bags", "car seat", "boiled sweets", "syringes"})
TEST_TOPICS = frozenset({"approved_exporter", "hs_structure"})
VAL_TOPICS = frozenset({"importer_knowledge"})
TEST_AGREEMENTS = frozenset({"Chile", "Ukraine"})  # of no_tools.OTHER_AGREEMENTS, by country
VAL_AGREEMENTS = frozenset({"Morocco"})

# Trajectories per kind in a training set of 2,000; validation and test are a tenth of it.
COUNTS = {
    "evaluation": 450,
    "subdivision": 150,
    "currency": 100,
    "statement": 300,
    "rule": 200,
    "rate": 160,
    "vat": 140,
    "parallel": 140,
    "concept": 110,
    "off_topic": 30,
    "out_of_scope": 60,
    "missing_info": 100,
    "api_error": 60,
}
# Shares of outcomes where chance alone would be lopsided; the rest is left as it comes.
QUOTAS = {
    "evaluation": {
        "originating": 0.45,
        "not_originating": 0.30,
        "by_hand": 0.20,
        "needs_fob": 0.05,
    },
    "statement": {
        "statement_issued": 0.45,
        "originating_needs_reference": 0.20,
        "not_originating": 0.20,
        "by_hand": 0.15,
    },
}


def split_of_product(product: Product) -> str:
    if product.en in TEST_PRODUCTS:
        return "test"
    return "val" if product.en in VAL_PRODUCTS else "train"


def _maker(kind: str, split: str) -> Callable[[random.Random], Trajectory]:
    """A function that makes one trajectory of this kind with this split's products/topics."""
    products = [p for p in PRODUCTS if split_of_product(p) == split]
    pairs = eligible_pairs(products)
    sub_pairs = [(p, a) for p, a in pairs if p.subdivisions.get(a)]
    topics = [
        t
        for t in no_tools.CONCEPTS
        if (t in TEST_TOPICS) == (split == "test") and (t in VAL_TOPICS) == (split == "val")
    ]
    others = [
        a
        for a in no_tools.OTHER_AGREEMENTS
        if (a[3] in TEST_AGREEMENTS) == (split == "test")
        and (a[3] in VAL_AGREEMENTS) == (split == "val")
    ]

    def lang(rng: random.Random) -> str:
        return rng.choice(("es", "en"))

    def make(rng: random.Random) -> Trajectory:
        today = random_today(rng)
        if kind in ("evaluation", "currency", "statement"):
            product, agreement = pick(rng, pairs)
            return evaluation_trajectory(
                rng,
                product,
                agreement,
                today,
                lang(rng),
                currency=kind == "currency",
                statement=kind == "statement",
            )
        if kind == "subdivision":
            product, agreement = pick(rng, sub_pairs)
            t = evaluation_trajectory(rng, product, agreement, today, lang(rng))
            t.kind = "subdivision"
            return t
        if kind == "rule":
            return lookups.rule_trajectory(
                rng, pick(rng, products), pick(rng, list(tools.AGREEMENTS)), today, lang(rng)
            )
        if kind == "rate":
            return lookups.rate_trajectory(rng, today, lang(rng))
        if kind == "vat":
            return lookups.vat_trajectory(rng, today, lang(rng))
        if kind == "parallel":
            return lookups.parallel_trajectory(rng, products, today, lang(rng))
        if kind == "api_error":
            return lookups.error_trajectory(rng, today, lang(rng))
        if kind == "concept":
            return no_tools.concept_trajectory(rng, today, lang(rng), pick(rng, topics))
        if kind == "off_topic":
            return no_tools.off_topic_trajectory(rng, today, lang(rng))
        if kind == "out_of_scope":
            with _only(no_tools, "OTHER_AGREEMENTS", tuple(others)):
                return no_tools.out_of_scope_trajectory(rng, pick(rng, products), today, lang(rng))
        if kind == "missing_info":
            product, agreement = pick(rng, pairs)
            return no_tools.missing_info_trajectory(rng, product, agreement, today, lang(rng))
        raise ValueError(kind)

    return make


class _only:
    """Temporarily narrow a module-level tuple (the agreements of this split)."""

    def __init__(self, module, name: str, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)


def build_split(split: str, scale: float, seed: int) -> list[Trajectory]:
    rng = random.Random(f"{seed}-{split}")
    out: list[Trajectory] = []
    for kind, count in COUNTS.items():
        n = max(1, round(count * scale))
        make = _maker(kind, split)
        quota = QUOTAS.get(kind)
        if quota is None:
            out += [make(rng) for _ in range(n)]
            continue
        wanted = {o: max(1, round(share * n)) for o, share in quota.items()}
        got: Counter[str] = Counter()
        attempts = 0
        while sum(got.values()) < n and attempts < 60 * n:
            attempts += 1
            t = make(rng)
            if got[t.outcome] < wanted.get(t.outcome, 0):
                got[t.outcome] += 1
                out.append(t)
        # A bucket that cannot be filled (no FOB-only rule among this split's products): top
        # up with whatever comes, so the kind keeps its size.
        while sum(got.values()) < n:
            t = make(rng)
            got[t.outcome] += 1
            out.append(t)
    rng.shuffle(out)
    for i, t in enumerate(out):
        t.id = f"{split}-{i:05d}"  # type: ignore[attr-defined]
    return out


def summary(trajectories: list[Trajectory]) -> dict:
    kinds = Counter(t.kind for t in trajectories)
    outcomes = Counter(f"{t.kind}/{t.outcome}" for t in trajectories)
    turns = Counter(sum(m["role"] == "assistant" for m in t.messages) for t in trajectories)
    return {
        "trajectories": len(trajectories),
        "assistant_turns": sum(k * v for k, v in turns.items()),
        "languages": dict(Counter(t.lang for t in trajectories)),
        "kinds": dict(sorted(kinds.items())),
        "outcomes": dict(sorted(outcomes.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/tool_sft")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-scale", type=float, default=1.0, help="1.0 = 2,000 trajectories")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for split, scale in (("train", args.train_scale), ("val", 0.1), ("test", 0.1)):
        trajectories = build_split(split, scale, args.seed)
        with (out / f"trajectories_{split}.jsonl").open("w", encoding="utf-8") as f:
            for t in trajectories:
                f.write(json.dumps({"id": t.id, **t.to_json()}, ensure_ascii=False) + "\n")  # type: ignore[attr-defined]
        report[split] = summary(trajectories)
        print(split, json.dumps(report[split]["kinds"]), report[split]["assistant_turns"], "turns")
    (out / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
