"""How much of the official rules of origin can the engine actually apply?

``evaluate_origin`` only decides a case when ``parse_rules_of_origin`` turns the rule text
into criteria the engine checks (CTH, CTSH, CC, MaxNOM, RVC, WO). Anything else goes to
``not_machine_readable`` and the tool asks for a manual review. That is the honest behaviour,
but it raises the question this script answers: **on what share of real rules can the system
give a verdict at all?**

It reads the rule sets already downloaded for the tool-use SFT
(``tool_use/sft/data/roo/<code>_<country>.json``, from the UK Trade Tariff API), parses each
one and counts. No model, no GPU, no network.

Run::

    uv run python -m tool_use.rule_coverage
    uv run python -m tool_use.rule_coverage --out reports/phase2_rule_coverage.json

Caveat on the sample: those products are the SFT catalogue, picked to be diverse and on
purpose to include headings with several subdivisions (textiles, food, vehicles). It is not a
random sample of the tariff, so the totals describe this catalogue, not the tariff as a whole.
The breakdown per HS chapter is the part that generalises.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tool_use.tools import AGREEMENTS, parse_rules_of_origin

ROO_DIR = Path("tool_use/sft/data/roo")
BY_COUNTRY = {name: AGREEMENTS[name].api_country for name in AGREEMENTS}
AGREEMENT_OF = {country: name for name, country in BY_COUNTRY.items()}

# Why a rule text cannot be turned into criteria. Checked in order; the first match wins.
# ``manufacturing_process`` goes first because those rules describe operations, not the origin
# or classification of materials: no amount of parsing turns them into something the engine
# can check. The rest are rules the engine almost understands, bar one condition.
REASONS = (
    ("manufacturing_process", ("knitting", "crocheting", "spinning", "weaving", "dyeing",
                               "making-up", "making up", "extrusion", "chemical reaction",
                               "purification", "isomer separation", "printing accompanied")),
    ("condition", ("provided that",)),
    ("weight_limit", ("net weight", "weight of")),
    ("specific_material_value", ("the value of non-originating materials of heading",
                                 "the value of all the materials of chapter",
                                 "the value of non-originating materials of chapter")),
    ("chapter_exclusion", ("except those of chapter", "from any other chapter")),
    ("exception", ("however,",)),
    ("material_must_originate", ("must be originating", "are originating")),
    ("wholly_obtained_variant", ("wholly obtained",)),
)


def reason_for(text: str) -> str:
    lowered = text.lower()
    for name, needles in REASONS:
        if any(needle in lowered for needle in needles):
            return name
    if re.search(r"\*and\*", lowered):
        return "several_conditions_joined_by_and"
    return "other"


def rule_files(directory: Path) -> list[tuple[str, str, Path]]:
    """Every downloaded rule set as (hs_code, agreement, path)."""
    found = []
    for path in sorted(directory.glob("*.json")):
        code, _, country = path.stem.partition("_")
        if country in AGREEMENT_OF:
            found.append((code, AGREEMENT_OF[country], path))
    return found


def measure(directory: Path = ROO_DIR) -> dict[str, Any]:
    """Parse every rule on disk and count what the engine could decide."""
    per_agreement: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_chapter: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    reasons: Counter[str] = Counter()
    examples: dict[str, str] = {}
    pairs = {"all_readable": 0, "some_readable": 0, "none_readable": 0}
    sets_readable = sets_total = failures = 0

    for code, agreement, path in rule_files(directory):
        try:
            parsed = parse_rules_of_origin(json.loads(path.read_text(encoding="utf-8")),
                                           agreement, code)
        except Exception as exc:  # noqa: BLE001 - a rule we cannot even parse is a finding
            failures += 1
            reasons[f"parse_error: {type(exc).__name__}"] += 1
            continue

        rule_sets = parsed["rule_sets"]
        readable = [s for s in rule_sets if s["criteria"]]
        sets_total += len(rule_sets)
        sets_readable += len(readable)
        per_agreement[agreement][0] += len(readable)
        per_agreement[agreement][1] += len(rule_sets)
        per_chapter[code[:2]][0] += len(readable)
        per_chapter[code[:2]][1] += len(rule_sets)

        if len(readable) == len(rule_sets):
            pairs["all_readable"] += 1
        elif readable:
            pairs["some_readable"] += 1
        else:
            pairs["none_readable"] += 1

        for rule_set in rule_sets:
            for text in rule_set["not_machine_readable"]:
                reason = reason_for(text)
                reasons[reason] += 1
                examples.setdefault(reason, f"[{code} {agreement}] {text}")

    def share(counts: dict[str, list[int]]) -> dict[str, dict[str, float | int]]:
        return {
            key: {"readable": ok, "total": total, "pct": round(100 * ok / total, 1)}
            for key, (ok, total) in sorted(counts.items())
            if total
        }

    n_pairs = sum(pairs.values())
    return {
        "products_x_agreements": n_pairs,
        "parse_failures": failures,
        "rule_sets": {
            "readable": sets_readable,
            "total": sets_total,
            "pct": round(100 * sets_readable / sets_total, 1) if sets_total else 0.0,
        },
        "pairs": {k: {"n": v, "pct": round(100 * v / n_pairs, 1)} for k, v in pairs.items()},
        "by_agreement": share(per_agreement),
        "by_hs_chapter": share(per_chapter),
        "reasons": dict(reasons.most_common()),
        "examples": examples,
    }


def report(result: dict[str, Any]) -> str:
    lines = [
        f"Reglas analizadas: {result['products_x_agreements']} pares producto+acuerdo "
        f"({result['rule_sets']['total']} conjuntos de reglas), "
        f"{result['parse_failures']} sin parsear.",
        "",
        f"Conjuntos con al menos un criterio aplicable: "
        f"{result['rule_sets']['readable']}/{result['rule_sets']['total']} "
        f"({result['rule_sets']['pct']} %)",
        "",
        "Por par producto+acuerdo:",
    ]
    labels = {
        "all_readable": "todas las subdivisiones aplicables",
        "some_readable": "algunas aplicables",
        "none_readable": "ninguna: revisión manual",
    }
    for key, label in labels.items():
        row = result["pairs"][key]
        lines.append(f"  {label:36} {row['n']:4d}  ({row['pct']} %)")

    for title, key in (("Por acuerdo", "by_agreement"), ("Por capítulo SA", "by_hs_chapter")):
        lines += ["", f"{title}:"]
        rows = sorted(result[key].items(), key=lambda kv: -kv[1]["total"])
        for name, row in rows:
            lines.append(f"  {name:8} {row['readable']:4d}/{row['total']:4d}  {row['pct']:5.1f} %")

    lines += ["", "Por qué no se puede aplicar:"]
    for reason, count in result["reasons"].items():
        lines.append(f"  {count:4d}  {reason}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", default=str(ROO_DIR), help="folder with the downloaded rules")
    parser.add_argument("--out", default="reports/phase2_rule_coverage.json")
    args = parser.parse_args()

    result = measure(Path(args.dir))
    print(report(result))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
