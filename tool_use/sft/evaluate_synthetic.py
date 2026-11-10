"""Evaluate a model on the synthetic test trajectories (products it was never trained on).

Each test trajectory is a question with a known right conversation. The model answers the
question in the real tool loop, inside the trajectory's offline world (same rules, ECB
rates, VIES answers and date), so the tools answer it as they answered the right
conversation. Then:

* ``selection_ok``: the first call is one the right conversation makes first; or, if the
  right answer needs no tool, the model calls none.
* ``calls_ok``: every call of the right conversation is among the model's (materials are
  compared by code, origin, value and declaration, not by name), and nothing forbidden
  happened: no statement where none was due, no statement on an id the model was not given
  by an originating evaluation, no made-up REX number. A first evaluation that only drew
  the list of subdivisions is not required: looking the rule up first is as good.
* ``answer_ok``: what the answer must say. For evaluations, the verdict of its first
  paragraph (yes, no, or none when the rule must be checked by hand or the FOB price is
  missing); the euro amount of a conversion; the holder or the invalidity of a VAT number;
  the file of an issued statement; the thresholds of a rule. The verdict is read in the
  first sentence (after a currency conversion, if any): that is where it goes, and the
  explanation after it may well say that a material is not originating.
* ``false_statement``: a statement issued where the right conversation issued none. The
  safety figure.

The scoring is a pure function of the trajectory and the model's calls and answer; the
tests feed it the right conversation and expect full marks.

Run (with the model chosen by the ``ARCA_TOOLS_*`` variables, as for the benchmark)::

    ARCA_TOOLS_ADAPTER=tool_use/weights/sft_think ARCA_TOOLS_THINKING=1 \\
    uv run --extra train python -m tool_use.sft.evaluate_synthetic \\
        --data outputs/tool_sft/trajectories_test.jsonl --out reports/phase2_sft_test_think.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from tool_use.evaluate import canonical, matches
from tool_use.executor import ToolLoop, ToolsResponse
from tool_use.registry import ToolRegistry
from tool_use.sft.generate import Offline, Trajectory, assistant_text
from tool_use.sft.render import load_trajectories
from tool_use.tools import default_registry

EVALUATION_KINDS = {"evaluation", "subdivision", "currency", "statement"}
NO_TOOL_KINDS = {"concept", "off_topic", "out_of_scope", "missing_info"}
EXPECTED_VERDICT = {
    "originating": "yes",
    "statement_issued": "yes",
    "originating_needs_reference": "yes",
    "not_originating": "no",
    "by_hand": "none",
    "needs_fob": "none",
}
NEGATIVE = (
    "not originating",
    "is not originating",
    "are not originating",
    "does not qualify",
    "do not qualify",
    "doesn't qualify",
    "not eligible",
    "no es originari",
    "no son originari",
    "no tiene origen",
    "no tienen origen",
    "no cumple las reglas",
    "no cumplen las reglas",
    "no califica",
)
POSITIVE = (
    "is originating",
    "are originating",
    "qualifies",
    "qualify for",
    "es originari",
    "son originari",
    "tiene origen preferencial",
    "tienen origen preferencial",
    "cumple las reglas",
    "cumplen las reglas",
)


# ----------------------------------------------------------------------------- the gold


def _calls_in(message: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        json.loads(body)
        for body in re.findall(r"<tool_call>\n(.*?)\n</tool_call>", message["content"], re.S)
    ]


def gold_calls(t: Trajectory) -> list[dict[str, Any]]:
    """The calls of the right conversation, each with its tool message ("result"/"error")."""
    calls, observations = [], []
    for m in t.messages:
        if m["role"] == "assistant":
            calls += _calls_in(m)
        elif m["role"] == "tool":
            observations.append(json.loads(m["content"]))
    return [{**c, **o} for c, o in zip(calls, observations, strict=True)]


def required(call: dict[str, Any]) -> bool:
    """A first evaluation that only drew the list of subdivisions is not required."""
    return not (call["name"] == "evaluate_origin" and "subdivisions" in call.get("error", ""))


def _expectation(call: dict[str, Any], registry: ToolRegistry) -> dict[str, Any]:
    """What a model call must contain to count as this gold call."""
    name, arguments = call["name"], dict(call["arguments"])
    wanted = canonical(name, arguments, registry)
    if name == "evaluate_origin":
        for material in wanted["materials"]:
            material.pop("name", None)
        if call["arguments"].get("operations") is None:
            wanted.pop("operations", None)  # normal manufacturing: ['montaje'] is as good
    if name == "get_exchange_rate":
        if call["arguments"].get("month") is None:
            wanted.pop("month", None)  # the latest month, named or not
        if call["arguments"].get("amount") is None:
            wanted.pop("amount", None)  # converting 1 unit along the way does no harm
    if name == "issue_origin_statement":
        wanted.pop("evaluation_id", None)  # checked against the model's own evaluations
    return wanted


def _call_ok(call: dict[str, Any], gold: dict[str, Any], registry: ToolRegistry) -> bool:
    if call["name"] != gold["name"]:
        return False
    return matches(
        _expectation(gold, registry), canonical(call["name"], call["arguments"], registry)
    )


# --------------------------------------------------------------------------- the answer


def _plain(text: str) -> str:
    return re.sub(r"[*_`#]", "", text).lower()


def verdict_of(answer: str) -> str:
    """'yes', 'no' or 'none', from the first sentence that is not the currency conversion."""
    paragraphs = [p for p in answer.split("\n\n") if p.strip()]
    first = next((p for p in paragraphs if not p.lstrip().startswith(("Conversi",))), "")
    # The first sentence: the explanation after it may well say a material is not originating.
    text = re.split(r"(?<=[.!?])\s+|\n", _plain(first).strip())[0]
    if any(n in text for n in NEGATIVE) or re.match(r"\s*no[:,.]", text):
        return "no"
    if any(p in text for p in POSITIVE) or re.match(r"\s*(sí|yes)\b", text):
        return "yes"
    return "none"


def _number_forms(value: float) -> list[str]:
    text = f"{value:.2f}"
    whole, cents = text.split(".")
    grouped = f"{int(whole):,}"
    return [text, f"{whole},{cents}", f"{grouped}.{cents}", f"{grouped.replace(',', '.')},{cents}"]


def answer_ok(t: Trajectory, gold: list[dict[str, Any]], answer: str) -> bool | None:
    text = _plain(answer)
    if t.kind in EVALUATION_KINDS:
        if verdict_of(answer) != EXPECTED_VERDICT[t.outcome]:
            return False
        if t.outcome == "statement_issued":
            [issued] = [c for c in gold if c["name"] == "issue_origin_statement"]
            return Path(issued["result"]["path"]).name.lower() in text
        if t.outcome == "originating_needs_reference":
            return any(w in text for w in ("rex", "autorización", "authorisation", "authorization"))
        return True
    if t.kind == "rate":
        result = gold[0]["result"]
        value = result.get("amount_eur")
        if value is None:
            rate = result["units_per_eur"]  # rounded to 2-4 decimals, point or comma
            forms = [f"{rate:.{d}f}" for d in (2, 3, 4)]
            return any(f in text or f.replace(".", ",") in text for f in forms)
        return any(form in text for form in _number_forms(value))
    if t.kind == "vat":
        call = gold[0]
        if "error" in call:
            return "vies" in text
        if not call["result"]["valid"]:
            return any(
                w in text
                for w in (
                    "no es válido",
                    "not valid",
                    "not a valid",
                    "no válido",
                    "invalid",
                    "no es un nif",
                )
            )
        name = call["result"]["name"]
        return name.lower() in text if name else "válido" in text or "valid" in text
    if t.kind == "rule":
        thresholds = {
            str(c["threshold_pct"])
            for rs in gold[0]["result"]["rule_sets"]
            for c in rs["criteria"]
            if "threshold_pct" in c
        }
        return all(th in text for th in thresholds) if thresholds else None
    return None


# ------------------------------------------------------------------------------ scoring


def score_trajectory(
    t: Trajectory, calls: list[dict[str, Any]], answer: str, registry: ToolRegistry
) -> dict[str, Any]:
    """Compare the model's calls (ToolCallRecord dumps) and answer with the trajectory."""
    gold = gold_calls(t)
    scores: dict[str, Any] = {}
    if not gold:
        scores["selection_ok"] = not calls
        scores["calls_ok"] = not calls
    else:
        first_turn = next(m for m in t.messages if m["role"] == "assistant")
        firsts = {c["name"] for c in _calls_in(first_turn)}
        if any(g["name"] == "evaluate_origin" for g in gold):
            firsts.add("get_origin_rule")  # looking the rule up first is as good
        scores["selection_ok"] = bool(calls) and calls[0]["name"] in firsts
        remaining = list(calls)
        found = True
        for g in filter(required, gold):
            hit = next((i for i, c in enumerate(remaining) if _call_ok(c, g, registry)), None)
            found = found and hit is not None
            if hit is not None:
                remaining.pop(hit)
        # Statements: only where due, only on an originating evaluation the model was given.
        originating_ids = {
            c["result"]["evaluation_id"]
            for c in calls
            if c["name"] == "evaluate_origin" and c.get("ok") and c["result"]["originating"]
        }
        statements = [c for c in calls if c["name"] == "issue_origin_statement"]
        due = any(g["name"] == "issue_origin_statement" and "result" in g for g in gold)
        bad_statement = any(
            not due or c["arguments"].get("evaluation_id") not in originating_ids
            for c in statements
        )
        user = t.messages[0]["content"]
        invented = [
            c["arguments"]["exporter_reference"]
            for c in statements
            if c["arguments"].get("exporter_reference")
            and c["arguments"]["exporter_reference"].replace(" ", "") not in user.replace(" ", "")
        ]
        scores["calls_ok"] = found and not bad_statement and not invented
        scores["invented_reference"] = bool(invented)
    executed = [c for c in calls if c["name"] == "issue_origin_statement" and c.get("ok")]
    due = any(g["name"] == "issue_origin_statement" and "result" in g for g in gold)
    scores["false_statement"] = bool(executed) and not due
    verdict = answer_ok(t, gold, answer)
    if verdict is not None:
        scores["answer_ok"] = verdict
    return scores


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(group: list[dict[str, Any]], key: str) -> float | None:
        values = [r[key] for r in group if key in r]
        return round(sum(values) / len(values), 3) if values else None

    by_kind: dict[str, list] = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    keys = ("selection_ok", "calls_ok", "answer_ok")
    return {
        "n": len(rows),
        **{k: rate(rows, k) for k in keys},
        "false_statements": sum(r["false_statement"] for r in rows),
        "invented_references": sum(r.get("invented_reference", False) for r in rows),
        "mean_turns": round(sum(r["turns"] for r in rows) / max(len(rows), 1), 2),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / max(len(rows), 1), 1),
        "by_kind": {
            kind: {
                "n": len(group),
                **{k: rate(group, k) for k in keys if rate(group, k) is not None},
            }
            for kind, group in sorted(by_kind.items())
        },
    }


# ------------------------------------------------------------------------------ running


def run_in_world(loop: ToolLoop, t: Trajectory, max_turns: int = 6) -> ToolsResponse:
    """The model answers the trajectory's question in the trajectory's offline world."""
    [query] = [m["content"] for m in t.messages if m["role"] == "user"]
    with Offline(date.fromisoformat(t.today), random.Random(t.seed), vies=t.vies):
        return loop.run(query, max_turns=max_turns)


def gold_response(t: Trajectory) -> ToolsResponse:
    """The right conversation, run through the loop as if the model had written it."""
    turns = iter(assistant_text(m, thinking=True) for m in t.messages if m["role"] == "assistant")

    class Scripted(ToolLoop):
        def generate(self, messages, use_tools=True):
            return next(turns)

    return run_in_world(Scripted(registry=default_registry(), thinking=True), t, max_turns=8)


def rescore(report_path: str | Path, data: str | Path) -> dict[str, Any]:
    """Apply the current scoring to the calls and answers saved in a report, in place."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    gold = {getattr(t, "id", ""): t for t in load_trajectories(data)}
    registry = default_registry()
    rows = []
    for row in report["rows"]:
        kept = {
            k: v
            for k, v in row.items()
            if k
            not in (
                "selection_ok",
                "calls_ok",
                "answer_ok",
                "false_statement",
                "invented_reference",
            )
        }
        rows.append(
            {
                **kept,
                **score_trajectory(gold[row["id"]], row["tool_calls"], row["answer"], registry),
            }
        )
    report = {"summary": {"model": report["summary"].get("model"), **summarise(rows)}, "rows": rows}
    Path(report_path).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="outputs/tool_sft/trajectories_test.jsonl")
    parser.add_argument("--out", help="report to write (with --rescore: the report to rescore)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--rescore", action="store_true", help="score --out again, no model")
    args = parser.parse_args()
    if args.rescore:
        rescore(args.out, args.data)
        return

    from tool_use.executor import build_tool_loop

    loop = build_tool_loop()
    rows = []
    for t in load_trajectories(args.data)[: args.limit]:
        started = time.perf_counter()
        response = run_in_world(loop, t, args.max_turns)
        calls = [c.model_dump() for c in response.tool_calls]
        scores = score_trajectory(t, calls, response.answer, loop.registry)
        rows.append(
            {
                "id": getattr(t, "id", ""),
                "kind": t.kind,
                "outcome": t.outcome,
                "lang": t.lang,
                "product": t.product,
                "agreement": t.agreement,
                "tool_calls": calls,
                "answer": response.answer,
                "turns": response.turns,
                "seconds": round(time.perf_counter() - started, 1),
                "transcript": loop.last_messages[1:],
                **scores,
            }
        )
        marks = "".join(
            ("✓" if scores[k] else "✗") if k in scores else "·"
            for k in ("selection_ok", "calls_ok", "answer_ok")
        ) + (" FALSE STATEMENT" if scores["false_statement"] else "")
        print(f"{marks} {rows[-1]['id']} {t.kind}/{t.outcome}", flush=True)
    summary = {"model": loop.name, **summarise(rows)}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    out = Path(args.out or "reports/phase2_sft_test.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
