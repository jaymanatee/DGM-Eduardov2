"""Phase 2 evaluation: does the model pick the right tool with the right arguments?

The benchmark is a JSONL file (see ``tool_use/bench/example_cases.jsonl``) where each line
holds a user query and the expected first tool call::

    {"query": "What's the temperature in Paris right now?",
     "expected_tool": "get_weather", "expected_arguments": {"location": "Paris, France"},
     "needs_tool": true}

Include cases where *no* tool is needed: a model that calls tools for everything is not
using them well. The two original metrics:

* tool selection accuracy: the first call names the expected tool (or there is no call when
  ``needs_tool`` is false);
* argument accuracy: among correct selections, the arguments match after normalisation.

Optional fields, for what the first call alone cannot show (``tool_use/bench/cases.jsonl``):

* ``expected_tool`` may be a list: any of them may come first (parallel calls have no order).
* ``expected_calls``: ``[{"tool": ..., "arguments": {...}}]``, every one must appear somewhere
  in the run, in any order. This is how chains and parallel calls are scored.
* ``optional_calls``: same format, calls the model may make or skip (looking the rule up before
  evaluate_origin, which finds it by itself); if one comes first, its arguments are checked.
* ``forbidden_tools``: tools that must not be called (e.g. issuing a statement for a product
  that is not originating).
* ``forbidden_arguments``: ``{"tool": ["argument"]}``, arguments that must stay empty because
  the user never gave them (an invented REX number on a statement).
* ``answer_contains``: strings the final answer must contain; an item that is a list means
  any of its alternatives. This is the metric that compares the model with and without tools.
* ``category``: free label, used to break the metrics down.

How arguments are compared: both sides go through the tool's own validation first, so what
the tool would accept as the same value is the same (``'8414.59'`` and ``'841459'``,
``'fr 40 303 265 045'`` and ``'FR40303265045'``); then only the keys in the expectation are
checked, and lists of objects match in any order. Strings ignore case and spacing.

Run::

    uv run --extra train python -m tool_use.evaluate --bench tool_use/bench/cases.jsonl \\
        --out reports/phase2_eval.json
    uv run --extra train python -m tool_use.evaluate --bench tool_use/bench/cases.jsonl \\
        --no-tools --out reports/phase2_eval_no_tools.json      # the baseline
    uv run python -m tool_use.evaluate --bench tool_use/bench/cases.jsonl \\
        --rescore reports/phase2_eval.json     # after editing the cases, without the model

The model and its settings come from the ``ARCA_TOOLS_*`` variables (see executor.py).
Statements on origin issued during the run go to a fresh folder under ``outputs/``, so two
runs never collide on the same invoice number.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from tool_use.registry import ToolRegistry


def previous_month(today: date | None = None) -> str:
    today = today or date.today()
    return f"{today.year - 1}-12" if today.month == 1 else f"{today.year}-{today.month - 1:02d}"


def load_bench(path: str | Path, today: date | None = None) -> list[dict[str, Any]]:
    """Read the cases; ``{previous_month}`` in a case is replaced by last month (YYYY-MM)."""
    text = Path(path).read_text(encoding="utf-8").replace("{previous_month}", previous_month(today))
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def normalise(value: Any) -> Any:
    """Make argument comparison tolerant to case, spacing and numeric formatting."""
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: normalise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalise(v) for v in value]
    return value


def canonical(name: str, arguments: dict[str, Any], registry: ToolRegistry | None) -> Any:
    """The arguments as the tool sees them after validation; unchanged if they do not validate."""
    tool = registry.tools.get(name) if registry else None
    if tool is not None and isinstance(arguments, dict):
        try:
            arguments = tool.validate(arguments)
        except Exception:  # noqa: BLE001 - invalid arguments are compared as written
            pass
    return normalise(arguments)


def matches(expected: Any, actual: Any) -> bool:
    """``expected`` is contained in ``actual``: dicts by key, lists element by element."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            k in actual and matches(v, actual[k]) for k, v in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return False
        remaining = list(actual)  # any order: each expected item takes one actual item
        for item in expected:
            hit = next((i for i, a in enumerate(remaining) if matches(item, a)), None)
            if hit is None:
                return False
            remaining.pop(hit)
        return True
    return expected == actual


def _call_matches(expected: dict[str, Any], call: dict[str, Any], registry) -> bool:
    if call["name"] != expected["tool"]:
        return False
    wanted = canonical(expected["tool"], expected.get("arguments", {}), None)
    return matches(wanted, canonical(call["name"], call["arguments"], registry))


def score_case(
    case: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    answer: str = "",
    registry: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Compare the calls the model made (and its answer) against the expectation."""
    first = tool_calls[0] if tool_calls else None
    scores: dict[str, Any] = {}
    if not case.get("needs_tool", True):
        scores["selection_ok"] = scores["arguments_ok"] = first is None
    else:
        options = case["expected_tool"]
        options = [options] if isinstance(options, str) else options
        scores["selection_ok"] = first is not None and first["name"] in options
        if not scores["selection_ok"]:
            scores["arguments_ok"] = False
        elif "expected_arguments" in case and isinstance(case["expected_tool"], str):
            expected = {"tool": first["name"], "arguments": case["expected_arguments"]}
            scores["arguments_ok"] = _call_matches(expected, first, registry)
        else:  # the first call's expectation is the matching entry of expected_calls
            expectations = case.get("expected_calls", []) + case.get("optional_calls", [])
            candidates = [e for e in expectations if e["tool"] == first["name"]]
            scores["arguments_ok"] = any(_call_matches(e, first, registry) for e in candidates)

    if "expected_calls" in case or "forbidden_tools" in case or "forbidden_arguments" in case:
        remaining = list(tool_calls)
        found = True
        for expected in case.get("expected_calls", []):
            hit = next(
                (i for i, c in enumerate(remaining) if _call_matches(expected, c, registry)), None
            )
            found = found and hit is not None
            if hit is not None:
                remaining.pop(hit)
        forbidden_tool = any(c["name"] in case.get("forbidden_tools", []) for c in tool_calls)
        invented = [
            f"{c['name']}.{arg}"
            for c in tool_calls
            for arg in case.get("forbidden_arguments", {}).get(c["name"], [])
            if c["arguments"].get(arg) not in (None, "")
        ]
        scores["calls_ok"] = found and not forbidden_tool and not invented
        scores["invented_arguments"] = invented

    if "answer_contains" in case:
        text = answer.lower()
        scores["answer_ok"] = all(
            any(alt.lower() in text for alt in ([item] if isinstance(item, str) else item))
            for item in case["answer_contains"]
        )
    return scores


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    scored = [r[key] for r in rows if key in r]
    return round(sum(scored) / len(scored), 3) if scored else None


def summarise(scores: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(len(scores), 1)
    summary: dict[str, Any] = {
        "tool_selection_accuracy": sum(s["selection_ok"] for s in scores) / n,
        "argument_accuracy": sum(s["arguments_ok"] for s in scores) / n,
        "n_cases": len(scores),
    }
    for key, label in (("calls_ok", "multi_call_accuracy"), ("answer_ok", "answer_accuracy")):
        if any(key in s for s in scores):
            summary[label] = _rate(scores, key)
            summary[f"n_{label}"] = sum(key in s for s in scores)
    if any("turns" in s for s in scores):
        calls = [c for s in scores for c in s.get("tool_calls", [])]
        summary["mean_turns"] = round(sum(s["turns"] for s in scores) / n, 2)
        summary["tool_calls"] = len(calls)
        summary["failed_tool_calls"] = sum(not c["ok"] for c in calls)
        summary["mean_seconds"] = round(sum(s["seconds"] for s in scores) / n, 1)
    by_category: dict[str, list] = defaultdict(list)
    for s in scores:
        if "category" in s:
            by_category[s["category"]].append(s)
    if by_category:
        summary["by_category"] = {
            name: {
                "n": len(rows),
                **{
                    k: _rate(rows, k)
                    for k in ("selection_ok", "arguments_ok", "calls_ok", "answer_ok")
                    if _rate(rows, k) is not None
                },
            }
            for name, rows in sorted(by_category.items())
        }
    return summary


_SCORE_KEYS = ("selection_ok", "arguments_ok", "calls_ok", "invented_arguments", "answer_ok")


def rescore(report_path: str | Path, bench_path: str | Path) -> dict[str, Any]:
    """Apply the current expectations to the calls and answers saved in a report, in place."""
    from tool_use.tools import default_registry

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in load_bench(bench_path)}
    rows = []
    for row in report["rows"]:
        case = cases[row["id"]]
        kept = {k: v for k, v in row.items() if k not in _SCORE_KEYS and k not in cases[row["id"]]}
        scores = score_case(case, row["tool_calls"], row["answer"], default_registry())
        rows.append({**case, **kept, **scores})
    keep = {k: report["summary"][k] for k in ("model", "tools") if k in report["summary"]}
    report = {"summary": {**keep, **summarise(rows)}, "rows": rows}
    Path(report_path).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bench", default="tool_use/bench/example_cases.jsonl")
    parser.add_argument("--out", default="reports/phase2_eval.json")
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--no-tools", action="store_true", help="baseline: no tools in the prompt")
    parser.add_argument("--limit", type=int, help="only the first N cases")
    parser.add_argument(
        "--rescore", metavar="REPORT", help="score a saved report again against --bench, no model"
    )
    args = parser.parse_args()
    if args.rescore:
        rescore(args.rescore, args.bench)
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.environ.setdefault("ARCA_STATEMENTS_DIR", f"outputs/bench_statements/{stamp}")
    from tool_use.executor import build_tool_loop

    loop = build_tool_loop()
    cases = load_bench(args.bench)[: args.limit]
    rows = []
    for case in cases:
        started = time.perf_counter()
        response = loop.run(case["query"], max_turns=args.max_turns, use_tools=not args.no_tools)
        calls = [c.model_dump() for c in response.tool_calls]
        scores = score_case(case, calls, response.answer, loop.registry)
        rows.append(
            {
                **case,
                "tool_calls": calls,
                "answer": response.answer,
                "turns": response.turns,
                "seconds": round(time.perf_counter() - started, 1),
                "transcript": loop.last_messages,
                **scores,
            }
        )
        marks = "".join(
            ("✓" if scores[k] else "✗") if k in scores else "·"
            for k in ("selection_ok", "arguments_ok", "calls_ok", "answer_ok")
        )
        print(f"{marks} {case.get('id', '')} {case['query'][:70]}")

    summary = {"model": loop.name, "tools": not args.no_tools, **summarise(rows)}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False, default=str)
    )


if __name__ == "__main__":
    main()
