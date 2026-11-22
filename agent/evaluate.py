"""Phase 4 evaluation: success rate, steps and tokens on multi-tool tasks, per brain.

Each line of the benchmark (``agent/bench/tasks.jsonl``) has a task, the tools that must be
involved, and a way to check the final answer automatically (a number, a substring, a regex,
or ``not`` / ``all`` / ``any`` of them). ``forbidden_tools`` fails a task if one of those tools
actually ran: a statement issued for a product that is not originating, or a verdict on a
value the user never gave. Run every brain::

    uv run python -m agent.evaluate --bench agent/bench/tasks.jsonl --brains base rlm thinking

and put the resulting table in your report next to the JSON-vs-code-agent comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path


CHECK_TYPES = {"contains", "regex", "number", "not", "all", "any"}


def _number_readings(token: str) -> set[float]:
    """'48,38' and '48.38' are 48.38; '1.042,28' and '1,042.28' are 1042.28: the agent answers
    in Spanish or in English, so both readings of the separators are tried."""
    readings = set()
    for thousands, decimal in ((",", "."), (".", ",")):
        try:
            readings.add(float(token.replace(thousands, "").replace(decimal, ".")))
        except ValueError:
            pass
    return readings


def matches(check: dict, final_answer: str) -> bool:
    """One check: a substring, a regex or a number, or ``not`` / ``all`` / ``any`` of checks."""
    kind, value = check.get("type"), check.get("value")
    if kind == "contains":
        return str(value).lower() in final_answer.lower()
    if kind == "regex":
        return re.search(value, final_answer) is not None
    if kind == "number":
        tolerance = check.get("tolerance", 0)
        return any(
            abs(reading - float(value)) <= tolerance
            for token in re.findall(r"-?\d[\d.,]*\d|\d", final_answer)
            for reading in _number_readings(token)
        )
    if kind == "not":
        return not matches(value, final_answer)
    if kind == "all":
        return all(matches(c, final_answer) for c in value)
    if kind == "any":
        return any(matches(c, final_answer) for c in value)
    raise ValueError(f"unknown check type {kind!r}: one of {sorted(CHECK_TYPES)}")


def check_answer(task: dict, final_answer: str) -> bool:
    """Automatic check of the final answer, according to the task's ``check`` field."""
    return matches(task.get("check", {}), final_answer)


def tools_used(steps: list[dict]) -> set[str]:
    return {s["action"]["name"] for s in steps if s.get("action")}


def tools_that_worked(steps: list[dict]) -> set[str]:
    """Tools whose call returned a result (the observation is the JSON of the phase-2 loop,
    with 'result' or 'error', maybe after a note in brackets)."""
    worked = set()
    for s in steps:
        if not s.get("action") or not s.get("observation"):
            continue
        text = s["observation"]
        start = text.find("{")
        try:
            payload, _ = json.JSONDecoder().raw_decode(text, start) if start >= 0 else ({}, 0)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "result" in payload:
            worked.add(s["action"]["name"])
    return worked


def task_passed(task: dict, response: dict) -> bool:
    """The answer passes its check and no tool in ``forbidden_tools`` actually ran (a statement
    issued for a product that is not originating, a verdict on an invented value...)."""
    forbidden = set(task.get("forbidden_tools", [])) & tools_that_worked(response["steps"])
    return response["succeeded"] and not forbidden and check_answer(task, response["final_answer"])


def summarize(rows: list[dict]) -> dict:
    """Success, steps, tokens and seconds over the rows of one brain, and success per category."""
    n = max(len(rows), 1)
    categories: dict[str, list[bool]] = {}
    for r in rows:
        categories.setdefault(r.get("category", "-"), []).append(r["correct"])
    return {
        "n_tasks": len(rows),
        "success_rate": sum(r["correct"] for r in rows) / n,
        "mean_steps": sum(r["n_steps"] for r in rows) / n,
        "mean_tokens": sum(r["tokens_used"] for r in rows) / n,
        "mean_seconds": sum(r.get("seconds", 0) for r in rows) / n,
        "required_tools_rate": sum(r["required_tools_used"] for r in rows) / n,
        "by_category": {c: sum(v) / len(v) for c, v in sorted(categories.items())},
    }


def load_tasks(bench: str | Path, ids: list[str] | None = None) -> list[dict]:
    tasks = [
        json.loads(line)
        for line in Path(bench).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if ids:
        unknown = set(ids) - {t.get("id") for t in tasks}
        if unknown:
            raise ValueError(f"unknown task ids: {sorted(unknown)}")
        tasks = [t for t in tasks if t.get("id") in ids]
    return tasks


def run_bench(
    agent, tasks: list[dict], brains: list[str], out: Path, max_steps: int, resume: bool
) -> dict:
    """Every task with every brain, saved after each task. ``agent`` is anything with
    ``run(task, max_steps, brain)``, ``label(brain)`` and ``tokens_used``: our ReAct agent or
    the smolagents CodeAgent of ``agent/code_agent.py``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text(encoding="utf-8")) if resume and out.exists() else {}
    for brain in brains:
        rows = results.get(brain, {}).get("rows", [])
        done = {r.get("id") for r in rows}
        for i, task in enumerate(tasks):
            task_id = task.get("id", str(i))
            if task_id in done:
                continue
            # issue_origin_statement never overwrites an invoice's statement: without a fresh
            # folder per brain and task, the second brain would find the first one's file.
            statements = Path("outputs/agent_eval") / brain / task_id
            shutil.rmtree(statements, ignore_errors=True)
            os.environ["ARCA_STATEMENTS_DIR"] = str(statements)
            start = time.perf_counter()
            try:
                response = agent.run(task["task"], max_steps=max_steps, brain=brain).model_dump()
                crash = None
            except Exception as error:  # one broken task must not end a run of hours
                response = {
                    "final_answer": "",
                    "steps": [],
                    "n_steps": 0,
                    "succeeded": False,
                    "model": agent.label(brain),
                    "tokens_used": agent.tokens_used,
                }
                crash = f"{type(error).__name__}: {error}"
            seconds = round(time.perf_counter() - start, 1)
            used = tools_used(response["steps"])
            row = {
                **task,
                **response,
                "correct": task_passed(task, response),
                "tools_used": sorted(used),
                "required_tools_used": set(task.get("required_tools", [])) <= used,
                "seconds": seconds,
                "crash": crash,
            }
            rows.append(row)
            print(
                f"{brain:>9} {task_id:<24} {'OK ' if row['correct'] else 'MAL'} "
                f"{row['n_steps']:>2} pasos {seconds:>7.1f} s" + (f"  {crash}" if crash else ""),
                flush=True,
            )
            # Saved after every task: a session that ends mid-run loses one task, not all.
            results[brain] = {**summarize(rows), "rows": rows}
            out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=list))
        summary = summarize(rows)
        results[brain] = {**summary, "rows": rows}
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=list))
        print(
            f"{brain:>9}: success {summary['success_rate']:.2f} "
            f"· steps {summary['mean_steps']:.1f} · {summary['mean_seconds']:.0f} s/task",
            flush=True,
        )
    return results


def rescore(out: Path, tasks: list[dict]) -> dict:
    """The saved answers checked again with the current checks of the bench (after fixing a
    check): no model, no GPU. Rows whose task is no longer in the bench are kept as they were."""
    by_id = {t.get("id"): t for t in tasks}
    results = json.loads(out.read_text(encoding="utf-8"))
    for brain, result in results.items():
        rows = []
        for row in result["rows"]:
            task = by_id.get(row.get("id"))
            if task is not None:
                row = {**row, **task}
                row["correct"] = task_passed(task, row)
                used = tools_used(row["steps"])
                row["required_tools_used"] = set(task.get("required_tools", [])) <= used
            rows.append(row)
        results[brain] = {**summarize(rows), "rows": rows}
        print(f"{brain:>9}: success {results[brain]['success_rate']:.2f} ({len(rows)} tasks)")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=list))
    return results


def bench_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bench", default="agent/bench/tasks.jsonl")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--ids", nargs="+", help="only these task ids (a quick check)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep the tasks already in --out and run only the rest (after a session ended)",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="check the answers saved in --out again with the current bench, without the model",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    bench_arguments(parser)
    parser.add_argument("--brains", nargs="+", default=["rlm"])
    parser.add_argument("--out", default="reports/phase4_eval.json")
    args = parser.parse_args()
    try:
        tasks = load_tasks(args.bench, args.ids)
    except ValueError as error:
        parser.error(str(error))
    if args.rescore:
        rescore(Path(args.out), tasks)
        return

    from agent.react_agent import ReActAgent

    agent = ReActAgent.from_env()
    run_bench(agent, tasks, args.brains, Path(args.out), args.max_steps, args.resume)


if __name__ == "__main__":
    main()