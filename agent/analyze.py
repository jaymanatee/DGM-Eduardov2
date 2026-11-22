"""Phase 4 analysis: the tables of the report from the saved runs, without the GPU.

Reads every ``reports/phase4_*.json`` written by ``agent/evaluate.py`` and
``agent/code_agent.py`` (or the files given) and writes ``reports/phase4_analysis.md``:

* success, steps, tokens and seconds per run and brain, and per category of task;
* the task-by-run matrix;
* argument accuracy: the ``evaluate_origin`` call against the ``reference_call`` of the task
  (agreement, product, prices, and every material's origin, value and declaration). A task can
  pass its check with wrong arguments (a verdict that happens to be right), and this finds it;
* how the agent fails: plain answers without ``final_answer``, refusals of our code checks
  (invented values, bill of materials, repeated calls), tool errors, steps run out;
* memory: how many times the memory was compressed and by how much;
* JSON vs code: tool calls per step.

    uv run python -m agent.analyze
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REPORTS = Path("reports")

# Markers our ReAct loop writes into the observations (agent/react_agent.py).
FAILURE_MARKERS = {
    "sin final_answer": "(answered without final_answer)",
    "recordatorio de final_answer": "[no tool called]",
    "llamada sin etiquetas recuperada": "[call written without <tool_call> tags, recovered]",
    "llamada repetida rechazada": '"error": "repeated call',
    "valor inventado rechazado": "not given by the user",
    "lista de materiales rechazada": "materials the user listed are missing",
    "producto como material rechazado": "is the product itself",
    "memoria comprimida": "[memory compressed:",
}


def _hs6(code) -> str:
    return re.sub(r"\D", "", str(code or ""))[:6]


def _close(a, b, rel: float = 0.01, absolute: float = 0.05) -> bool:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= max(absolute, rel * abs(b))


def _payload(observation: str) -> dict:
    start = (observation or "").find("{")
    if start < 0:
        return {}
    try:
        payload, _ = json.JSONDecoder().raw_decode(observation, start)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def evaluate_calls(row: dict) -> list[tuple[dict, bool]]:
    """Every evaluate_origin call of a run, with whether it returned a result."""
    return [
        (s["action"].get("arguments") or {}, "result" in _payload(s.get("observation", "")))
        for s in row.get("steps", [])
        if s.get("action") and s["action"].get("name") == "evaluate_origin"
    ]


def argument_errors(expected: dict, got: dict) -> list[str]:
    """What differs between the reference evaluate_origin call and the agent's."""
    errors = []
    if got.get("agreement") != expected["agreement"]:
        errors.append("acuerdo")
    if _hs6(got.get("product_hs_code")) != expected["product_hs_code"]:
        errors.append("código del producto")
    if not _close(got.get("exw_price_eur"), expected["exw_price_eur"]):
        errors.append("precio franco fábrica")
    if "fob_price_eur" in expected and not _close(
        got.get("fob_price_eur"), expected["fob_price_eur"]
    ):
        errors.append("precio FOB")
    if "subdivision" in expected and got.get("subdivision") != expected["subdivision"]:
        errors.append("subdivisión")
    if "operations" in expected:
        operations = got.get("operations") or []
        operations = [operations] if isinstance(operations, str) else operations
        if set(operations) != set(expected["operations"]):
            errors.append("operaciones")
    materials = [m for m in got.get("materials") or [] if isinstance(m, dict)]
    by_code = {_hs6(m.get("hs_code")): m for m in materials}
    want = {m["hs_code"]: m for m in expected["materials"]}
    if set(by_code) != set(want):
        errors.append("materiales que sobran o faltan")
    for code, m in want.items():
        g = by_code.get(code)
        if g is None:
            continue
        if g.get("origin") != m["origin"]:
            errors.append(f"origen de {code}")
        if not _close(g.get("value_eur"), m["value_eur"]):
            errors.append(f"valor de {code}")
        if bool(g.get("supplier_declaration")) != m["supplier_declaration"]:
            errors.append(f"declaración de {code}")
    return errors


def argument_check(row: dict) -> dict | None:
    """None if the task has no reference call. Otherwise whether the best evaluate_origin call
    (the last one that ran, or the last attempt) had the right arguments, and what was wrong."""
    reference = row.get("reference_call")
    if not reference:
        return None
    calls = evaluate_calls(row)
    if not calls:
        return {"called": False, "exact": False, "errors": ["no llamó a evaluate_origin"]}
    worked = [args for args, ok in calls if ok]
    args = worked[-1] if worked else calls[-1][0]
    errors = argument_errors(reference["arguments"], args)
    return {"called": True, "ran": bool(worked), "exact": not errors, "errors": errors}


def failure_counts(rows: list[dict]) -> Counter:
    """Tasks where each marker appears at least once, plus a few outcomes."""
    counts: Counter = Counter()
    for row in rows:
        text = "\n".join(s.get("observation") or "" for s in row.get("steps", []))
        for label, marker in FAILURE_MARKERS.items():
            if marker in text:
                counts[label] += 1
        errors = sum(
            1
            for s in row.get("steps", [])
            if s.get("action") and "error" in _payload(s.get("observation", ""))
        )
        if errors:
            counts["alguna herramienta devolvió error"] += 1
        if not row.get("succeeded", True):
            counts["sin respuesta (pasos agotados o atascado)"] += 1
        if row.get("crash"):
            counts["excepción"] += 1
    return counts


def compressions(rows: list[dict]) -> list[tuple[int, int]]:
    found = []
    for row in rows:
        for s in row.get("steps", []):
            text = s.get("observation") or ""
            for a, b in re.findall(r"memory compressed: (\d+) -> (\d+)", text):
                found.append((int(a), int(b)))
    return found


def load_runs(paths: list[Path]) -> dict[str, dict]:
    """'file:brain' -> the brain's results, in a stable order."""
    runs = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for brain, result in data.items():
            runs[f"{path.stem.removeprefix('phase4_')}:{brain}"] = result
    return runs


def _pct(x: float) -> str:
    return f"{100 * x:.0f} %"


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def report(runs: dict[str, dict]) -> str:
    out = ["# Fase 4 · Análisis de las ejecuciones", ""]
    names = list(runs)

    summary = []
    for name, res in runs.items():
        rows = res["rows"]
        checks = [c for c in map(argument_check, rows) if c is not None]
        exact = sum(c["exact"] for c in checks)
        lucky = sum(
            1
            for r in rows
            if r["correct"] and (c := argument_check(r)) and c["called"] and not c["exact"]
        )
        model = rows[0].get("model", "") if rows else ""
        summary.append(
            [
                f"`{name}`",
                model,
                f"{_pct(res['success_rate'])} ({sum(r['correct'] for r in rows)}/{len(rows)})",
                f"{res['mean_steps']:.1f}",
                f"{res['mean_tokens']:.0f}",
                f"{res.get('mean_seconds', 0):.0f}",
                f"{exact}/{len(checks)}",
                lucky,
            ]
        )
    out += [
        "## Resumen",
        "",
        _table(
            ["Ejecución", "Modelo", "Éxito", "Pasos", "Tokens", "s/tarea",
             "Argumentos exactos", "Acierta con argumentos mal"],
            summary,
        ),  # fmt: skip
        "",
        "*Argumentos exactos*: la llamada a `evaluate_origin` coincide con la de referencia "
        "(acuerdo, producto, precios y cada material). *Acierta con argumentos mal*: la tarea "
        "pasa su comprobación aunque los argumentos estén mal.",
        "",
    ]

    categories = sorted({r.get("category", "-") for res in runs.values() for r in res["rows"]})
    out += ["## Éxito por categoría", ""]
    out.append(
        _table(
            ["Categoría", *[f"`{n}`" for n in names]],
            [
                [c, *[_pct(runs[n].get("by_category", {}).get(c, 0)) for n in names]]
                for c in categories
            ],
        )
    )
    out.append("")

    ids = list(dict.fromkeys(r["id"] for res in runs.values() for r in res["rows"]))
    out += ["## Tarea a tarea", ""]
    matrix = []
    for task_id in ids:
        cells = []
        for n in names:
            row = next((r for r in runs[n]["rows"] if r["id"] == task_id), None)
            if row is None:
                cells.append("")
                continue
            mark = "✓" if row["correct"] else "✗"
            check = argument_check(row)
            if check and check["called"] and not check["exact"]:
                mark += " ⚠"
            cells.append(f"{mark} {row['n_steps']}")
        matrix.append([task_id, *cells])
    out.append(_table(["Tarea", *[f"`{n}`" for n in names]], matrix))
    out += [
        "",
        "✓/✗ éxito y número de pasos; ⚠ argumentos de `evaluate_origin` distintos de la referencia.",
        "",
    ]

    out += ["## Errores en los argumentos de `evaluate_origin`", ""]
    for n in names:
        errors: Counter = Counter()
        for r in runs[n]["rows"]:
            check = argument_check(r)
            for e in (check or {}).get("errors", []):
                errors[re.sub(r" de \d{6}$", " de un material", e)] += 1
        listed = ", ".join(f"{e} ({k})" for e, k in errors.most_common()) or "ninguno"
        out.append(f"- `{n}`: {listed}")
    out.append("")

    out += ["## Cómo falla (tareas en las que aparece cada cosa)", ""]
    counts = {n: failure_counts(runs[n]["rows"]) for n in names}
    labels = sorted({k for c in counts.values() for k in c})
    out.append(
        _table(
            ["", *[f"`{n}`" for n in names]],
            [[k, *[counts[n][k] for n in names]] for k in labels],
        )
    )
    out.append("")

    memory = {n: compressions(runs[n]["rows"]) for n in names}
    if any(memory.values()):
        out += ["## Memoria", ""]
        out.append(
            _table(
                ["Ejecución", "Compresiones", "Tokens antes (media)", "Después (media)"],
                [
                    [f"`{n}`", len(c), f"{sum(a for a, _ in c) / len(c):.0f}",
                     f"{sum(b for _, b in c) / len(c):.0f}"]
                    for n, c in memory.items() if c
                ],  # fmt: skip
            )
        )
        out.append("")

    code = [n for n in names if ":code_" in n]
    if code:
        out += ["## JSON frente a código", ""]
        rows = []
        for n in names:
            calls = sum(len([s for s in r["steps"] if s.get("action")]) for r in runs[n]["rows"])
            steps = sum(r["n_steps"] for r in runs[n]["rows"]) or 1
            multi = sum(
                1
                for r in runs[n]["rows"]
                for k in Counter(
                    s["thought"] for s in r["steps"] if s.get("action") and n in code
                ).values()
                if k > 1
            )
            rows.append([f"`{n}`", f"{calls / steps:.2f}", multi if n in code else "-"])
        out.append(
            _table(["Ejecución", "Llamadas por paso", "Pasos con varias llamadas"], rows)
        )
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--out", type=Path, default=REPORTS / "phase4_analysis.md")
    args = parser.parse_args()
    files = args.files or sorted(
        p for p in REPORTS.glob("phase4_*.json") if "smoke" not in p.name
    )
    if not files:
        parser.error("no reports/phase4_*.json yet")
    text = report(load_runs(files))
    args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()