"""Phase 1 analysis of an evaluation file: what each model gets right, how it fails, and why.

Run after ``rlm/evaluate.py``::

    uv run python -m rlm.analyze_eval reports/phase1_eval_test.json
    uv run python -m rlm.analyze_eval reports/phase1_eval_test_ood.json --no-teacher

No GPU needed: it only reads the per-example verdicts that ``evaluate.py`` saved. It prints
markdown tables and writes them to ``reports/phase1_analysis_<data>.md``, ready to quote in
EXPERIMENTS.md and the report:

1. pass@1 per model with a 95 % confidence interval (Wilson), strict and lenient.
2. Paired comparison of two models on the same problems: fixed, broken, both right, both wrong.
3. Failure types per model (no ``<answer>``, ``<tool_call>`` instead of ``<answer>``, cut by the
   token limit, wrong verdict, wrong criterion, wrong percentage...).
4. pass@1 per group (expected verdict, kind of criterion, every field of ``branches``) next to
   the teacher's acceptance rate on the distillation problems (``sft_traces.jsonl``).
5. Length: tokens of right and wrong answers, share cut by the limit.
6. Right for the right reasons? Heuristic checks on the reasoning of correct answers: does it
   name the rule's heading, write the expected percentage, write the expected sum of
   non-originating materials. Approximate by design: numbers are matched with a tolerance and
   the EXW price is read from the statement with a regex.
7. Candidate failures to read by hand (problem index, expected and predicted answer).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

CLASS_CHANGE = {"CC", "CTH", "CTSH"}
VALUE = {"MAXNOM", "RVC"}
MIN_GROUP = 5

FAILURE_PREFIXES = {
    "la respuesta no es": "json_invalido",
    "falta": "campo_ausente",
    "veredicto": "veredicto_incorrecto",
    "criterio": "criterio_incorrecto",
    "porcentaje": "porcentaje_incorrecto",
}


# ----------------------------------------------------------------------------- helpers


def as_dict(value) -> dict | None:
    """Answers and branches may be stored as JSON text or as objects."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % confidence interval of a proportion (Wilson score): honest with n = 200."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def pct(x: float) -> str:
    return f"{100 * x:.1f} %".replace(".", ",")


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def parse_number(token: str) -> float | None:
    """'1.234,56', '1,234.56', '287,46', '46.0' -> float. None if it is not a number."""
    token = token.strip(".,")
    if not token:
        return None
    if "," in token and "." in token:
        decimal = "," if token.rfind(",") > token.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        token = token.replace(thousands, "").replace(decimal, ".")
    elif "," in token:
        head, _, tail = token.rpartition(",")
        token = token.replace(",", ".") if len(tail) <= 2 and "," not in head else token.replace(",", "")
    elif token.count(".") > 1:
        token = token.replace(".", "")
    try:
        return float(token)
    except ValueError:
        return None


def numbers_in(text: str) -> list[float]:
    out = []
    for token in re.findall(r"\d[\d.,]*", text):
        value = parse_number(token)
        if value is not None:
            out.append(value)
    return out


# ----------------------------------------------------------------------------- grouping


def answer_groups(answer) -> list[str]:
    """Expected verdict and which kind of criterion makes the product originating."""
    truth = as_dict(answer)
    if truth is None or "originario" not in truth:
        return []
    labels = [f"originario={truth['originario']}"]
    if not truth["originario"]:
        return labels + ["tipo=no_originario"]
    valid = {str(c).upper() for c in truth.get("criterios_validos", [])}
    if valid and valid <= CLASS_CHANGE:
        labels.append("tipo=solo_cambio_de_clasificacion")
    elif valid & VALUE:
        labels.append("tipo=criterio_de_valor")
    else:
        labels.append("tipo=otro")
    return labels


def branch_groups(branches) -> list[str]:
    """Every scalar field of ``branches``; list fields give one label per element."""
    data = as_dict(branches) or {}
    labels = []
    for key, value in data.items():
        if isinstance(value, bool):  # trap flags: only the problems that have the trap
            if value:
                labels.append(key)
            continue
        for v in value if isinstance(value, list) else [value]:
            if v is None or isinstance(v, (str, int, float)):
                labels.append(f"{key}={v}")
    return labels


def groups_of(answer, branches) -> list[str]:
    return list(dict.fromkeys(["todos", *answer_groups(answer), *branch_groups(branches)]))


# ----------------------------------------------------------------------------- failures


def failure_type(row: dict) -> str:
    if row["is_correct"]:
        return "acierto"
    detail = row.get("detail") or ""
    raw = row.get("raw") or ""
    if detail.startswith("no <answer>") or row.get("predicted") is None:
        if "<tool_call>" in raw:
            return "tool_call_en_vez_de_answer"
        if row.get("truncated"):
            return "cortada_sin_answer"
        if "</think>" not in raw:
            return "sin_cerrar_think"
        return "sin_answer"
    for prefix, label in FAILURE_PREFIXES.items():
        if detail.startswith(prefix):
            return label
    return "otro"


# ----------------------------------------------------------------------------- reasoning checks

EXW_RE = [
    re.compile(r"[Pp]recio[^0-9\n]{0,60}?(\d[\d.,]*)\s*€"),
    re.compile(r"(?:franco f[áa]brica|EXW)[^0-9\n]{0,40}?(\d[\d.,]*)\s*€"),
]
HEADING_RE = re.compile(r"partida\s+(\d{4})")


def exw_price(question: str) -> float | None:
    for regex in EXW_RE:
        match = regex.search(question)
        if match:
            value = parse_number(match.group(1))
            if value:
                return value
    return None


def reasoning_checks(row: dict) -> dict[str, bool | None]:
    """Heuristic: is the expected work visible in the reasoning, not only in the answer?"""
    raw = row.get("raw") or ""
    think = raw.split("</think>")[0]
    truth = as_dict(row.get("expected")) or {}
    found = numbers_in(think)
    out: dict[str, bool | None] = {}

    heading = HEADING_RE.search(row.get("question") or "")
    out["cita_partida"] = heading.group(1) in think if heading else None

    expected_pct = truth.get("pct_no_originario")
    if isinstance(expected_pct, (int, float)) and expected_pct > 0:
        out["porcentaje_en_razonamiento"] = any(abs(x - expected_pct) <= 0.5 for x in found)
        exw = exw_price(row.get("question") or "")
        if exw:
            expected_sum = expected_pct * exw / 100
            tol = max(0.5, 0.005 * expected_sum)
            out["suma_en_razonamiento"] = any(abs(x - expected_sum) <= tol for x in found)
        else:
            out["suma_en_razonamiento"] = None
    else:
        out["porcentaje_en_razonamiento"] = None
        out["suma_en_razonamiento"] = None
    return out


# ----------------------------------------------------------------------------- sections


def section_pass1(results: dict) -> str:
    rows = []
    for name, res in results.items():
        r = res["rows"]
        n, k = len(r), sum(x["is_correct"] for x in r)
        kl = sum(x.get("is_correct_lenient", x["is_correct"]) for x in r)
        lo, hi = wilson(k, n)
        rows.append([name, n, f"{k} ({pct(k / n)})", f"{pct(lo)} – {pct(hi)}", f"{kl} ({pct(kl / n)})",
                     pct(sum(x.get("has_valid_format", False) for x in r) / n),
                     pct(sum(x.get("truncated", False) for x in r) / n)])
    n = len(next(iter(results.values()))["rows"])
    half = 100 * 1.96 * math.sqrt(0.25 / n) if n else 0
    return "## 1. pass@1\n\n" + table(
        ["Modelo", "n", "Aciertos", "IC 95 %", "Lenient", "Formato válido", "Cortadas"], rows
    ) + (f"\n\nCon n = {n}, el intervalo llega a medir unos ±{half:.0f} puntos: diferencias de "
         "pass@1 menores que eso entre dos modelos no son concluyentes por sí solas (la "
         "comparación por parejas de la sección 2 es más fina).")


def section_paired(results: dict, a: str, b: str) -> str:
    ra, rb = results[a]["rows"], results[b]["rows"]
    if len(ra) != len(rb) or any(x["question"] != y["question"] for x, y in zip(ra, rb)):
        return f"## 2. {a} frente a {b}\n\nNo son los mismos problemas en el mismo orden: no se compara."
    counts = Counter((x["is_correct"], y["is_correct"]) for x, y in zip(ra, rb))
    both, only_a, only_b, none = counts[(True, True)], counts[(True, False)], counts[(False, True)], counts[(False, False)]
    # Exact McNemar test (two-sided) on the discordant pairs.
    d = only_a + only_b
    p = min(1.0, 2 * sum(math.comb(d, i) for i in range(min(only_a, only_b) + 1)) / 2**d) if d else 1.0
    return f"## 2. {a} frente a {b} (mismos problemas)\n\n" + table(
        ["", "Problemas"],
        [[f"Aciertan los dos", both], [f"Solo {b} (arreglados)", only_b],
         [f"Solo {a} (estropeados)", only_a], ["Fallan los dos", none]],
    ) + f"\n\nTest de McNemar exacto sobre los {d} problemas en que discrepan: p = {p:.3g}".replace(".", ",") + "."


def section_failures(results: dict) -> str:
    types = Counter()
    per_model = {}
    for name, res in results.items():
        c = Counter(failure_type(r) for r in res["rows"])
        per_model[name] = c
        types.update(c)
    order = [t for t, _ in types.most_common() if t != "acierto"]
    rows = [[t] + [per_model[m].get(t, 0) for m in results] for t in order]
    return "## 3. Tipos de fallo\n\n" + table(["Tipo", *results], rows) + (
        "\n\n`tool_call_en_vez_de_answer`: escribe `<tool_call>` tras `</think>`. "
        "`cortada_sin_answer`: llega al límite de tokens sin responder."
    )


def teacher_rates(path: Path) -> dict[str, tuple[int, int]]:
    rates: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for g in groups_of(row.get("answer"), row.get("branches")):
                rates[g][0] += bool(row.get("verified"))
                rates[g][1] += 1
    return {g: (k, n) for g, (k, n) in rates.items()}


def section_groups(results: dict, teacher: dict | None) -> str:
    per_model: dict[str, dict[str, list[bool]]] = {}
    labels: dict[str, int] = {}
    for name, res in results.items():
        g: dict[str, list[bool]] = defaultdict(list)
        for r in res["rows"]:
            for label in groups_of(r["expected"], r.get("branches")):
                g[label].append(r["is_correct"])
        per_model[name] = g
        for label, v in g.items():
            labels[label] = len(v)
    header = ["Grupo", "n", *results] + (["Profesor (aceptación)"] if teacher else [])
    rows = []
    for label in sorted(labels, key=lambda s: (s != "todos", s)):
        if labels[label] < MIN_GROUP:
            continue
        row = [label, labels[label]]
        row += [pct(sum(per_model[m][label]) / len(per_model[m][label])) if per_model[m][label] else "—"
                for m in results]
        if teacher:
            k, n = teacher.get(label, (0, 0))
            row.append(f"{pct(k / n)} (n={n})" if n else "—")
        rows.append(row)
    note = (
        "\n\nProfesor: trazas verificadas / generadas en los 500 problemas de destilación "
        "(Qwen3-4B, temperatura 0,6, problemas de `train`). Orientativo: no son los mismos "
        "problemas ni la misma forma de generar."
        if teacher else ""
    )
    return f"## 4. pass@1 por grupo (n ≥ {MIN_GROUP})\n\n" + table(header, rows) + note


def section_length(results: dict) -> str:
    rows = []
    for name, res in results.items():
        ok = [r["n_tokens"] for r in res["rows"] if r["is_correct"]]
        ko = [r["n_tokens"] for r in res["rows"] if not r["is_correct"]]
        all_ = sorted(r["n_tokens"] for r in res["rows"])
        p95 = all_[int(0.95 * (len(all_) - 1))] if all_ else 0
        rows.append([name, f"{statistics.median(ok):.0f}" if ok else "—",
                     f"{statistics.median(ko):.0f}" if ko else "—", p95,
                     pct(sum(r.get("truncated", False) for r in res["rows"]) / len(res["rows"]))])
    return "## 5. Longitud\n\n" + table(
        ["Modelo", "Mediana (aciertos)", "Mediana (fallos)", "p95", "Cortadas"], rows
    )


def section_reasoning(results: dict) -> str:
    keys = ["cita_partida", "porcentaje_en_razonamiento", "suma_en_razonamiento"]
    rows = []
    for name, res in results.items():
        correct = [r for r in res["rows"] if r["is_correct"]]
        checks = [reasoning_checks(r) for r in correct]
        row = [name, len(correct)]
        for key in keys:
            vals = [c[key] for c in checks if c[key] is not None]
            row.append(f"{pct(sum(vals) / len(vals))} (n={len(vals)})" if vals else "—")
        dubious = sum(
            1 for c in checks
            if c["porcentaje_en_razonamiento"] is False and c["suma_en_razonamiento"] is not True
        )
        row.append(dubious)
        rows.append(row)
    return "## 6. ¿Acierta por las razones correctas? (solo respuestas acertadas, heurístico)\n\n" + table(
        ["Modelo", "Aciertos", "Cita la partida", "Porcentaje en el razonamiento",
         "Suma en el razonamiento", "Aciertos sin el cálculo a la vista"], rows
    ) + (
        "\n\nSe busca en el `<think>` el porcentaje esperado (±0,5 puntos) y la suma esperada de "
        "materiales no originarios (porcentaje × precio EXW leído del enunciado, ±0,5 %). Los "
        "casos con 0 % no se comprueban. La última columna son aciertos cuyo razonamiento no "
        "contiene el porcentaje ni la suma: candidatos a acierto por casualidad, para leer a mano."
    )


def section_candidates(results: dict, focus: str, per_type: int = 3) -> str:
    rows = results[focus]["rows"]
    by_type: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_type[failure_type(r)].append(i)
    out = [f"## 7. Fallos para leer a mano ({focus})\n"]
    for t, idx in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        if t == "acierto":
            continue
        out.append(f"**{t}** ({len(idx)} problemas): índices {idx[:15]}{' …' if len(idx) > 15 else ''}\n")
        for i in idx[:per_type]:
            r = rows[i]
            groups = ", ".join(g for g in groups_of(r["expected"], r.get("branches")) if g != "todos")
            predicted = (r.get("predicted") or r.get("raw", "")[-160:]).replace("\n", " ")
            out.append(f"- Problema {i} [{groups}]\n  - esperado: `{r['expected']}`\n"
                       f"  - modelo: `{predicted[:200]}`\n  - motivo: {r.get('detail') or '—'}")
        out.append("")
    out.append(
        f"Para ver el razonamiento completo: `uv run --extra train python -m rlm.try_model <índice>`"
        f" (o leer `raw` en el JSON)."
    )
    return "\n".join(out)


# ----------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("eval_json", help="file written by rlm/evaluate.py")
    parser.add_argument("--teacher", default="rlm/data/sft_traces.jsonl")
    parser.add_argument("--no-teacher", action="store_true", help="skip the teacher column")
    parser.add_argument("--compare", nargs=2, default=None, metavar=("A", "B"),
                        help="models for the paired comparison (default: the first two)")
    parser.add_argument("--focus", default=None, help="model whose failures are listed (default: last)")
    parser.add_argument("--out", default=None, help="default: reports/phase1_analysis_<data>.md")
    args = parser.parse_args()

    path = Path(args.eval_json)
    results = json.loads(path.read_text(encoding="utf-8"))
    if not results:
        raise SystemExit("el fichero no tiene ningún modelo evaluado")
    names = list(results)

    teacher = None
    if not args.no_teacher and Path(args.teacher).exists():
        teacher = teacher_rates(Path(args.teacher))

    sections = [f"# Análisis de `{path.name}`\n\nModelos: {', '.join(names)}.", section_pass1(results)]
    pair = args.compare or (names[:2] if len(names) >= 2 else None)
    if pair:
        sections.append(section_paired(results, *pair))
    sections += [
        section_failures(results),
        section_groups(results, teacher),
        section_length(results),
        section_reasoning(results),
        section_candidates(results, args.focus or names[-1]),
    ]
    text = "\n\n".join(sections) + "\n"
    print(text)

    stem = path.stem.replace("phase1_eval_", "")
    out = Path(args.out or f"reports/phase1_analysis_{stem}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()