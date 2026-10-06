"""Problemas verificables de origen preferencial: ¿es originario este producto?

Sigue el patrón de la plantilla: tres funciones y la clase base hace el resto.

    sample_params(rng, split)  -> los datos de un problema (regla, producto, materiales, precios…)
    solve(params)              -> la respuesta correcta, con el motor de reglas de abajo
    render(params, rng)        -> el enunciado en lenguaje natural, varias plantillas

``solve`` es a la vez la implementación de referencia y el verificador: la etiqueta sale del
motor, no de una anotación manual. Por eso los tests del motor importan más que los del generador.

La semántica (qué cuenta como originario, cómo se calcula el porcentaje, qué criterio se
reporta) está documentada en ``origin_engine.py``.

Catálogo de reglas PROVISIONAL: las dos primeras salen de los ejemplos de la propuesta y las
demás son de relleno para ejercitar el motor. Hay que sustituirlas por reglas reales del TCA /
API del UK Trade Tariff (``source="placeholder"`` marca las que faltan).

    uv run python -m rlm.generate_problems --n 4000 --split train --out rlm/data/train.jsonl
    uv run python -m rlm.generate_problems --n 200  --split test  --out rlm/data/test.jsonl
    uv run python -m rlm.generate_problems --n 100  --split ood   --out rlm/data/test_ood.jsonl

``train`` y ``test`` comparten distribución. El split ``ood`` todavía no está definido
(reglas al estilo UE-Corea y UE-Mercosur): ``sample_params`` lo dice con un error explícito.
"""

from __future__ import annotations

import argparse
import json
import random
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from rlm.origin_engine import (
    INSUFFICIENT_OPERATIONS,
    ORIGINATING_ORIGINS,
    ORIGINS,
    SUFFICIENT_OPERATIONS,
    VALUE_CRITERIA,
    Alt,
    Material,
    Rule,
    _frac,
    evaluate,
    pct,
)


@dataclass
class Problem:
    question: str
    answer: str
    params: dict[str, Any]
    template_id: int
    branches: dict[str, str] = field(default_factory=dict)


class ProblemGenerator(ABC):
    """Subclass this for your domain. Three methods, and the base class does the rest."""

    name: str = "generator"

    @abstractmethod
    def sample_params(self, rng: random.Random, split: str) -> dict[str, Any]:
        """One problem's data. ``split`` lets you hold a region out for the OOD set."""

    @abstractmethod
    def solve(self, params: dict[str, Any]) -> tuple[str, dict[str, str]]:
        """Reference implementation. Returns the answer and which branches were taken."""

    @abstractmethod
    def render(self, params: dict[str, Any], rng: random.Random) -> tuple[str, int]:
        """The statement in natural language. Returns the text and the template used."""

    def key(self, params: dict[str, Any]) -> str:
        """Deduplication key. Hash the *parameters*, never the text."""
        return json.dumps(params, sort_keys=True, default=str)

    def generate(self, n: int, split: str, seed: int = 0) -> list[Problem]:
        rng = random.Random(f"{seed}-{split}")
        seen: set[str] = set()
        problems: list[Problem] = []
        attempts = 0
        while len(problems) < n and attempts < 200 * n:
            attempts += 1
            params = self.sample_params(rng, split)
            fingerprint = self.key(params)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            answer, branches = self.solve(params)
            question, template_id = self.render(params, rng)
            problems.append(Problem(question, answer, params, template_id, branches))
        if len(problems) < n:
            raise RuntimeError(
                f"only {len(problems)} unique problems after {attempts} attempts: "
                "your parameter space is too small, widen sample_params"
            )
        return problems


# --------------------------------------------------------------------------- catálogo (provisional)

AGREEMENTS = {
    "TCA": ("Acuerdo de Comercio y Cooperación UE-Reino Unido (TCA)", "Reino Unido"),
    "UE-JP": ("Acuerdo UE-Japón", "Japón"),
    "UE-KR": ("Acuerdo UE-Corea del Sur", "Corea del Sur"),
    "CETA": ("Acuerdo UE-Canadá (CETA)", "Canadá"),
}

RULES: dict[str, Rule] = {
    r.rule_id: r
    for r in (
        # Las dos primeras salen de los ejemplos de la propuesta.
        Rule("TCA-8507", "8507", "TCA", (Alt("CTH"), Alt("MAXNOM", pct(50))), source="propuesta"),
        Rule("TCA-8414", "8414", "TCA", (Alt("CTH"),), source="propuesta"),
        # Relleno: sustituir por reglas reales.
        Rule("UE-JP-8501", "8501", "UE-JP", (Alt("CTH", excluded_headings=("8503",)), Alt("RVC", pct(50), basis="FOB")),),
        Rule("UE-KR-9403", "9403", "UE-KR", (Alt("CC"), Alt("MAXNOM", pct(40)))),
        Rule("CETA-8413", "8413", "CETA", (Alt("CTSH", excluded_headings=("8482",)),)),
        Rule("UE-KR-8479", "8479", "UE-KR", (Alt("MAXNOM", pct(40)),)),
    )
}


@dataclass(frozen=True)
class ProductTemplate:
    name: str
    code: str
    exw_range: tuple[int, int]  # euros, se muestrea de 10 en 10
    components: tuple[tuple[str, str], ...]  # (nombre, subpartida SA)


PRODUCTS: dict[str, ProductTemplate] = {
    "TCA-8507": ProductTemplate(
        "batería de ion-litio",
        "850760",
        (80, 600),
        (
            ("celdas de ion-litio", "850760"),
            ("partes de acumuladores", "850790"),
            ("sistema electrónico de gestión", "853710"),
            ("carcasa de aluminio", "761699"),
            ("lámina separadora de plástico", "392099"),
            ("mazo de cableado", "854442"),
            ("conectores", "853690"),
        ),
    ),
    "TCA-8414": ProductTemplate(
        "ventilador industrial",
        "841459",
        (120, 900),
        (
            ("motor eléctrico", "850110"),
            ("rodete", "841490"),
            ("carcasa de chapa", "732690"),
            ("cojinetes", "848210"),
            ("condensador de arranque", "853221"),
            ("rejilla protectora", "732619"),
            ("cable de alimentación", "854442"),
        ),
    ),
    "UE-JP-8501": ProductTemplate(
        "motor eléctrico monofásico",
        "850140",
        (150, 1200),
        (
            ("devanado de cobre", "854411"),
            ("chapa magnética", "722519"),
            ("rodamientos", "848210"),
            ("carcasa de fundición", "732599"),
            ("rotor prefabricado", "850190"),
            ("piezas de estator y rotor", "850300"),
            ("caja de bornes", "853890"),
        ),
    ),
    "UE-KR-9403": ProductTemplate(
        "mueble de oficina de madera",
        "940330",
        (90, 700),
        (
            ("tablero de fibra de madera", "441114"),
            ("herrajes", "830242"),
            ("estructura metálica", "940390"),
            ("asiento tapizado", "940180"),
            ("tejido de tapizado", "540752"),
            ("barniz", "320890"),
            ("tornillería", "731815"),
        ),
    ),
    "CETA-8413": ProductTemplate(
        "bomba centrífuga",
        "841370",
        (200, 1500),
        (
            ("cuerpo de bomba prefabricado", "841370"),
            ("partes de bombas", "841391"),
            ("motor eléctrico", "850110"),
            ("juntas de caucho", "401693"),
            ("eje de acero", "722840"),
            ("rodamientos", "848210"),
            ("sello mecánico", "848420"),
        ),
    ),
    "UE-KR-8479": ProductTemplate(
        "máquina de envasado",
        "847989",
        (500, 4000),
        (
            ("controlador programable", "853710"),
            ("motor eléctrico", "850110"),
            ("estructura de acero", "730890"),
            ("reductor", "848340"),
            ("sensores", "903289"),
            ("subconjunto mecánico", "847990"),
            ("bomba de vacío", "841410"),
        ),
    ),
}

ORIGIN_WEIGHTS = (40, 12, 40, 8)  # UE, SOCIO, TERCERO, DESCONOCIDO
P_DECLARED = 0.85
P_SAME_HEADING = 0.4
P_EXCLUDED_HEADING = 0.5
P_SMALL_SAME_HEADING = 0.5  # el material de la misma partida pesa poco: la tolerancia lo rescata
P_EXACT_THRESHOLD = 0.20  # total no originario justo en el umbral
P_OVER_THRESHOLD = 0.12  # total por encima del umbral pero dentro de la tolerancia (trampa)
P_INSUFFICIENT_ONLY = 0.10
P_NO_TOLERANCE = 0.15

OPERATION_LABEL = {
    "fabricacion": "fabricación",
    "montaje": "montaje",
    "mecanizado": "mecanizado",
    "embalaje": "embalaje",
    "etiquetado": "etiquetado",
    "pintado": "pintado",
    "limpieza": "limpieza",
}


def fmt_code(code: str) -> str:
    return f"{code[:4]}.{code[4:]}"


def fmt_eur(value: float) -> str:
    text = f"{value:,.2f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(value: Fraction) -> str:
    percent = value * 100
    return str(percent.numerator) if percent.denominator == 1 else f"{float(percent):g}"


def describe_alternative(alt: Alt) -> str:
    except_text = ""
    if alt.excluded_headings:
        heads = ", ".join(alt.excluded_headings)
        plural = "partidas" if len(alt.excluded_headings) > 1 else "partida"
        except_text = f", salvo a partir de {'las' if len(alt.excluded_headings) > 1 else 'la'} {plural} {heads}"
    if alt.kind == "WO":
        return "producto obtenido en su totalidad (WO)"
    if alt.kind == "CC":
        return f"cambio de capítulo (CC){except_text}"
    if alt.kind == "CTH":
        return f"cambio de partida (CTH){except_text}"
    if alt.kind == "CTSH":
        return f"cambio de subpartida (CTSH){except_text}"
    assert alt.threshold is not None
    basis = "precio franco fábrica (EXW)" if alt.basis == "EXW" else "precio FOB"
    if alt.kind == "MAXNOM":
        return (
            f"valor máximo de materiales no originarios (MaxNOM) del {fmt_pct(alt.threshold)} % "
            f"del {basis}"
        )
    return (
        f"valor de contenido regional (RVC) de al menos el {fmt_pct(alt.threshold)} %, "
        f"es decir, (precio FOB - valor de materiales no originarios) / precio FOB "
        f">= {fmt_pct(alt.threshold)} %"
    )


def describe_rule(rule: Rule) -> str:
    return "; o ".join(describe_alternative(alt) for alt in rule.alternatives) + "."


# --------------------------------------------------------------------------- generador


class OriginGenerator(ProblemGenerator):
    """¿Es originario? Regla de la partida, lista de materiales, tolerancia y operaciones.

    La regla literal va siempre en el enunciado: el modelo no puede memorizar cientos de reglas,
    tiene que leerla y aplicarla.
    """

    name = "origin"

    def sample_params(self, rng: random.Random, split: str) -> dict[str, Any]:
        if split == "ood":
            raise NotImplementedError(
                "el split ood (reglas estilo UE-Corea y UE-Mercosur) aún no está definido"
            )
        rule_id = rng.choice(sorted(RULES))
        rule, product = RULES[rule_id], PRODUCTS[rule_id]
        exw = rng.randrange(product.exw_range[0], product.exw_range[1] + 1, 10)
        fob = round(exw * rng.uniform(1.03, 1.12))
        heading = product.code[:4]
        excluded = {h for alt in rule.alternatives for h in alt.excluded_headings}

        pool = list(product.components)
        chosen: list[tuple[str, str]] = []
        same_heading = [c for c in pool if c[1][:4] == heading]
        if same_heading and rng.random() < P_SAME_HEADING:
            chosen.append(rng.choice(same_heading))
        banned = [c for c in pool if c[1][:4] in excluded and c not in chosen]
        if banned and rng.random() < P_EXCLUDED_HEADING:
            chosen.append(rng.choice(banned))
        size = max(len(chosen), rng.randint(3, min(6, len(pool))))
        rest = [c for c in pool if c not in chosen]
        chosen += rng.sample(rest, size - len(chosen))
        rng.shuffle(chosen)

        shares = [rng.uniform(0.04, 0.28) for _ in chosen]
        scale = min(1.0, 0.9 / sum(shares))
        materials = []
        for (name, code), share in zip(chosen, shares):
            origin = rng.choices(ORIGINS, weights=ORIGIN_WEIGHTS)[0]
            declared = origin in ORIGINATING_ORIGINS and rng.random() < P_DECLARED
            materials.append(
                {
                    "name": name,
                    "code": code,
                    "origin": origin,
                    "value": round(exw * share * scale, 2),
                    "declared": declared,
                }
            )
        tolerance_pct = 0 if rng.random() < P_NO_TOLERANCE else 10

        same_in_list = [m for m in materials if m["code"][:4] == heading]
        if same_in_list and rng.random() < P_SMALL_SAME_HEADING:
            small = rng.choice(same_in_list)
            small["origin"], small["declared"] = "TERCERO", False
            small["value"] = round(exw * rng.uniform(0.02, 0.095), 2)

        draw = rng.random()
        if draw < P_EXACT_THRESHOLD:
            self._force_exact(rule, materials, exw, fob, Fraction(0))
        elif draw < P_EXACT_THRESHOLD + P_OVER_THRESHOLD and tolerance_pct:
            extra = Fraction(tolerance_pct, 100) * exw * Fraction(str(round(rng.uniform(0.2, 0.8), 2)))
            self._force_exact(rule, materials, exw, fob, extra)

        if rng.random() < P_INSUFFICIENT_ONLY:
            operations = rng.sample(INSUFFICIENT_OPERATIONS, rng.randint(1, 2))
        else:
            operations = rng.sample(SUFFICIENT_OPERATIONS, rng.randint(1, 2))
            if rng.random() < 0.3:
                operations.append(rng.choice(INSUFFICIENT_OPERATIONS))
            rng.shuffle(operations)

        return {
            "rule_id": rule_id,
            "agreement": rule.agreement,
            "product_name": product.name,
            "product_code": product.code,
            "exw": exw,
            "fob": fob,
            "tolerance_pct": tolerance_pct,
            "operations": operations,
            "materials": materials,
        }

    @staticmethod
    def _force_exact(
        rule: Rule, materials: list[dict[str, Any]], exw: int, fob: int, extra: Fraction
    ) -> None:
        """Ajusta un material para que el total no originario caiga en el umbral (+ ``extra`` €).

        ``extra`` = 0 da el umbral exacto; con ``extra`` > 0 el total supera el umbral pero cabe
        en la tolerancia, que no debe rescatarlo (MaxNOM/RVC no se amplían).
        """
        alt = next((a for a in rule.alternatives if a.kind in VALUE_CRITERIA), None)
        if alt is None or alt.threshold is None:
            return
        denominator = exw if alt.basis == "EXW" else fob
        target = (
            alt.threshold * denominator
            if alt.kind == "MAXNOM"
            else (1 - alt.threshold) * denominator
        ) + extra
        non_orig = [
            m for m in materials if not (m["origin"] in ORIGINATING_ORIGINS and m["declared"])
        ]
        if not non_orig:
            materials[0]["origin"], materials[0]["declared"] = "TERCERO", False
            non_orig = [materials[0]]
        adjuster, others = non_orig[0], non_orig[1:]
        while sum((_frac(m["value"]) for m in others), Fraction(0)) >= target:
            for m in others:
                m["value"] = round(m["value"] / 2, 2)
        adjuster["value"] = round(
            float(target - sum((_frac(m["value"]) for m in others), Fraction(0))), 2
        )

    def solve(self, params: dict[str, Any]) -> tuple[str, dict[str, str]]:
        rule = RULES[params["rule_id"]]
        materials = [Material.from_dict(m) for m in params["materials"]]
        evaluation = evaluate(
            rule,
            params["product_code"],
            params["exw"],
            params["fob"],
            materials,
            params["operations"],
            Fraction(params["tolerance_pct"], 100),
        )
        answer = json.dumps(
            {
                "originario": evaluation.originating,
                "criterio": evaluation.criterion,
                "pct_no_originario": evaluation.pct_non_originating,
                "criterios_validos": evaluation.satisfied,
            },
            ensure_ascii=False,
        )
        branches = {
            "regla": rule.rule_id,
            "originario": str(evaluation.originating),
            "criterio_aplicado": evaluation.criterion,
            "criterios_validos": "|".join(evaluation.satisfied) or "NINGUNO",
        }
        branches.update({name: str(value) for name, value in evaluation.flags.items()})
        return answer, branches

    def render(self, params: dict[str, Any], rng: random.Random) -> tuple[str, int]:
        rule = RULES[params["rule_id"]]
        agreement_name, partner = AGREEMENTS[params["agreement"]]
        origin_label = {
            "UE": "UE",
            "SOCIO": partner,
            "TERCERO": "tercer país",
            "DESCONOCIDO": "origen desconocido",
        }
        code = fmt_code(params["product_code"])
        materials = params["materials"]

        def declaration(m: dict[str, Any]) -> str:
            if m["origin"] not in ORIGINATING_ORIGINS:
                return ""
            return "con declaración de proveedor" if m["declared"] else "sin declaración de proveedor"

        product_block = (
            f"Una pyme española exporta a {partner} el producto «{params['product_name']}» "
            f"(subpartida {code}) al amparo del {agreement_name}. "
            f"Precio franco fábrica (EXW): {fmt_eur(params['exw'])} €. "
            f"Precio FOB: {fmt_eur(params['fob'])} €."
        )
        rule_block = f"Regla de origen de la partida {params['product_code'][:4]}: {describe_rule(rule)}"
        tolerance_text = (
            f"Tolerancia: los materiales no originarios que incumplen el cambio de clasificación "
            f"pueden usarse si su valor total no supera el {params['tolerance_pct']} % del precio "
            f"EXW. La tolerancia no amplía los umbrales de valor (MaxNOM, RVC)."
            if params["tolerance_pct"]
            else "No se aplica tolerancia en este caso."
        )
        insufficient = ", ".join(OPERATION_LABEL[o] for o in INSUFFICIENT_OPERATIONS)
        general_block = (
            "Reglas generales:\n"
            f"- Son originarios los materiales de la UE o de {partner} con declaración de proveedor. "
            "Los materiales de terceros países, los de origen desconocido y los de la UE o "
            "del país socio sin declaración de proveedor se consideran no originarios.\n"
            f"- {tolerance_text}\n"
            f"- Las operaciones insuficientes ({insufficient}) por sí solas no confieren origen."
        )
        operations_block = "Operaciones realizadas: " + ", ".join(
            OPERATION_LABEL[o] for o in params["operations"]
        ) + "."
        ask = (
            "Decide si el producto es originario. Da la respuesta final como JSON con las claves "
            '"originario" (true/false), "criterio" (uno de los criterios de la regla que se '
            "cumplan: CTH, CTSH, CC, MAXNOM, RVC o WO; NINGUNO si no es originario) y "
            '"pct_no_originario" (valor total de materiales no originarios sobre el precio EXW, '
            "en %, con dos decimales)."
        )

        template_id = rng.randrange(3)
        if template_id == 0:
            lines = []
            for m in materials:
                extra = declaration(m)
                lines.append(
                    f"- {m['name']} (SA {fmt_code(m['code'])}): origen {origin_label[m['origin']]}, "
                    f"{fmt_eur(m['value'])} €" + (f", {extra}." if extra else ".")
                )
            materials_block = "Lista de materiales utilizados:\n" + "\n".join(lines)
            sections = [product_block, rule_block, general_block, materials_block, operations_block, ask]
        elif template_id == 1:
            rows = ["| Material | SA | Origen | Valor (€) | Declaración de proveedor |", "|---|---|---|---|---|"]
            for m in materials:
                extra = declaration(m)
                decl = ("sí" if m["declared"] else "no") if extra else "—"
                rows.append(
                    f"| {m['name']} | {fmt_code(m['code'])} | {origin_label[m['origin']]} | "
                    f"{fmt_eur(m['value'])} | {decl} |"
                )
            materials_block = "Materiales utilizados:\n" + "\n".join(rows)
            sections = [rule_block, product_block, materials_block, operations_block, general_block, ask]
        else:
            parts = []
            for m in materials:
                extra = declaration(m)
                parts.append(
                    f"{m['name']} ({fmt_code(m['code'])}, {origin_label[m['origin']]}, "
                    f"{fmt_eur(m['value'])} €" + (f", {extra})" if extra else ")")
                )
            materials_block = "Materiales: " + "; ".join(parts) + "."
            sections = [product_block, materials_block, operations_block, rule_block, general_block, ask]
        return "\n\n".join(sections), template_id


def describe(problems: list[Problem]) -> dict[str, Any]:
    """Branch coverage and leakage check: the two numbers I will ask you about."""
    counters: dict[str, Counter] = {}
    for problem in problems:
        for branch, value in problem.branches.items():
            counters.setdefault(branch, Counter())[value] += 1
    leaked = sum(1 for p in problems if p.answer in p.question)
    return {
        "n_problems": len(problems),
        "n_templates": len({p.template_id for p in problems}),
        "branches": {k: dict(v) for k, v in counters.items()},
        "answer_leaked_in_statement": leaked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=800, help="how many unique problems")
    parser.add_argument("--split", choices=["train", "test", "ood"], default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="rlm/data/train.jsonl")
    args = parser.parse_args()

    generator = OriginGenerator()
    problems = generator.generate(args.n, args.split, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for problem in problems:
            row = asdict(problem)
            row["split"] = args.split
            row["label_source"] = "generator"
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(describe(problems), indent=2, ensure_ascii=False))
    print(f"\n{len(problems)} problemas -> {out}")
    print("Ejemplo:\n" + problems[0].question + f"\nRespuesta: {problems[0].answer}")


if __name__ == "__main__":
    main()
