"""El generador es a la vez el dataset y la verdad: si falla, falla todo lo que viene detrás."""

import hashlib
import json

from rlm.generate_problems import OOD_RULES, RULES, OriginGenerator, describe, to_jsonl
from rlm.verifier import OriginVerifier

GEN = OriginGenerator()
VERIFIER = OriginVerifier()

# Huella de los 200 primeros problemas de train (semilla 0). Las trazas destiladas y los
# adaptadores entrenados dependen de que train.jsonl no cambie. Si este test falla, algún cambio
# ha alterado el sorteo o el texto de train: o lo deshaces, o asumes que hay que volver a
# destilar y actualizas la constante a propósito.
#
#     uv run python -m rlm.generate_problems --n 200 --split train --out /tmp/t.jsonl
#     sha256sum /tmp/t.jsonl
TRAIN_200_SHA256 = "e02c26b5b174854c3926e11cec6f65193fdd0591ddca9a7f8b166cc0f417d89a"

TRAPS = (
    "material_misma_partida",
    "subdivision_distinta",
    "excepcion_partida",
    "proveedor_sin_declaracion",
    "origen_desconocido",
    "operacion_insuficiente",
    "tolerancia_usada",
    "umbral_exacto",
    "tolerancia_no_amplia_maxnom",
)


def material(name: str, code: str, origin: str, value: float, declared: bool = False) -> dict:
    return {"name": name, "code": code, "origin": origin, "value": value, "declared": declared}


def solve(rule_id: str, product_code: str, exw: int, materials: list[dict], **overrides) -> dict:
    params = {
        "rule_id": rule_id,
        "product_code": product_code,
        "exw": exw,
        "fob": round(exw * 1.05),
        "tolerance_pct": 10,
        "operations": ["fabricacion"],
        "materials": materials,
    }
    params.update(overrides)
    return json.loads(GEN.solve(params)[0])


def test_train_fingerprint_is_frozen():
    content = to_jsonl(GEN.generate(200, "train", seed=0), "train")
    assert hashlib.sha256(content.encode("utf-8")).hexdigest() == TRAIN_200_SHA256


def test_generation_is_deterministic_and_a_smaller_n_is_a_prefix():
    first = GEN.generate(60, "train", seed=7)
    second = GEN.generate(60, "train", seed=7)
    assert [p.question for p in first] == [p.question for p in second]
    assert len({GEN.key(p.params) for p in first}) == 60
    # Lo que permite destilar solo los N primeros problemas de train.
    shorter = GEN.generate(20, "train", seed=7)
    assert [p.question for p in shorter] == [p.question for p in first[:20]]


def test_the_two_examples_of_the_proposal():
    # Batería: las celdas chinas son de la misma partida y pesan el 38 %, así que CTH falla,
    # pero el total no originario es el 46 % y cabe en el MaxNOM del 50 %.
    battery = solve(
        "TCA-8507",
        "850760",
        250,
        [
            material("celdas", "850760", "TERCERO", 95),
            material("placa", "853710", "TERCERO", 20),
        ],
    )
    assert battery["originario"] is True
    assert battery["criterio"] == "MAXNOM"
    assert battery["pct_no_originario"] == 46.0

    # Ventilador: el rodete es de la misma partida, pero pesa el 7,5 % y la tolerancia lo rescata.
    fan = solve(
        "TCA-8414",
        "841459",
        400,
        [
            material("motor", "850110", "TERCERO", 120),
            material("rodete", "841490", "TERCERO", 30),
        ],
    )
    assert fan["originario"] is True
    assert fan["criterio"] == "CTH"
    assert fan["pct_no_originario"] == 37.5


def test_traps_change_the_answer():
    cells = [material("celdas", "850760", "TERCERO", 130)]  # 52 % de 250
    # La tolerancia no amplía el MaxNOM: 52 % no cabe en el 50 % aunque haya un 10 % de tolerancia.
    assert solve("TCA-8507", "850760", 250, cells)["originario"] is False
    # Un material de la UE sin declaración de proveedor cuenta como no originario.
    eu = [material("celdas", "850760", "UE", 130, declared=False)]
    assert solve("TCA-8507", "850760", 250, eu)["pct_no_originario"] == 52.0
    declared = [dict(eu[0], declared=True)]
    assert solve("TCA-8507", "850760", 250, declared)["pct_no_originario"] == 0.0
    # Solo operaciones insuficientes: no es originario aunque los números cuadren.
    fine = [material("placa", "853710", "TERCERO", 20)]
    assert solve("TCA-8507", "850760", 250, fine)["originario"] is True
    assert solve("TCA-8507", "850760", 250, fine, operations=["embalaje"])["criterio"] == "NINGUNO"


def test_statements_are_diverse_and_do_not_leak_the_answer():
    stats = describe(GEN.generate(300, "train", seed=3))
    assert stats["n_templates"] == 3
    assert stats["answer_leaked_in_statement"] == 0
    assert set(stats["branches"]["regla"]) == set(RULES)
    assert set(stats["branches"]["originario"]) == {"True", "False"}


def test_every_trap_appears_in_at_least_8_percent_of_train():
    problems = GEN.generate(1000, "train", seed=0)
    for trap in TRAPS:
        share = sum(p.branches[trap] == "True" for p in problems) / len(problems)
        assert share >= 0.08, f"{trap}: {share:.1%}"


def test_answers_are_checkable_with_the_origin_verifier():
    for split in ("test", "ood"):
        for problem in GEN.generate(40, split, seed=5):
            truth = json.loads(problem.answer)
            assert VERIFIER.is_correct(problem.answer, problem.answer)
            # Cualquier criterio válido se acepta; el veredicto contrario, nunca.
            for criterion in truth["criterios_validos"]:
                other = json.dumps({**truth, "criterio": criterion})
                assert VERIFIER.is_correct(other, problem.answer)
            flipped = {**truth, "originario": not truth["originario"]}
            assert not VERIFIER.is_correct(json.dumps(flipped), problem.answer)


def test_ood_is_a_family_absent_from_train_and_test():
    ood = GEN.generate(100, "ood", seed=0)
    seen = GEN.generate(300, "train", seed=0) + GEN.generate(100, "test", seed=0)

    assert {p.params["rule_id"] for p in ood} == set(OOD_RULES)
    assert {p.params["rule_id"] for p in seen} == set(RULES)
    assert not set(OOD_RULES) & set(RULES)
    # Ni el acuerdo ni las partidas del producto se han visto antes.
    assert not {p.params["agreement"] for p in ood} & {p.params["agreement"] for p in seen}
    assert not {r.heading for r in OOD_RULES.values()} & {r.heading for r in RULES.values()}

    # La clave ``wording`` y la rama ``redaccion`` solo existen en ood: train queda intacto.
    assert all("wording" not in p.params and "redaccion" not in p.branches for p in seen)
    assert {p.branches["redaccion"] for p in ood} == {"siglas", "descriptiva"}


def test_ood_descriptive_wording_hides_the_acronyms_in_the_rule():
    for problem in GEN.generate(100, "ood", seed=0):
        rule_line = next(
            line for line in problem.question.split("\n") if line.startswith("Regla de origen")
        )
        acronyms = ("CC", "CTH", "CTSH", "MaxNOM", "RVC")
        has_acronym = any(f"({code}" in rule_line for code in acronyms)
        assert has_acronym == (problem.params["wording"] == "siglas")


def test_ood_keeps_both_verdicts_and_every_trap():
    stats = describe(GEN.generate(100, "ood", seed=0))
    assert stats["answer_leaked_in_statement"] == 0
    assert set(stats["branches"]["originario"]) == {"True", "False"}
    for trap in TRAPS:
        assert stats["branches"][trap].get("True", 0) > 0, trap
