"""Tests for the verifiers. Add one class of tests per domain verifier you write."""

from __future__ import annotations

import json

import pytest

from rlm.generate_problems import OriginGenerator
from rlm.verifier import ExactMatchVerifier, NumericVerifier, OriginVerifier


def test_numeric_verifier_exact():
    v = NumericVerifier()
    assert v.is_correct("42", "42")
    assert v.is_correct("$42.00", "42")
    assert not v.is_correct("41", "42")
    assert not v.is_correct(None, "42")
    assert not v.is_correct("forty-two", "42")


def test_numeric_verifier_with_tolerance():
    v = NumericVerifier(tolerance=0.05)
    assert v.is_correct("3.14", "3.1416")
    assert not v.is_correct("3.0", "3.1416")


def test_verify_extracts_from_full_completion():
    result = NumericVerifier().verify("<think>...</think><answer>18</answer>", "18")
    assert result.is_correct and result.predicted == "18"
    missing = NumericVerifier().verify("no tags at all", "18")
    assert not missing.is_correct and "no <answer>" in missing.detail


def test_exact_match_ignores_case_and_spacing():
    v = ExactMatchVerifier()
    assert v.is_correct("  Madrid ", "madrid")
    assert not v.is_correct("Barcelona", "Madrid")


# --- OriginVerifier ---------------------------------------------------------------------------
# ``expected`` es el ``answer`` del dataset (con ``criterios_validos``). La respuesta del modelo
# solo lleva tres campos: ``originario``, ``criterio`` y ``pct_no_originario``.

VERIFIER = OriginVerifier()

# Producto originario por CTH o por MaxNOM (varios criterios válidos).
EXPECTED_MULTI = json.dumps(
    {
        "originario": True,
        "criterio": "CTH",
        "pct_no_originario": 40.0,
        "criterios_validos": ["CTH", "MAXNOM"],
    }
)
# Producto no originario.
EXPECTED_NO = json.dumps(
    {"originario": False, "criterio": "NINGUNO", "pct_no_originario": 40.0, "criterios_validos": []}
)


def completion(answer: str | None) -> str:
    """Una generación completa del modelo. ``None`` = sin bloque <answer>."""
    if answer is None:
        return "<think>solo razono y no contesto</think>"
    return f"<think>razonamiento</think><answer>{answer}</answer>"


def answer(originario=True, criterio="CTH", pct=40.0) -> str:
    return json.dumps({"originario": originario, "criterio": criterio, "pct_no_originario": pct})


ACCEPTED = [
    pytest.param(EXPECTED_MULTI, answer(), id="correcta-canonica"),
    pytest.param(EXPECTED_MULTI, answer(criterio="MAXNOM"), id="otro-criterio-valido"),
    pytest.param(EXPECTED_MULTI, answer(criterio="cth"), id="criterio-en-minusculas"),
    pytest.param(EXPECTED_MULTI, answer(pct=40.5), id="pct-en-el-limite-superior"),
    pytest.param(EXPECTED_MULTI, answer(pct=39.5), id="pct-en-el-limite-inferior"),
    pytest.param(EXPECTED_MULTI, answer(pct="40,3 %"), id="pct-como-texto-con-coma-y-%"),
    pytest.param(EXPECTED_MULTI, answer(originario="true"), id="booleano-como-texto"),
    pytest.param(
        EXPECTED_MULTI, "```json\n" + answer() + "\n```", id="con-fences-de-codigo"
    ),
    pytest.param(EXPECTED_NO, answer(False, "NINGUNO"), id="no-originario-correcto"),
]

# (expected, texto de <answer>, fragmento que tiene que aparecer en el motivo)
REJECTED = [
    pytest.param(EXPECTED_MULTI, answer(False, "NINGUNO"), "veredicto incorrecto", id="veredicto-contrario"),
    pytest.param(EXPECTED_NO, answer(True, "CTH"), "veredicto incorrecto", id="dice-originario-y-no-lo-es"),
    pytest.param(EXPECTED_MULTI, answer(criterio="RVC"), "no válido", id="criterio-que-no-se-cumple"),
    pytest.param(EXPECTED_MULTI, answer(criterio="NINGUNO"), "no válido", id="originario-sin-criterio"),
    pytest.param(EXPECTED_NO, answer(False, "CTH"), "no válido", id="no-originario-con-criterio"),
    pytest.param(EXPECTED_MULTI, answer(pct=40.51), "fuera de", id="pct-justo-fuera-por-arriba"),
    pytest.param(EXPECTED_MULTI, answer(pct=39.49), "fuera de", id="pct-justo-fuera-por-abajo"),
    pytest.param(EXPECTED_MULTI, '{"originario": true, "criterio": "CTH"', "JSON", id="json-roto"),
    pytest.param(EXPECTED_MULTI, "[1, 2]", "JSON", id="answer-no-es-un-objeto"),
    pytest.param(EXPECTED_MULTI, "originario: sí", "JSON", id="texto-libre"),
    pytest.param(EXPECTED_MULTI, '{"originario": true, "criterio": "CTH"}', "pct_no_originario", id="falta-pct"),
    pytest.param(EXPECTED_MULTI, '{"criterio": "CTH", "pct_no_originario": 40}', "originario", id="falta-originario"),
    pytest.param(EXPECTED_MULTI, '{"originario": true, "pct_no_originario": 40}', "criterio", id="falta-criterio"),
    pytest.param(EXPECTED_MULTI, answer(originario=1), "originario", id="originario-como-1"),
    pytest.param(EXPECTED_MULTI, answer(originario=None), "originario", id="originario-null"),
    pytest.param(EXPECTED_MULTI, answer(criterio=3), "criterio", id="criterio-numerico"),
    pytest.param(EXPECTED_MULTI, answer(pct=True), "pct_no_originario", id="pct-booleano"),
    pytest.param(EXPECTED_MULTI, answer(pct="mucho"), "pct_no_originario", id="pct-no-numerico"),
]


@pytest.mark.parametrize("expected, model_answer", ACCEPTED)
def test_accepts(expected, model_answer):
    result = VERIFIER.verify(completion(model_answer), expected)
    assert result.is_correct, result.detail
    assert result.detail == ""


@pytest.mark.parametrize("expected, model_answer, reason", REJECTED)
def test_rejects_and_says_why(expected, model_answer, reason):
    result = VERIFIER.verify(completion(model_answer), expected)
    assert not result.is_correct
    assert reason in result.detail


def test_rejects_completion_without_answer_block():
    result = VERIFIER.verify(completion(None), EXPECTED_MULTI)
    assert not result.is_correct
    assert result.predicted is None
    assert "no <answer> block found" in result.detail


def test_last_answer_block_is_the_one_that_counts():
    wrong, right = answer(False, "NINGUNO"), answer()
    text = f"<answer>{wrong}</answer> pensándolo mejor <answer>{right}</answer>"
    assert VERIFIER.verify(text, EXPECTED_MULTI).is_correct


def test_last_number_trick_does_not_work():
    """Con el accuracy_reward numérico de la plantilla, esto pasaría: solo mira el último número."""
    assert not VERIFIER.is_correct(answer(False, "NINGUNO", 40.0), EXPECTED_MULTI)


def test_is_correct_matches_verify():
    for expected, model_answer in [(EXPECTED_MULTI, answer()), (EXPECTED_MULTI, answer(False, "NINGUNO"))]:
        assert VERIFIER.is_correct(model_answer, expected) == VERIFIER.verify(
            completion(model_answer), expected
        ).is_correct


def test_is_correct_with_none_prediction():
    assert VERIFIER.is_correct(None, EXPECTED_MULTI) is False


def test_malformed_expected_is_a_dataset_bug_not_a_wrong_answer():
    with pytest.raises(ValueError):
        VERIFIER.is_correct(answer(), "esto no es un json")


def test_custom_tolerance():
    strict = OriginVerifier(pct_tolerance=0.1)
    assert strict.is_correct(answer(pct=40.1), EXPECTED_MULTI)
    assert not strict.is_correct(answer(pct=40.2), EXPECTED_MULTI)


# --- contra el dataset real -------------------------------------------------------------------


@pytest.fixture(scope="module")
def problems():
    return OriginGenerator().generate(300, "test", seed=0)


def model_view(expected: str) -> dict:
    """Lo que tendría que contestar el modelo: las tres claves, sin criterios_validos."""
    truth = json.loads(expected)
    return {k: truth[k] for k in ("originario", "criterio", "pct_no_originario")}


def test_every_dataset_answer_accepts_itself(problems):
    for problem in problems:
        model_answer = json.dumps(model_view(problem.answer))
        result = VERIFIER.verify(completion(model_answer), problem.answer)
        assert result.is_correct, (problem.answer, result.detail)


def test_flipping_the_verdict_is_always_rejected(problems):
    for problem in problems:
        flipped = model_view(problem.answer)
        flipped["originario"] = not flipped["originario"]
        assert not VERIFIER.is_correct(json.dumps(flipped), problem.answer), problem.answer


def test_every_alternative_criterion_is_accepted(problems):
    multi = [p for p in problems if len(json.loads(p.answer)["criterios_validos"]) > 1]
    assert multi, "el dataset debería tener casos con varios criterios válidos"
    for problem in multi:
        truth = json.loads(problem.answer)
        for criterion in truth["criterios_validos"]:
            model_answer = json.dumps({**model_view(problem.answer), "criterio": criterion})
            assert VERIFIER.is_correct(model_answer, problem.answer), (criterion, problem.answer)
