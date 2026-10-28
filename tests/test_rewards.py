"""Tests for the format and accuracy rewards used in phase 1 and in the smoke test."""

import json

from rlm.rewards import (
    accuracy_reward,
    domain_reward,
    extract_answer,
    follows_origin_schema,
    format_reward,
    normalize_number,
    origin_accuracy_reward,
    rule_criteria,
    thinking_length,
)

GOOD = "<think>2 + 2 is 4.</think><answer>4</answer>"
GOOD_SPACED = "  <think>\nStep 1.\nStep 2.\n</think>\n<answer> 4 </answer>\n"
NO_THINK = "The answer is <answer>4</answer>"
DOUBLE_ANSWER = "<think>a</think><answer>4</answer><answer>5</answer>"
BOXED = "Let me compute. \\boxed{1,234}"


def test_format_reward_accepts_strict_structure():
    assert format_reward([None, None], [GOOD, GOOD_SPACED]) == [1.0, 1.0]


def test_format_reward_rejects_missing_or_extra_blocks():
    assert format_reward([None] * 3, [NO_THINK, DOUBLE_ANSWER, "just text"]) == [0.0, 0.0, 0.0]


def test_format_reward_handles_conversational_completions():
    completion = [{"role": "assistant", "content": GOOD}]
    assert format_reward([None], [completion]) == [1.0]


def test_extract_answer_prefers_answer_block_then_boxed():
    assert extract_answer(GOOD) == "4"
    assert extract_answer(BOXED) == "1,234"
    assert extract_answer(DOUBLE_ANSWER) == "5"  # the last block wins
    assert extract_answer("nothing here") is None


def test_normalize_number_strips_currency_and_separators():
    assert normalize_number("$1,234.50") == "1234.5"
    assert normalize_number("The total is 18 apples") == "18"
    assert normalize_number("18.0") == "18"
    assert normalize_number("-3") == "-3"
    assert normalize_number("no digits") is None


def test_accuracy_reward_matches_numerically():
    completions = [
        GOOD,
        "<think>x</think><answer>$4.00</answer>",
        "<think>x</think><answer>5</answer>",
    ]
    assert accuracy_reward([None] * 3, completions, answer=["4", "4", "4"]) == [1.0, 1.0, 0.0]


def test_accuracy_reward_is_zero_without_answer_block():
    assert accuracy_reward([None], ["<think>4</think>"], answer=["4"]) == [0.0]


def test_thinking_length_counts_tokens_in_think_block():
    assert thinking_length(GOOD) == 5  # '2', '+', '2', 'is', '4.'
    assert thinking_length("no think block") == 0


# --------------------------------------------------------- origin domain

# TCA-8507: "CTH o MaxNOM". The expected answer is the dataset's, with criterios_validos.
PARAMS_8507 = {"rule_id": "TCA-8507"}
EXPECTED = json.dumps(
    {"originario": True, "criterio": "CTH", "pct_no_originario": 56.9, "criterios_validos": ["CTH"]}
)


def _origin(answer: dict | str) -> str:
    text = answer if isinstance(answer, str) else json.dumps(answer)
    return f"<think>razonamiento</think><answer>{text}</answer>"


def test_origin_accuracy_reward_uses_the_origin_verifier():
    completions = [
        _origin({"originario": True, "criterio": "CTH", "pct_no_originario": 56.9}),
        _origin({"originario": True, "criterio": "CTH", "pct_no_originario": "57.1"}),  # ±0.5
        # Right percentage, wrong verdict: accuracy_reward would pay it, this one must not.
        _origin({"originario": False, "criterio": "NINGUNO", "pct_no_originario": 56.9}),
        "<think>sin respuesta</think>",
    ]
    assert origin_accuracy_reward([None] * 4, completions, answer=[EXPECTED] * 4) == [
        1.0, 1.0, 0.0, 0.0,
    ]
    assert accuracy_reward([None], completions[2:3], answer=[EXPECTED]) == [1.0]


def test_rule_criteria_reads_the_generator_rules():
    assert rule_criteria(PARAMS_8507) == {"CTH", "MAXNOM"}
    assert rule_criteria({"rule_id": "MERCOSUR-9401"}) == {"CC"}  # ood rules too
    assert rule_criteria({"rule_id": "NO-EXISTE"}) is None
    assert rule_criteria(None) is None


def test_follows_origin_schema_accepts_coherent_answers():
    criteria = {"CTH", "MAXNOM"}
    ok = [
        {"originario": True, "criterio": "MAXNOM", "pct_no_originario": 46.0},
        {"originario": True, "criterio": "CTH", "pct_no_originario": "56.90"},  # as the teacher
        {"originario": False, "criterio": "NINGUNO", "pct_no_originario": 0},
    ]
    for answer in ok:
        assert follows_origin_schema(json.dumps(answer), criteria), answer


def test_follows_origin_schema_rejects_incoherent_or_malformed_answers():
    criteria = {"CTH", "MAXNOM"}
    bad = [
        {"originario": True, "criterio": "CTSH", "pct_no_originario": 10.0},  # not in the rule
        {"originario": True, "criterio": "NINGUNO", "pct_no_originario": 10.0},
        {"originario": False, "criterio": "MAXNOM", "pct_no_originario": 74.6},  # E1's error
        {"originario": "true", "criterio": "CTH", "pct_no_originario": 10.0},
        {"originario": True, "criterio": "cth", "pct_no_originario": 10.0},
        {"originario": True, "criterio": "CTH", "pct_no_originario": "10,5 %"},
        {"originario": True, "criterio": "CTH", "pct_no_originario": 771.81},
        {"originario": True, "criterio": "CTH", "pct_no_originario": True},
        {"originario": True, "criterio": "CTH"},
        {"originario": True, "criterio": "CTH", "pct_no_originario": 10.0, "nota": "x"},
    ]
    for answer in bad:
        assert not follows_origin_schema(json.dumps(answer), criteria), answer
    assert not follows_origin_schema('```json\n{"originario": false}\n```', criteria)
    assert not follows_origin_schema("no es JSON", criteria)
    assert not follows_origin_schema(None, criteria)


def test_domain_reward_checks_the_rule_of_each_problem():
    answer = {"originario": True, "criterio": "CTSH", "pct_no_originario": 10.0}
    completions = [_origin(answer)] * 3
    params = [PARAMS_8507, {"rule_id": "CETA-8413"}, None]  # CTSH: not in 8507, the rule of 8413
    assert domain_reward([None] * 3, completions, params=params) == [0.0, 1.0, 1.0]
    assert domain_reward([None], completions[:1]) == [1.0]  # without params, no rule check


def test_domain_reward_ignores_correctness():
    wrong_but_well_formed = _origin(
        {"originario": False, "criterio": "NINGUNO", "pct_no_originario": 3.0}
    )
    assert domain_reward([None], [wrong_but_well_formed], params=[PARAMS_8507]) == [1.0]
    assert origin_accuracy_reward([None], [wrong_but_well_formed], answer=[EXPECTED]) == [0.0]
