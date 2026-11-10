"""Scoring of the phase 2 benchmark, and a sanity check of tool_use/bench/cases.jsonl."""

from datetime import date

import pytest

from tool_use.evaluate import load_bench, previous_month, score_case, summarise
from tool_use.tools import default_registry

REGISTRY = default_registry()


def call(name, ok=True, **arguments):
    return {"name": name, "arguments": arguments, "ok": ok}


def test_original_format_still_scores_the_first_call():
    case = {
        "query": "q",
        "expected_tool": "get_weather",
        "expected_arguments": {"location": "Paris"},
    }
    assert score_case(case, [call("get_weather", location=" paris ")]) == {
        "selection_ok": True,
        "arguments_ok": True,
    }
    assert score_case(case, [call("calculator", expression="1")])["selection_ok"] is False
    assert score_case({"query": "q", "needs_tool": False}, [])["selection_ok"] is True


def test_arguments_are_compared_as_the_tool_validates_them():
    case = {"expected_tool": "get_origin_rule", "expected_arguments": {"hs_code": "841459"}}
    made = [call("get_origin_rule", hs_code="8414.59.00", agreement="TCA")]
    assert score_case(case, made, registry=REGISTRY)["arguments_ok"]
    assert not score_case(case, made)["arguments_ok"]  # without the tool's validation
    vat = {
        "expected_tool": "check_vat_number",
        "expected_arguments": {"vat_number": "FR40303265045"},
    }
    spaced = [call("check_vat_number", vat_number="fr 40 303 265 045")]
    assert score_case(vat, spaced, registry=REGISTRY)["arguments_ok"]


def test_lists_match_in_any_order_and_only_listed_keys_count():
    case = {
        "expected_tool": "evaluate_origin",
        "expected_calls": [
            {
                "tool": "evaluate_origin",
                "arguments": {"criteria": [{"kind": "CTH"}, {"kind": "WO"}]},
            }
        ],
    }
    made = [call("evaluate_origin", criteria=[{"kind": "WO"}, {"kind": "CTH", "basis": "EXW"}])]
    scores = score_case(case, made)
    assert scores["arguments_ok"] and scores["calls_ok"]
    shorter = [call("evaluate_origin", criteria=[{"kind": "CTH"}])]
    assert not score_case(case, shorter)["calls_ok"]


def test_parallel_calls_in_any_order():
    case = {
        "expected_tool": ["get_exchange_rate", "check_vat_number"],
        "expected_calls": [
            {"tool": "get_exchange_rate", "arguments": {"currency": "JPY"}},
            {"tool": "check_vat_number", "arguments": {"vat_number": "ESA28015865"}},
        ],
    }
    made = [
        call("check_vat_number", vat_number="ESA28015865"),
        call("get_exchange_rate", currency="jpy"),
    ]
    assert score_case(case, made) == {
        "selection_ok": True,
        "arguments_ok": True,
        "calls_ok": True,
        "invented_arguments": [],
    }
    assert not score_case(case, made[:1])["calls_ok"]


def test_optional_calls_are_checked_only_if_made():
    case = {
        "expected_tool": ["get_origin_rule", "evaluate_origin"],
        "expected_calls": [{"tool": "evaluate_origin", "arguments": {"agreement": "TCA"}}],
        "optional_calls": [{"tool": "get_origin_rule", "arguments": {"hs_code": "841459"}}],
    }
    direct = [call("evaluate_origin", agreement="TCA")]
    assert score_case(case, direct)["arguments_ok"] and score_case(case, direct)["calls_ok"]
    looked_up = [call("get_origin_rule", hs_code="841459"), *direct]
    assert score_case(case, looked_up)["arguments_ok"]
    wrong_code = [call("get_origin_rule", hs_code="850760"), *direct]
    assert not score_case(case, wrong_code)["arguments_ok"]
    assert score_case(case, wrong_code)["calls_ok"]  # the evaluation itself was right


def test_forbidden_tools_and_invented_arguments():
    case = {
        "expected_tool": "evaluate_origin",
        "forbidden_tools": ["issue_origin_statement"],
        "forbidden_arguments": {"issue_origin_statement": ["exporter_reference"]},
    }
    assert score_case(case, [call("evaluate_origin")])["calls_ok"]
    issued = [call("evaluate_origin"), call("issue_origin_statement", exporter_reference="")]
    assert not score_case(case, issued)["calls_ok"]  # forbidden tool
    case.pop("forbidden_tools")
    assert score_case(case, issued)["calls_ok"]  # empty reference is fine
    invented = [call("issue_origin_statement", exporter_reference="EX-123456")]
    scores = score_case(case, invented)
    assert not scores["calls_ok"]
    assert scores["invented_arguments"] == ["issue_origin_statement.exporter_reference"]


def test_answer_contains_with_alternatives():
    case = {"needs_tool": False, "answer_contains": [["282.63", "282,63"], "EUR"]}
    assert score_case(case, [], "Son 282,63 eur")["answer_ok"]
    assert not score_case(case, [], "Son 282 EUR")["answer_ok"]


def test_summary_breaks_down_by_category():
    rows = [
        {"category": "a", "selection_ok": True, "arguments_ok": True, "answer_ok": False},
        {"category": "b", "selection_ok": False, "arguments_ok": False, "calls_ok": True},
    ]
    summary = summarise(rows)
    assert summary["tool_selection_accuracy"] == 0.5
    assert summary["multi_call_accuracy"] == 1.0 and summary["n_multi_call_accuracy"] == 1
    assert summary["by_category"]["a"] == {
        "n": 1,
        "selection_ok": 1.0,
        "arguments_ok": 1.0,
        "answer_ok": 0.0,
    }


def test_previous_month():
    assert previous_month(date(2026, 11, 9)) == "2026-10"
    assert previous_month(date(2027, 1, 3)) == "2026-12"


BENCH = load_bench("tool_use/bench/cases.jsonl", today=date(2026, 11, 9))


def test_bench_has_31_cases_with_known_tools():
    assert len(BENCH) == 31 and len({c["id"] for c in BENCH}) == 31
    assert sum(not c.get("needs_tool", True) for c in BENCH) >= 6
    for case in BENCH:
        expected = case.get("expected_tool", [])
        names = [expected] if isinstance(expected, str) else expected
        names += [e["tool"] for e in case.get("expected_calls", [])]
        names += case.get("forbidden_tools", []) + list(case.get("forbidden_arguments", {}))
        assert set(names) <= set(REGISTRY.tools), case["id"]


@pytest.mark.parametrize(
    "case", [c for c in BENCH if c.get("needs_tool", True)], ids=lambda c: c["id"]
)
def test_a_perfect_run_scores_full_marks(case):
    """Replay the expected calls as if the model had made them: every metric must pass."""
    calls = []
    if "expected_arguments" in case:
        calls.append(call(case["expected_tool"], **case["expected_arguments"]))
    calls += [call(e["tool"], **e["arguments"]) for e in case.get("expected_calls", [])]
    if "expected_arguments" not in case:  # parallel or chain: make the first call a valid one
        first = case["expected_tool"]
        first = first if isinstance(first, str) else first[0]
        calls.sort(key=lambda c: c["name"] != first)
    scores = score_case(case, calls, registry=REGISTRY)
    assert scores["selection_ok"] and scores["arguments_ok"], scores
    assert scores.get("calls_ok", True), scores
    # The single-call expectations are complete calls: the tool itself must accept them.
    if "expected_arguments" in case:
        REGISTRY.tools[case["expected_tool"]].validate(case["expected_arguments"])
