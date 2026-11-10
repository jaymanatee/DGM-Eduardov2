"""Synthetic data for SFT on tool use: the product catalogue and the rules saved on disk."""

import json
import random
import re
from datetime import date

import pytest

from tool_use.evaluate import load_bench
from tool_use.sft import build, generate, no_tools
from tool_use.sft.catalog import BENCH_HEADINGS, PRODUCTS
from tool_use.sft.fetch import ecb_path, load_rule, roo_path
from tool_use.sft.generate import assistant_text, evaluation_trajectory, replay
from tool_use.tools import AGREEMENTS, ECB_CURRENCIES, get_exchange_rate

TODAY = date(2026, 11, 10)
PAIRS = [(p, a) for p in PRODUCTS for a in AGREEMENTS]


def bench_product_headings():
    headings = set()
    for case in load_bench("tool_use/bench/cases.jsonl", today=TODAY):
        expectations = case.get("expected_calls", []) + case.get("optional_calls", [])
        argument_sets = [e["arguments"] for e in expectations]
        argument_sets.append(case.get("expected_arguments", {}))
        for arguments in argument_sets:
            for key in ("hs_code", "product_hs_code"):
                if key in arguments:
                    headings.add(re.sub(r"\D", "", arguments[key])[:4])
    return headings


def test_no_training_product_shares_a_heading_with_a_benchmark_product():
    assert bench_product_headings() <= BENCH_HEADINGS
    clashes = [p.en for p in PRODUCTS if p.code[:4] in BENCH_HEADINGS]
    assert clashes == []


def test_catalogue_is_well_formed():
    assert len({(p.en, p.code) for p in PRODUCTS}) == len(PRODUCTS) >= 70
    for product in PRODUCTS:
        assert re.fullmatch(r"\d{6}", product.code), product.en
        assert product.exw_range[0] < product.exw_range[1], product.en
        assert len(product.materials) >= 2, product.en
        for material in product.materials:
            assert re.fullmatch(r"\d{6}", material.code), (product.en, material.en)
        assert set(product.subdivisions) <= set(AGREEMENTS), product.en


@pytest.mark.parametrize("product, agreement", PAIRS, ids=lambda x: getattr(x, "code", x))
def test_every_rule_is_on_disk_and_every_subdivision_choice_is_annotated(product, agreement):
    assert roo_path(product.code, agreement).exists()
    sets = load_rule(product.code, agreement, TODAY)["rule_sets"]
    if len(sets) == 1:
        assert agreement not in product.subdivisions, "annotation for a rule with one set"
        return
    assert agreement in product.subdivisions, f"{product.en}: which of {len(sets)} sets?"
    choice = product.subdivisions[agreement]
    if choice is not None:
        number, reason = choice
        assert 1 <= number <= len(sets) and reason


def test_every_ecb_currency_is_on_disk():
    assert all(ecb_path(currency).exists() for currency in ECB_CURRENCIES)


# ------------------------------------------------------------------- generated trajectories


def _trajectories(n=90, seed=11):
    rng = random.Random(seed)
    pairs = generate.eligible_pairs(PRODUCTS)
    out = []
    for i in range(n):
        product, agreement = generate.pick(rng, pairs)
        kind = ("evaluation", "currency", "statement")[i % 3]
        out.append(
            evaluation_trajectory(
                rng, product, agreement, generate.random_today(rng), rng.choice(("es", "en")),
                currency=kind == "currency", statement=kind == "statement",
            )
        )  # fmt: skip
    return out


TRAJECTORIES = _trajectories()


def _calls(message):
    return [
        json.loads(b) for b in re.findall(r"<tool_call>\n(.*?)\n</tool_call>", message["content"])
    ]


def test_the_generated_kinds_and_outcomes_all_occur():
    seen = {(t.kind, t.outcome) for t in TRAJECTORIES}
    assert {("statement", "statement_issued"), ("statement", "originating_needs_reference")} <= seen
    assert {"originating", "not_originating", "by_hand"} <= {o for _, o in seen}


@pytest.mark.parametrize("t", TRAJECTORIES[:30], ids=lambda t: f"{t.kind}-{t.product}")
def test_the_real_tool_loop_reproduces_every_trajectory(t):
    """The loop, fed the trajectory's turns as model output, ends with the same messages."""
    expected = []
    for m in t.messages:
        if m["role"] == "assistant":
            m = {"role": "assistant", "content": assistant_text(m, thinking=True)}
        expected.append(m)
    assert replay(t) == expected


def _results(messages, tool="evaluate_origin"):
    return [
        json.loads(m["content"]).get("result")
        for m in messages
        if m["role"] == "tool" and m["name"] == tool
    ]


def test_the_answer_says_what_the_tool_said():
    for t in TRAJECTORIES:
        final = t.messages[-1]
        assert final["role"] == "assistant" and "<tool_call>" not in final["content"]
        paragraphs = final["content"].split("\n\n")
        verdict = next(p for p in paragraphs if not p.startswith(("Conversión", "Conversion")))
        last = _results(t.messages)[-1]
        if last is None:  # an error: by hand, or the FOB price is missing; never a verdict
            assert not verdict.startswith(("Sí", "Yes", "No:")), verdict
            assert any(w in verdict for w in ("a mano", "by hand", "FOB")), verdict
        else:
            assert verdict.startswith(("Sí", "Yes")) == last["originating"], verdict
            assert verdict.startswith("No:") == (not last["originating"]), verdict


def test_statements_follow_the_rules():
    for t in TRAJECTORIES:
        assistant = [m for m in t.messages if m["role"] == "assistant"]
        issued = [m for m in assistant if "issue_origin_statement" in m["content"]]
        if t.kind != "statement":
            assert not issued
            continue
        for message in issued:
            [call] = _calls(message)  # alone in its message
            evaluations = _results(t.messages[: t.messages.index(message)])
            assert evaluations[-1]["originating"]
            assert call["arguments"]["evaluation_id"] == evaluations[-1]["evaluation_id"]
            big = call["arguments"]["consignment_value_eur"] > 6000
            assert not big or call["arguments"].get("exporter_reference")
            # The reference, if any, is the one the user wrote.
            reference = call["arguments"].get("exporter_reference")
            assert reference is None or reference in t.messages[0]["content"]


def test_the_product_code_is_never_a_material_code():
    code = {p.en: p.code for p in PRODUCTS}
    for t in TRAJECTORIES:
        for m in t.messages:
            if m["role"] != "assistant":
                continue
            for call in _calls(m):
                if call["name"] == "evaluate_origin":
                    assert call["arguments"]["product_hs_code"] == code[t.product]


def test_no_value_of_the_benchmark_shows_up_in_the_trajectories():
    bench = []
    for case in load_bench("tool_use/bench/cases.jsonl", today=TODAY):
        for e in case.get("expected_calls", []):
            if e["tool"] == "issue_origin_statement":
                bench += [v for v in e["arguments"].values() if isinstance(v, str) and len(v) > 4]
    assert bench
    text = json.dumps([t.messages for t in TRAJECTORIES], ensure_ascii=False)
    assert [v for v in bench if v in text] == []


def test_offline_ecb_publishes_a_month_only_once_it_is_over():
    with generate.Offline(date(2026, 3, 10), random.Random(0)):
        assert get_exchange_rate(currency="USD", month="2026-02")["month"] == "2026-02"
        assert get_exchange_rate(currency="USD")["month"] == "2026-02"
        with pytest.raises(ValueError, match="once the month is over"):
            get_exchange_rate(currency="USD", month="2026-03")


# ------------------------------------------------------------------------- the whole dataset


SMALL = {split: build.build_split(split, 0.03, seed=5) for split in ("train", "val", "test")}
NO_TOOL_KINDS = {"concept", "off_topic", "out_of_scope", "missing_info"}


def test_every_kind_is_in_every_split():
    for split, trajectories in SMALL.items():
        assert {t.kind for t in trajectories} == set(build.COUNTS), split


def test_splits_share_no_product_topic_or_uncovered_agreement():
    def seen(split, attribute):
        return {getattr(t, attribute) for t in SMALL[split]} - {""}

    products = {s: seen(s, "product") for s in SMALL}
    assert products["test"] <= build.TEST_PRODUCTS and products["val"] <= build.VAL_PRODUCTS
    assert not products["train"] & (build.TEST_PRODUCTS | build.VAL_PRODUCTS)
    topics = {s: {t.outcome for t in SMALL[s] if t.kind == "concept"} for s in SMALL}
    assert topics["test"] <= build.TEST_TOPICS and not topics["train"] & build.TEST_TOPICS
    countries = {c[3] for c in no_tools.OTHER_AGREEMENTS if c[3] in build.TEST_AGREEMENTS}
    train_text = json.dumps([t.messages for t in SMALL["train"]], ensure_ascii=False)
    assert not any(c in train_text for c in countries)


def test_no_tool_trajectories_call_nothing_and_answer_in_one_turn():
    for trajectories in SMALL.values():
        for t in trajectories:
            if t.kind in NO_TOOL_KINDS:
                assert [m["role"] for m in t.messages] == ["user", "assistant"], t.kind
                assert "<tool_call>" not in t.messages[1]["content"]


def test_one_trajectory_of_each_kind_replays_through_the_real_loop():
    firsts = {}
    for t in SMALL["train"]:
        firsts.setdefault(t.kind, t)
    for t in firsts.values():
        expected = [
            {"role": "assistant", "content": assistant_text(m, thinking=True)}
            if m["role"] == "assistant"
            else m
            for m in t.messages
        ]
        assert replay(t) == expected, t.kind


def test_no_benchmark_question_or_value_in_any_kind():
    values, queries = [], []
    for case in load_bench("tool_use/bench/cases.jsonl", today=TODAY):
        queries.append(case["query"])
        expectations = case.get("expected_calls", []) + case.get("optional_calls", [])
        for arguments in [e["arguments"] for e in expectations] + [
            case.get("expected_arguments", {})
        ]:
            values += [v for v in arguments.values() if isinstance(v, str) and len(v) >= 8]
    text = json.dumps([t.messages for ts in SMALL.values() for t in ts], ensure_ascii=False)
    assert [v for v in values if v in text] == []
    users = {t.messages[0]["content"] for ts in SMALL.values() for t in ts}
    assert not users & set(queries)
    concept_questions = {q for c in no_tools.CONCEPTS.values() for q in c[0] + c[1]}
    assert not concept_questions & set(queries)
