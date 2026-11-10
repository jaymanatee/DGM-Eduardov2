"""Synthetic data for SFT on tool use: the product catalogue and the rules saved on disk."""

import re
from datetime import date

import pytest

from tool_use.evaluate import load_bench
from tool_use.sft.catalog import BENCH_HEADINGS, PRODUCTS
from tool_use.sft.fetch import ecb_path, load_rule, roo_path
from tool_use.tools import AGREEMENTS, ECB_CURRENCIES

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
