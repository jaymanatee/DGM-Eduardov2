"""Tests for the rule-coverage measurement (tool_use/rule_coverage.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tool_use.rule_coverage import ROO_DIR, measure, reason_for, rule_files

TEXTS = {
    "manufacturing_process": "Knitting or crocheting and making-up (including cutting).",
    "condition": "A change from any other heading, provided that the net value is lower.",
    "specific_material_value": (
        "the value of non-originating materials of heading 1701 used does not exceed 30%"
    ),
    "chapter_exclusion": "Manufacture from materials of any heading, except those of chapter 11.",
    "exception": "However, non-originating mustard flour may be used.",
    "other": "A change from within any of these subheading or any other subheading.",
}


@pytest.mark.parametrize("expected, text", TEXTS.items())
def test_reason_for_classifies_known_texts(expected, text):
    assert reason_for(text) == expected


def test_every_downloaded_rule_file_is_named_as_expected():
    found = rule_files(ROO_DIR)
    assert found, f"no rules in {ROO_DIR}: run tool_use.sft.fetch first"
    assert all(code.isdigit() and len(code) == 6 for code, _, _ in found)


def test_measure_counts_every_rule_set_once():
    result = measure(ROO_DIR)
    sets = result["rule_sets"]
    assert 0 < sets["readable"] <= sets["total"]
    assert sum(row["n"] for row in result["pairs"].values()) == result["products_x_agreements"]
    for key in ("by_agreement", "by_hs_chapter"):
        assert sum(row["total"] for row in result[key].values()) == sets["total"]


def test_textile_chapters_have_no_machine_readable_rule():
    """Chapters 61-63 state their rules as manufacturing operations (knitting, spinning,
    dyeing), not as classification changes or value thresholds, so the engine cannot check
    any of them. It is a limit of the domain, not a parsing bug."""
    by_chapter = measure(ROO_DIR)["by_hs_chapter"]
    for chapter in ("61", "62", "63"):
        if chapter in by_chapter:
            assert by_chapter[chapter]["readable"] == 0


def test_missing_folder_gives_an_empty_measurement(tmp_path: Path):
    assert rule_files(tmp_path) == []