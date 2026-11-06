"""Checks for the /reasoning wiring that do not need a GPU or the trained adapter."""

import json

import pytest

from rlm.inference import VERIFIERS, ReasoningModel, split_thinking

EXPECTED = json.dumps(
    {
        "originario": True,
        "criterio": "CTSH",
        "pct_no_originario": 0.0,
        "criterios_validos": ["CTSH"],
    }
)


def test_origin_verifier_is_registered_and_default(tmp_path, monkeypatch):
    (tmp_path / "adapter_config.json").write_text('{"base_model_name_or_path": "Qwen/Qwen3-1.7B"}')
    monkeypatch.setenv("ARCA_RLM_ADAPTER", str(tmp_path))
    monkeypatch.delenv("ARCA_RLM_VERIFIER", raising=False)
    monkeypatch.delenv("ARCA_RLM_BASE_MODEL", raising=False)
    model = ReasoningModel.from_env()
    assert model.verifier_name == "origin"
    assert model.base_model == "Qwen/Qwen3-1.7B"
    assert "origin" in VERIFIERS


def test_missing_adapter_is_reported_as_pending(monkeypatch):
    monkeypatch.delenv("ARCA_RLM_ADAPTER", raising=False)
    with pytest.raises(NotImplementedError):
        ReasoningModel.from_env()


def test_split_and_verify_a_domain_answer():
    raw = (
        "<think>Todos los materiales son originarios: CTSH se cumple.</think>\n"
        '<answer>{"originario": true, "criterio": "CTSH", "pct_no_originario": 0.0}</answer>'
    )
    thinking, answer = split_thinking(raw)
    assert thinking.startswith("Todos los materiales")
    assert json.loads(answer)["criterio"] == "CTSH"
    assert VERIFIERS["origin"]().verify(raw, EXPECTED).is_correct
