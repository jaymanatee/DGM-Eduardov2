"""Verifier interface for reinforcement learning with verifiable rewards.

A verifier answers one question deterministically: *is this answer correct for
this problem?* Everything in phase 1 hangs on it. If the verifier is sloppy, the
model will learn to exploit the sloppiness instead of learning to reason.

We ship two verifiers:

* ``NumericVerifier`` compares numbers after normalisation. It is what GSM8K needs
  and what the smoke test uses.
* ``ExactMatchVerifier`` compares normalised strings. Useful for multiple-choice
  or short factual answers.

Your domain verifier goes in this module too. Subclass ``Verifier``, implement
``is_correct`` and add a test for it in ``tests/test_verifier.py``. Common shapes:
run unit tests on generated code, execute a SQL query and compare result sets,
validate a JSON document against a schema, check that a date falls in a range.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from rlm.rewards import extract_answer, normalize_number


@dataclass(frozen=True)
class VerificationResult:
    """What a verifier reports back. ``detail`` is free text for logging and debugging."""

    is_correct: bool
    predicted: str | None
    expected: str
    detail: str = ""


class Verifier(ABC):
    """Base class for all verifiers."""

    name: str = "verifier"

    @abstractmethod
    def is_correct(self, predicted: str | None, expected: str) -> bool:
        """Return True when ``predicted`` should be accepted as a correct answer."""

    def verify(self, completion: str, expected: str) -> VerificationResult:
        """Extract the final answer from a full completion and check it."""
        predicted = extract_answer(completion)
        ok = self.is_correct(predicted, expected)
        detail = "no <answer> block found" if predicted is None else ""
        return VerificationResult(ok, predicted, expected, detail)


class NumericVerifier(Verifier):
    """Numeric comparison with an optional absolute tolerance."""

    name = "numeric"

    def __init__(self, tolerance: float = 0.0):
        self.tolerance = tolerance

    def is_correct(self, predicted: str | None, expected: str) -> bool:
        if predicted is None:
            return False
        p, e = normalize_number(predicted), normalize_number(expected)
        if p is None or e is None:
            return False
        if self.tolerance == 0.0:
            return p == e
        return abs(float(p) - float(e)) <= self.tolerance


class ExactMatchVerifier(Verifier):
    """Case- and whitespace-insensitive string comparison."""

    name = "exact_match"

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def is_correct(self, predicted: str | None, expected: str) -> bool:
        if predicted is None:
            return False
        return self._normalize(predicted) == self._normalize(expected)


# --------------------------------------------------------- IMPLEMENTATION


PCT_TOLERANCE = 0.5
NO_CRITERION = "NINGUNO"


class OriginVerifier(Verifier):
    """Origen preferencial: la respuesta es un JSON con tres campos.

    ``expected`` es el ``answer`` del dataset: el JSON correcto más ``criterios_validos``.
    Es tolerante con el formato (fences de código, "46,0 %", "true" como texto) y estricto
    con el contenido: veredicto exacto, criterio dentro de los válidos y porcentaje a ±0,5.
    """

    name = "origin"

    def __init__(self, pct_tolerance: float = PCT_TOLERANCE):
        self.pct_tolerance = pct_tolerance

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _as_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        return None

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.replace("%", "").replace(",", ".").strip())
            except ValueError:
                return None
        return None

    def check(self, predicted: str | None, expected: str) -> tuple[bool, str]:
        """Devuelve (correcto, motivo). El motivo sirve para el análisis de fallos."""
        if predicted is None:
            return False, "no <answer> block found"
        truth = self._parse_json(expected)
        if truth is None:
            raise ValueError(f"expected no es un JSON válido: {expected!r}")
        answer = self._parse_json(predicted)
        if answer is None:
            return False, "la respuesta no es un JSON válido"

        verdict = self._as_bool(answer.get("originario"))
        if verdict is None:
            return False, "falta 'originario' o no es booleano"
        criterion = answer.get("criterio")
        if not isinstance(criterion, str):
            return False, "falta 'criterio' o no es texto"
        pct = self._as_float(answer.get("pct_no_originario"))
        if pct is None:
            return False, "falta 'pct_no_originario' o no es un número"

        if verdict != truth["originario"]:
            return False, f"veredicto incorrecto: {verdict}"
        valid = truth["criterios_validos"] if truth["originario"] else [NO_CRITERION]
        if criterion.strip().upper() not in valid:
            return False, f"criterio {criterion!r} no válido (válidos: {valid})"
        if abs(pct - truth["pct_no_originario"]) > self.pct_tolerance + 1e-9:
            return False, f"porcentaje fuera de ±{self.pct_tolerance}: {pct} vs {truth['pct_no_originario']}"
        return True, ""

    def is_correct(self, predicted: str | None, expected: str) -> bool:
        return self.check(predicted, expected)[0]

    def verify(self, completion: str, expected: str) -> VerificationResult:
        predicted = extract_answer(completion)
        ok, detail = self.check(predicted, expected)
        return VerificationResult(ok, predicted, expected, detail)
