"""Verifiable reward functions for reasoning training (RLVR).

These are the two rewards DeepSeek used for R1-Zero, exactly as discussed in
class: a *format* reward that checks the ``<think>...</think><answer>...</answer>``
structure, and an *accuracy* reward that compares the final answer against the
ground truth using a deterministic verifier.

For our domain (preferential origin, see EXPERIMENTS.md) there are two more:
``origin_accuracy_reward``, the accuracy reward with ``OriginVerifier`` instead of a
numeric comparison, and ``domain_reward``, which checks that the answer respects the JSON
schema and is coherent with the rule of the problem.

All functions follow the signature expected by ``trl.GRPOTrainer``:

    reward_fn(prompts, completions, **kwargs) -> list[float]

``kwargs`` receives every extra column of the training dataset (for example
``answer``), which is how the ground truth reaches the reward function.

Completions may arrive as plain strings or as conversational messages
(``[{"role": "assistant", "content": "..."}]``). ``_completion_text`` handles both.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

# Full structure: a single think block followed by a single answer block, nothing else.
# The tempered groups ``(?:(?!</?tag>).)*`` forbid a second opening/closing tag inside a block,
# so "<answer>4</answer><answer>5</answer>" is rejected instead of silently accepted.
FORMAT_PATTERN = re.compile(
    r"^\s*<think>(?P<think>(?:(?!</?think>).)*)</think>"
    r"\s*<answer>(?P<answer>(?:(?!</?answer>).)*)</answer>\s*$",
    re.DOTALL,
)
# Looser pattern used to *extract* an answer even when the format is imperfect.
ANSWER_PATTERN = re.compile(r"<answer>(?P<answer>.*?)</answer>", re.DOTALL)
# Fallback for math-style completions that finish with \boxed{...}.
BOXED_PATTERN = re.compile(r"\\boxed\{(?P<answer>[^{}]*)\}")
# Something that looks like a number: optional sign, digits, thousands separators, decimals.
NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
# The answer of the origin domain: exactly these keys.
ORIGIN_KEYS = {"originario", "criterio", "pct_no_originario"}
NO_CRITERION = "NINGUNO"
# A percentage written as text, e.g. "40.03": digits and an optional decimal point, nothing else.
PLAIN_NUMBER = re.compile(r"^\d+(?:\.\d+)?$")


def _completion_text(completion: str | Sequence[dict]) -> str:
    """Return the text of a completion, whether it is a string or a list of messages."""
    if isinstance(completion, str):
        return completion
    parts = []
    for message in completion:
        if isinstance(message, dict) and message.get("role", "assistant") == "assistant":
            parts.append(str(message.get("content", "")))
    return "\n".join(parts)


def has_valid_format(text: str) -> bool:
    """True when the completion is exactly ``<think>...</think><answer>...</answer>``."""
    return FORMAT_PATTERN.match(text) is not None


def extract_answer(text: str) -> str | None:
    """Extract the final answer from a completion.

    Priority: the ``<answer>`` block, then a ``\\boxed{}`` expression. Returns ``None``
    when neither is present, so that callers can distinguish "no answer" from "wrong answer".
    """
    matches = ANSWER_PATTERN.findall(text)
    if matches:
        # The *last* block: models sometimes echo the format instructions before answering.
        return matches[-1].strip()
    boxed = BOXED_PATTERN.findall(text)
    if boxed:
        return boxed[-1].strip()
    return None


def normalize_number(text: str) -> str | None:
    """Pull the last number out of a piece of text and normalise it.

    ``"The answer is $1,234.50."`` -> ``"1234.5"``; ``"12"`` -> ``"12"``; no number -> ``None``.
    We take the *last* number because models often restate intermediate values before
    committing to a final one.
    """
    numbers = NUMBER_PATTERN.findall(text.replace("$", ""))
    if not numbers:
        return None
    raw = numbers[-1].replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def numbers_match(predicted: str | None, expected: str | None) -> bool:
    """Compare two answers numerically after normalisation."""
    if predicted is None or expected is None:
        return False
    return normalize_number(predicted) == normalize_number(expected)


def format_reward(prompts: Sequence, completions: Sequence, **kwargs) -> list[float]:
    """1.0 if the completion respects the think/answer format, else 0.0."""
    return [1.0 if has_valid_format(_completion_text(c)) else 0.0 for c in completions]


def accuracy_reward(
    prompts: Sequence, completions: Sequence, answer: Sequence[str], **kwargs
) -> list[float]:
    """1.0 if the extracted final answer matches the ground truth numerically, else 0.0.

    ``answer`` is the dataset column holding the ground truth (one per completion).
    """
    rewards = []
    for completion, expected in zip(completions, answer, strict=True):
        predicted = extract_answer(_completion_text(completion))
        rewards.append(1.0 if numbers_match(predicted, expected) else 0.0)
    return rewards


def origin_accuracy_reward(
    prompts: Sequence, completions: Sequence, answer: Sequence[str], **kwargs
) -> list[float]:
    """1.0 if ``OriginVerifier`` accepts the answer, else 0.0.

    ``accuracy_reward`` only compares the last number, which in our JSON is the percentage:
    a wrong verdict with the right percentage would get the full reward. This one uses the
    same verifier as ``distill.py`` and ``evaluate.py``, so GRPO optimises what pass@1 measures.
    Binary on purpose: partial credit (e.g. for the verdict alone) would pay "always NINGUNO".
    """
    from rlm.verifier import OriginVerifier  # here, not at the top: verifier.py imports this module

    verifier = OriginVerifier()
    return [
        1.0 if verifier.verify(_completion_text(c), expected).is_correct else 0.0
        for c, expected in zip(completions, answer, strict=True)
    ]


def rule_criteria(params: dict | None) -> set[str] | None:
    """Criteria the rule of a problem admits (``{"CTH", "MAXNOM"}``), or None if unknown."""
    from rlm.generate_problems import ALL_RULES

    if not isinstance(params, dict) or params.get("rule_id") not in ALL_RULES:
        return None
    return {alt.kind for alt in ALL_RULES[params["rule_id"]].alternatives}


def follows_origin_schema(answer: str | None, criteria: set[str] | None = None) -> bool:
    """True when the answer is the JSON the statement asks for, and coherent with itself.

    Unlike ``OriginVerifier``, which is lenient with the form, this is strict: a bare JSON
    object (no code fences), exactly the three keys, ``originario`` a JSON boolean,
    ``criterio`` NINGUNO if and only if the product is not originating and otherwise one of
    the rule's criteria (``criteria``; any criterion if None), and a percentage between 0 and
    100. The percentage may be a number or a plain numeric string ("40.03"): the statement
    asks for two decimals, and 63 % of the teacher's verified traces write it as a string.
    """
    if answer is None:
        return False
    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        return False
    if not isinstance(data, dict) or set(data) != ORIGIN_KEYS:
        return False

    verdict, criterion, pct = data["originario"], data["criterio"], data["pct_no_originario"]
    if not isinstance(verdict, bool) or not isinstance(criterion, str):
        return False
    if verdict:
        if criterion == NO_CRITERION or (criteria is not None and criterion not in criteria):
            return False
    elif criterion != NO_CRITERION:
        return False

    if isinstance(pct, str) and PLAIN_NUMBER.match(pct):
        pct = float(pct)
    if isinstance(pct, bool) or not isinstance(pct, (int, float)):
        return False
    return 0 <= pct <= 100


def domain_reward(
    prompts: Sequence, completions: Sequence, params: Sequence[dict] | None = None, **kwargs
) -> list[float]:
    """1.0 if the answer follows the origin schema (``follows_origin_schema``), else 0.0.

    Our third reward: whoever calls ``/reasoning`` parses this JSON, so it must be well
    formed and must not contradict itself or the rule, whether or not it is correct. It
    needs no ground truth, so it also gives signal on wrong answers, and it punishes the
    errors we saw in the distillation: criteria the rule does not have (statement E4) and a
    failed criterion instead of NINGUNO (E1). ``params`` is the dataset column with the rule.
    """
    if params is None:
        params = [None] * len(completions)
    return [
        1.0 if follows_origin_schema(extract_answer(_completion_text(c)), rule_criteria(p)) else 0.0
        for c, p in zip(completions, params, strict=True)
    ]


def thinking_length(text: str) -> int:
    """Number of whitespace-separated tokens inside the ``<think>`` block (0 if absent)."""
    match = FORMAT_PATTERN.match(text)
    if match is None:
        match = re.search(r"<think>(?P<think>.*?)</think>", text, re.DOTALL)
    if match is None:
        return 0
    return len(match.group("think").split())
