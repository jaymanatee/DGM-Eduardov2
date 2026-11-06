"""Inference for the phase-1 reasoning model, and the object behind ``POST /reasoning``.

``ReasoningModel`` loads a base model plus (optionally) your LoRA adapter, generates
with the R1-Zero system prompt and splits the output into thinking / answer. When the
request carries the expected answer, it also runs the verifier and returns the verdict.

Configuration comes from environment variables so that the same code runs on your
laptop, on the DGX and inside Docker:

* ``ARCA_RLM_BASE_MODEL``  base model id (default: the one in the adapter's ``adapter_config.json``)
* ``ARCA_RLM_ADAPTER``     path to the trained LoRA adapter (final: ``rlm/weights/grpo_pps3``)
* ``ARCA_RLM_VERIFIER``    ``origin`` (default), ``numeric`` or ``exact_match``

Generation matches ``rlm/evaluate.py`` (same prompt, adapter merged, greedy, 3,072 tokens), so
the verdicts of ``/reasoning`` are comparable with the pass@1 in ``reports/``.

Try it from the command line::

    ARCA_RLM_ADAPTER=rlm/weights/grpo_pps3 \\
        uv run --extra train python -m rlm.inference "<problem statement>"
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from api.schemas import ReasoningResponse, VerifierVerdict
from rlm import disable_cudnn_attention
from rlm.data import build_prompt
from rlm.evaluate import render_prompt
from rlm.rewards import extract_answer, has_valid_format
from rlm.verifier import ExactMatchVerifier, NumericVerifier, OriginVerifier, Verifier

VERIFIERS: dict[str, type[Verifier]] = {
    "origin": OriginVerifier,
    "numeric": NumericVerifier,
    "exact_match": ExactMatchVerifier,
}

# The evaluation's limit: with fewer tokens, many correct answers would be cut off.
DEFAULT_MAX_NEW_TOKENS = 3072

THINK_PATTERN = re.compile(r"<think>(?P<think>.*?)</think>", re.DOTALL)


def split_thinking(raw: str) -> tuple[str, str]:
    """Return (thinking, answer) from a raw completion, tolerating imperfect formats."""
    match = THINK_PATTERN.search(raw)
    thinking = match.group("think").strip() if match else ""
    answer = extract_answer(raw)
    if answer is None:
        # No <answer> block: fall back to whatever comes after </think>, or the whole text.
        answer = raw.split("</think>")[-1].strip() if match else raw.strip()
    return thinking, answer


def _base_model_from_adapter(adapter: str) -> str:
    """PEFT stores the base model id in adapter_config.json; use it unless overridden."""
    config = Path(adapter) / "adapter_config.json"
    if config.exists():
        base = json.loads(config.read_text()).get("base_model_name_or_path")
        if base:
            return base
    return "Qwen/Qwen3-0.6B"


@dataclass
class ReasoningModel:
    base_model: str
    adapter_path: str | None = None
    verifier_name: str = "origin"

    @classmethod
    def from_env(cls) -> ReasoningModel:
        adapter = os.environ.get("ARCA_RLM_ADAPTER") or None
        if adapter is None:
            raise NotImplementedError(
                "Phase 1 is not wired yet: set ARCA_RLM_ADAPTER to the folder of the trained "
                "LoRA adapter, rlm/weights/grpo_pps3 (see .env.example)."
            )
        if not Path(adapter).exists():
            raise FileNotFoundError(f"ARCA_RLM_ADAPTER points to a missing folder: {adapter}")
        base_model = os.environ.get("ARCA_RLM_BASE_MODEL") or _base_model_from_adapter(adapter)
        return cls(
            base_model=base_model,
            adapter_path=adapter,
            verifier_name=os.environ.get("ARCA_RLM_VERIFIER", "origin"),
        )

    def load(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        disable_cudnn_attention()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model, dtype=dtype, device_map=device
        )
        if self.adapter_path:
            # Merged as in evaluate.py: otherwise rounding differs and greedy outputs diverge.
            model = PeftModel.from_pretrained(model, self.adapter_path).merge_and_unload()
        self.model = model.eval()
        self.verifier = VERIFIERS[self.verifier_name]()

    def generate(
        self, question: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    ) -> tuple[str, int]:
        """Return the raw completion and the number of generated tokens (greedy)."""
        import torch

        text = render_prompt(self.tokenizer, build_prompt(question))
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                # Override the sampling defaults in Qwen3's generation_config, as evaluate.py does.
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True), int(new_tokens.numel())

    def answer(
        self,
        question: str,
        expected_answer: str | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> ReasoningResponse:
        raw, n_tokens = self.generate(question, max_new_tokens)
        thinking, answer = split_thinking(raw)
        verdict = None
        if expected_answer is not None:
            result = self.verifier.verify(raw, expected_answer)
            verdict = VerifierVerdict(
                is_correct=result.is_correct,
                predicted=result.predicted,
                expected=result.expected,
                detail=result.detail,
            )
        return ReasoningResponse(
            thinking=thinking,
            answer=answer,
            raw=raw,
            has_valid_format=has_valid_format(raw),
            verifier=verdict,
            tokens_generated=n_tokens,
            model=f"{self.base_model}+{Path(self.adapter_path).name}"
            if self.adapter_path
            else self.base_model,
        )


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "If 5x - 3 = 12, what is the value of 5x + 3?"
    rlm = ReasoningModel.from_env()
    rlm.load()
    print(rlm.answer(question, expected_answer=None).model_dump_json(indent=2))
