"""Turn trajectories into SFT examples: one per assistant turn, as the tool loop renders it.

The prompt of each example is not built here with a copy of the chat template: the real
ToolLoop runs the trajectory (its turns stand in for the model's output, the tools run
offline on the trajectory's day) and every text the loop would hand to the model is kept.
So training sees exactly what the loop shows at inference, system prompt and date included,
in either variant:

* ``think``: each turn is ``<think>`` short reasoning ``</think>`` and then the calls or the
  answer; the loop runs with thinking on, so earlier turns keep their reasoning.
* ``nothink``: the same turns without the reasoning; the loop runs with thinking off, so
  the chat template opens an empty ``<think></think>`` before each generation.

The completion ends with ``<|im_end|>``, the token that closes an assistant turn in Qwen3
(TRL would add it otherwise). The loss covers only the completion.

Run::

    uv run --extra train python -m tool_use.sft.render --data outputs/tool_sft
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from datetime import date
from pathlib import Path
from typing import Any

from tool_use import tools
from tool_use.executor import DEFAULT_MODEL, ToolLoop
from tool_use.sft.generate import Offline, Trajectory, assistant_text

END = "<|im_end|>"
VARIANTS = {"think": True, "nothink": False}


def load_trajectories(path: str | Path) -> list[Trajectory]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        data = json.loads(line)
        trajectory_id = data.pop("id", "")
        t = Trajectory(**data)
        t.id = trajectory_id  # type: ignore[attr-defined]
        out.append(t)
    return out


def render_trajectory(t: Trajectory, tokenizer: Any, thinking: bool) -> list[dict[str, Any]]:
    """The (prompt, completion) pairs of a trajectory, rendered by the real loop."""
    turns = [assistant_text(m, thinking) for m in t.messages if m["role"] == "assistant"]
    prompts: list[str] = []

    class Capture(ToolLoop):
        def generate(self, messages: list[dict[str, Any]], use_tools: bool = True) -> str:
            prompts.append(self.render(messages, use_tools))
            return turns[len(prompts) - 1]

    loop = Capture(registry=tools.default_registry(), thinking=thinking)
    loop.tokenizer = tokenizer
    [query] = [m["content"] for m in t.messages if m["role"] == "user"]
    with Offline(date.fromisoformat(t.today), random.Random(t.seed), vies=t.vies):
        loop.run(query, max_turns=8)
    if len(prompts) != len(turns):
        name = getattr(t, "id", "")
        raise AssertionError(f"{name}: {len(prompts)} prompts for {len(turns)} turns")
    return [
        {
            "prompt": prompt,
            "completion": turn + END,
            "trajectory": getattr(t, "id", ""),
            "turn": i,
            "kind": t.kind,
            "lang": t.lang,
        }
        for i, (prompt, turn) in enumerate(zip(prompts, turns, strict=True))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="outputs/tool_sft")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="whose tokenizer and chat template")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = Path(args.data)
    stats: dict[str, Any] = {}
    for variant in args.variants:
        for split in ("train", "val", "test"):
            examples = []
            for t in load_trajectories(data / f"trajectories_{split}.jsonl"):
                examples += render_trajectory(t, tokenizer, VARIANTS[variant])
            lengths = [len(tokenizer(e["prompt"] + e["completion"])["input_ids"]) for e in examples]
            completions = [len(tokenizer(e["completion"])["input_ids"]) for e in examples]
            with (data / f"sft_{variant}_{split}.jsonl").open("w", encoding="utf-8") as f:
                for e in examples:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            stats[f"{variant}/{split}"] = {
                "examples": len(examples),
                "tokens_median": statistics.median(lengths),
                "tokens_max": max(lengths),
                "completion_tokens_median": statistics.median(completions),
                "completion_tokens_max": max(completions),
                "completion_tokens_total": sum(completions),
            }
            print(variant, split, json.dumps(stats[f"{variant}/{split}"]))
    (data / "sft_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
