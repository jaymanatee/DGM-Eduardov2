"""Try a phase 1 model on one problem and watch it reason token by token.

Run::

    uv run --extra train python -m rlm.try_model 5                     # SFT, test problem 5
    uv run --extra train python -m rlm.try_model 5 --adapter none      # same problem, base model
    uv run --extra train python -m rlm.try_model --question "..."      # your own statement

Same prompt and decoding as ``rlm/evaluate.py``, so what you see here is what the evaluation
measures. The model was trained on statements written like the generator's (rule, how the
rule applies, EXW price, bill of materials); a loose question without those data will get
an answer, but not a reliable one: that is the agent's job in phase 4.
"""

from __future__ import annotations

import argparse
import json
import time

from rlm.data import build_prompt
from rlm.evaluate import load_model, render_prompt
from rlm.verifier import OriginVerifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("index", nargs="?", type=int, default=0, help="problem number in --data")
    parser.add_argument("--data", default="rlm/data/test.jsonl")
    parser.add_argument("--question", default=None, help="your own statement instead of --data")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", default="rlm/weights/sft_lora", help="'none' for the base model")
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    args = parser.parse_args()

    import torch
    from transformers import TextStreamer

    expected = None
    if args.question:
        question = args.question
    else:
        with open(args.data, encoding="utf-8") as handle:
            row = json.loads(handle.readlines()[args.index])
        question = row["question"]
        answer = row["answer"]
        expected = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)

    print(question, "\n" + "=" * 80, flush=True)
    model, tokenizer = load_model(args.model, None if args.adapter == "none" else args.adapter)
    inputs = tokenizer(render_prompt(tokenizer, build_prompt(question)), return_tensors="pt")
    inputs = inputs.to(model.device)

    start = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
            streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True),
        )
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print("=" * 80)
    print(f"{len(new_tokens)} tokens in {time.time() - start:.0f} s")

    if expected is not None:
        result = OriginVerifier().verify(completion, expected)
        print("expected:", expected)
        print("verdict: ", "CORRECT" if result.is_correct else f"WRONG ({result.detail})")


if __name__ == "__main__":
    main()