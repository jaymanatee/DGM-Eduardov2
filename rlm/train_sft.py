"""Phase 1, step 2: cold-start supervised fine-tuning on verified reasoning traces.

This is the "SFT round 1" box of the DeepSeek pipeline. The input is the JSONL produced
by ``rlm/distill.py``: problems from your domain with a teacher-generated reasoning trace
that *passed your verifier*. The output is a LoRA adapter that already speaks the
``<think>…</think><answer>…</answer>`` format before we ever run GRPO.

Run::

    uv run --extra train python -m rlm.train_sft --data rlm/data/sft_traces.jsonl \
        --output rlm/weights/sft_lora

What is decided for you: the trainer (``trl.SFTTrainer``), LoRA, the conversational
format (``prompt`` + ``completion`` columns, so that the loss is computed only on the
assistant turn). What you decide: the data, the base model, epochs, learning rate,
sequence length. Write down in EXPERIMENTS.md why you chose them.

Our choices (see EXPERIMENTS.md):

* All verified traces, up to 4 per problem (option A). Capping at 2 per problem is a
  planned ablation.
* A held-out validation split *by problem*, not by trace: the 4 traces of a problem are
  near-duplicates, so splitting them across train and eval would make ``eval_loss``
  look better than it is. The split hashes the question text, so it is deterministic and
  does not depend on the order of the file.
* ``max_length=4096``: measured with the Qwen3 chat template, prompt + trace is 1,983
  tokens at the median and 4,021 at most, so no trace is truncated (the answer sits at
  the end, a truncated trace would teach the model to never answer).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset

from rlm.data import build_prompt


def load_verified_traces(path: str | Path):
    """Load the distilled JSONL and keep only the traces that passed the verifier."""
    dataset = load_dataset("json", data_files=str(path), split="train")
    if "verified" in dataset.column_names:
        dataset = dataset.filter(lambda ex: bool(ex["verified"]))
    return dataset


def to_conversational(dataset):
    """Turn traces into TRL's prompt/completion conversational format."""
    return dataset.map(
        lambda ex: {
            "prompt": build_prompt(ex["question"]),
            "completion": [{"role": "assistant", "content": ex["trace"]}],
        },
        remove_columns=dataset.column_names,
    )


def load_sft_dataset(path: str | Path):
    """Verified traces in conversational format (kept for code that imports it)."""
    return to_conversational(load_verified_traces(path))


def _in_eval(question: str, eval_fraction: float) -> bool:
    """Deterministic per-problem assignment: same question, same side, every run."""
    digest = int(hashlib.sha256(question.encode("utf-8")).hexdigest(), 16)
    return (digest % 10_000) < eval_fraction * 10_000


def split_by_problem(dataset, eval_fraction: float):
    """Split traces into train / eval so that no problem appears on both sides."""
    if eval_fraction <= 0:
        return dataset, None
    train = dataset.filter(lambda ex: not _in_eval(ex["question"], eval_fraction))
    evaluation = dataset.filter(lambda ex: _in_eval(ex["question"], eval_fraction))
    return train, evaluation


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    traces = load_verified_traces(args.data)
    train_raw, eval_raw = split_by_problem(traces, args.eval_fraction)
    for name, part in (("train", train_raw), ("eval", eval_raw)):
        if part is not None:
            n_problems = len(set(part["question"]))
            print(f"{name}: {len(part)} verified traces from {n_problems} problems")

    train_dataset = to_conversational(train_raw)
    eval_dataset = to_conversational(eval_raw) if eval_raw is not None else None

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    config = SFTConfig(
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        model_init_kwargs={"dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32},
        # Validation on held-out problems, to catch over-fitting (eval_loss going up while
        # the training loss keeps going down). eval_on_start gives the "before" point.
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps,
        eval_on_start=eval_dataset is not None,
        per_device_eval_batch_size=1,
        # No packing: sequences reach 4,021 tokens, so there is little room to pack, and it
        # would mix the end of one trace with the start of another inside a sample.
        # No assistant_only_loss: the prompt/completion format already restricts the loss
        # to the completion.
    )
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output)

    # Loss history for the curves in EXPERIMENTS.md (train loss and eval_loss per step).
    history = Path(args.output) / "log_history.json"
    history.write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    print(f"adapter saved to {args.output}; loss history in {history}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data", required=True, help="JSONL with question / trace / answer / verified"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output", default="rlm/weights/sft_lora")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--eval-fraction", type=float, default=0.1,
        help="fraction of problems held out for eval_loss (0 disables validation)",
    )
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--resume-from-checkpoint", default=None, help="checkpoint-XXX folder")
    train(parser.parse_args())


if __name__ == "__main__":
    main()