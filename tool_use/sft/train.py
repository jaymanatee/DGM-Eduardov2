"""SFT for tool use: the phase 1 model, trained further on the synthetic tool trajectories.

The starting point is the final phase 1 adapter (``rlm/weights/grpo_pps3``, LoRA on
Qwen3-1.7B), as the phase 2 README asks: the same LoRA keeps training, so the result is one
adapter that the tool loop loads with ``ARCA_TOOLS_ADAPTER``. ``--init-adapter none`` trains a
fresh LoRA on the base model instead, for comparison.

Data: the examples of ``render.py`` (``outputs/tool_sft/sft_<variant>_<split>.jsonl``), one per
assistant turn, prompt and completion; the loss covers only the completion. Serve the
``think`` adapter with ``ARCA_TOOLS_THINKING=1`` and the ``nothink`` one with ``0``: each was
trained with the chat template rendered that way.

Run (DGX, ~2.5 h per variant and epoch on the 16 GB partition)::

    uv run --extra train python -m tool_use.sft.train --variant think \\
        --output tool_use/weights/sft_think
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_INIT = "rlm/weights/grpo_pps3"


def load_split(data: Path, variant: str, split: str, limit: int | None = None):
    from datasets import load_dataset

    dataset = load_dataset(
        "json", data_files=str(data / f"sft_{variant}_{split}.jsonl"), split="train"
    )
    dataset = dataset.select_columns(["prompt", "completion"])
    if limit is not None and limit < len(dataset):
        dataset = dataset.shuffle(seed=0).select(range(limit))
    return dataset


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    data = Path(args.data)
    train_dataset = load_split(data, args.variant, "train")
    eval_dataset = load_split(data, args.variant, "val", args.eval_limit)
    print(f"train: {len(train_dataset)} examples; eval: {len(eval_dataset)}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if args.init_adapter != "none":
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
        peft_config = None
    else:
        model = args.model
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,  # the longest example is 4,446 tokens: nothing is cut
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        eval_on_start=True,
        per_device_eval_batch_size=1,
        save_strategy="steps",
        save_steps=args.save_steps,  # DGX sessions end: resume with --resume-from-checkpoint
        save_total_limit=2,
        report_to="none",
        model_init_kwargs={"dtype": dtype} if peft_config is not None else None,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output)
    history = Path(args.output) / "log_history.json"
    history.write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    print(f"adapter saved to {args.output}; loss history in {history}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variant", choices=("think", "nothink"), required=True)
    parser.add_argument("--data", default="outputs/tool_sft")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--init-adapter", default=DEFAULT_INIT, help="'none' for a fresh LoRA")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="stop early (smoke test)")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=4608)
    parser.add_argument("--eval-steps", type=int, default=40)
    parser.add_argument("--eval-limit", type=int, default=150, help="validation examples")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--resume-from-checkpoint", default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
