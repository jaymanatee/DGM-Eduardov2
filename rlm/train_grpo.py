"""Phase 1, step 3: reinforcement learning with verifiable rewards using GRPO.

Starting point: the SFT adapter from ``train_sft.py`` (or the base model, if you want to
reproduce the R1-Zero experiment and see what happens without cold start). Output: your
final reasoning model, ``rlm/weights/final_rlm_lora``.

Run::

    uv run --extra train python -m rlm.train_grpo --data rlm/data/train.jsonl \
        --init-adapter rlm/weights/sft_lora

The smoke test (``smoke/smoke_grpo.py``) is the minimal version of this script on GSM8K.
Here you add what makes it *yours*:

1. Your domain dataset (``rlm/data.py::load_domain_dataset``) and your verifier.
2. A third, domain-specific reward (``domain_reward``). Think about what a good
   answer looks like for your user beyond being correct: language, units, length,
   citing a source, respecting a schema. Justify it in EXPERIMENTS.md and, if you use
   ``reward_weights``, justify those too.
3. The hyper-parameters. Group size, completion length, learning rate and KL
   coefficient all change what the model learns. Change one thing at a time and log it.

Our choices (see EXPERIMENTS.md):

* Three rewards, all in ``rlm/rewards.py``: ``format_reward``, ``origin_accuracy_reward``
  (``OriginVerifier``, binary) and ``domain_reward`` (JSON schema and coherence with the
  rule). Weights 0.2 / 1.0 / 0.3: accuracy must dominate, or a well-formed wrong answer
  would pay almost as much as a right one.
* No length penalty for now. GRPO normalises rewards within each group, so in a group
  where every answer gets the same accuracy, a small length term would decide the whole
  advantage. Cut answers already score 0 on all three rewards.
* ``--max-completion-length 3072``: the teacher's verified traces reach 3,072 tokens, and an
  answer cut before ``<answer>`` never gets any reward.
* The same chat template as ``distill.py`` and ``evaluate.py`` (``enable_thinking=True``).

Training in several sessions (the DGX's 16 GB sessions last 8 hours at most): run the
same command every session with ``--resume-from-checkpoint last --max-hours 7.5``. The
first session starts from scratch; each one stops cleanly before its time runs out, after
saving a checkpoint, and the next one continues from it with the same learning-rate
schedule, as if the run had never stopped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rlm import disable_cudnn_attention
from rlm.data import load_domain_dataset, load_gsm8k
from rlm.rewards import accuracy_reward, domain_reward, format_reward, origin_accuracy_reward


def resolve_resume(value: str | None, output: str) -> str | None:
    """``last`` -> newest ``checkpoint-*`` in ``output``, or None to start from scratch."""
    if value != "last":
        return value
    from transformers.trainer_utils import get_last_checkpoint

    last = get_last_checkpoint(output) if Path(output).is_dir() else None
    print(f"resuming from {last}" if last else f"no checkpoint in {output}: starting from scratch")
    return last


def time_limit_callback(max_hours: float):
    """Stop after the step that crosses ``max_hours``, saving a checkpoint first.

    The session would otherwise kill the run mid-step, losing every step since the last
    checkpoint (or corrupting the checkpoint being written).
    """
    from transformers import TrainerCallback

    class TimeLimit(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self.start = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            if time.monotonic() - self.start >= max_hours * 3600:
                print(f"time limit of {max_hours} h reached at step {state.global_step}")
                control.should_save = True
                control.should_training_stop = True
            return control

    return TimeLimit()


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    disable_cudnn_attention()  # 19x faster generation on the DGX

    if args.data == "gsm8k":
        dataset = load_gsm8k("train", n_examples=args.n_examples, seed=args.seed)
        reward_funcs, reward_weights = [format_reward, accuracy_reward], None
    else:
        dataset = load_domain_dataset(args.data)
        if args.n_examples is not None:
            dataset = dataset.shuffle(seed=args.seed).select(range(args.n_examples))
        reward_funcs = [format_reward, origin_accuracy_reward, domain_reward]
        reward_weights = args.reward_weights
    print(f"{len(dataset)} training problems")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GRPOConfig(
        output_dir=args.output,
        max_steps=args.steps,
        learning_rate=args.learning_rate,
        # Each step generates prompts_per_step groups of num_generations answers at once, then
        # trains on them batch_size at a time. With 8 x 3,072 tokens in one batch the fp32
        # logits alone take 14 GiB; generating more at once is almost free (EXPERIMENTS.md).
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.prompts_per_step * args.num_generations // args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        epsilon=0.2,
        bf16=device == "cuda",
        gradient_checkpointing=device == "cuda",
        logging_steps=1,
        save_steps=args.save_steps,
        save_strategy="steps",
        report_to="none",
        seed=args.seed,
        log_completions=True,
        num_completions_to_print=2,
        model_init_kwargs={"dtype": torch.bfloat16 if device == "cuda" else torch.float32},
        reward_weights=reward_weights,
        # Same prompt as distill.py and evaluate.py: without it, Qwen3 skips the reasoning.
        chat_template_kwargs={"enable_thinking": True},
    )

    if args.init_adapter and args.init_adapter != "none":
        # Continue training the SFT adapter: load base + adapter as a trainable PeftModel.
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
        peft_config = None
    else:
        model = args.model
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=2 * args.lora_rank,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
        callbacks=[time_limit_callback(args.max_hours)] if args.max_hours else None,
    )
    trainer.train(resume_from_checkpoint=resolve_resume(args.resume_from_checkpoint, args.output))
    if trainer.state.global_step < args.steps:
        # Stopped by --max-hours: the checkpoint is the result, not a final adapter.
        print(f"stopped at step {trainer.state.global_step} of {args.steps}; run the same "
              "command again with --resume-from-checkpoint last to continue")
        return
    trainer.save_model(args.output)

    # Reward, length and loss per step, for the curves in EXPERIMENTS.md.
    history = Path(args.output) / "log_history.json"
    history.write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    print(f"final adapter saved to {args.output}; history in {history}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data", default="rlm/data/train.jsonl", help="'gsm8k' or path to your domain JSONL"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--init-adapter", default="rlm/weights/sft_lora",
        help="SFT adapter to start from; 'none' for the base model (R1-Zero)",
    )
    parser.add_argument("--output", default="rlm/weights/final_rlm_lora")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num-generations", type=int, default=8, help="answers per problem")
    parser.add_argument("--prompts-per-step", type=int, default=1, help="problems per step")
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="answers per forward/backward pass; must divide prompts-per-step x num-generations",
    )
    parser.add_argument("--max-completion-length", type=int, default=3072)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0, help="KL coefficient (0 disables it)")
    parser.add_argument(
        "--reward-weights", type=float, nargs=3, default=[0.2, 1.0, 0.3],
        metavar=("FORMAT", "ACCURACY", "DOMAIN"),
        help="weights of the three domain rewards (ignored with gsm8k)",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--n-examples", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="checkpoint-XXX folder to continue an interrupted run, or 'last' for the newest "
        "one in --output (starts from scratch if there is none)",
    )
    parser.add_argument(
        "--max-hours", type=float, default=None,
        help="save a checkpoint and stop after this many hours of training (8 h sessions)",
    )
    parser.add_argument("--seed", type=int, default=0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
