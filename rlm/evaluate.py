"""Phase 1 evaluation: pass@1 of base vs SFT vs GRPO on a held-out set, plus training curves.

Run::

    uv run --extra train python -m rlm.evaluate --data rlm/data/test.jsonl \
        --adapters base=none sft=rlm/weights/sft_lora grpo=rlm/weights/final_rlm_lora

It writes ``reports/phase1_eval_<data>.json`` with per-example verdicts (so you can do the
failure analysis) and ``reports/phase1_pass1_<data>.png`` with the bar chart. Results are
*merged* into the JSON: you can evaluate ``base`` today and ``sft`` tomorrow and the file
keeps both, so the comparison table always shows every model evaluated on that data.

``--history rlm/weights/sft_lora/log_history.json`` plots the training curves (loss and
eval_loss for SFT; reward and completion length for GRPO) instead of evaluating.

Design choices (see EXPERIMENTS.md):

* Every model is evaluated with exactly the same prompt as training: ``build_prompt`` +
  the Qwen3 chat template with ``enable_thinking=True``. With ``False`` the template
  inserts an empty ``<think></think>`` and the model would skip its reasoning.
* Greedy decoding by default, so pass@1 is reproducible. ``truncated`` in the summary
  shows whether the model loops until ``--max-new-tokens`` (Qwen recommends sampling in
  thinking mode for that reason; ``--temperature 0.6`` is the alternative).
* Two metrics: ``pass@1`` (strict: the answer must be inside ``<answer>``, as
  ``/reasoning`` requires) and ``pass@1_lenient`` (also reads the last JSON after
  ``</think>``). The gap between them separates format errors from reasoning errors.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from rlm import disable_cudnn_attention
from rlm.data import load_domain_dataset, load_gsm8k
from rlm.verifier import NumericVerifier, OriginVerifier, Verifier

VERIFIERS = {"origin": OriginVerifier, "numeric": NumericVerifier}
FORMAT_RE = re.compile(r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL)
JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


# ----------------------------------------------------------------------------- prompt


def render_prompt(tokenizer, messages: list[dict]) -> str:
    """The single place where chat messages become the text the model sees.

    Same settings as ``distill.py`` and the SFT data: generation prompt on, thinking on.
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )


# ----------------------------------------------------------------------------- scoring


def has_valid_format(text: str) -> bool:
    """``<think>…</think>`` followed by ``<answer>…</answer>`` and nothing else."""
    return bool(FORMAT_RE.match(text))


def lenient_completion(text: str) -> str:
    """If the model reasons but forgets ``<answer>``, wrap the last JSON after ``</think>``.

    Only for the lenient metric: pass@1 always uses the raw completion.
    """
    if "<answer>" in text:
        return text
    found = JSON_RE.findall(text.split("</think>")[-1])
    return f"<answer>{found[-1]}</answer>" if found else text


def _clean(text: str, tokenizer) -> str:
    """Drop end-of-turn and padding tokens kept by ``skip_special_tokens=False``."""
    for token in {tokenizer.eos_token, tokenizer.pad_token, "<|im_end|>", "<|endoftext|>"}:
        if token:
            text = text.replace(token, "")
    return text.strip()


# ----------------------------------------------------------------------------- generation


def load_model(base_model: str, adapter: str | None):
    """Base model in bf16, with the LoRA adapter merged on top when one is given."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    disable_cudnn_attention()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="auto"
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"model on {model.device} ({torch.cuda.get_device_name(0)}, "
              f"{free / 2**30:.1f} of {total / 2**30:.1f} GB free)", flush=True)
    else:
        print("WARNING: no GPU visible, generating on CPU (hours, not minutes)", flush=True)
    return model, tokenizer


def evaluate_model(
    base_model: str,
    adapter: str | None,
    dataset,
    verifier: Verifier,
    max_new_tokens: int,
    batch_size: int = 8,
    temperature: float = 0.0,
) -> list[dict]:
    """One generation per example, one verdict each.

    Returns one dict per example with ``question``, ``expected``, ``raw``, ``predicted``,
    ``is_correct``, ``is_correct_lenient``, ``has_valid_format``, ``n_tokens``,
    ``truncated``, ``detail`` and ``branches`` (when the dataset has them).
    """
    import torch

    model, tokenizer = load_model(base_model, adapter)
    gen_kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": tokenizer.pad_token_id}
    if temperature > 0:
        torch.manual_seed(0)
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95, top_k=20)
    else:  # override the sampling defaults in Qwen3's generation_config
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)

    from transformers import StoppingCriteria, StoppingCriteriaList

    class Progress(StoppingCriteria):
        """Never stops generation: prints how far the batch is and the time per token."""

        def __init__(self, every: int = 256):
            self.every, self.steps, self.start = every, 0, time.time()

        def __call__(self, input_ids, scores, **kwargs):
            self.steps += 1
            if self.steps % self.every == 0:
                ms = 1000 * (time.time() - self.start) / self.steps
                print(f"    ... {self.steps} tokens ({ms:.0f} ms/token)", flush=True)
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    examples = [dataset[i] for i in range(len(dataset))]
    rows: list[dict] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        texts = [render_prompt(tokenizer, ex["prompt"]) for ex in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs, **gen_kwargs, stopping_criteria=StoppingCriteriaList([Progress()])
            )
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        print(f"    generated {new_tokens.shape[1]} tokens in {time.time() - t0:.0f} s; verifying",
              flush=True)

        for ex, seq in zip(batch, new_tokens):
            n_tokens = int((seq != tokenizer.pad_token_id).sum())
            raw = _clean(tokenizer.decode(seq, skip_special_tokens=False), tokenizer)
            result = verifier.verify(raw, ex["answer"])
            rows.append(
                {
                    "question": ex["prompt"][-1]["content"],
                    "expected": ex["answer"],
                    "raw": raw,
                    "predicted": result.predicted,
                    "is_correct": result.is_correct,
                    "is_correct_lenient": verifier.verify(
                        lenient_completion(raw), ex["answer"]
                    ).is_correct,
                    "has_valid_format": has_valid_format(raw),
                    "n_tokens": n_tokens,
                    "truncated": n_tokens >= max_new_tokens,
                    "detail": result.detail,
                    "branches": ex.get("branches"),
                }
            )
        done = len(rows)
        ok = sum(r["is_correct"] for r in rows)
        print(f"  [{done}/{len(examples)}] {ok} correct so far", flush=True)

    del model
    torch.cuda.empty_cache()
    return rows


# ----------------------------------------------------------------------------- metrics


def pass_at_1(rows: list[dict]) -> float:
    return sum(r["is_correct"] for r in rows) / max(len(rows), 1)


def summarize(rows: list[dict]) -> dict:
    n = max(len(rows), 1)
    return {
        "n": len(rows),
        "pass@1": pass_at_1(rows),
        "pass@1_lenient": sum(r["is_correct_lenient"] for r in rows) / n,
        "valid_format": sum(r["has_valid_format"] for r in rows) / n,
        "truncated": sum(r["truncated"] for r in rows) / n,
        "mean_tokens": sum(r["n_tokens"] for r in rows) / n,
    }


def _groups(row: dict) -> list[str]:
    """Labels a row belongs to: expected verdict plus every scalar field in ``branches``."""
    labels = []
    try:
        labels.append(f"originario={json.loads(row['expected'])['originario']}")
    except (ValueError, TypeError, KeyError):
        pass
    branches = row.get("branches") or {}
    if isinstance(branches, str):
        try:
            branches = json.loads(branches)
        except ValueError:
            branches = {}
    for key, value in branches.items():
        if isinstance(value, bool):  # trap flags: only the problems that have the trap
            if value:
                labels.append(key)
            continue
        for v in value if isinstance(value, list) else [value]:
            if v is None or isinstance(v, (str, int, float)):
                labels.append(f"{key}={v}")
    return list(dict.fromkeys(labels))  # a label counts once per row


def breakdown(rows: list[dict]) -> dict:
    """pass@1 per group (rule, trap, wording, expected verdict...)."""
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        for label in _groups(row):
            groups[label].append(row["is_correct"])
    return {k: {"n": len(v), "pass@1": sum(v) / len(v)} for k, v in sorted(groups.items())}


# ----------------------------------------------------------------------------- plots


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no está instalado (uv add matplotlib): no se dibuja la figura")
        return None
    return plt


def plot_pass1(results: dict, out_png: Path) -> None:
    plt = _pyplot()
    if plt is None:
        return
    names = list(results)
    strict = [results[n]["summary"]["pass@1"] for n in names]
    lenient = [results[n]["summary"]["pass@1_lenient"] for n in names]
    xs = range(len(names))
    fig, ax = plt.subplots(figsize=(1.8 * len(names) + 2, 3.5))
    ax.bar([x - 0.2 for x in xs], strict, width=0.4, label="pass@1")
    ax.bar([x + 0.2 for x in xs], lenient, width=0.4, label="pass@1 lenient", alpha=0.6)
    ax.set_xticks(list(xs), names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("aciertos")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"figure -> {out_png}")


def plot_history(path: str | Path) -> None:
    """Training curves from a ``log_history.json`` (SFT or GRPO)."""
    plt = _pyplot()
    if plt is None:
        return
    history = json.loads(Path(path).read_text(encoding="utf-8"))
    keys = sorted(
        {
            k
            for h in history
            for k, v in h.items()
            if isinstance(v, (int, float))
            and (k in ("loss", "eval_loss") or "reward" in k or "length" in k)
        }
    )
    if not keys:
        print("no hay curvas que dibujar en", path)
        return
    fig, axes = plt.subplots(len(keys), 1, figsize=(7, 2.4 * len(keys)), sharex=True, squeeze=False)
    for ax, key in zip(axes[:, 0], keys):
        points = [(h["step"], h[key]) for h in history if key in h and "step" in h]
        ax.plot(*zip(*points), marker="o", markersize=3)
        ax.set_ylabel(key)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("step")
    fig.tight_layout()
    out_png = Path(path).with_suffix(".png")
    fig.savefig(out_png, dpi=120)
    print(f"curves -> {out_png}")


# ----------------------------------------------------------------------------- main


def main() -> None:
    # `kill -USR1 <pid>` prints where the process is (all threads) without stopping it.
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="rlm/data/test.jsonl", help="'gsm8k' or a JSONL")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--adapters", nargs="+", default=["base=none"],
        help="name=path pairs; use 'none' for the bare base model",
    )
    parser.add_argument("--n-examples", type=int, default=None, help="default: all")
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    parser.add_argument("--verifier", choices=sorted(VERIFIERS), default="origin")
    parser.add_argument("--out", default=None, help="default: reports/phase1_eval_<data>.json")
    parser.add_argument("--history", default=None, help="plot a log_history.json and exit")
    args = parser.parse_args()

    if args.history:
        plot_history(args.history)
        return

    if args.data == "gsm8k":
        dataset = load_gsm8k("test", n_examples=args.n_examples or 200)
        verifier: Verifier = NumericVerifier()
        stem = "gsm8k"
    else:
        dataset = load_domain_dataset(args.data)
        if args.n_examples:
            dataset = dataset.select(range(min(args.n_examples, len(dataset))))
        verifier = VERIFIERS[args.verifier]()
        stem = Path(args.data).stem

    out = Path(args.out or f"reports/phase1_eval_{stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}

    for pair in args.adapters:
        name, path = pair.split("=", 1)
        print(f"== {name} ({path}) on {len(dataset)} problems")
        rows = evaluate_model(
            args.model,
            None if path == "none" else path,
            dataset,
            verifier,
            args.max_new_tokens,
            batch_size=args.batch_size,
            temperature=args.temperature,
        )
        results[name] = {
            "adapter": path,
            "settings": {
                "model": args.model,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
            },
            "summary": summarize(rows),
            "breakdown": breakdown(rows),
            "rows": rows,
        }
        # Save after every model: a cut DGX session keeps the ones already finished.
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'modelo':>8} | pass@1 | lenient | formato | cortadas | tokens")
    for name, res in results.items():
        s = res["summary"]
        print(
            f"{name:>8} | {s['pass@1']:6.3f} | {s['pass@1_lenient']:7.3f} | "
            f"{s['valid_format']:7.3f} | {s['truncated']:8.3f} | {s['mean_tokens']:6.0f}"
        )
    print("\npass@1 por grupo (n >= 5):")
    for name, res in results.items():
        print(f"  {name}")
        for label, g in res["breakdown"].items():
            if g["n"] >= 5:
                print(f"    {label:<40} {g['pass@1']:.3f}  (n={g['n']})")
    print(f"details -> {out}")
    plot_pass1(results, out.with_name(out.name.replace("phase1_eval", "phase1_pass1")).with_suffix(".png"))


if __name__ == "__main__":
    main()