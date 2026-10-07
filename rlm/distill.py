"""Phase 1, step 1b: generate reasoning traces with a teacher model and keep the verified ones.

This is what Sky-T1, OpenThoughts and DeepSeek's cold start have in common: a strong
model writes solutions with visible reasoning, a verifier throws away the wrong ones,
and what survives becomes SFT data. Here the teacher is any model that can think in the
``<think>…</think><answer>…</answer>`` format (Qwen3 in thinking mode works well; a
DeepSeek-R1 distilled model too).

Run::

    uv run python -m rlm.distill --data rlm/data/train.jsonl --teacher Qwen/Qwen3-4B \
        --samples 4 --output rlm/data/sft_traces.jsonl

Output: one JSON line per generated trace with ``question``, ``answer``, ``trace``,
``verified`` and ``teacher``. Report in EXPERIMENTS.md the acceptance rate: it is your
first measurement of how hard your domain is.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from rlm.data import build_prompt, load_domain_dataset
from rlm.verifier import NumericVerifier, OriginVerifier, Verifier

VERIFIERS = {"origin": OriginVerifier, "numeric": NumericVerifier}

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{(.*)\}", re.DOTALL)
JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def canonicalize(text: str) -> tuple[str | None, str]:
    """Rewrite a raw teacher completion as ``<think>…</think><answer>…</answer>``.

    Returns ``(trace, status)``. ``trace`` is None when the completion cannot be repaired
    (truncated before ``</think>``, empty reasoning, or no recoverable answer).
    """
    match = THINK_RE.search(text)
    if match is not None:
        reasoning, rest = match.group(1), text[match.end():]
    elif "</think>" in text:
        reasoning, rest = text.split("</think>", 1)
        reasoning = reasoning.replace("<think>", "")
    else:
        return None, "truncated_or_no_think"

    reasoning = reasoning.strip()
    if not reasoning:
        return None, "empty_think"

    answer = None
    if (m := ANSWER_RE.search(rest)) is not None:
        answer = m.group(1)
    elif (m := BOXED_RE.search(rest)) is not None:
        answer = m.group(1)
    elif matches := JSON_RE.findall(rest):
        answer = matches[-1]
    if answer is None or not answer.strip():
        return None, "no_answer"

    return f"<think>\n{reasoning}\n</think>\n<answer>\n{answer.strip()}\n</answer>", "ok"


def _clean(text: str, tokenizer) -> str:
    """Remove end-of-sequence and padding tokens left by ``skip_special_tokens=False``."""
    for token in {tokenizer.eos_token, tokenizer.pad_token, "<|im_end|>", "<|endoftext|>"}:
        if token:
            text = text.replace(token, "")
    return text.strip()


def _as_expected(answer) -> str:
    """The verifiers expect a string; the JSONL may store the origin answer as an object."""
    return answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)


def _reason(detail: str) -> str:
    """Collapse the verifier's free-text detail into a small set of rejection categories."""
    prefixes = {
        "no <answer>": "no_answer",
        "la respuesta no es": "bad_json",
        "falta": "missing_field",
        "veredicto": "wrong_verdict",
        "criterio": "wrong_criterion",
        "porcentaje": "wrong_pct",
    }
    for prefix, category in prefixes.items():
        if detail.startswith(prefix):
            return category
    return "wrong_answer"


def _load_checkpoint(path: Path | None) -> list[dict]:
    """Read rows already generated in a previous (interrupted) run."""
    if path is None or not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break  # last line cut by a crash
    return rows


def generate_traces(
    dataset,
    teacher: str,
    samples: int,
    max_new_tokens: int,
    verifier: Verifier,
    batch_size: int = 2,
    checkpoint: Path | None = None,
) -> list[dict]:
    """Sample ``samples`` completions per problem from the teacher and verify each one.

    Every batch is appended to ``checkpoint`` so a killed DGX session can resume where it
    stopped. Rejected traces are kept too (``verified: false``) with a ``status`` that says
    why, which is what the acceptance analysis in EXPERIMENTS.md is built from.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    problems = [dataset[i] for i in range(len(dataset))]
    rows = _load_checkpoint(checkpoint)
    start_idx = max((r["problem_idx"] for r in rows), default=-1) + 1
    if start_idx:
        print(f"resuming from problem {start_idx} ({len(rows)} traces already in checkpoint)")

    tokenizer = AutoTokenizer.from_pretrained(teacher)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        teacher, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    for start in range(start_idx, len(problems), batch_size):
        batch = problems[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_prompt(p["question"]),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,  # Qwen3 thinking mode; ignored by other templates
            )
            for p in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                num_return_sequences=samples,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = outputs[:, inputs["input_ids"].shape[1] :]

        batch_rows = []
        for k, seq in enumerate(new_tokens):
            problem = batch[k // samples]  # generate groups the samples of each prompt
            raw = _clean(tokenizer.decode(seq, skip_special_tokens=False), tokenizer)
            n_tokens = int((seq != tokenizer.pad_token_id).sum())
            trace, status = canonicalize(raw)

            verified, detail = False, ""
            if trace is not None:
                try:
                    result = verifier.verify(trace, _as_expected(problem["answer"]))
                    verified, detail = result.is_correct, result.detail
                    if not verified:
                        status = _reason(detail)
                except Exception as exc:  # a crashing verifier must not kill a long run
                    status, detail = "verifier_error", f"{type(exc).__name__}: {exc}"

            batch_rows.append(
                {
                    "problem_idx": start + k // samples,
                    "question": problem["question"],
                    "answer": problem["answer"],
                    "trace": trace if trace is not None else raw,
                    "verified": verified,
                    "teacher": teacher,
                    "status": status,
                    "detail": detail,
                    "n_tokens": n_tokens,
                    "truncated": n_tokens >= max_new_tokens,
                }
            )

        rows.extend(batch_rows)
        if checkpoint is not None:
            with checkpoint.open("a", encoding="utf-8") as handle:
                for row in batch_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        done = min(start + batch_size, len(problems))
        kept = sum(r["verified"] for r in rows)
        print(f"[{done}/{len(problems)}] {kept}/{len(rows)} verified so far", flush=True)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, help="domain JSONL with question / answer")
    parser.add_argument("--teacher", default="Qwen/Qwen3-4B")
    parser.add_argument("--samples", type=int, default=4, help="traces per problem")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2, help="problems per generate call")
    parser.add_argument(
        "--verifier", choices=sorted(VERIFIERS), default="origin",
        help="origin for your domain, numeric for the GSM8K control set",
    )
    parser.add_argument("--output", default="rlm/data/sft_traces.jsonl")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = out.with_name(out.stem + ".partial.jsonl")

    dataset = load_domain_dataset(args.data)
    verifier = VERIFIERS[args.verifier]()
    traces = generate_traces(
        dataset,
        args.teacher,
        args.samples,
        args.max_new_tokens,
        verifier,
        batch_size=args.batch_size,
        checkpoint=checkpoint,
    )

    with out.open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    checkpoint.unlink(missing_ok=True)

    kept = sum(1 for t in traces if t["verified"])
    n_problems = len({t["problem_idx"] for t in traces})
    solved = len({t["problem_idx"] for t in traces if t["verified"]})
    print(
        f"{kept}/{len(traces)} traces verified ({100 * kept / max(len(traces), 1):.1f}%) -> {out}"
    )
    print(
        f"{solved}/{n_problems} problems with at least one verified trace "
        f"({100 * solved / max(n_problems, 1):.1f}%)"
    )
    print("status breakdown:", dict(Counter(t["status"] for t in traces).most_common()))


if __name__ == "__main__":
    main()