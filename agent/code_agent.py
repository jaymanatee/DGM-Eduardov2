"""The contrast the README asks for: smolagents' ``CodeAgent`` on the same tasks and tools.

Our agent writes one JSON tool call per step; CodeAct (Wang et al., 2024) says that writing
Python instead is better: several calls in one step, variables between them, loops, fewer
steps. Here the same Qwen3-1.7B (same brain options as ``ReActAgent``) gets the same five
tools plus ``search_knowledge_base`` and the same domain rules, but as a smolagents
``CodeAgent``. What it does NOT get is our loop's code checks (repeated calls, invented REX
numbers, the bill-of-materials check, the action report): it is smolagents as it comes, so the
comparison measures the JSON loop with its checks against code actions without them.

Tool calls are recorded as they run, and written as steps in the format of
``agent/evaluate.py``, so the same checks and summaries apply::

    uv run --extra train --extra rag --extra agent python -m agent.code_agent \\
        --brains base --out reports/phase4_codeagent.json

``--brains`` takes base, thinking or rlm (rlm with ARCA_AGENT_ADAPTER, as in the ReAct agent).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.evaluate import bench_arguments, load_tasks, run_bench
from agent.prompts import AGENT_DOMAIN_CONTEXT
from agent.react_agent import DEFAULT_ADAPTER, DEFAULT_MODEL, search_knowledge_base
from tool_use.registry import ToolRegistry
from tool_use.tools import default_registry

JSON_TYPES = {"string", "boolean", "integer", "number", "array", "object"}


def split_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """The description before 'Args:', and each argument's text from the Google-style 'Args:'
    section (what the JSON agent reads inside the description, smolagents wants per input)."""
    head, _, args = doc.partition("\nArgs:")
    docs: dict[str, str] = {}
    name = None
    for line in args.splitlines():
        match = re.match(r"^\s{2,8}(\w+):\s*(.*)$", line)
        if match and (name is None or len(line) - len(line.lstrip()) <= 4):
            name, text = match.groups()
            docs[name] = text
        elif name and line.strip():
            docs[name] += " " + line.strip()
    return head.strip(), docs


def _input_spec(prop: dict[str, Any], defs: dict[str, Any], doc: str = "") -> tuple[str, str]:
    """smolagents type and a description for one property of a pydantic JSON schema."""
    options = prop.get("anyOf", [prop])
    kinds = [o.get("type", "object" if "$ref" in o else "any") for o in options]
    kind = next((k for k in kinds if k != "null"), "any")
    parts = [doc or prop.get("description", "")]
    for option in options:
        if "enum" in option:
            parts.append("One of: " + ", ".join(map(str, option["enum"])) + ".")
        items = option.get("items", {})
        if "$ref" in items:  # a list of BomLine: say which keys each object has
            fields = defs[items["$ref"].split("/")[-1]]["properties"]
            parts.append(
                "A list of objects with keys "
                + "; ".join(
                    f"{name}" + (f" ({'|'.join(f['enum'])})" if "enum" in f else "")
                    for name, f in fields.items()
                )
                + "."
            )
    return (kind if kind in JSON_TYPES else "any"), " ".join(p for p in parts if p)


def smol_tools(registry: ToolRegistry, calls: list[dict], agent_ref: list) -> list:
    """Our registry as smolagents tools. Each call goes through ``registry.call`` (same
    validation as the JSON agent), is recorded in ``calls`` and raises on error, which the
    CodeAgent shows to the model as the observation."""
    from smolagents import Tool

    tools = []
    for name, ours in registry.tools.items():
        schema = ours.args_model.model_json_schema()
        defs = schema.get("$defs", {})
        required = set(schema.get("required", []))
        description, arg_docs = split_docstring(ours.description)
        inputs = {}
        for arg, prop in schema.get("properties", {}).items():
            kind, text = _input_spec(prop, defs, arg_docs.get(arg, ""))
            inputs[arg] = {"type": kind, "description": text or arg}
            if arg not in required:
                inputs[arg]["nullable"] = True

        def forward(self, **kwargs):
            arguments = {k: v for k, v in kwargs.items() if v is not None}
            record = registry.call(self.name, arguments)
            agent = agent_ref[0] if agent_ref else None
            calls.append(
                {
                    "name": self.name,
                    "arguments": arguments,
                    "ok": record.ok,
                    "result": record.result if record.ok else None,
                    "error": record.error,
                    "step": getattr(agent, "step_number", None),
                }
            )
            if not record.ok:
                raise RuntimeError(record.error)
            return record.result

        cls = type(
            f"Smol_{name}",
            (Tool,),
            {
                "name": name,
                "description": description,
                "inputs": inputs,
                "output_type": "any",
                "skip_forward_signature_validation": True,
                "forward": forward,
            },
        )
        tools.append(cls())
    return tools


class CodeAgentRunner:
    """Same interface as ``ReActAgent`` for ``run_bench``: ``run``, ``label``, ``tokens_used``.
    One model in memory; the adapter is switched off for base and thinking, as in ReActAgent."""

    def __init__(self, model_id: str, adapter_path: str | None, rlm_thinking: bool):
        from smolagents import TransformersModel

        from rlm import disable_cudnn_attention

        disable_cudnn_attention()
        self.model_id, self.adapter_path, self.rlm_thinking = model_id, adapter_path, rlm_thinking
        self.model = TransformersModel(
            model_id=model_id, device_map="cuda", torch_dtype="bfloat16", max_new_tokens=1024
        )
        if adapter_path:
            from peft import PeftModel

            self.model.model = PeftModel.from_pretrained(self.model.model, adapter_path)
        self.registry = default_registry().register(search_knowledge_base)
        self.tokens_used = 0

    def label(self, brain: str) -> str:
        adapter = f"+{self.adapter_path}" if brain == "rlm" else ""
        return f"smolagents CodeAgent, {self.model_id}{adapter} ({brain})"

    def run(self, task: str, max_steps: int = 10, brain: str = "base"):
        from contextlib import nullcontext

        from smolagents import CodeAgent

        if brain not in ("base", "thinking", "rlm"):
            raise ValueError(f"unknown brain {brain!r}")
        if brain == "rlm" and not self.adapter_path:
            raise ValueError("brain 'rlm' needs ARCA_AGENT_ADAPTER")
        thinking = brain == "thinking" or (brain == "rlm" and self.rlm_thinking)
        self.model.apply_chat_template_kwargs = {"enable_thinking": thinking}
        if isinstance(getattr(self.model, "kwargs", None), dict):  # read by generate()
            self.model.kwargs["max_new_tokens"] = 3072 if thinking else 1024
        calls: list[dict] = []
        agent_ref: list = []
        agent = CodeAgent(
            tools=smol_tools(self.registry, calls, agent_ref),
            model=self.model,
            max_steps=max_steps,
            additional_authorized_imports=["json"],
            instructions=AGENT_DOMAIN_CONTEXT.format(today=date.today().isoformat()),
            verbosity_level=0,
        )
        agent_ref.append(agent)
        adapter_off = self.adapter_path and brain != "rlm"
        with self.model.model.disable_adapter() if adapter_off else nullcontext():
            result = agent.run(task, return_full_result=True)

        usage = result.token_usage
        self.tokens_used = (
            getattr(usage, "total_tokens", None) or usage.input_tokens + usage.output_tokens
            if usage
            else 0
        )
        code_trace = [
            {
                "step": getattr(s, "step_number", None),
                "code": getattr(s, "code_action", None),
                "observations": (getattr(s, "observations", None) or "")[:2000],
                "error": str(s.error) if getattr(s, "error", None) else None,
            }
            for s in agent.memory.steps
            if hasattr(s, "code_action")
        ]
        steps = [
            {
                "thought": f"[code step {c['step']}]",
                "action": {"name": c["name"], "arguments": c["arguments"]},
                "observation": json.dumps(
                    {"name": c["name"], "result": c["result"]}
                    if c["ok"]
                    else {"name": c["name"], "error": c["error"]},
                    ensure_ascii=False,
                    default=str,
                ),
            }
            for c in calls
        ]
        response = {
            "final_answer": "" if result.output is None else str(result.output),
            "steps": steps,
            "n_steps": len(code_trace),
            "succeeded": result.state == "success",
            "model": self.label(brain),
            "tokens_used": self.tokens_used,
            "code_trace": code_trace,
        }
        return SimpleNamespace(model_dump=lambda: response)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    bench_arguments(parser)
    parser.add_argument("--brains", nargs="+", default=["base"])
    parser.add_argument("--out", default="reports/phase4_codeagent.json")
    args = parser.parse_args()
    try:
        tasks = load_tasks(args.bench, args.ids)
    except ValueError as error:
        parser.error(str(error))

    adapter = os.environ.get("ARCA_AGENT_ADAPTER", DEFAULT_ADAPTER)
    runner = CodeAgentRunner(
        model_id=os.environ.get("ARCA_AGENT_MODEL") or DEFAULT_MODEL,
        adapter_path=None if adapter in ("", "none") else adapter,
        rlm_thinking=os.environ.get("ARCA_AGENT_RLM_THINKING", "1") == "1",
    )
    # The brains are stored as code_<brain>, so the file never mixes with the ReAct results.
    run_bench(
        _Prefixed(runner),
        tasks,
        [f"code_{b}" for b in args.brains],
        Path(args.out),
        args.max_steps,
        args.resume,
    )


class _Prefixed:
    """``run_bench`` names the folders and rows by brain: 'code_base' runs brain 'base'."""

    def __init__(self, runner: CodeAgentRunner):
        self.runner = runner

    @property
    def tokens_used(self) -> int:
        return self.runner.tokens_used

    def label(self, brain: str) -> str:
        return self.runner.label(brain.removeprefix("code_"))

    def run(self, task: str, max_steps: int, brain: str):
        return self.runner.run(task, max_steps=max_steps, brain=brain.removeprefix("code_"))


if __name__ == "__main__":
    main()