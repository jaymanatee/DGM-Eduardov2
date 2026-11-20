"""The ReAct agent: Thought -> Action -> Observation, in a loop, with every tool you built.

This is where phases 1 to 3 meet. The brain is a model (``brain="base"``, your phase-1
RLM with ``brain="rlm"``, or a thinking model with ``brain="thinking"``); the hands are the
phase-2 registry plus ``search_knowledge_base`` from phase 3; the loop is yours to write.

``ReActAgent.run`` is what ``POST /agent`` calls. Until it is implemented the endpoint
answers 501. Reuse what you already have: ``tool_use.parser.parse_tool_calls`` to read
the model's output, ``ToolRegistry.call`` to execute, ``ConversationMemory`` to keep the
history within budget.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as ToolTimeout

from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import date

from agent.memory import ConversationMemory
from agent.prompts import AGENT_DOMAIN_CONTEXT, NO_CALL_NUDGE, REACT_SYSTEM_PROMPT
from api.schemas import AgentAction, AgentResponse, AgentStep, ToolCallRecord
from tool_use.executor import ToolLoop
from tool_use.parser import ParsedOutput, parse_tool_calls
from tool_use.registry import ToolRegistry, tool
from tool_use.tools import default_registry


_knowledge_base = None


@tool
def final_answer(answer: str) -> str:
    """Return the final answer to the user and stop the loop.

    Args:
        answer: The complete answer to the task.
    """
    return answer

def knowledge_base():
    """The phase-3 retrievers, built once on first use (no generator model: the agent's
    brain reads the passages itself)."""
    global _knowledge_base
    if _knowledge_base is None:
        from rag.generate import RagPipeline

        pipeline = RagPipeline.from_env()
        pipeline.load(generator=False)
        _knowledge_base = pipeline
    return _knowledge_base

@tool
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search the EU rules-of-origin knowledge base (agreement texts and official guides)
    and return the most relevant passages with their ids.

    Use it for questions about how the rules work (tolerance, cumulation, proofs of origin,
    REX, deadlines...), not for the rule of a specific HS code or for an origin verdict:
    those come from get_origin_rule and evaluate_origin.

    Args:
        query: What to look for, phrased as a question. Name the agreement if there is one.
        top_k: How many passages to return (1 to 8).
    """
    from rag.generate import format_chunks

    top_k = max(1, min(int(top_k), 8))
    hits = knowledge_base().retrieve(query, top_k=top_k, retriever="hybrid")
    if not hits:
        return "No passage found."
    return format_chunks(hits)

def recover_untagged_call(
    raw: str, parsed: ParsedOutput, registry: ToolRegistry
    ) -> tuple[str, ParsedOutput]:
    """A call written without the <tool_call> tags, as in "Action: {json}", becomes a call.

    The ReAct prompt says "Action:", and the base and thinking brains copy it and drop the
    tags. Accepted only for a JSON object that names a registered tool and has an
    'arguments' object, so ordinary prose never qualifies. The output is rewritten with the
    tags, so the history shows the model the right format from then on.
    """
    decoder = json.JSONDecoder()
    text = parsed.text
    for match in re.finditer(r"\{", text):
        try:
            payload, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if (
            isinstance(payload, dict)
            and payload.get("name") in registry.tools
            and isinstance(payload.get("arguments"), dict)
        ):
            prose = re.sub(r"Action:\s*$", "", text[: match.start()].strip()).strip()
            think = f"<think>\n{parsed.thinking}\n</think>\n\n" if parsed.thinking else ""
            call = json.dumps(
                {"name": payload["name"], "arguments": payload["arguments"]}, ensure_ascii=False
            )
            fixed = think + (prose + "\n" if prose else "") + f"<tool_call>\n{call}\n</tool_call>"
            return fixed, parse_tool_calls(fixed)
    return raw, parsed


# ---------------------------------------------------------------------------------- brains

# The three brains share one backbone, so that only one thing changes between them:
#
#   base      Qwen3-1.7B as released, answering without reasoning
#   thinking  the same weights, reasoning in <think> before every step
#   rlm       the same weights + our LoRA adapter (phase 1 + tool-use SFT of phase 2)
#
# The adapter is loaded once on top of the backbone and switched off for base and thinking
# (``disable_adapter``), so one model sits in the 16 GB GPU and changing brain is free.

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_ADAPTER = "tool_use/weights/sft_think"

@dataclass(frozen=True)
class Brain:
    adapter: bool   # LoRA adapter active
    thinking: bool   # enable_thinking

def brains_from_env() -> dict[str, Brain]:
    """``ARCA_AGENT_RLM_THINKING=0`` for an adapter trained without thinking (sft_nothink)."""
    return {
        "base": Brain(adapter=False, thinking=False),
        "thinking": Brain(adapter=False, thinking=True),
        "rlm": Brain(adapter=True, thinking=os.environ.get("ARCA_AGENT_RLM_THINKING", "1") == "1"),
    }

@dataclass
class ReActAgent:
    registry: ToolRegistry = field(default_factory=default_registry)
    token_budget: int = 6000
    model_id: str = DEFAULT_MODEL
    adapter_path: str | None = DEFAULT_ADAPTER
    brains: dict[str, Brain] = field(default_factory=brains_from_env)
    max_new_tokens: int | None = None  # per step; default 1024, or 3072 when thinking
    model: object = None
    tokenizer: object = None
    tokens_used: int = 0  # prompt + generated tokens of the current run, for the evaluation
    tool_timeout: float = 45.0  # seconds a tool may take before the agent stops waiting for it

    @classmethod
    def from_env(cls) -> ReActAgent:
        agent = cls(model_id=os.environ.get("ARCA_AGENT_MODEL") or DEFAULT_MODEL,
            # Empty or "none": no adapter at all (then brain="rlm" is refused).
            adapter_path=os.environ.get("ARCA_AGENT_ADAPTER", DEFAULT_ADAPTER),
            max_new_tokens=int(os.environ.get("ARCA_AGENT_MAX_NEW_TOKENS") or 0) or None,
            tool_timeout=float(os.environ.get("ARCA_AGENT_TOOL_TIMEOUT") or 45),
        )
        if agent.adapter_path in ("", "none"):
            agent.adapter_path = None
        agent.registry.register(final_answer, search_knowledge_base)
        knowledge_base() # build rag now since build time should not count as tool use time
        agent.load()
        return agent

    def load(self) -> None:
        """The backbone once, with the adapter on top, unmerged so that it can be switched off."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from rlm import disable_cudnn_attention

        disable_cudnn_attention()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype, device_map=device)
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        self.model = model.eval()

    def system_prompt(self, max_steps: int) -> str:
        """The ReAct loop and the tools (agent/prompts.py), then the domain rules."""
        react = REACT_SYSTEM_PROMPT.format(
            max_steps=max_steps, tools_description=self.registry.render_for_prompt()
        )
        return react + "\n" + AGENT_DOMAIN_CONTEXT.format(today=date.today().isoformat())

    def brain(self, name: str) -> Brain:
        if name not in self.brains:
            raise ValueError(f"unknown brain {name!r}: one of {', '.join(self.brains)}")
        brain = self.brains[name]
        if brain.adapter and not self.adapter_path:
            raise ValueError(f"brain {name!r} needs an adapter: set ARCA_AGENT_ADAPTER")
        return brain

    def generate(self, messages: list[dict], brain: str) -> str:
        """One greedy generation with the chosen brain. The <think> block, if any, is kept:
        ``parse_tool_calls`` separates it from the text and the tool calls."""
        import torch

        if self.model is None:
            raise RuntimeError("call load() first")
        config = self.brain(brain)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config.thinking,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        # base and thinking run the backbone alone: the adapter is switched off, not unloaded.
        adapter_off = self.adapter_path and not config.adapter
        with torch.no_grad(), self.model.disable_adapter() if adapter_off else nullcontext():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens or (3072 if config.thinking else 1024),
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                # One action per step: stop at the end of the first call, so the model can
                # neither write a second one nor invent the observation that follows.
                stop_strings=["</tool_call>"],
                tokenizer=self.tokenizer,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        # <think> and <tool_call> are not special tokens in Qwen3: they survive the decode.
        self.tokens_used += inputs["input_ids"].shape[1] + len(new_tokens)
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text)["input_ids"])

    def label(self, brain: str) -> str:
        config = self.brains[brain]
        adapter = f"+{self.adapter_path}" if config.adapter else ""
        return f"{self.model_id}{adapter} ({brain}, thinking {'on' if config.thinking else 'off'})"

    def call_tool(self, name: str, arguments: dict) -> ToolCallRecord:
        """``registry.call`` with a time limit. A tool cannot be killed from Python: on a
        time-out the agent stops waiting and tells the model, and the call finishes (or not)
        in the background. Its result is then thrown away."""
        pool = ThreadPoolExecutor(max_workers=1)
        started = time.perf_counter()
        future = pool.submit(self.registry.call, name, arguments)
        try:
            return future.result(timeout=self.tool_timeout)
        except ToolTimeout:
            return ToolCallRecord(
                name=name,
                arguments=arguments,
                ok=False,
                error=f"time-out: {name} did not answer in {self.tool_timeout:.0f} s. "
                "You may try once more; if it fails again, do not answer from memory: "
                "call final_answer saying that the information could not be retrieved.",
                latency_ms=1000 * (time.perf_counter() - started),
            )
        finally:
            pool.shutdown(wait=False)

    def run(self, task: str, max_steps: int = 10, brain: str = "rlm") -> AgentResponse:
        """Thought -> Action -> Observation, until final_answer or max_steps."""
        self.brain(brain)
        self.tokens_used = 0
        memory = ConversationMemory(
            self.system_prompt(max_steps),
            self.count_tokens,
            self.token_budget,
        )
        memory.add("user", task)
        steps: list[AgentStep] = []

        def response(answer: str, n_steps: int, succeeded: bool = True) -> AgentResponse:
            return AgentResponse(
                final_answer=answer,
                steps=steps,
                n_steps=n_steps,
                succeeded=succeeded,
                model=self.label(brain),
                tokens_used=self.tokens_used,
            )

        nudged = False  # remainder to call final_answer, sent 0 or 1 times
        done: dict[str, int] = {}  # successful calls: name + arguments -> step that made it
        repeats = 0
        max_repeats = 2  # a third identical call means the loop is stuck

        for step in range(1, max_steps + 1):
            raw = self.generate(memory.render(), brain)
            parsed = parse_tool_calls(raw)
            recovered = False
            if not parsed.tool_calls:
                fixed, parsed = recover_untagged_call(raw, parsed, self.registry)
                recovered, raw = fixed != raw, fixed
            # The reasoning: the <think> block and/or the "Thought: ..." text before the call.
            thought = "\n".join(t for t in (parsed.thinking, parsed.text.strip()) if t)

            if not parsed.tool_calls:
                if not nudged:
                    # First plain answer: remind the loop's rules once and generate again.
                    nudged = True
                    steps.append(
                        AgentStep(thought=thought, observation=f"[no tool called] {NO_CALL_NUDGE}")
                    )
                    memory.add("assistant", raw)
                    memory.add("user", NO_CALL_NUDGE)
                    continue
                steps.append(
                    AgentStep(thought=thought, observation="(answered without final_answer)")
                )
                answer = parsed.text.strip()
                return response(answer, step, succeeded=bool(answer))

            call = parsed.tool_calls[0]
            if call.error is not None:  # malformed <tool_call>: the model hears what was wrong
                record = ToolCallRecord(
                    name=call.name or "<invalid>", arguments=call.arguments, ok=False,
                    error=call.error,
                )
            else:
                key = json.dumps(
                    {"name": call.name, "arguments": call.arguments},
                    sort_keys=True,
                    ensure_ascii=False,
                )
                if key in done:
                    # Same call, same arguments, and it already worked: not run again.
                    repeats += 1
                    record = ToolCallRecord(
                        name=call.name,
                        arguments=call.arguments,
                        ok=False,
                        error=f"repeated call: you already called {call.name} with these "
                        f"arguments in step {done[key]} and have its result above. Use it, "
                        "change the arguments, or call final_answer.",
                    )
                else:
                    record = self.call_tool(call.name, call.arguments)
                    if record.ok:
                        done[key] = step
            observation = ToolLoop.observation(record)  # same format as in the phase-2 training
            note = "[call written without <tool_call> tags, recovered] " if recovered else ""
            steps.append(
                AgentStep(
                    thought=thought,
                    action=AgentAction(name=record.name, arguments=call.arguments),
                    observation=note + observation["content"],
                )
            )
            if repeats >= max_repeats:
                return response("", step, succeeded=False)  # stuck: the trace shows the loop
            if record.ok and record.name == "final_answer":
                return response(str(record.result), step)

            memory.add("assistant", raw)
            memory.add(**observation)  # role "tool", the tool name and the result

        return response("", max_steps, succeeded=False)
