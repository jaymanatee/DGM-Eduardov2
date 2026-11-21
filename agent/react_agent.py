"""The ReAct agent behind POST /agent: Thought -> Action -> Observation, in a loop.

This is where phases 1 to 3 meet. The brain is Qwen3-1.7B in one of three modes (``base``,
``thinking``, or ``rlm`` with our phase-2 adapter, which starts from the phase-1 GRPO); the
hands are the phase-2 tools; the knowledge base of phase 3 is one more tool,
``search_knowledge_base``; ``final_answer`` ends the loop.

The model understands the task, picks the tools, fills in their arguments and writes the
answer. What can be checked by code is checked by code, and each check was added after a
test where the model failed at it: calls written without tags are recovered; a call repeated
with the same arguments is refused; values only the user can give (REX number, invoice,
exporter) must appear in the task; the bill of materials must be the user's; the outcome of
the actions is reported by the code, not by the model; the memory is summarised from the real
calls, not by the model. The code refuses and explains, it never writes the call for the
model. Configuration: the ARCA_AGENT_* variables (see .env.example).
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
from tool_use.executor import CONFIRMATION_POLICIES, ToolLoop
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

# Arguments of an action that the user has to give: a value for them that does not appear
# in the task was made up by the model (it invented REX numbers in three runs in a row).
USER_GIVEN = {"issue_origin_statement": ("exporter_reference", "invoice_number", "exporter_name")}


def _plain(text: str) -> str:
    return re.sub(r"[\s.\-_/]", "", text).upper()


def invented_arguments(name: str, arguments: dict, task: str) -> list[str]:
    """The user-given arguments of an action whose value is not in the task."""
    return [
        argument
        for argument in USER_GIVEN.get(name, ())
        if isinstance(arguments.get(argument), str)
        and arguments[argument].strip()
        and _plain(arguments[argument]) not in _plain(task)
    ]

# HS codes the user wrote (6 digits, "8414.59", "8414 59" or "841459"), not prices: a number
# followed by € or EUR is an amount, even if it looks like a code ("1200.50 €").
_HS_IN_TEXT = re.compile(r"\b(\d{4})[.\s]?(\d{2})\b(?!\s*(?:€|EUR|eur|euros))")


def _hs6(code) -> str:
    return re.sub(r"\D", "", str(code or ""))[:6]

def bom_problems(name: str, arguments: dict, task: str) -> list[str]:
    """..."""  # (el mismo docstring)
    if name != "evaluate_origin":
        return []
    product = _hs6(arguments.get("product_hs_code"))
    exw = arguments.get("exw_price_eur") or 0
    materials = [m for m in arguments.get("materials") or [] if isinstance(m, dict)]
    problems = []
    keep = []
    for m in materials:
        value = m.get("value_eur") or 0
        if product and _hs6(m.get("hs_code")) == product and exw and value >= exw:
            problems.append(
                f"'{m.get('name')}' is the product itself (same HS code {product}, worth the "
                "whole ex-works price), not one of its materials: REMOVE it from materials"
            )
        else:
            keep.append(m.get("name"))
    listed = {_hs6(m.get("hs_code")) for m in materials}
    in_task = {a + b for a, b in _HS_IN_TEXT.findall(task)} - {product}
    missing = sorted(in_task - listed)
    if missing:
        problems.append(
            f"materials the user listed are missing (HS {', '.join(missing)}): ADD them, "
            "with the origin and value the user gave"
        )
    if problems and keep:
        problems.append(
            "call evaluate_origin again with the same arguments, keeping these materials: "
            + ", ".join(f"'{k}'" for k in keep)
        )
    return problems

def _transient(error: str) -> bool:
    """A failure that may not happen again with the same arguments: the agent's time-out or a
    service that did not answer (ServiceUnavailable, raised by tools._http_get)."""
    return error.startswith("time-out") or "ServiceUnavailable" in error

def clean_answer(text: str) -> str:
    """A plain-text answer without the leftovers seen in tests: a closing </think> that was
    never opened (the parser only removes complete blocks) and the classic ReAct prefix
    "Final Answer:"."""
    text = re.sub(r"^\s*</think>\s*", "", text)
    return re.sub(r"^\s*Final Answer\s*:\s*", "", text, flags=re.IGNORECASE).strip()

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
    confirm: str = "auto"  # policy for tools with requires_confirmation: auto or deny

    @classmethod
    def from_env(cls) -> ReActAgent:
        model_id = os.environ.get("ARCA_AGENT_MODEL") or DEFAULT_MODEL
        if model_id.lower() in ("off", "none"):
            # Like ARCA_TOOLS_MODEL=off in phase 2: the API answers 501 for /agent instead of
            # loading two models and the RAG (the test suite sets it in conftest.py).
            raise NotImplementedError(
                f"Phase 4 is disabled: ARCA_AGENT_MODEL is 'off'. Set it to a model id, "
                f"e.g. {DEFAULT_MODEL}."
            )
        agent = cls(model_id=model_id,
            # Empty or "none": no adapter at all (then brain="rlm" is refused).
            adapter_path=os.environ.get("ARCA_AGENT_ADAPTER", DEFAULT_ADAPTER),
            max_new_tokens=int(os.environ.get("ARCA_AGENT_MAX_NEW_TOKENS") or 0) or None,
            tool_timeout=float(os.environ.get("ARCA_AGENT_TOOL_TIMEOUT") or 45),
            confirm=os.environ.get("ARCA_AGENT_CONFIRM", "auto"),
            token_budget=int(os.environ.get("ARCA_AGENT_TOKEN_BUDGET") or 6000),
        )
        if agent.confirm not in CONFIRMATION_POLICIES:
            raise ValueError(f"ARCA_AGENT_CONFIRM must be one of {sorted(CONFIRMATION_POLICIES)}")
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

    def summarize(self, messages: list[dict]) -> str:
        """The old turns condensed for ConversationMemory.compress, without the model.
        A model summary was tried first (the base brain with SUMMARY_PROMPT): Qwen3-1.7B did
        not summarise, it carried on the agent's conversation, and invented calls, numbers, a
        verdict and the instruction to change the values the user gave; once in the summary,
        the agent read them as its own memory and got stuck (see EXPERIMENTS). Here each old
        call becomes one line, built from what really happened: the tool, its arguments and
        the start of its result or error. Nothing can be invented, and it costs no tokens.
        """
        lines = []
        call = None
        for message in messages:
            if message["role"] == "assistant":
                parsed = parse_tool_calls(message["content"])
                call = parsed.tool_calls[0] if parsed.tool_calls else None
            elif message["role"] == "tool":
                try:
                    payload = json.loads(message["content"])
                except json.JSONDecodeError:
                    payload = {"result": message["content"]}
                outcome = (
                    f"error: {payload['error']}" if "error" in payload
                    else f"ok: {json.dumps(payload.get('result'), ensure_ascii=False)}"
                )
                arguments = json.dumps(call.arguments, ensure_ascii=False) if call else "{}"
                name = message.get("name") or (call.name if call else "?")
                lines.append(f"- {name}({arguments[:300]}) -> {outcome[:400]}")
                call = None
            # The reminder to call final_answer (role user) carries no fact: left out.
        return "\n".join(lines)

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

    def confirmed(self, name: str, arguments: dict) -> bool | None:
        """None for a tool without effects; for one with effects (issue_origin_statement),
        whether the policy lets it run. POST /agent has no human in the loop: "auto" takes
        the user's request as the confirmation, "deny" refuses every action."""
        tool = self.registry.tools.get(name)
        if tool is None or not tool.requires_confirmation:
            return None
        return CONFIRMATION_POLICIES[self.confirm](name, arguments)

    def action_report(self, actions: list[ToolCallRecord]) -> str:
        """What really happened to the actions (tools with effects), written by the code from
        the trace and appended to the final answer. The model's own account is not reliable:
        in a test it said a statement had been issued after the tool had refused it twice."""
        if not actions:
            return ""
        issued = [a for a in actions if a.ok]
        if issued:
            paths = ", ".join(
                str(a.result.get("path", "")) if isinstance(a.result, dict) else str(a.result)
                for a in issued
            )
            return f"\n\n[Comprobado por el sistema] Declaración de origen emitida: {paths}"
        return (
            "\n\n[Comprobado por el sistema] La declaración de origen NO se ha emitido. "
            f"Último error: {(actions[-1].error or '')[:300]}"
        )

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

        actions: list[ToolCallRecord] = []  # every call to a tool with effects, for the report
        pending = ""  # the plain answer the model gave when it got the reminder, as a fallback
        nudged = False  # remainder to call final_answer, sent 0 or 1 times
        done: dict[str, int] = {}  # successful calls: name + arguments -> step that made it
        failed: dict[str, tuple[int, str]] = {}  # failed calls: name + arguments -> (step, error)
        retried: set[str] = set()  # transient failures already retried once
        repeats = 0
        max_repeats = 2  # a third identical call means the loop is stuck

        def response(answer: str, n_steps: int, succeeded: bool = True) -> AgentResponse:
            report = self.action_report(actions)
            return AgentResponse(
                final_answer=(answer + report) if answer else report.strip(),
                steps=steps,
                n_steps=n_steps,
                succeeded=succeeded,
                model=self.label(brain),
                tokens_used=self.tokens_used,
            )

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
                    pending = clean_answer(parsed.text)
                    steps.append(
                        AgentStep(thought=thought, observation=f"[no tool called] {NO_CALL_NUDGE}")
                    )
                    memory.add("assistant", raw)
                    memory.add("user", NO_CALL_NUDGE)
                    continue
                steps.append(
                    AgentStep(thought=thought, observation="(answered without final_answer)")
                )
                answer = clean_answer(parsed.text)
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
                elif key in failed and (not _transient(failed[key][1]) or key in retried):
                    # Same call, same arguments, and it already failed: it would fail again.
                    repeats += 1
                    record = ToolCallRecord(
                        name=call.name,
                        arguments=call.arguments,
                        ok=False,
                        error=f"repeated call: this same call failed in step {failed[key][0]} "
                        f"with: {failed[key][1][:200]}. Running it again would fail the same "
                        "way: change the arguments as that error says, or call final_answer.",
                    )
                elif invented := invented_arguments(call.name, call.arguments, task):
                    # Values the user must give and did not: refused before running anything.
                    fix = (
                        'Call issue_origin_statement again with "exporter_reference": "" (empty); '
                        "if this consignment needs one, the tool will say so."
                        if invented == ["exporter_reference"]
                        else "Call final_answer asking the user for it."
                    )
                    record = ToolCallRecord(
                        name=call.name,
                        arguments=call.arguments,
                        ok=False,
                        error=f"not executed: {', '.join(invented)} not given by the user (the "
                        f"values must be copied from the task, never made up). Nothing was done. "
                        f"{fix}",
                    )
                elif problems := bom_problems(call.name, call.arguments, task):
                    record = ToolCallRecord(
                        name=call.name,
                        arguments=call.arguments,
                        ok=False,
                        error="not executed: " + "; ".join(problems) + ". The user's bill of "
                        "materials is complete: do not ask the user for it.",
                    )
                else:
                    confirmed = self.confirmed(call.name, call.arguments)
                    if confirmed is False:
                        record = ToolCallRecord(
                            name=call.name,
                            arguments=call.arguments,
                            ok=False,
                            error=f"not executed: {call.name} has effects outside this "
                            "conversation, and actions are disabled on this server "
                            "(ARCA_AGENT_CONFIRM=deny). Nothing was done: call final_answer "
                            "with what you found and say clearly that the statement was not "
                            "issued and has to be issued by hand.",
                        )
                    else:
                        record = self.call_tool(call.name, call.arguments)
                        if (
                            not record.ok
                            and call.name == "issue_origin_statement"
                            and "is not a REX number" in (record.error or "")
                        ):
                            # The tool's own message (phase 2) says "leave it empty or ask them
                            # for it", and the base brain asked. Same explicit hint as for an
                            # invented REX: the model still writes the call.
                            record.error += (
                                ' Call issue_origin_statement again with "exporter_reference": "" '
                                "(empty); if this consignment needs one, the tool will say so."
                            )
                        if record.ok:
                            done[key] = step
                if not record.ok and not (record.error or "").startswith("repeated call"):
                    if key in failed:
                        retried.add(key)  # second failure of a transient error: no more retries
                    failed[key] = (step, record.error or "")
            observation = ToolLoop.observation(record)  # same format as in the phase-2 training
            tool_def = self.registry.tools.get(record.name)
            if tool_def is not None and tool_def.requires_confirmation:
                actions.append(record)  # issued, refused by the tool, or denied by the policy
            note = "[call written without <tool_call> tags, recovered] " if recovered else ""
            if call.error is None and self.confirmed(call.name, call.arguments):
                note += f"[confirmed automatically: ARCA_AGENT_CONFIRM={self.confirm}] "
            steps.append(
                AgentStep(
                    thought=thought,
                    action=AgentAction(name=record.name, arguments=call.arguments),
                    observation=note + observation["content"],
                )
            )
            if repeats >= max_repeats:
                # Stuck: the answer given before the reminder, if any; the trace shows the loop.
                return response(pending, step, succeeded=bool(pending))
            if record.ok and record.name == "final_answer":
                return response(str(record.result), step)

            memory.add("assistant", raw)
            memory.add(**observation)  # role "tool", the tool name and the result
            if self.token_budget:  # 0: no compression
                before = memory.tokens()
                if memory.compress(self.summarize):
                    steps[-1].observation += (
                        f" [memory compressed: {before} -> {memory.tokens()} tokens]"
                    )
                    # The stubs tell the model it may search again: that must not be refused
                    # as a repeated call.
                    done = {
                        k: s
                        for k, s in done.items()
                        if json.loads(k)["name"] != "search_knowledge_base"
                    }

        if pending:
            # Out of steps after the reminder (in the test, two final_answer calls with broken
            # JSON): the plain answer the model had already given beats an empty one.
            steps.append(
                AgentStep(
                    thought="",
                    observation="(out of steps: returning the answer given before the reminder)",
                )
            )
            return response(pending, max_steps)
        return response("", max_steps, succeeded=False)
