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
from agent.router import ROUTED_NOTE
from api.schemas import AgentAction, AgentResponse, AgentStep, ToolCallRecord
from tool_use.executor import CONFIRMATION_POLICIES, ToolLoop
from tool_use.parser import ParsedOutput, ToolCall, parse_tool_calls
from tool_use.registry import ToolRegistry, tool
from tool_use.tools import ECB_CURRENCIES, default_registry

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
    return recover_named_call(raw, parsed, registry)


# "Action: call `evaluate_origin` with {...}": the tool named in prose, its arguments after it.
_NAMED_CALL = re.compile(r"`?(?P<name>[a-z][a-z_]+)`?\s*(?:with|con)\s*(?=\{)")


def recover_named_call(
    raw: str, parsed: ParsedOutput, registry: ToolRegistry
) -> tuple[str, ParsedOutput]:
    """v2: a registered tool named in prose and followed by its arguments becomes a call.

    In the v1 run, sft_nothink wrote 'Action: call `evaluate_origin` with {...}', with
    "150 / 0.9153" inside the JSON, and that text became its final answer. With valid JSON it is
    a call; with broken JSON it is a malformed call, so the model hears what to fix instead of
    the loop taking the text for the answer."""
    for match in _NAMED_CALL.finditer(parsed.text):
        name = match.group("name")
        if name not in registry.tools:
            continue
        start = match.end()
        try:
            arguments, _ = json.JSONDecoder().raw_decode(parsed.text, start)
        except json.JSONDecodeError as error:
            near = parsed.text[max(start, error.pos - 30) : error.pos + 30].replace("\n", " ")
            bad = ToolCall(
                name=name,
                error=f"the arguments of {name} are not valid JSON ({error.msg}, near: {near!r})."
                " Write each value as a plain number or string, not an expression: convert "
                "amounts with get_exchange_rate and copy its amount_eur.",
            )
            return raw, ParsedOutput(parsed.thinking, parsed.text, [bad])
        if isinstance(arguments, dict):
            prose = parsed.text[: match.start()].strip()
            think = f"<think>\n{parsed.thinking}\n</think>\n\n" if parsed.thinking else ""
            call = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
            fixed = think + (prose + "\n" if prose else "") + f"<tool_call>\n{call}\n</tool_call>"
            return fixed, parse_tool_calls(fixed)
    return raw, parsed


def where_json_breaks(raw: str) -> str:
    """' (near: '...')' for the first <tool_call> block whose JSON does not parse, or ''."""
    for block in re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", raw, flags=re.DOTALL):
        try:
            json.loads(block)
        except json.JSONDecodeError as error:
            near = block[max(0, error.pos - 30) : error.pos + 30].replace("\n", " ")
            return f" (near: {near!r})"
    return ""

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

def _amounts(text: str) -> set[float]:
    """Every number of the text, read with a decimal point and with a decimal comma
    ('4.200' is 4.2 and 4200; '18,500' is 18.5 and 18500)."""
    values = set()
    for token in re.findall(r"\d[\d.,]*\d|\d", text):
        for thousands, decimal in ((",", "."), (".", ",")):
            try:
                values.add(float(token.replace(thousands, "").replace(decimal, ".")))
            except ValueError:
                pass
    return values


def consignment_not_given(name: str, arguments: dict, task: str) -> bool:
    """An issue_origin_statement whose consignment value is not a number of the task.

    The phase-2 SFT data samples the consignment value without looking at the price: in 18 of
    its 135 training statements it is below the ex-works price of one product. In the synthetic
    test (2026-11-23) sft_nothink passed a unit price as the consignment, so a consignment that
    needed the exporter reference (above 6,000 EUR) went out without it."""
    if name != "issue_origin_statement":
        return False
    value = arguments.get("consignment_value_eur")
    if not isinstance(value, (int, float)):
        return False
    return not any(abs(value - v) < 0.005 for v in _amounts(task))


# HS codes the user wrote (6 digits, "8414.59", "8414 59" or "841459"), not prices: a number
# followed by € or EUR is an amount, even if it looks like a code ("1200.50 €").
_HS_IN_TEXT = re.compile(r"\b(\d{4})[.\s]?(\d{2})\b(?!\s*(?:€|EUR|eur|euros))")


def _hs6(code) -> str:
    return re.sub(r"\D", "", str(code or ""))[:6]

def bom_problems(name: str, arguments: dict, task: str) -> list[str]:
    """Blocking problems of an evaluate_origin call: a material with the product's own HS code
    worth the whole ex-works price (the model copied the product into its own bill of
    materials). Codes of the task missing from the materials are not blocking: see
    ``unlisted_codes``."""
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
            # v1: not just "remove it". In veredicto-04 (lamps bought finished and repacked)
            # the bought lamp IS a material, and removing it would leave only EU packaging:
            # a false "originating". There the error was the ex-works price (purchase price).
            problems.append(
                f"'{m.get('name')}' has the product's own HS code ({product}) and is worth the "
                "whole ex-works price. If it is the product itself, REMOVE it from materials. If "
                "the user bought it finished and only packs, labels or cleans it, keep it, use "
                "as exw_price_eur the price at which the user sells, and pass 'operations'"
            )
        else:
            keep.append(m.get("name"))
    if problems and keep:
        problems.append(
            "call evaluate_origin again with the same arguments, keeping these materials: "
            + ", ".join(f"'{k}'" for k in keep)
        )
    return problems


def unlisted_codes(name: str, arguments: dict, task: str) -> list[str]:
    """HS codes written in the task that are neither the product nor a material of the call.

    Until v0 this refused the call. It also refused right calls: a task that mentions a former
    classification ("lo tenía clasificado en la 8418.21") or a product code the model got
    wrong, where the "missing material" was the real product. Now it is a note added to the
    observation, and the model decides."""
    if name != "evaluate_origin":
        return []
    product = _hs6(arguments.get("product_hs_code"))
    listed = {
        _hs6(m.get("hs_code")) for m in arguments.get("materials") or [] if isinstance(m, dict)
    }
    return sorted({a + b for a, b in _HS_IN_TEXT.findall(task)} - listed - {product})


def unlisted_note(codes: list[str]) -> str:
    return (
        f"\nNote: the task also mentions HS {', '.join(codes)}, which is not in this call (neither "
        "the product nor a material). If it is a material the user listed, call evaluate_origin "
        "again adding it with the origin and value the user gave. If it is the product, fix "
        "product_hs_code. If it is something else (a former classification, another product), "
        "ignore this note."
    )


# Currencies an amount can come in: the ECB ones plus others users write. An amount in one of
# them must go through get_exchange_rate before evaluate_origin (v1: in cambio-01 the model
# used 150 USD as 150 EUR, and the verdict flipped).
FOREIGN_CURRENCIES = ECB_CURRENCIES | frozenset(
    "VND ARS CLP COP PEN UYU EGP MAD TND AED SAR PKR BDT TWD RUB UAH NGN KES".split()
)
_AMOUNT = re.compile(r"\d[\d.,]*\s*([A-Z]{3})\b")


def foreign_amounts(task: str) -> set[str]:
    """Currencies other than the euro in which the task gives an amount ("150 USD")."""
    return {c for c in _AMOUNT.findall(task) if c in FOREIGN_CURRENCIES}


def _transient(error: str) -> bool:
    """A failure that may not happen again with the same arguments: the agent's time-out or a
    service that did not answer (ServiceUnavailable, raised by tools._http_get)."""
    return error.startswith("time-out") or "ServiceUnavailable" in error

def _payload_error(observation: str) -> str | None:
    """The 'error' of a step's observation (the phase-2 JSON, maybe after notes in brackets)."""
    start = observation.find("{")
    if start < 0:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(observation, start)
    except json.JSONDecodeError:
        return None
    return payload.get("error") if isinstance(payload, dict) else None


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
    router: object = None  # agent.router.KnowledgeRouter, or None: the model alone decides

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
        # On by default since v3 (EXPERIMENTS.md, 2026-11-23); ARCA_AGENT_ROUTER=0 turns it off.
        if os.environ.get("ARCA_AGENT_ROUTER", "1") == "1":
            from agent.router import KnowledgeRouter

            agent.router = KnowledgeRouter.from_examples()
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
        broken: dict[str, int] = {}  # malformed calls: error -> first step it was seen
        to_convert = foreign_amounts(task)  # currencies with an amount in the task
        converted: set[str] = set()  # currencies get_exchange_rate has converted in this run
        refused_unconverted: set[str] = set()  # evaluate_origin calls refused for that

        def stuck_answer() -> str:
            """v1: when the loop ends without any answer from the model, the user still learns
            why (in v0 the answer was empty in 6 of 20 tasks of the base brain)."""
            last = next((a for a in reversed(steps) if a.action is not None), None)
            error = ""
            if last is not None and last.observation:
                error = (_payload_error(last.observation) or "")[:400]
            return (
                "No he podido completar la tarea. "
                + (f"Último error de {last.action.name}: {error}" if error else "")
            ).strip()

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

        # The knowledge router (agent/router.py): searches made by the loop before the model's
        # first step, in the trace like any other call and counted in max_steps.
        routed = self.router.queries(task) if self.router is not None else []
        for query, probability in routed[: max(0, max_steps - 1)]:
            arguments = {"query": query, "top_k": 3}
            record = self.call_tool("search_knowledge_base", arguments)
            observation = ToolLoop.observation(record)
            if record.ok:
                observation["content"] += ROUTED_NOTE.format(query=query)
            steps.append(
                AgentStep(
                    thought=f"[knowledge router, p={probability:.2f}] The task asks how the "
                    "rules work: the loop searches the knowledge base before the first step.",
                    action=AgentAction(name="search_knowledge_base", arguments=arguments),
                    observation=observation["content"],
                )
            )
            call_text = json.dumps(
                {"name": "search_knowledge_base", "arguments": arguments}, ensure_ascii=False
            )
            memory.add("assistant", f"<tool_call>\n{call_text}\n</tool_call>")
            memory.add(**observation)
            key = json.dumps(
                {"name": "search_knowledge_base", "arguments": arguments},
                sort_keys=True,
                ensure_ascii=False,
            )
            done[key] = len(steps)

        for step in range(len(steps) + 1, max_steps + 1):
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
                error = call.error
                if error.startswith("tool_call is not valid JSON"):
                    error += where_json_breaks(raw)  # v2: where, not only what
                # v2: the same broken call again is a repetition too. In the v1 run, the base
                # brain wrote one broken JSON 9 times and spent all 10 steps.
                if error in broken:
                    repeats += 1
                    error += (
                        f" This is the same broken call as in step {broken[error]}: write it "
                        "again as valid JSON, or call final_answer."
                    )
                else:
                    broken[error] = step
                record = ToolCallRecord(
                    name=call.name or "<invalid>", arguments=call.arguments, ok=False,
                    error=error,
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
                elif consignment_not_given(call.name, call.arguments, task):
                    record = ToolCallRecord(
                        name=call.name,
                        arguments=call.arguments,
                        ok=False,
                        error="not executed: consignment_value_eur "
                        f"{call.arguments.get('consignment_value_eur')} is not a value the user "
                        "gave. Copy the value of the consignment (the invoice) from the task; if "
                        "the user did not give it, call final_answer asking for it. Nothing was "
                        "issued.",
                    )
                elif call.name == "evaluate_origin" and (missing := to_convert - converted):
                    # v1: amounts in another currency must be converted first.
                    if key in refused_unconverted:
                        repeats += 1
                    refused_unconverted.add(key)
                    record = ToolCallRecord(
                        name=call.name,
                        arguments=call.arguments,
                        ok=False,
                        error="not executed: the task gives amounts in "
                        f"{', '.join(sorted(missing))}. Convert each one first with "
                        "get_exchange_rate (currency, the month the user names, amount) and "
                        "use its amount_eur as value_eur. If "
                        "get_exchange_rate cannot convert a currency, call final_answer asking "
                        "the user for that value in euros.",
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
                            if call.name == "get_exchange_rate":
                                converted.add(str(call.arguments.get("currency", "")).upper())
                unconverted = (record.error or "").startswith(
                    "not executed: the task gives amounts"
                )
                if (
                    not record.ok
                    and not (record.error or "").startswith("repeated call")
                    and not unconverted  # refused for the state of the run, not the arguments
                ):
                    if key in failed:
                        retried.add(key)  # second failure of a transient error: no more retries
                    failed[key] = (step, record.error or "")
            observation = ToolLoop.observation(record)  # same format as in the phase-2 training
            executed = call.error is None and not (record.error or "").startswith(
                ("not executed", "repeated call")
            )
            if executed and (codes := unlisted_codes(call.name, call.arguments, task)):
                observation["content"] += unlisted_note(codes)  # v1: a note, not a refusal
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
                # Without one, the last error explained (v1): still not a success.
                return response(pending or stuck_answer(), step, succeeded=bool(pending))
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
        return response(stuck_answer(), max_steps, succeeded=False)