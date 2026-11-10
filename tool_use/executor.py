"""The tool-use workflow: the five steps from class, implemented by hand, with no framework.

    1. Get the user message together with the system prompt that carries the tool definitions.
    2. The model generates: text, or one or more <tool_call> blocks.
    3. The application executes the calls (``ToolRegistry.call``) and never trusts the
       arguments blindly: they are validated first.
    4. Each result goes back into the conversation as a message with role ``tool``.
    5. The model is prompted again; if it answers without calling tools, we are done.

Steps 2-4 repeat until the model answers or ``max_turns`` is reached (multi-step tool use).
Several <tool_call> blocks in one generation are executed in the same turn, concurrently
(parallel calls), and their results go back in the order the model wrote them.

Everything that goes wrong is an observation for the model, not an exception: a
<tool_call> that is not valid JSON, an unknown tool, arguments that fail validation, an API
that times out, an action that was not confirmed. The record also lands in
``ToolsResponse.tool_calls`` with ``ok=False``, so the caller sees the failed attempt too.

``run_tool_loop`` is what ``POST /tools`` calls. Configuration (environment variables):

* ``ARCA_TOOLS_MODEL``    model id (default ``Qwen/Qwen3-1.7B``; ``off`` disables phase 2,
                          which is what the test suite does so it never loads a model)
* ``ARCA_TOOLS_ADAPTER``  optional LoRA adapter, merged into the model as in phase 1
* ``ARCA_TOOLS_PROMPT``   ``slides`` (default: the template below, written by hand) or
                          ``native`` (Qwen's chat template renders the tools itself)
* ``ARCA_TOOLS_THINKING`` ``1`` (default) lets Qwen3 think before each turn, ``0`` turns it
                          off: on the DGX, 42 % of chains right with it, 25 % without
* ``ARCA_TOOLS_MAX_NEW_TOKENS`` per generation (default 1024, or 4096 when thinking)
* ``ARCA_TOOLS_CONFIRM``  ``auto`` (default) or ``deny``: what happens with tools registered
                          with ``requires_confirmation`` (see ``CONFIRMATION_POLICIES``)

Try it from the command line; it prints the whole conversation::

    uv run --extra train python -m tool_use.executor "Is 8507.60 under CETA ..."
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any

from api.schemas import ToolCallRecord, ToolsResponse
from tool_use.parser import ParsedOutput, ToolCall, format_tool_result, parse_tool_calls
from tool_use.registry import ToolRegistry
from tool_use.tools import default_registry

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"

SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant designed to help users.
You will be given a task to solve. To help you, you have access to a set of tools.
Each tool is a function and has a description explaining what it does and the inputs it
expects:
---
{tools_description}
---
You should decide if you want to call a tool. If so, your call must be formatted as follows:
<tool_call>
{{"name": "tool name", "arguments": {{"argument": "value"}}}}
</tool_call>
You may emit several <tool_call> blocks in one message if the calls are independent.
After the tool calls are executed, you will receive their results and can continue.
When you have everything you need, answer the user directly without calling tools."""

# What the model needs to know about the domain and that the tool docstrings cannot say:
# today's date (for "last month"), and when *not* to reach for a tool.
#
# v1 (after reports/phase2_v0_tools.json): the old line "Convert prices in other currencies to
# EUR before evaluating origin" made the model call get_exchange_rate for prices already in
# euros; the model also chained dependent calls in one message and invented ids and numbers.
DOMAIN_CONTEXT = """Today is {today}. You assist EU exporters with preferential origin under
four trade agreements, and only these: TCA (EU-UK), UE-JP (EU-Japan), UE-KR (EU-Korea) and
CETA (EU-Canada). If the user asks about another agreement, say it is not covered, without
calling a tool.
Rules of origin, exchange rates and VAT data must come from the tools, never from memory.
In a tool call, use only values the user gave or a tool returned: never invent a code, a
number or an id. If something you need is missing, ask the user.
Several <tool_call> blocks in one message are only for independent lookups. When a call
needs the result of another one (the criteria of a rule, an evaluation_id), wait for that
result and make the call in your next message.
Never tell the user that something was done (a statement issued, a number checked) unless
the tool result says so; if a tool returned an error, say what failed.
Questions about general concepts can be answered directly, without tools. Answer in the
language of the user."""

NO_TOOLS_PROMPT = "You are an AI assistant designed to help users."


def build_system_prompt(registry: ToolRegistry, today: date | None = None) -> str:
    """Step 1: the system prompt with the tool definitions."""
    tools = SYSTEM_PROMPT_TEMPLATE.format(tools_description=registry.render_for_prompt())
    return tools + "\n\n" + DOMAIN_CONTEXT.format(today=(today or date.today()).isoformat())


# ------------------------------------------------------------------------- confirmation

# Phase 2 has no human in the loop: POST /tools takes one query and returns one answer. The
# user's request is taken as the confirmation ("auto"), and the action tool guards itself
# (issue_origin_statement only accepts an originating evaluation, never overwrites, and asks
# for the REX number above 6,000 EUR). "deny" shows what the model does when it is refused.
# Phase 4 replaces this with a real question to the user.
ConfirmFn = Callable[[str, dict[str, Any]], bool]
CONFIRMATION_POLICIES: dict[str, ConfirmFn] = {
    "auto": lambda name, arguments: True,
    "deny": lambda name, arguments: False,
}


def _not_confirmed(call: ToolCall) -> ToolCallRecord:
    return ToolCallRecord(
        name=call.name,
        arguments=call.arguments,
        ok=False,
        error=(
            f"not executed: {call.name} has effects outside this conversation and the user did "
            "not confirm it. Tell the user what you would do and ask them to confirm."
        ),
    )


def _final_text(text: str) -> str:
    """The answer without a <think> the model opened and never closed.

    With thinking on, the model sometimes writes its final answer inside an unclosed <think>
    (2 of 30 cases on the DGX). The parser only removes complete blocks, so the user would
    get a stray tag; what follows it is the answer.
    """
    stripped = text.lstrip()
    if stripped.startswith("<think>") and "</think>" not in stripped:
        return stripped.removeprefix("<think>").strip()
    return text


def _not_alone(call: ToolCall) -> ToolCallRecord:
    return ToolCallRecord(
        name=call.name,
        arguments=call.arguments,
        ok=False,
        error=(
            f"not executed: {call.name} has effects, so it must be the only call in its "
            "message. Read the results of the other calls first, then call it alone if they "
            "allow it."
        ),
    )


# --------------------------------------------------------------------------------- loop


class ToolLoop:
    """Holds the model and the registry and runs the five-step workflow."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        model_id: str | None = None,
        adapter_path: str | None = None,
        prompt_style: str = "slides",
        thinking: bool = True,
        max_new_tokens: int | None = None,
        confirm: ConfirmFn | None = None,
    ):
        if prompt_style not in ("slides", "native"):
            raise ValueError(f"prompt_style must be 'slides' or 'native', not {prompt_style!r}")
        self.registry = registry or default_registry()
        self.model_id = model_id or DEFAULT_MODEL
        self.adapter_path = adapter_path
        self.prompt_style = prompt_style
        self.thinking = thinking
        # Thinking needs room: with 1024 tokens the reasoning is cut before the call.
        self.max_new_tokens = max_new_tokens or (4096 if thinking else 1024)
        self.confirm = confirm or CONFIRMATION_POLICIES["auto"]
        self.model = None
        self.tokenizer = None
        # The last conversation, raw model outputs included: for debugging and the evaluation.
        self.last_messages: list[dict[str, Any]] = []

    @classmethod
    def from_env(cls) -> ToolLoop:
        model_id = os.environ.get("ARCA_TOOLS_MODEL", DEFAULT_MODEL)
        if model_id.lower() in ("", "off", "none"):
            raise NotImplementedError(
                "Phase 2 is disabled: ARCA_TOOLS_MODEL is 'off'. Set it to a model id, "
                f"e.g. {DEFAULT_MODEL}."
            )
        policy = os.environ.get("ARCA_TOOLS_CONFIRM", "auto")
        if policy not in CONFIRMATION_POLICIES:
            raise ValueError(f"ARCA_TOOLS_CONFIRM must be one of {sorted(CONFIRMATION_POLICIES)}")
        return cls(
            model_id=model_id,
            adapter_path=os.environ.get("ARCA_TOOLS_ADAPTER") or None,
            prompt_style=os.environ.get("ARCA_TOOLS_PROMPT", "slides"),
            thinking=os.environ.get("ARCA_TOOLS_THINKING", "1") != "0",
            max_new_tokens=int(os.environ.get("ARCA_TOOLS_MAX_NEW_TOKENS") or 0) or None,
            confirm=CONFIRMATION_POLICIES[policy],
        )

    @property
    def name(self) -> str:
        """Model label for the response: id, adapter and the settings that change behaviour."""
        label = self.model_id + (f"+{self.adapter_path}" if self.adapter_path else "")
        return f"{label} ({self.prompt_style} prompt, thinking {'on' if self.thinking else 'off'})"

    # -- step 1 and 2 -----------------------------------------------------------

    def load(self) -> None:
        """Load tokenizer and model (bf16 on GPU), merging the adapter if there is one."""
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

            model = PeftModel.from_pretrained(model, self.adapter_path).merge_and_unload()
        self.model = model.eval()

    def initial_messages(self, query: str, use_tools: bool = True) -> list[dict[str, Any]]:
        """Step 1. With the native prompt the tools are added later, by the chat template."""
        if not use_tools:
            system = NO_TOOLS_PROMPT
        elif self.prompt_style == "slides":
            system = build_system_prompt(self.registry)
        else:
            system = NO_TOOLS_PROMPT + "\n\n" + DOMAIN_CONTEXT.format(today=date.today())
        return [{"role": "system", "content": system}, {"role": "user", "content": query}]

    def render(self, messages: list[dict[str, Any]], use_tools: bool = True) -> str:
        """The text the model sees. ``tools=`` only in native mode, so nothing is duplicated."""
        tools = self.registry.definitions() if use_tools and self.prompt_style == "native" else None
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )

    def generate(self, messages: list[dict[str, Any]], use_tools: bool = True) -> str:
        """Step 2: one greedy generation from the conversation so far."""
        import torch

        if self.model is None:
            raise RuntimeError("call load() first")
        inputs = self.tokenizer(self.render(messages, use_tools), return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        # <tool_call> and <think> are not special tokens in Qwen3: they survive the decode.
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # -- step 3 and 4 -----------------------------------------------------------

    def execute(self, call: ToolCall) -> ToolCallRecord:
        """Step 3 for one call: parse error, confirmation, then the registry (which validates)."""
        if call.error is not None:
            return ToolCallRecord(
                name=call.name or "<invalid>", arguments=call.arguments, ok=False, error=call.error
            )
        tool = self.registry.tools.get(call.name)
        if (
            tool is not None
            and tool.requires_confirmation
            and not self.confirm(call.name, call.arguments)
        ):
            return _not_confirmed(call)
        return self.registry.call(call.name, call.arguments)

    def execute_all(self, calls: list[ToolCall]) -> list[ToolCallRecord]:
        """Calls from the same generation are independent: run them concurrently, keep order.

        Except actions (``requires_confirmation``): one that comes with other calls is not run.
        Its arguments cannot depend on results the model has not seen yet, and in the v0
        benchmark they did: the model wrote rule, evaluation and statement in one message, with
        a guessed evaluation_id. Lookups in parallel stay allowed; the action has to come alone.
        """
        if len(calls) == 1:
            return [self.execute(calls[0])]

        def run(call: ToolCall) -> ToolCallRecord:
            tool = self.registry.tools.get(call.name)
            if call.error is None and tool is not None and tool.requires_confirmation:
                return _not_alone(call)
            return self.execute(call)

        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            return list(pool.map(run, calls))

    @staticmethod
    def observation(record: ToolCallRecord) -> dict[str, Any]:
        """Step 4: the ``tool`` message. Named, because parallel results arrive together."""
        payload = {"name": record.name}
        payload.update({"result": record.result} if record.ok else {"error": record.error})
        return {"role": "tool", "name": record.name, "content": format_tool_result("", payload)}

    def recover_bare_calls(self, raw: str, parsed: ParsedOutput) -> tuple[str, ParsedOutput]:
        """A reply that is nothing but tool-call JSON, without the <tool_call> tags, is a call.

        Qwen3-1.7B sometimes drops the tags (2 of 30 cases in reports/phase2_v1_tools.json)
        and the parser, rightly, sees an answer. Accepted only when the whole reply (after
        <think>, and inside a ```json fence if there is one) is one JSON object per line, or a
        list, each naming a registered tool with an 'arguments' object, so prose never
        qualifies. The reply is rewritten with the tags, so the next turns show the model the
        right format. Anything else is returned unchanged.
        """
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", parsed.text.strip())
        try:
            payloads = json.loads(text)
        except json.JSONDecodeError:
            try:
                payloads = [json.loads(line) for line in text.splitlines() if line.strip()]
            except json.JSONDecodeError:
                return raw, parsed
        payloads = payloads if isinstance(payloads, list) else [payloads]
        if not payloads or not all(
            isinstance(p, dict)
            and p.get("name") in self.registry.tools
            and isinstance(p.get("arguments"), dict)
            for p in payloads
        ):
            return raw, parsed
        think = f"<think>\n{parsed.thinking}\n</think>\n\n" if "<think>" in raw else ""
        calls = "\n".join(
            "<tool_call>\n"
            + json.dumps({"name": p["name"], "arguments": p["arguments"]}, ensure_ascii=False)
            + "\n</tool_call>"
            for p in payloads
        )
        return think + calls, parse_tool_calls(think + calls)

    # -- step 5: the loop -------------------------------------------------------

    def run(self, query: str, max_turns: int = 5, use_tools: bool = True) -> ToolsResponse:
        """Generate, execute, feed back; stop when the model answers without calling tools.

        ``use_tools=False`` is the baseline for the comparison: no definitions in the prompt,
        one generation, and any <tool_call> the model writes anyway is left unexecuted.
        """
        messages = self.initial_messages(query, use_tools)
        self.last_messages = messages
        records: list[ToolCallRecord] = []
        for turn in range(1, max_turns + 1):
            raw = self.generate(messages, use_tools)
            parsed = parse_tool_calls(raw)
            message = {"role": "assistant", "content": raw}
            if use_tools and not parsed.tool_calls:
                message["content"], parsed = self.recover_bare_calls(raw, parsed)
                if message["content"] != raw:  # kept for the analysis; the template ignores it
                    message["untagged_original"] = raw
            messages.append(message)
            if not use_tools or not parsed.tool_calls:
                return ToolsResponse(
                    answer=_final_text(parsed.text), tool_calls=records, turns=turn, model=self.name
                )
            # Malformed calls count too: the model has to hear what was wrong to fix it.
            for record in self.execute_all(parsed.tool_calls):
                records.append(record)
                messages.append(self.observation(record))
        # Out of turns: whatever text came with the last calls, flagged as unfinished.
        flag = f"[stopped after {max_turns} turns without a final answer]"
        return ToolsResponse(
            answer=f"{flag} {_final_text(parsed.text)}".strip(),
            tool_calls=records,
            turns=max_turns,
            model=self.name,
        )


_loop: ToolLoop | None = None


def build_tool_loop() -> ToolLoop:
    """Create and load the loop once; the API keeps the instance."""
    global _loop
    if _loop is None:
        loop = ToolLoop.from_env()
        loop.load()
        _loop = loop
    return _loop


def run_tool_loop(query: str, max_turns: int = 5) -> ToolsResponse:
    """Convenience entry point for scripts and the evaluation."""
    return build_tool_loop().run(query, max_turns=max_turns)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query")
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--no-tools", action="store_true", help="baseline without tools")
    args = parser.parse_args()

    loop = build_tool_loop()
    response = loop.run(args.query, max_turns=args.max_turns, use_tools=not args.no_tools)
    for message in loop.last_messages[1:]:
        print(f"── {message['role']} ".ljust(80, "─"))
        print(message["content"])
    print("─" * 80)
    for record in response.tool_calls:
        status = "ok" if record.ok else f"ERROR {record.error}"
        print(f"{record.name} {record.latency_ms:.0f} ms {status}")
    print(f"{response.turns} turns · {response.model}")


if __name__ == "__main__":
    main()


__all__ = ["ToolCallRecord", "ToolLoop", "build_system_prompt", "build_tool_loop", "run_tool_loop"]
