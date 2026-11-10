"""The five-step loop, with a scripted model in place of Qwen: no GPU, no network."""

import json
import time

import pytest

from tool_use.executor import ToolLoop, build_system_prompt
from tool_use.registry import ToolRegistry, tool
from tool_use.tools import default_registry


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def slow_echo(text: str) -> str:
    """Wait 0.3 s and return the text."""
    time.sleep(0.3)
    return text


@tool
def broken() -> str:
    """Always fails, like an API that is down."""
    raise RuntimeError("service unavailable")


@tool(requires_confirmation=True)
def write_note(text: str) -> str:
    """Write a note somewhere outside the conversation."""
    return f"written: {text}"


def call(name, **arguments):
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tool_call>"


class ScriptedLoop(ToolLoop):
    """Returns the scripted outputs in order and keeps what the model was shown each turn."""

    def __init__(self, outputs, **kwargs):
        registry = ToolRegistry().register(add, slow_echo, broken, write_note)
        super().__init__(registry=registry, **kwargs)
        self.outputs = list(outputs)
        self.seen: list[list[dict]] = []

    def generate(self, messages, use_tools=True):
        self.seen.append([dict(m) for m in messages])
        return self.outputs.pop(0)


def tool_messages(messages):
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]


def test_answer_without_tools_is_one_turn():
    loop = ScriptedLoop(["<think>\nEasy.\n</think>\n\nA tool call is a request."])
    response = loop.run("What is a tool call?")
    assert response.answer == "A tool call is a request."
    assert response.tool_calls == [] and response.turns == 1


def test_single_call_result_goes_back_as_tool_message():
    loop = ScriptedLoop([call("add", a=2, b=3), "The sum is 5."])
    response = loop.run("2 + 3?")
    assert response.answer == "The sum is 5." and response.turns == 2
    [record] = response.tool_calls
    assert record.ok and record.result == 5 and record.latency_ms >= 0
    # Second generation sees: system, user, the assistant's call and the tool result.
    second = loop.seen[1]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool"]
    assert tool_messages(second) == [{"name": "add", "result": 5}]


def test_multi_step_uses_the_previous_result():
    loop = ScriptedLoop([call("add", a=1, b=2), call("add", a=3, b=4), "Done: 7."])
    response = loop.run("chain")
    assert [r.result for r in response.tool_calls] == [3, 7]
    assert response.turns == 3


def test_parallel_calls_run_concurrently_and_keep_order():
    loop = ScriptedLoop([call("slow_echo", text="a") + "\n" + call("slow_echo", text="b"), "ok"])
    started = time.perf_counter()
    response = loop.run("two at once")
    assert time.perf_counter() - started < 0.55  # sequentially it would take 0.6 s
    assert [r.result for r in response.tool_calls] == ["a", "b"]
    assert [m["result"] for m in tool_messages(loop.seen[1])] == ["a", "b"]


def test_errors_are_observations_and_the_model_can_retry():
    loop = ScriptedLoop(
        [
            "<tool_call>\n{not json}\n</tool_call>",
            call("add", a="two", b=1) + call("nope") + call("broken"),
            call("add", a=2, b=1),
            "3",
        ]
    )
    response = loop.run("retry")
    assert [r.ok for r in response.tool_calls] == [False, False, False, False, True]
    errors = [r.error for r in response.tool_calls[:4]]
    assert "not valid JSON" in errors[0]
    assert "invalid arguments" in errors[1]
    assert "unknown tool" in errors[2]
    assert "RuntimeError: service unavailable" in errors[3]
    assert "error" in tool_messages(loop.seen[1])[0]
    assert response.answer == "3"


def test_confirmation_policy():
    approved = ScriptedLoop([call("write_note", text="hi"), "Written."])
    assert approved.run("note").tool_calls[0].result == "written: hi"

    denied = ScriptedLoop(
        [call("write_note", text="hi"), "Shall I write it?"], confirm=lambda n, a: False
    )
    [record] = denied.run("note").tool_calls
    assert not record.ok and "did not confirm" in record.error
    # Tools without the flag are not asked about.
    other = ScriptedLoop([call("add", a=1, b=1), "2"], confirm=lambda n, a: False)
    assert other.run("x").tool_calls[0].ok


def test_out_of_turns_is_flagged():
    loop = ScriptedLoop([call("add", a=1, b=1)] * 2)
    response = loop.run("loop forever", max_turns=2)
    assert response.turns == 2 and len(response.tool_calls) == 2
    assert response.answer.startswith("[stopped after 2 turns")


def test_without_tools_calls_are_not_executed():
    loop = ScriptedLoop([call("add", a=1, b=1)])
    response = loop.run("x", use_tools=False)
    assert response.tool_calls == [] and response.turns == 1
    assert "add" not in loop.seen[0][0]["content"]


def test_system_prompt_lists_the_domain_tools_and_today():
    from datetime import date

    prompt = build_system_prompt(default_registry(), today=date(2026, 11, 9))
    for name in default_registry().tools:
        assert f'"name": "{name}"' in prompt
    assert "Today is 2026-11-09" in prompt and "<tool_call>" in prompt


def test_from_env(monkeypatch):
    monkeypatch.setenv("ARCA_TOOLS_MODEL", "off")
    with pytest.raises(NotImplementedError):
        ToolLoop.from_env()
    monkeypatch.setenv("ARCA_TOOLS_MODEL", "Qwen/Qwen3-0.6B")
    monkeypatch.setenv("ARCA_TOOLS_PROMPT", "native")
    monkeypatch.setenv("ARCA_TOOLS_CONFIRM", "deny")
    loop = ToolLoop.from_env()
    assert loop.model_id == "Qwen/Qwen3-0.6B" and loop.prompt_style == "native"
    assert loop.confirm("issue_origin_statement", {}) is False
    monkeypatch.setenv("ARCA_TOOLS_CONFIRM", "maybe")
    with pytest.raises(ValueError):
        ToolLoop.from_env()


def test_bare_json_call_without_tags_is_recovered():
    bare = '{"name": "add", "arguments": {"a": 2, "b": 3}}'
    loop = ScriptedLoop([bare, "5"])
    response = loop.run("2 + 3?")
    assert [r.result for r in response.tool_calls] == [5] and response.answer == "5"
    # The history shows the model the tagged form, not what it wrote.
    assert loop.seen[1][2]["content"] == "<tool_call>\n" + bare + "\n</tool_call>"
    assert loop.seen[1][2]["untagged_original"] == bare


def test_bare_json_variants_fenced_list_and_lines():
    fenced = '<think>\nok\n</think>\n\n```json\n{"name": "add", "arguments": {"a": 1, "b": 1}}\n```'
    one = '{{"name": "add", "arguments": {{"a": {}, "b": {}}}}}'
    as_list = "[" + one.format(1, 2) + ", " + one.format(2, 2) + "]"
    lines = one.format(1, 3) + "\n" + one.format(3, 3)
    for raw, expected in ((fenced, [2]), (as_list, [3, 4]), (lines, [4, 6])):
        response = ScriptedLoop([raw, "done"]).run("x")
        assert [r.result for r in response.tool_calls] == expected, raw


def test_json_that_is_not_a_tool_call_stays_an_answer():
    for answer in (
        '{"name": "unknown_tool", "arguments": {}}',
        '{"rate": 1.17, "currency": "GBP"}',
        'Call {"name": "add", "arguments": {"a": 1, "b": 1}} if you like.',
    ):
        response = ScriptedLoop([answer]).run("x")
        assert response.tool_calls == [] and response.answer == answer
    baseline = ScriptedLoop(['{"name": "add", "arguments": {"a": 1, "b": 1}}'])
    assert baseline.run("x", use_tools=False).tool_calls == []


def test_an_action_is_only_run_alone():
    together = call("add", a=1, b=1) + call("write_note", text="hi")
    loop = ScriptedLoop([together, call("write_note", text="hi"), "Written."])
    response = loop.run("note")
    first, refused, alone = response.tool_calls
    assert first.ok and first.result == 2
    assert not refused.ok and "only call in its message" in refused.error
    assert alone.ok and alone.result == "written: hi"
    # Lookups in parallel are still fine.
    parallel = ScriptedLoop([call("add", a=1, b=1) + call("add", a=2, b=2), "ok"])
    assert all(r.ok for r in parallel.run("x").tool_calls)


def test_thinking_gets_more_tokens():
    assert ToolLoop(registry=ToolRegistry()).max_new_tokens == 1024
    assert ToolLoop(registry=ToolRegistry(), thinking=True).max_new_tokens == 4096
    assert (
        ToolLoop(registry=ToolRegistry(), thinking=True, max_new_tokens=2048).max_new_tokens == 2048
    )
