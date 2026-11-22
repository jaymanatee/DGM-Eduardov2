"""Phase 4 without a model: the memory policy, the checks on the model's calls, and the ReAct
loop driven by a scripted model (its outputs are written here, step by step)."""

import json
import time

from agent.evaluate import check_answer
from agent.memory import ConversationMemory
from agent.react_agent import (
    ReActAgent,
    _transient,
    bom_problems,
    foreign_amounts,
    unlisted_codes,
    clean_answer,
    final_answer,
    invented_arguments,
    recover_untagged_call,
)
from tool_use.parser import parse_tool_calls
from tool_use.registry import ToolRegistry, tool

# ------------------------------------------------------------------------------------ memory


def test_memory_keeps_the_task_and_the_last_messages():
    memory = ConversationMemory("sys", count_tokens=len, token_budget=60, keep_last=2)
    for i in range(6):
        memory.add("user" if i % 2 == 0 else "assistant", f"message number {i} " * 2)
    assert memory.over_budget()
    assert memory.compress(lambda old: f"{len(old)} old messages summarised")
    # The task (first message) and the last two stay verbatim; the three in between are summarised.
    assert [m["content"] for m in memory.messages] == [
        "message number 0 " * 2,
        "message number 4 " * 2,
        "message number 5 " * 2,
    ]
    assert "3 old messages" in memory.summary
    assert memory.render()[0]["role"] == "system" and "Summary" in memory.render()[0]["content"]


def test_memory_does_nothing_under_budget():
    memory = ConversationMemory("sys", count_tokens=len, token_budget=10_000)
    memory.add("user", "hello")
    assert not memory.compress(lambda old: "unused")


def test_memory_cuts_old_passages_first_and_keeps_their_ids():
    words = lambda text: len(text.split())  # noqa: E731
    passages = json.dumps(
        {
            "name": "search_knowledge_base",
            "result": "[abc-0001] (from tca/rules.md)\n"
            + "text " * 300
            + "\n\n[abc-0002] (from guides/g.pdf)\n"
            + "text " * 300,
        }
    )
    memory = ConversationMemory("system", words, token_budget=500, keep_last=4)
    memory.add("user", "the task")
    memory.add("assistant", "call 1")
    memory.add("tool", passages, name="search_knowledge_base")
    for i in (2, 3):
        memory.add("assistant", f"call {i}")
        memory.add("tool", f"short result {i}", name="evaluate_origin")
    summarised = []
    assert memory.compress(lambda old: summarised.append(old) or "summary")
    assert not summarised  # the cut was enough: no summary
    stub = memory.messages[2]["content"]
    assert "[abc-0001] tca/rules.md" in stub and "[abc-0002] guides/g.pdf" in stub
    assert "text text" not in stub
    assert memory.messages[0]["content"] == "the task"


def test_check_answer_kinds():
    assert check_answer({"check": {"type": "contains", "value": "Madrid"}}, "It is in madrid.")
    assert check_answer({"check": {"type": "number", "value": 7}}, "The name has 7 letters")
    assert check_answer({"check": {"type": "number", "value": 21.5, "tolerance": 0.5}}, "≈ 21.2 °C")
    assert not check_answer({"check": {"type": "regex", "value": r"\bF\b"}}, "no unit here")


# ------------------------------------------------------------------- checks on the calls

TASK = (
    "Evalúa un ventilador de pie (SA 8414.59), precio franco fábrica 180 €, con motor eléctrico "
    "chino (8501.40) de 60 € y aspas chinas (8414.90) de 25 €. Emite la declaración en la "
    "factura VT-2026-0001: exportador Ventiladores Turia SL, Valencia, valor del envío 4.000 €."
)
MOTOR = {"name": "motor", "hs_code": "850140", "origin": "TERCERO", "value_eur": 60}
BLADES = {"name": "aspas", "hs_code": "841490", "origin": "TERCERO", "value_eur": 25}
FAN = {"name": "ventilador", "hs_code": "841459", "origin": "UE", "value_eur": 180}
EVALUATION = {"agreement": "TCA", "product_hs_code": "8414.59", "exw_price_eur": 180}


def test_bom_accepts_the_users_materials():
    assert bom_problems("evaluate_origin", {**EVALUATION, "materials": [MOTOR, BLADES]}, TASK) == []
    assert bom_problems("get_origin_rule", {"hs_code": "841459"}, TASK) == []


def test_bom_refuses_the_product_as_its_own_material():
    problems = bom_problems(
        "evaluate_origin", {**EVALUATION, "materials": [MOTOR, BLADES, FAN]}, TASK
    )
    assert any("product itself" in p and "REMOVE" in p for p in problems)
    assert "'motor', 'aspas'" in problems[-1]  # which materials to keep


def test_codes_not_in_the_call_are_listed_and_prices_are_not():
    """v1: no longer a refusal (bom_problems), a note: see test_unlisted_code_is_a_note."""
    assert bom_problems("evaluate_origin", {**EVALUATION, "materials": [MOTOR]}, TASK) == []
    assert unlisted_codes("evaluate_origin", {**EVALUATION, "materials": [MOTOR]}, TASK) == [
        "841490"
    ]
    # A price that looks like a code is not a material the user listed.
    priced = TASK + " Otro gasto: 1200.50 €."
    assert unlisted_codes("evaluate_origin", {**EVALUATION, "materials": [MOTOR, BLADES]}, priced) == []


def test_invented_arguments_are_the_ones_not_in_the_task():
    statement = {"invoice_number": "VT-2026-0001", "exporter_name": "Ventiladores Turia S.L."}
    assert invented_arguments(
        "issue_origin_statement", {**statement, "exporter_reference": "REX-123456"}, TASK
    ) == ["exporter_reference"]
    assert (
        invented_arguments("issue_origin_statement", {**statement, "exporter_reference": ""}, TASK)
        == []
    )
    with_rex = TASK + " REX ESREX B46.123.456"
    assert (
        invented_arguments(
            "issue_origin_statement",
            {**statement, "exporter_reference": "ESREXB46123456"},
            with_rex,
        )
        == []
    )
    assert invented_arguments("evaluate_origin", {"agreement": "X"}, TASK) == []


def test_untagged_call_is_recovered_and_prose_is_not():
    raw = 'Thought: I search.\nAction:\n{"name": "final_answer", "arguments": {"answer": "hi"}}'
    registry = ToolRegistry().register(final_answer)
    fixed, parsed = recover_untagged_call(raw, parse_tool_calls(raw), registry)
    assert "<tool_call>" in fixed and parsed.tool_calls[0].name == "final_answer"
    prose = "The answer is {not json} and that is all."
    assert recover_untagged_call(prose, parse_tool_calls(prose), registry)[0] == prose


def test_clean_answer_removes_format_marks_only():
    assert clean_answer("</think>\n\nFinal Answer: It is originating.") == "It is originating."
    text = "It is originating. Final Answer: it is."  # content in the middle is kept
    assert clean_answer(text) == text


def test_transient_errors():
    assert _transient("time-out: x did not answer")
    assert _transient("ServiceUnavailable: ECB")
    assert not _transient("ValueError: bad arguments")


# ---------------------------------------------------------------- the loop, scripted


@tool
def get_origin_rule(hs_code: str) -> dict:
    """Rule of origin of an HS code.

    Args:
        hs_code: The code.
    """
    return {"hs_code": hs_code, "rule": "CTSH"}


@tool
def slow_tool(x: str) -> str:
    """A tool that never answers in time.

    Args:
        x: Anything.
    """
    time.sleep(2)
    return x


@tool(requires_confirmation=True)
def issue_origin_statement(
    evaluation_id: str,
    invoice_number: str,
    exporter_name: str,
    place: str,
    consignment_value_eur: float,
    exporter_reference: str = "",
) -> dict:
    """Write the statement on origin.

    Args:
        evaluation_id: The evaluation.
        invoice_number: The invoice.
        exporter_name: The exporter.
        place: The place.
        consignment_value_eur: The value.
        exporter_reference: The REX number, if the user gave one.
    """
    if exporter_reference and not exporter_reference.startswith("ES"):
        raise ValueError(f"{exporter_reference!r} is not a REX number")
    return {"path": f"outputs/origin_statements/{invoice_number}.txt"}


def call(name: str, **arguments) -> str:
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tool_call>"


STATEMENT = {
    "evaluation_id": "EV-1",
    "invoice_number": "VT-2026-0001",
    "exporter_name": "Ventiladores Turia SL",
    "place": "Valencia",
    "consignment_value_eur": 4000,
}


def payload(step) -> dict:
    """The observation of a step as the model got it, without the trace's notes in front."""
    return json.loads(step.observation[step.observation.index("{") :])


def scripted(*outputs: str, **settings) -> ReActAgent:
    """An agent whose model writes ``outputs`` in order. No model, no network."""
    registry = ToolRegistry().register(
        final_answer, get_origin_rule, slow_tool, issue_origin_statement
    )
    agent = ReActAgent(registry=registry, token_budget=0, **settings)
    replies = iter(outputs)
    agent.generate = lambda messages, brain: next(replies)
    return agent


def test_final_answer_ends_the_loop_with_the_trace():
    agent = scripted(call("get_origin_rule", hs_code="841459"), call("final_answer", answer="CTSH"))
    r = agent.run(TASK, brain="base")
    assert r.succeeded and r.final_answer == "CTSH" and r.n_steps == 2
    assert [s.action.name for s in r.steps] == ["get_origin_rule", "final_answer"]
    assert '"rule": "CTSH"' in r.steps[0].observation


def test_plain_answer_gets_one_reminder_then_is_accepted():
    r = scripted("It is originating.", "Still plain text.").run(TASK, brain="base")
    assert "[no tool called]" in r.steps[0].observation
    assert r.steps[1].observation == "(answered without final_answer)"
    assert r.final_answer == "Still plain text." and r.succeeded


def test_answer_before_the_reminder_survives_running_out_of_steps():
    agent = scripted("It is originating.", call("get_origin_rule", hs_code="1"))
    r = agent.run(TASK, max_steps=2, brain="base")
    assert r.final_answer == "It is originating."
    assert "out of steps" in r.steps[-1].observation


def test_a_failed_call_is_not_run_twice():
    bad = call("issue_origin_statement", **STATEMENT, exporter_reference="XX")  # not in the task
    r = scripted(bad, bad, bad).run(TASK, brain="base")
    assert "not given by the user" in r.steps[0].observation
    assert "repeated call" in r.steps[1].observation and "repeated call" in r.steps[2].observation
    assert not r.succeeded  # stuck: cut after the second repetition
    # v1: the user still learns why, from the last error, instead of an empty answer.
    assert r.final_answer.startswith("No he podido completar la tarea")
    assert "issue_origin_statement" in r.final_answer


def test_invented_rex_refused_then_empty_rex_issues_and_the_report_says_so():
    agent = scripted(
        call("issue_origin_statement", **STATEMENT, exporter_reference="REX-123456"),
        call("issue_origin_statement", **STATEMENT, exporter_reference=""),
        call("final_answer", answer="Done."),
    )
    r = agent.run(TASK, brain="base")
    assert '"exporter_reference": ""' in payload(r.steps[0])["error"]  # the call to make next
    assert r.final_answer.startswith("Done.")
    assert (
        "Declaración de origen emitida: outputs/origin_statements/VT-2026-0001.txt"
        in r.final_answer
    )


def test_the_report_contradicts_a_false_claim():
    rex = TASK + " REX XXREX1"  # given by the user, but refused by the tool
    agent = scripted(
        call("issue_origin_statement", **STATEMENT, exporter_reference="XXREX1"),
        call("final_answer", answer="The statement has been issued."),
    )
    r = agent.run(rex, brain="base")
    assert r.final_answer.startswith("The statement has been issued.")  # the model's text, as is
    assert "NO se ha emitido" in r.final_answer
    assert (
        '"exporter_reference": ""' in payload(r.steps[0])["error"]
    )  # hint added to the tool's error


def test_deny_policy_does_not_run_the_action():
    agent = scripted(
        call("issue_origin_statement", **STATEMENT),
        call("final_answer", answer="Not issued."),
        confirm="deny",
    )
    r = agent.run(TASK, brain="base")
    assert "actions are disabled" in r.steps[0].observation
    assert "NO se ha emitido" in r.final_answer


def test_a_slow_tool_times_out_and_the_loop_goes_on():
    agent = scripted(
        call("slow_tool", x="a"), call("final_answer", answer="No data."), tool_timeout=0.1
    )
    r = agent.run(TASK, brain="base")
    assert '"error": "time-out' in r.steps[0].observation
    assert r.final_answer == "No data."


def test_summary_is_built_from_the_real_calls():
    agent = scripted()
    lines = agent.summarize(
        [
            {"role": "assistant", "content": call("get_origin_rule", hs_code="841459")},
            {
                "role": "tool",
                "name": "get_origin_rule",
                "content": '{"name": "get_origin_rule", "result": {"rule": "CTSH"}}',
            },
            {"role": "user", "content": "reminder"},
            {"role": "assistant", "content": call("slow_tool", x="a")},
            {
                "role": "tool",
                "name": "slow_tool",
                "content": '{"name": "slow_tool", "error": "time-out"}',
            },
        ]
    )
    assert lines.splitlines() == [
        '- get_origin_rule({"hs_code": "841459"}) -> ok: {"rule": "CTSH"}',
        '- slow_tool({"x": "a"}) -> error: time-out',
    ]


# ------------------------------------------------------------------------- v1 of the loop


@tool
def evaluate_origin(
    agreement: str, product_hs_code: str, exw_price_eur: float, materials: list
) -> dict:
    """Origin verdict.

    Args:
        agreement: The agreement.
        product_hs_code: The product.
        exw_price_eur: The price.
        materials: The materials.
    """
    return {"originating": True, "pct_non_originating": 47.2}


@tool
def get_exchange_rate(currency: str, month: str = "", amount: float = 0) -> dict:
    """Exchange rate.

    Args:
        currency: The currency.
        month: The month.
        amount: The amount.
    """
    return {"currency": currency, "amount_eur": round(amount / 1.15, 2)}


def scripted_v1(*outputs: str) -> ReActAgent:
    agent = scripted(*outputs)
    agent.registry.register(evaluate_origin, get_exchange_rate)
    return agent


FORMER = TASK + " Antes lo clasificábamos en la 8414.51."
FULL = {**EVALUATION, "materials": [MOTOR, BLADES]}


def test_unlisted_code_is_a_note_not_a_refusal():
    r = scripted_v1(call("evaluate_origin", **FULL), call("final_answer", answer="Sí.")).run(
        FORMER, brain="base"
    )
    assert '"pct_non_originating": 47.2' in r.steps[0].observation  # it ran
    assert "Note: the task also mentions HS 841451" in r.steps[0].observation
    assert r.succeeded and r.final_answer == "Sí."


def test_foreign_amounts():
    assert foreign_amounts("partes compradas por 150 USD y 2,400 CNY; precio 600 €") == {
        "USD",
        "CNY",
    }
    assert foreign_amounts("precio franco fábrica 1.800 EUR, 1.880 € FOB") == set()
    assert foreign_amounts("difusores por 650.000 VND") == {"VND"}


def test_evaluation_waits_for_the_conversion_and_then_runs():
    task = TASK + " Las aspas se pagaron 29 USD en septiembre de 2026."
    r = scripted_v1(
        call("evaluate_origin", **FULL),
        call("get_exchange_rate", currency="USD", month="2026-09", amount=29),
        call("evaluate_origin", **FULL),  # same arguments: not refused as a repeated call
        call("final_answer", answer="Sí."),
    ).run(task, brain="base")
    assert "the task gives amounts in USD" in r.steps[0].observation
    assert '"pct_non_originating": 47.2' in r.steps[2].observation
    assert r.succeeded


def test_unconverted_evaluation_repeated_gets_stuck():
    task = TASK + " Las aspas se pagaron 29 USD."
    again = call("evaluate_origin", **FULL)
    r = scripted_v1(again, again, again, again).run(task, brain="base")
    assert not r.succeeded and r.n_steps == 3
    assert "USD" in r.final_answer and r.final_answer.startswith("No he podido")