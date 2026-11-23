"""The knowledge router (agent/router.py): what counts as a question for the knowledge base,
and the loop searching before the model's first step."""

import json
from pathlib import Path

import pytest

from agent.react_agent import ReActAgent, final_answer
from agent.router import KnowledgeRouter, question_clauses, routable, with_agreement
from tool_use.registry import ToolRegistry, tool

BENCH = Path(__file__).resolve().parents[1] / "agent" / "bench" / "tasks.jsonl"


@pytest.fixture(scope="module")
def router() -> KnowledgeRouter:
    return KnowledgeRouter.from_examples()  # a few seconds: trained on first use


def test_question_clauses_split_a_second_question():
    clauses = question_clauses(
        "Price 300 EUR. Are they originating under CETA, and how long do we have to keep the "
        "origin declaration? Thanks."
    )
    assert [c for c, _ in clauses] == [
        "Are they originating under CETA",
        "how long do we have to keep the origin declaration?",
    ]
    assert all(s.startswith("Are they") for _, s in clauses)  # the sentence is searched
    spanish = question_clauses("Y si lo es: no somos exportador autorizado, ¿podemos extenderla?")
    assert [c for c, _ in spanish][-1] == "¿podemos extenderla?"


def test_verdict_and_short_questions_are_not_routed():
    assert not routable("Are the packs originating under the TCA?")
    assert not routable("¿Son originarias según el CETA?")
    assert not routable("¿Es válido?")
    # A concept about non-originating materials is a question for the knowledge base.
    assert routable("What is the general tolerance for non-originating materials under CETA?")


def test_the_agreement_of_the_task_goes_into_a_query_that_lacks_it():
    task = "Exportamos a Corea del Sur. ¿Cuánto tiempo hay que guardar la declaración?"
    assert with_agreement("¿Cuánto tiempo hay que guardar la declaración?", task).endswith(
        "(EU-Korea FTA)"
    )
    assert with_agreement("How long under CETA?", task) == "How long under CETA?"


def test_router_on_the_phase4_bench(router):
    """It was trained on other data (agent/router.py): here it must route the knowledge tasks
    and leave alone every task the tools answer."""
    routed = {}
    for t in map(json.loads, BENCH.read_text(encoding="utf-8").splitlines()):
        routed[t["id"]] = (t["category"], router.queries(t["task"]))
    for task_id, (category, queries) in routed.items():
        if category == "conocimiento":
            assert queries, task_id
        elif category != "imposible":
            assert not queries, (task_id, queries)


SEARCHES = []


@tool
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search.

    Args:
        query: The question.
        top_k: How many passages.
    """
    SEARCHES.append(query)
    return "[korea-0027] (from korea/protocol.md) ... 1 000 US dollars ..."


class FixedRouter:
    def queries(self, task):
        return [("¿Hasta qué valor? (EU-Korea FTA)", 0.9)]


def test_the_loop_searches_before_the_first_step():
    SEARCHES.clear()
    registry = ToolRegistry().register(final_answer, search_knowledge_base)
    agent = ReActAgent(registry=registry, token_budget=0, router=FixedRouter())
    seen = []

    def generate(messages, brain):
        seen.append(messages)
        answer = {"name": "final_answer", "arguments": {"answer": "1.000 USD"}}
        return f"<tool_call>\n{json.dumps(answer)}\n</tool_call>"

    agent.generate = generate
    r = agent.run("¿Hasta qué valor pueden entrar en Corea los pequeños paquetes?", brain="base")
    assert SEARCHES == ["¿Hasta qué valor? (EU-Korea FTA)"]
    assert r.steps[0].action.name == "search_knowledge_base"
    assert r.steps[0].thought.startswith("[knowledge router")
    assert "1 000 US dollars" in r.steps[0].observation
    assert "Note: the loop searched" in r.steps[0].observation
    assert r.n_steps == 2 and r.final_answer == "1.000 USD"  # the search counts as a step
    assert any("1 000 US dollars" in m["content"] for m in seen[0])  # the model saw it


def test_the_routed_search_counts_towards_max_steps():
    registry = ToolRegistry().register(final_answer, search_knowledge_base)
    agent = ReActAgent(registry=registry, token_budget=0, router=FixedRouter())
    generations = []
    agent.generate = lambda messages, brain: generations.append(1) or "Nada."
    r = agent.run("¿Hasta qué valor?", brain="base", max_steps=2)
    # One step for the search, one generation for the model, and no more.
    assert r.n_steps == 2 and len(generations) == 1
    assert [s.action.name if s.action else None for s in r.steps][:2] == [
        "search_knowledge_base",
        None,
    ]