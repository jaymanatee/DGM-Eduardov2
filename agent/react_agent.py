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

import os
from contextlib import nullcontext
from dataclasses import dataclass, field

from agent.prompts import REACT_SYSTEM_PROMPT
from api.schemas import AgentResponse, AgentStep
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
    steps: list[AgentStep] = field(default_factory=list)
    model_id: str = DEFAULT_MODEL
    adapter_path: str | None = DEFAULT_ADAPTER
    brains: dict[str, Brain] = field(default_factory=brains_from_env)
    max_new_tokens: int | None = None  # per step; default 1024, or 3072 when thinking
    model: object = None
    tokenizer: object = None

    @classmethod
    def from_env(cls) -> ReActAgent:
        agent = cls(model_id=os.environ.get("ARCA_AGENT_MODEL") or DEFAULT_MODEL,
            # Empty or "none": no adapter at all (then brain="rlm" is refused).
            adapter_path=os.environ.get("ARCA_AGENT_ADAPTER", DEFAULT_ADAPTER),
            max_new_tokens=int(os.environ.get("ARCA_AGENT_MAX_NEW_TOKENS") or 0) or None,
        )
        if agent.adapter_path in ("", "none"):
            agent.adapter_path = None
        agent.registry.register(final_answer, search_knowledge_base)
        agent.load()
        return agent

    def load(self) -> None:
        """Tu turno: load the model(s) behind BRAINS. Loading lazily per brain is fine."""
        raise NotImplementedError("Phase 4 is not implemented yet: see agent/README.md.")

    def system_prompt(self, max_steps: int) -> str:
        return REACT_SYSTEM_PROMPT.format(
            max_steps=max_steps, tools_description=self.registry.render_for_prompt()
        )

    def generate(self, messages: list[dict], brain: str) -> str:
        """Tu turno: one generation with the chosen brain."""
        raise NotImplementedError

    def run(self, task: str, max_steps: int = 10, brain: str = "rlm") -> AgentResponse:
        """Tu turno: the ReAct loop.

        Sketch::

            memory = ConversationMemory(self.system_prompt(max_steps), count_tokens, budget)
            memory.add("user", task)
            for step in range(max_steps):
                raw = self.generate(memory.render(), brain)
                parsed = parse_tool_calls(raw)
                thought = parsed.thinking or parsed.text
                if not parsed.has_tool_calls:
                    # the model answered without the final_answer tool: accept it, but flag it
                    ...
                call = parsed.tool_calls[0]
                if call.name == "final_answer":
                    return AgentResponse(final_answer=..., steps=..., n_steps=step + 1, ...)
                record = self.registry.call(call.name, call.arguments)
                observation = str(record.result if record.ok else record.error)
                step_record = AgentStep(thought=thought, action=AgentAction(...), observation=obs)
                self.steps.append(step_record)
                memory.add("assistant", raw); memory.add("tool", observation, name=call.name)
                memory.compress(self.summarize)
            return AgentResponse(final_answer="", steps=self.steps, n_steps=max_steps,
                                 succeeded=False)

        Things to decide and document: timeouts per tool, what happens on repeated identical
        calls (loop detection), and how tools with ``requires_confirmation`` are handled.
        """
        raise NotImplementedError
