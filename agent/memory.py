"""Conversation memory with a token budget.

The context window is the agent's working memory, and we saw in class that stuffing it
does not work: things in the middle get lost. ``ConversationMemory`` keeps the message
list, counts tokens, and when the budget is exceeded it asks you (``summarize``) to
compress the oldest turns into a summary that stays at the top of the conversation.
"""
from __future__ import annotations

import json
import re

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class ConversationMemory:
    system_prompt: str
    count_tokens: Callable[[str], int]
    token_budget: int = 6000
    keep_last: int = 4  # most recent messages (the last two steps) that are never summarised
    pin_first: bool = True  # the task (first user message) is never summarised
    shrink_tools: tuple[str, ...] = ("search_knowledge_base",)  # observations cut first
    messages: list[dict] = field(default_factory=list)
    summary: str = ""

    def add(self, role: str, content: str, **extra) -> None:
        self.messages.append({"role": role, "content": content, **extra})

    def tokens(self) -> int:
        text = self.system_prompt + self.summary + "".join(m["content"] for m in self.messages)
        return self.count_tokens(text)

    def over_budget(self) -> bool:
        return self.tokens() > self.token_budget

    def render(self) -> list[dict]:
        """Messages to send to the model: system (+ summary) followed by the live messages."""
        system = self.system_prompt
        if self.summary:
            system += f"\n\nSummary of the conversation so far:\n{self.summary}"
        return [{"role": "system", "content": system}, *self.messages]

    def compress(self, summarize: Callable[[list[dict]], str]) -> bool:
        """Our policy, applied only when over budget. Returns True if anything changed.

        1. The task (first user message) and the last ``keep_last`` messages stay verbatim:
        the task holds the exact prices, codes and invoice data, and the last steps are
        what the model is working on.
        2. Cheap first: old observations of ``shrink_tools`` (the knowledge-base passages,
        the bulk of the context) become a one-line stub. They were read already, and
        what mattered from them is in what the model wrote afterwards.
        3. Only if still over budget, the old turns become a summary written by the model
        (``summarize``, with SUMMARY_PROMPT), kept at the top of the conversation.
        """
        if not self.over_budget():
            return False
        head = self.messages[:1] if self.pin_first else []
        if len(self.messages) <= len(head) + self.keep_last:
            return False  # nothing old enough to touch
        body = self.messages[len(head) : -self.keep_last]
        recent = self.messages[-self.keep_last :]

        shrunk = [self._shrink(m) for m in body]
        changed = shrunk != body
        self.messages = head + shrunk + recent
        if not self.over_budget():
            return changed

        new_summary = summarize(shrunk)
        self.summary = f"{self.summary}\n{new_summary}".strip()
        self.messages = head + recent
        return True

    def _shrink(self, message: dict) -> dict:
        """An old observation of a bulky tool, replaced by a stub that keeps which passages
        they were (id and source, a few tokens) but not their text."""
        if message.get("role") != "tool" or message.get("name") not in self.shrink_tools:
            return message
        try:
            result = json.loads(message["content"]).get("result", "")
        except (json.JSONDecodeError, AttributeError):
            result = ""
        if not isinstance(result, str) or result.startswith("[passages already read"):
            return message  # an error, or already shrunk
        # format_chunks writes one header per passage: "[<id>] (from <source>)".
        seen = re.findall(r"^\[([^\]]+)\] \(from ([^)]+)\)", result, flags=re.MULTILINE)
        listed = "; ".join(f"[{chunk_id}] {source}" for chunk_id, source in seen)
        stub = json.dumps(
            {
                "name": message["name"],
                "result": f"[passages already read, omitted to save memory: {listed}. "
                "Search again if you need their text]",
            },
            ensure_ascii=False,
        )
        return {**message, "content": stub}
