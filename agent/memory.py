"""Conversation memory with a token budget.

The context window is the agent's working memory, and we saw in class that stuffing it
does not work: things in the middle get lost. ``ConversationMemory`` keeps the message
list, counts tokens, and when the budget is exceeded it asks you (``summarize``) to
compress the oldest turns into a summary that stays at the top of the conversation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class ConversationMemory:
    system_prompt: str
    count_tokens: Callable[[str], int]
    token_budget: int = 6000
    keep_last: int = 6  # most recent messages that are never summarised
    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    compressions: int = 0

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
        """Our policy: *when* — only when over budget; *what* — everything between the pinned
        messages and the last ``keep_last`` ones.

        Pinned messages (``pinned=True``, the user's tasks) are never summarised: a
        paraphrased task is how an agent drifts away from what was asked, and a task is short.
        The ``keep_last`` most recent messages stay verbatim because the next step depends on
        them (the last tool call and its observation). Tool observations are already
        truncated when they enter the memory (``ReActAgent``), so a single long FAERS listing
        cannot by itself force a summary. Returns True if a compression happened.
        """
        if not self.over_budget() or len(self.messages) <= self.keep_last:
            return False
        head, recent = self.messages[: -self.keep_last], self.messages[-self.keep_last :]
        pinned = [m for m in head if m.get("pinned")]
        old = [m for m in head if not m.get("pinned")]
        if not old:
            return False
        new_summary = summarize(old)
        self.summary = f"{self.summary}\n{new_summary}".strip()
        self.messages = pinned + recent
        self.compressions += 1
        return True

    def model_messages(self) -> list[dict]:
        """``render()`` without our bookkeeping keys, ready for the chat template."""
        return [{k: v for k, v in m.items() if k != "pinned"} for m in self.render()]
