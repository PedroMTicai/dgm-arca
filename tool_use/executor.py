"""The tool-use workflow: the five steps from class, implemented by hand, with no framework.

    1. Get the user message together with the system prompt that carries the tool definitions.
    2. The model generates: text, or one or more <tool_call> blocks.
    3. The application executes the calls (``ToolRegistry.call``) and never trusts the
       arguments blindly: they are validated first.
    4. Each result goes back into the conversation as a message with role ``tool``.
    5. The model is prompted again; if it answers without calling tools, we are done.

Steps 2-4 repeat until the model answers or ``max_turns`` is reached (multi-step tool use).
Several <tool_call> blocks in one generation are executed in the same turn (parallel calls).

How the definitions reach the model. We pass ``tools=registry.definitions()`` to Qwen's chat
template instead of pasting JSON into our own prompt. The template renders them inside the
system message, between ``<tools></tools>``, with the instruction to answer with
``<tool_call>{"name": ..., "arguments": ...}</tool_call>``; it is the format Qwen3 was trained
on, and ``tool_use.parser`` reads exactly that. Our ``SYSTEM_PROMPT`` (the domain role and the
rules about when to use a tool) goes before it in the same system message, and tool results
come back as ``tool`` messages, which the template wraps in ``<tool_response>``. Run
``uv run python -m tool_use.executor --show-prompt`` to see the full rendered prompt.

Errors are observations, never crashes: an invalid JSON block, an unknown tool, arguments
that fail validation and exceptions inside a tool all go back to the model as the ``tool``
message, so it can correct itself in the next turn.
"""

from __future__ import annotations

import os

from api.schemas import ToolCallRecord, ToolsResponse
from tool_use.parser import format_tool_result, parse_tool_calls
from tool_use.registry import ToolRegistry
from tool_use.tools import default_registry

SYSTEM_PROMPT = """You are a pharmacovigilance assistant for a drug safety technician in Spain.
Answer in the language of the user. You have tools: use them when the answer depends on
external data (FAERS report counts), on a calculation (Naranjo score, PRR, doses) or when the
user asks you to create a report. Do not call a tool for general questions you can answer
directly, and never invent tool results: if a tool returns an error, fix the arguments or
explain the problem. Drug names may be given in Spanish; reactions must be MedDRA Preferred
Terms in English. When you have the information you need, answer briefly and give the numbers
the tools returned."""


class ToolLoop:
    """Holds the model and the registry and runs the five-step workflow."""

    def __init__(self, registry: ToolRegistry | None = None, model_id: str | None = None):
        self.registry = registry or default_registry()
        self.model_id = model_id
        self.llm = None
        self.enable_thinking = os.environ.get("ARCA_TOOLS_THINKING", "0") == "1"

    def load(self) -> None:
        """The shared chat model (``common.llm``): Qwen3-1.7B unless ARCA_LLM_MODEL is set."""
        from common.llm import get_chat_model

        self.llm = get_chat_model(self.model_id)

    def messages_for(self, query: str) -> list[dict]:
        """Step 1: system prompt + user message (the tool definitions go via the template)."""
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]

    def generate(self, messages: list[dict], use_tools: bool = True) -> tuple[str, int]:
        """Step 2: one generation from the conversation so far."""
        return self.llm.generate(
            messages,
            max_new_tokens=768,
            enable_thinking=self.enable_thinking,
            tools=self.registry.definitions() if use_tools else None,
        )

    def run(self, query: str, max_turns: int = 5, use_tools: bool = True) -> ToolsResponse:
        """The loop. ``use_tools=False`` is the baseline: same model, no tool definitions."""
        messages = self.messages_for(query)
        records: list[ToolCallRecord] = []
        tokens = 0
        for turn in range(1, max_turns + 1):
            raw, n_tokens = self.generate(messages, use_tools=use_tools)
            tokens += n_tokens
            parsed = parse_tool_calls(raw)
            if not use_tools or not parsed.tool_calls:
                return ToolsResponse(
                    answer=parsed.text,
                    tool_calls=records,
                    turns=turn,
                    model=self.llm.name,
                    tokens_generated=tokens,
                )
            messages.append({"role": "assistant", "content": raw})
            for call in parsed.tool_calls:  # parallel calls: all executed in this turn
                if call.error:  # step 3 never runs on a block we could not parse
                    record = ToolCallRecord(
                        name=call.name or "?", arguments={}, ok=False, error=call.error
                    )
                else:
                    record = self.registry.call(call.name, call.arguments)
                records.append(record)
                payload = record.result if record.ok else {"error": record.error}
                messages.append(
                    {
                        "role": "tool",
                        "name": record.name,
                        "content": format_tool_result(record.name, payload),
                    }
                )
        # Out of turns: the last generation still asked for tools. Ask once more without
        # tools so the user gets an answer built on what was gathered, and say so.
        messages.append(
            {"role": "user", "content": "Answer now with the information you already have."}
        )
        raw, n_tokens = self.generate(messages, use_tools=False)
        return ToolsResponse(
            answer=parse_tool_calls(raw).text,
            tool_calls=records,
            turns=max_turns,
            model=self.llm.name,
            tokens_generated=tokens + n_tokens,
            finished=False,
        )


_loop: ToolLoop | None = None


def build_tool_loop() -> ToolLoop:
    """Create and load the loop once; the API keeps the instance."""
    global _loop
    if _loop is None:
        if os.environ.get("ARCA_TOOLS_DISABLED"):
            # Used by the test suite: /health would otherwise load the chat model.
            raise NotImplementedError("Phase 2 disabled by ARCA_TOOLS_DISABLED.")
        loop = ToolLoop()
        loop.load()
        _loop = loop
    return _loop


def run_tool_loop(query: str, max_turns: int = 5, use_tools: bool = True) -> ToolsResponse:
    """Convenience entry point for scripts and the evaluation."""
    return build_tool_loop().run(query, max_turns=max_turns, use_tools=use_tools)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the tool loop on one query.")
    parser.add_argument("query", nargs="*")
    parser.add_argument("--show-prompt", action="store_true", help="print the rendered prompt")
    parser.add_argument("--no-tools", action="store_true", help="baseline without tools")
    args = parser.parse_args()
    query = (
        " ".join(args.query)
        or "¿Cuántas notificaciones de somnolencia con sertralina hay en FAERS?"
    )
    loop = build_tool_loop()
    if args.show_prompt:
        print(loop.llm.render(loop.messages_for(query), tools=loop.registry.definitions()))
    else:
        print(loop.run(query, use_tools=not args.no_tools).model_dump_json(indent=2))


__all__ = ["ToolCallRecord", "ToolLoop", "build_tool_loop", "run_tool_loop"]
