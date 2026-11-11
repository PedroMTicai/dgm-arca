"""The contrast of phase 4: the same tools and the same brains behind a *code* agent.

Our ReAct agent (``react_agent.py``) is a JSON agent: each step the model writes one tool call
as JSON, the harness runs it and pastes the result back. A code agent (CodeAct, Wang et al.
2024) writes a Python snippet instead, in which tools are plain functions::

    counts = faers_reaction_counts(drug="sertralina", reaction="Somnolence")
    prr = disproportionality_prr(a=counts["reports_with_drug_and_reaction"], ...)
    final_answer(f"PRR = {prr['prr']}")

so it can chain several calls, keep intermediate values in variables and compute with them in
one step. We use smolagents' ``CodeAgent`` (allowed in this phase as a contrast) with:

* **the same brains**: ``LocalModel`` adapts our shared ``ChatModel`` (``common/llm.py``) to
  smolagents, so ``base``, ``rlm`` and ``thinking`` are the very same weights as in ReAct;
* **the same tools**: ``RegistryTool`` wraps every tool of our registry (FAERS, Naranjo, PRR,
  ICSR, calculator, ``search_knowledge_base``, ``code_meddra``). Arguments are validated by
  our Pydantic models, errors are raised so the agent sees them as the step's error;
* **the same permission policy**: ``create_icsr_report`` is validated and queued in
  ``pending_actions`` instead of being executed, as in ReAct with ``allow_actions=False``.

What differs is only the action format (code vs JSON) and smolagents' own system prompt, so
the comparison in ``agent/evaluate.py`` (brains ``code-base``, ``code-rlm``, ``code-thinking``)
measures the agent design, not the model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from smolagents import CodeAgent, Model
from smolagents import Tool as SmolTool
from smolagents.memory import ActionStep
from smolagents.models import (
    ChatMessage,
    MessageRole,
    get_clean_message_list,
    tool_role_conversions,
)
from smolagents.monitoring import LogLevel, TokenUsage

from agent.react_agent import Brain, ReActAgent, code_meddra, search_knowledge_base
from common.llm import strip_thinking
from tool_use.registry import Tool, ToolRegistry
from tool_use.tools import default_registry

INSTRUCTIONS = """You assist pharmacovigilance technicians. Answer in the language of the user.
Every number in your answer must come from a tool result, never from memory. When you use the
knowledge base, cite the passage ids you relied on. Actions with an effect (create_icsr_report)
are queued for human confirmation: say so, do not call them again. If the task cannot be done
with the available tools or data (wrong country, wrong year, no tool for it), say exactly what
is missing in the final answer instead of guessing."""

JSON_TYPES = {"string", "integer", "number", "boolean", "array", "object"}


def tool_inputs(tool: Tool) -> dict[str, dict[str, Any]]:
    """smolagents ``inputs`` from our Pydantic schema: type, description, nullable."""
    schema = tool.args_model.model_json_schema()
    required = set(schema.get("required", []))
    inputs = {}
    for name, prop in schema.get("properties", {}).items():
        options = [p.get("type") for p in prop.get("anyOf", [prop]) if p.get("type") != "null"]
        kind = options[0] if options and options[0] in JSON_TYPES else "any"
        description = prop.get("description") or name
        if "enum" in prop:
            description += f" One of: {', '.join(map(str, prop['enum']))}."
        inputs[name] = {"type": kind, "description": description}
        if name not in required:
            inputs[name]["nullable"] = True
    return inputs


class RegistryTool(SmolTool):
    """One tool of our registry as a smolagents tool (a function the code can call)."""

    skip_forward_signature_validation = True  # forward(**kwargs) serves every tool
    output_type = "any"

    def __init__(self, tool: Tool, runner: CodeAgentRunner):
        self.tool, self.runner = tool, runner
        self.name, self.description = tool.name, tool.description
        self.inputs = tool_inputs(tool)
        super().__init__()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        arguments = dict(zip(self.inputs, args, strict=False)) | kwargs  # positional calls too
        return self.runner.call(self.tool, arguments)


class LocalModel(Model):
    """Our shared ``ChatModel`` behind smolagents' ``Model`` interface."""

    def __init__(self, brain: Brain, max_new_tokens: int):
        super().__init__(flatten_messages_as_text=True, model_id=brain.llm.name)
        self.brain, self.max_new_tokens = brain, max_new_tokens

    def generate(self, messages, stop_sequences=None, response_format=None, **kwargs):
        chat = get_clean_message_list(
            messages, role_conversions=tool_role_conversions, flatten_messages_as_text=True
        )
        raw, generated = self.brain.llm.generate(
            chat,
            max_new_tokens=self.max_new_tokens,
            enable_thinking=self.brain.thinking,
            stop=stop_sequences,
        )
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=strip_thinking(raw),
            raw=raw,
            token_usage=TokenUsage(self.brain.llm.last_prompt_tokens, generated),
        )


@dataclass
class CodeAgentRunner:
    """Runs a task with smolagents' ``CodeAgent`` and returns the same fields as ReAct."""

    registry: ToolRegistry = field(default_factory=default_registry)
    react: ReActAgent | None = None  # only for its brains (shared weights) and timeouts
    calls: list[dict] = field(default_factory=list)
    pending: list[dict] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> CodeAgentRunner:
        runner = cls(react=ReActAgent(registry=ToolRegistry()))
        runner.registry.register(search_knowledge_base)
        if os.environ.get("ARCA_RLM_ADAPTER"):
            runner.registry.register(code_meddra)
        runner.react.registry = runner.registry
        runner.react.load()  # warm up the knowledge base, as in ReAct
        return runner

    def call(self, tool: Tool, arguments: dict) -> Any:
        """Execute one call with ReAct's policy: confirmation, timeout, errors raised."""
        if tool.requires_confirmation:
            validated = tool.validate(arguments)  # invalid arguments raise, as in ReAct
            self.pending.append({"name": tool.name, "arguments": validated})
            self.calls.append({"action": {"name": tool.name, "arguments": validated}})
            return (
                f"NOT EXECUTED: {tool.name} has an effect and is queued for human "
                "confirmation. Tell the user it is pending; do not call it again."
            )
        record = self.react.execute(tool.name, arguments)
        self.calls.append({"action": {"name": tool.name, "arguments": arguments}})
        if not record.ok:
            raise RuntimeError(record.error)
        return record.result

    def run(self, task: str, max_steps: int = 8, brain: str = "thinking") -> dict:
        mind = self.react.brain_for(brain)
        self.calls, self.pending = [], []
        agent = CodeAgent(
            tools=[RegistryTool(t, self) for t in self.registry.tools.values()],
            model=LocalModel(mind, max_new_tokens=2048 if mind.thinking else 1024),
            instructions=INSTRUCTIONS,
            max_steps=max_steps,
            verbosity_level=LogLevel.OFF,
            additional_authorized_imports=["math", "statistics"],
        )
        result = agent.run(task, return_full_result=True)
        n_steps = sum(isinstance(s, ActionStep) for s in agent.memory.steps)
        usage = result.token_usage
        return {
            "final_answer": "" if result.output is None else str(result.output),
            "steps": self.calls,
            "n_steps": n_steps,
            "succeeded": result.state == "success",
            "tokens_used": usage.total_tokens if usage else 0,
            "stop_reason": "final_answer" if result.state == "success" else "max_steps",
            "pending_actions": self.pending,
        }
