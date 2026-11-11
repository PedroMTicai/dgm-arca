"""The ReAct agent: Thought -> Action -> Observation, in a loop, with every tool we built.

This is the "minimal harness" of the course: the model only produces text; everything that
turns it into an agent lives here. Phases 1 to 3 meet in it:

* **Brain** (``brain=``): ``base`` is Qwen3-1.7B without thinking, the model phase 1 started
  from (``ARCA_RLM_BASE_MODEL``); ``rlm`` is the same model with our phase-1 LoRA adapter
  (``ARCA_RLM_ADAPTER``, a folder or a Hub id); ``thinking`` is Qwen3-1.7B with its native
  thinking mode on (``ARCA_THINKING_MODEL``). base vs rlm answers "did our training help
  the agent?"; thinking is the native reasoning model to compare against.
* **Hands**: the phase-2 registry (FAERS, Naranjo, PRR, ICSR, calculator), the phase-3
  retriever as ``search_knowledge_base``, ``final_answer``, and ``code_meddra`` (the phase-1
  model as a tool) when the adapter is available.

What makes it a harness and not just a loop:

* **Permissions.** Tools registered with ``requires_confirmation`` (``create_icsr_report``)
  are not executed unless the request says ``allow_actions=True``. Otherwise the call is
  validated, returned in ``pending_actions`` for a human to confirm, and the model is told
  so. In phase 2 we saw Qwen3-1.7B invent ``patient_age: 0`` to get past the ICSR
  validation: an action with an effect cannot depend on the model's judgement alone.
* **Timeouts** per tool (``TOOL_TIMEOUTS``): a slow API becomes an observation, not a hang.
* **Error recovery**: invalid JSON, unknown tools, invalid arguments and exceptions are all
  observations the model reads and can correct.
* **Loop detection**: the same call with the same arguments a second time gets a warning
  observation; a third time stops the run (``stop_reason="loop_detected"``).
* **Memory** (``agent/memory.py``): the task is pinned, long observations are truncated
  before they enter the context, and old turns are summarised by the brain itself when the
  token budget is exceeded. With ``session_id`` the memory survives between requests
  (multi-turn).
* **Trace**: every step (thought, action with arguments, real observation) is returned.
"""

from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import Field

from agent.memory import ConversationMemory
from agent.prompts import REACT_SYSTEM_PROMPT, SUMMARY_PROMPT
from api.schemas import AgentAction, AgentResponse, AgentStep, ToolCallRecord
from common.llm import ChatModel, get_chat_model, strip_thinking
from tool_use.parser import parse_tool_calls
from tool_use.registry import ToolRegistry, tool
from tool_use.tools import default_registry

MAX_OBSERVATION_CHARS = 1500
TOOL_TIMEOUTS = {"faers_reaction_counts": 60.0, "search_knowledge_base": 60.0, "code_meddra": 120.0}
DEFAULT_TIMEOUT = 15.0


@tool
def final_answer(
    answer: Annotated[
        str, Field(description="The complete answer to the task, in the language of the user.")
    ],
) -> str:
    """Return the final answer to the user and stop the loop.

    Call it as soon as you can answer, and also when the task is impossible: then say what
    is missing instead of guessing.
    """
    return answer


@tool
def search_knowledge_base(
    query: Annotated[str, Field(description="What to look for, phrased as a question.")],
    top_k: Annotated[int, Field(ge=1, le=10, description="How many passages to return.")] = 4,
) -> str:
    """Search the pharmacovigilance knowledge base: Spanish summaries of product
    characteristics (fichas técnicas: indications, dosing, contraindications, interactions,
    pregnancy, driving, adverse reactions with frequencies, overdose) and EMA
    pharmacovigilance guidelines (reporting rules, deadlines, definitions).

    Returns the most relevant passages with their ids. Cite the ids you use.
    """
    from rag.generate import RagPipeline

    global _RAG
    if _RAG is None:
        pipeline = RagPipeline()
        pipeline.load(load_generator=False)  # the agent is the generator here
        _RAG = pipeline
    hits = _RAG.retrieve(query, top_k=top_k, retriever="hybrid")
    return "\n\n".join(
        f"[{h.chunk.id}] ({h.chunk.context or h.chunk.source})\n{h.chunk.text}" for h in hits
    )


_RAG = None


@tool
def code_meddra(
    report: Annotated[
        str, Field(description="The adverse event report, in Spanish, as the reporter wrote it.")
    ],
) -> str:
    """Code an adverse event report into MedDRA Preferred Terms with our phase-1 reasoning
    model (trained to apply the MedDRA coding rules). Returns the terms separated by ';'.
    """
    from rlm.inference import ReasoningModel

    global _RLM
    if _RLM is None:
        model = ReasoningModel.from_env()
        model.load()
        _RLM = model
    return _RLM.answer(report).answer


_RLM = None


@dataclass
class Brain:
    """A chat model plus how to call it."""

    llm: ChatModel
    thinking: bool = False


def _brain_specs() -> dict[str, dict]:
    base = os.environ.get("ARCA_RLM_BASE_MODEL", "Qwen/Qwen3-1.7B")
    return {
        "base": {"model_id": base},
        "rlm": {"model_id": base, "adapter": os.environ.get("ARCA_RLM_ADAPTER") or None},
        "thinking": {
            "model_id": os.environ.get("ARCA_THINKING_MODEL", "Qwen/Qwen3-1.7B"),
            "thinking": True,
        },
    }


@dataclass
class ReActAgent:
    registry: ToolRegistry = field(default_factory=default_registry)
    token_budget: int = 6000
    brains: dict[str, Brain] = field(default_factory=dict)
    sessions: dict[str, ConversationMemory] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> ReActAgent:
        if os.environ.get("ARCA_AGENT_DISABLED"):
            # Used by the test suite: /health would otherwise load the models.
            raise NotImplementedError("Phase 4 disabled by ARCA_AGENT_DISABLED.")
        agent = cls(token_budget=int(os.environ.get("ARCA_AGENT_TOKEN_BUDGET", "6000")))
        agent.registry.register(final_answer, search_knowledge_base)
        if os.environ.get("ARCA_RLM_ADAPTER"):
            agent.registry.register(code_meddra)
        agent.load()
        return agent

    def load(self) -> None:
        """Warm up the knowledge base (embedder + index) so that the first
        ``search_knowledge_base`` call is not eaten by its timeout; brains load lazily on
        first use (``brain_for``)."""
        global _RAG
        if _RAG is None and "search_knowledge_base" in self.registry.tools:
            from rag.generate import RagPipeline

            pipeline = RagPipeline()
            pipeline.load(load_generator=False)
            _RAG = pipeline

    def brain_for(self, name: str) -> Brain:
        if name not in self.brains:
            specs = _brain_specs()
            if name not in specs:
                raise ValueError(f"unknown brain {name!r}; use {sorted(specs)}")
            spec = specs[name]
            if name == "rlm" and not spec.get("adapter"):
                raise ValueError(
                    "brain 'rlm' needs ARCA_RLM_ADAPTER (the phase-1 LoRA adapter folder)"
                )
            llm = get_chat_model(spec["model_id"], spec.get("adapter"))
            self.brains[name] = Brain(llm=llm, thinking=spec.get("thinking", False))
        return self.brains[name]

    def system_prompt(self, max_steps: int) -> str:
        return REACT_SYSTEM_PROMPT.format(
            max_steps=max_steps, tools_description=self.registry.render_for_prompt()
        )

    def generate(self, messages: list[dict], brain: Brain) -> tuple[str, int]:
        """One generation. Returns the raw text and the tokens it cost (prompt + output)."""
        raw, generated = brain.llm.generate(
            messages,
            max_new_tokens=1024 if brain.thinking else 512,
            enable_thinking=brain.thinking,
        )
        return raw, brain.llm.last_prompt_tokens + generated

    def summarize(self, brain: Brain, old: list[dict]) -> str:
        """The memory policy's summariser: the brain itself, with ``SUMMARY_PROMPT``."""
        history = "\n".join(f"{m['role']}: {m['content']}" for m in old)
        raw, _ = brain.llm.generate(
            [{"role": "user", "content": SUMMARY_PROMPT.format(history=history)}],
            max_new_tokens=300,
        )
        return strip_thinking(raw)

    def execute(self, name: str, arguments: dict) -> ToolCallRecord:
        """``registry.call`` with a timeout. The worker thread is left behind on timeout
        (Python cannot kill it), but the agent moves on with an error observation."""
        timeout = TOOL_TIMEOUTS.get(name, DEFAULT_TIMEOUT)
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(self.registry.call, name, arguments)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            return ToolCallRecord(
                name=name,
                arguments=arguments,
                ok=False,
                error=f"timed out after {timeout:.0f} s",
                latency_ms=timeout * 1000,
            )
        finally:
            pool.shutdown(wait=False)

    def run(
        self,
        task: str,
        max_steps: int = 10,
        brain: str = "rlm",
        session_id: str | None = None,
        allow_actions: bool = False,
    ) -> AgentResponse:
        """The ReAct loop."""
        mind = self.brain_for(brain)
        if session_id and session_id in self.sessions:
            memory = self.sessions[session_id]
            memory.system_prompt = self.system_prompt(max_steps)
        else:
            session_id = session_id or uuid.uuid4().hex[:12]
            memory = ConversationMemory(
                self.system_prompt(max_steps), mind.llm.count_tokens, self.token_budget
            )
            self.sessions[session_id] = memory
        memory.add("user", task, pinned=True)

        steps: list[AgentStep] = []
        pending: list[AgentAction] = []
        tokens = 0
        seen_calls: dict[str, int] = {}

        def respond(answer: str, ok: bool, reason: str) -> AgentResponse:
            memory.add("assistant", answer)
            return AgentResponse(
                final_answer=answer,
                steps=steps,
                n_steps=len(steps),
                succeeded=ok,
                model=mind.llm.name + (" (thinking)" if mind.thinking else ""),
                tokens_used=tokens,
                stop_reason=reason,
                pending_actions=pending,
                session_id=session_id,
                memory_compressions=memory.compressions,
            )

        for _ in range(max_steps):
            raw, cost = self.generate(memory.model_messages(), mind)
            tokens += cost
            parsed = parse_tool_calls(raw)
            thought = parsed.thinking or parsed.text
            if not parsed.tool_calls:
                # Answered in plain text: accept it as the answer, but record that the model
                # did not follow the protocol (it shows up in stop_reason and the benchmark).
                steps.append(AgentStep(thought=thought))
                return respond(parsed.text or thought, True, "answered_without_final_answer")
            # ReAct is one action per step, then look at the observation. Qwen sometimes
            # writes several calls at once, final_answer included, i.e. it answers before
            # seeing any result. Only the first call runs; the rest are dropped from the
            # history and the model is told so.
            call, dropped = parsed.tool_calls[0], len(parsed.tool_calls) - 1
            end_first = raw.find("</tool_call>")
            memory.add("assistant", raw[: end_first + len("</tool_call>")] if dropped else raw)
            note = (
                f"\nnote: only your first tool call was executed ({dropped} more ignored). "
                "One tool per step; wait for the observation."
                if dropped
                else ""
            )
            if call.error:
                observation = f"error: {call.error}{note}"
                steps.append(AgentStep(thought=thought, observation=observation))
                memory.add("tool", observation, name="parser")
                memory.compress(lambda old: self.summarize(mind, old))
                continue
            action = AgentAction(name=call.name, arguments=call.arguments)
            if call.name == "final_answer":
                steps.append(AgentStep(thought=thought, action=action))
                return respond(str(call.arguments.get("answer", "")), True, "final_answer")

            key = json.dumps([call.name, call.arguments], sort_keys=True, default=str)
            seen_calls[key] = seen_calls.get(key, 0) + 1
            if seen_calls[key] == 3:
                steps.append(
                    AgentStep(
                        thought=thought,
                        action=action,
                        observation="stopped: same call repeated 3 times",
                    )
                )
                return respond(
                    "No he podido completar la tarea: el agente repetía la misma "
                    "acción sin avanzar.",
                    False,
                    "loop_detected",
                )

            observation = self.observe(call.name, call.arguments, allow_actions, pending)
            if seen_calls[key] == 2:
                observation += (
                    "\nwarning: you already made this exact call; use the "
                    "result you have or change the arguments."
                )
            observation += note
            steps.append(AgentStep(thought=thought, action=action, observation=observation))
            memory.add("tool", observation, name=call.name)
            memory.compress(lambda old: self.summarize(mind, old))

        return respond(
            "No he podido terminar la tarea en el número máximo de pasos.", False, "max_steps"
        )

    def observe(
        self, name: str, arguments: dict, allow_actions: bool, pending: list[AgentAction]
    ) -> str:
        """Run (or hold) one tool call and turn the outcome into the observation text."""
        spec = self.registry.tools.get(name)
        if spec is not None and spec.requires_confirmation and not allow_actions:
            try:
                spec.validate(arguments)
            except Exception as exc:  # noqa: BLE001 - validation errors are observations
                return f"error: invalid arguments: {exc}"
            pending.append(AgentAction(name=name, arguments=arguments))
            return (
                "not executed: this action has side effects and needs human "
                "confirmation. It has been queued for review; tell the user it is pending."
            )
        record = self.execute(name, arguments)
        text = (
            json.dumps(record.result, ensure_ascii=False, default=str)
            if record.ok
            else f"error: {record.error}"
        )
        if len(text) > MAX_OBSERVATION_CHARS:
            text = text[:MAX_OBSERVATION_CHARS] + f"… [truncated, {len(text)} chars]"
        return text
