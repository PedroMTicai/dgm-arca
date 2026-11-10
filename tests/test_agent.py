"""Memory mechanics and the benchmark checker are given; the loop is yours."""

from agent.evaluate import check_answer
from agent.memory import ConversationMemory


def test_memory_compresses_old_messages_when_over_budget():
    memory = ConversationMemory("sys", count_tokens=len, token_budget=60, keep_last=2)
    for i in range(6):
        memory.add("user" if i % 2 == 0 else "assistant", f"message number {i} " * 2)
    assert memory.over_budget()
    compressed = memory.compress(lambda old: f"{len(old)} old messages summarised")
    assert compressed
    assert len(memory.messages) == 2
    assert "4 old messages" in memory.summary
    assert memory.render()[0]["role"] == "system" and "Summary" in memory.render()[0]["content"]


def test_memory_does_nothing_under_budget():
    memory = ConversationMemory("sys", count_tokens=len, token_budget=10_000)
    memory.add("user", "hello")
    assert not memory.compress(lambda old: "unused")


def test_check_answer_kinds():
    assert check_answer({"check": {"type": "contains", "value": "Madrid"}}, "It is in madrid.")
    assert check_answer({"check": {"type": "number", "value": 7}}, "The name has 7 letters")
    assert check_answer({"check": {"type": "number", "value": 21.5, "tolerance": 0.5}}, "≈ 21.2 °C")
    assert not check_answer({"check": {"type": "regex", "value": r"\bF\b"}}, "no unit here")


# ---------------------------------------------------------------- the ReAct loop, scripted
import json  # noqa: E402

import pytest  # noqa: E402

from agent import react_agent  # noqa: E402
from agent.react_agent import Brain, ReActAgent, final_answer  # noqa: E402
from tool_use import tools  # noqa: E402
from tool_use.registry import ToolRegistry, tool  # noqa: E402


class _ScriptedLLM:
    name = "scripted"
    last_prompt_tokens = 100

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def generate(self, messages, max_new_tokens=512, enable_thinking=False, tools=None):
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0), 10

    def count_tokens(self, text):
        return len(text.split())


def _call(name, **arguments):
    return f"<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"


def _agent(replies, *extra_tools, budget=6000):
    registry = ToolRegistry().register(
        final_answer, tools.calculator, tools.create_icsr_report, *extra_tools
    )
    agent = ReActAgent(registry=registry, token_budget=budget)
    agent.brains["base"] = Brain(llm=_ScriptedLLM(replies))
    return agent


def test_agent_acts_observes_and_answers():
    agent = _agent(
        [
            "Thought: compute.\n" + _call("calculator", expression="10*23*2"),
            _call("final_answer", answer="460 mg al día"),
        ]
    )
    r = agent.run("dosis", brain="base")
    assert r.stop_reason == "final_answer" and r.final_answer == "460 mg al día"
    assert r.steps[0].action.name == "calculator" and r.steps[0].observation == "460.0"
    assert r.tokens_used == 2 * 110 and r.succeeded


def test_only_the_first_call_of_a_step_runs():
    agent = _agent(
        [
            _call("calculator", expression="2+2") + _call("final_answer", answer="5"),
            _call("final_answer", answer="4"),
        ]
    )
    r = agent.run("suma", brain="base")
    assert r.final_answer == "4"  # the premature final_answer was dropped
    assert "only your first tool call" in r.steps[0].observation


def test_side_effect_tools_wait_for_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "ICSR_DIR", tmp_path)
    args = dict(
        reporter_type="consumer",
        patient_age=45,
        patient_sex="female",
        suspect_drug="sertralina",
        reactions=["Somnolence"],
        serious=False,
        outcome="recovered",
        narrative="x",
    )
    agent = _agent([_call("create_icsr_report", **args), _call("final_answer", answer="pendiente")])
    r = agent.run("registra", brain="base")
    assert [a.name for a in r.pending_actions] == ["create_icsr_report"]
    assert "not executed" in r.steps[0].observation and not list(tmp_path.iterdir())
    agent = _agent([_call("create_icsr_report", **args), _call("final_answer", answer="hecho")])
    r = agent.run("registra", brain="base", allow_actions=True)
    assert not r.pending_actions and len(list(tmp_path.iterdir())) == 1


def test_errors_are_observations_and_loops_are_stopped():
    bad = _call("calculator", expression="__import__('os')")
    agent = _agent([bad, bad, bad])
    r = agent.run("bucle", brain="base")
    assert r.steps[0].observation.startswith("error:")
    assert "already made this exact call" in r.steps[1].observation
    assert r.stop_reason == "loop_detected" and not r.succeeded


def test_max_steps_and_plain_text_answers():
    agent = _agent([_call("calculator", expression=f"{i}+1") for i in range(3)])
    r = agent.run("x", brain="base", max_steps=3)
    assert r.stop_reason == "max_steps" and r.n_steps == 3 and not r.succeeded
    agent = _agent(["La respuesta es 4."])
    r = agent.run("x", brain="base")
    assert (
        r.stop_reason == "answered_without_final_answer" and r.final_answer == "La respuesta es 4."
    )


def test_slow_tools_time_out(monkeypatch):
    import time

    @tool
    def slow(seconds: float) -> str:
        """Sleep."""
        time.sleep(seconds)
        return "done"

    monkeypatch.setitem(react_agent.TOOL_TIMEOUTS, "slow", 0.2)
    agent = _agent([_call("slow", seconds=2), _call("final_answer", answer="ok")], slow)
    r = agent.run("x", brain="base")
    assert "timed out" in r.steps[0].observation


def test_sessions_keep_memory_and_the_task_stays_pinned():
    agent = _agent([_call("final_answer", answer="uno"), _call("final_answer", answer="dos")])
    first = agent.run("primera pregunta", brain="base")
    agent.run("segunda pregunta", brain="base", session_id=first.session_id)
    second_prompt = agent.brains["base"].llm.seen[-1]
    assert any(m["content"] == "primera pregunta" for m in second_prompt)
    assert all("pinned" not in m for m in second_prompt)  # bookkeeping never reaches the model


def test_memory_summary_keeps_pinned_task():
    memory = ConversationMemory("sys", count_tokens=len, token_budget=80, keep_last=2)
    memory.add("user", "the task " * 3, pinned=True)
    for i in range(6):
        memory.add("tool", f"observation {i} " * 3)
    assert memory.compress(lambda old: f"{len(old)} summarised")
    assert memory.messages[0]["content"].startswith("the task") and memory.compressions == 1
    assert "4 summarised" in memory.summary


def test_unknown_brain_is_rejected():
    with pytest.raises(ValueError):
        ReActAgent(registry=ToolRegistry()).brain_for("magic")
