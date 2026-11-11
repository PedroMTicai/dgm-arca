"""The code agent (smolagents CodeAgent) over our registry, with a scripted model."""

from typing import Annotated

import pytest
from pydantic import Field

pytest.importorskip("smolagents")

from agent.code_agent import CodeAgentRunner, RegistryTool, tool_inputs  # noqa: E402
from agent.react_agent import Brain, ReActAgent  # noqa: E402
from tool_use import tools  # noqa: E402
from tool_use.registry import ToolRegistry, tool  # noqa: E402


class _ScriptedLLM:
    name = "scripted"
    last_prompt_tokens = 100

    def __init__(self, replies):
        self.replies, self.stops = list(replies), []

    def generate(self, messages, max_new_tokens=512, enable_thinking=False, stop=None, **kw):
        self.stops.append(stop)
        return self.replies.pop(0), 10


@tool(requires_confirmation=True)
def notify(
    recipient: Annotated[str, Field(description="Who to notify.")],
) -> str:
    """Send a notification (an action with an effect)."""
    raise AssertionError("must not run without confirmation")


def _runner(replies, *extra):
    registry = ToolRegistry().register(tools.calculator, *extra)
    runner = CodeAgentRunner(registry=registry, react=ReActAgent(registry=registry))
    runner.react.brains["base"] = Brain(llm=_ScriptedLLM(replies))
    return runner


def _code(source):
    return f"Thought: do it.\n<code>\n{source}\n</code>"


def test_tool_inputs_follow_the_pydantic_schema():
    inputs = tool_inputs(tools.faers_reaction_counts)
    assert inputs["drug"]["type"] == "string" and "nullable" not in inputs["drug"]
    assert inputs["reaction"].get("nullable") is True
    RegistryTool(tools.faers_reaction_counts, runner=None)  # smolagents accepts the wrapper


def test_code_agent_calls_tools_from_code_and_answers():
    runner = _runner([_code("dose = calculator(expression='10*23*2')\nfinal_answer(f'{dose} mg')")])
    r = runner.run("dosis", brain="base")
    assert r["succeeded"] and r["stop_reason"] == "final_answer"
    assert r["final_answer"] == "460.0 mg"
    assert r["steps"][0]["action"]["name"] == "calculator" and r["n_steps"] == 1
    assert r["tokens_used"] == 110
    assert "Observation:" in runner.react.brains["base"].llm.stops[0]


def test_code_agent_sees_tool_errors_and_recovers():
    runner = _runner(
        [
            _code("print(calculator(expression='import os'))"),
            _code("final_answer('could not compute')"),
        ]
    )
    r = runner.run("dosis", brain="base")
    assert r["final_answer"] == "could not compute" and r["n_steps"] == 2


def test_code_agent_queues_actions_for_confirmation():
    runner = _runner(
        [_code("print(notify(recipient='AEMPS'))"), _code("final_answer('pendiente')")], notify
    )
    r = runner.run("avisa", brain="base")
    assert r["pending_actions"] == [{"name": "notify", "arguments": {"recipient": "AEMPS"}}]
    assert r["final_answer"] == "pendiente"


def test_cut_at_stop_ignores_the_thinking_block():
    from common.llm import cut_at_stop

    text = "<think>Observation: not yet</think>code\nObservation: invented"
    assert cut_at_stop(text, ["Observation:"]) == "<think>Observation: not yet</think>code\n"
    assert cut_at_stop("plain", None) == "plain"
    assert cut_at_stop("<think>Observation: still thinking", ["Observation:"]).endswith("thinking")


def test_adapter_can_be_a_hub_id_or_a_folder(tmp_path):
    from rlm.inference import is_hub_id

    assert is_hub_id("someone/arca-meddra-qwen3-1.7b")
    assert not is_hub_id(str(tmp_path))  # an existing folder
    assert not is_hub_id("rlm/weights/missing_lora")  # a missing local path, not the Hub
