"""Phase 2: the domain tools and the five-step loop, without network or models."""

import json

import pytest

from tool_use import tools
from tool_use.evaluate import check_answer, contains_number, score_case
from tool_use.executor import ToolLoop
from tool_use.registry import ToolRegistry


def test_spanish_drug_names_map_to_faers_names():
    assert tools.faers_drug_name("Sertralina") == "sertraline"
    assert tools.faers_drug_name("ácido valproico") == "valproic acid"
    # US adopted names, not translations
    assert tools.faers_drug_name("paracetamol") == "acetaminophen"
    assert tools.faers_drug_name("salbutamol") == "albuterol"
    assert tools.faers_drug_name("Sertraline") == "sertraline"  # English passes through


@pytest.mark.parametrize(
    ("answers", "score", "category"),
    [
        ({}, 0, "doubtful"),
        ({"appeared_after_drug": "no"}, -1, "doubtful"),
        ({"appeared_after_drug": "yes", "alternative_causes": "yes"}, 1, "possible"),
        (
            {
                "previous_reports": "yes",
                "appeared_after_drug": "yes",
                "improved_on_withdrawal": "yes",
                "reappeared_on_rechallenge": "yes",
                "alternative_causes": "no",
            },
            8,
            "probable",
        ),
        (
            {
                q: "yes"
                for q in tools.NARANJO_POINTS
                if q not in ("alternative_causes", "reappeared_with_placebo")
            }
            | {"alternative_causes": "no", "reappeared_with_placebo": "no"},
            13,
            "definite",
        ),
    ],
)
def test_naranjo_scores_and_categories(answers, score, category):
    result = tools.naranjo_causality(**answers)
    assert result["score"] == score and result["category"] == category


def test_naranjo_rejects_answers_outside_the_schema():
    record = (
        ToolRegistry()
        .register(tools.naranjo_causality)
        .call("naranjo_causality", {"appeared_after_drug": "maybe"})
    )
    assert not record.ok and "invalid arguments" in record.error


def test_prr_matches_the_formula_and_flags_signals():
    result = tools.disproportionality_prr(
        reports_with_drug_and_reaction=20,
        reports_with_drug=1000,
        reports_with_reaction=5000,
        total_reports=1_000_000,
    )
    assert result["prr"] == pytest.approx((20 / 1000) / (4980 / 999_000), rel=1e-3)
    assert result["signal"] and result["ci95"][0] > 1
    weak = tools.disproportionality_prr(
        reports_with_drug_and_reaction=3,
        reports_with_drug=10_000,
        reports_with_reaction=30_000,
        total_reports=10_000_000,
    )
    assert not weak["signal"]
    with pytest.raises(ValueError):
        tools.disproportionality_prr(
            reports_with_drug_and_reaction=50,
            reports_with_drug=10,
            reports_with_reaction=100,
            total_reports=1000,
        )


def test_icsr_is_written_and_validated(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "ICSR_DIR", tmp_path)
    args = dict(
        reporter_type="consumer",
        patient_age=45,
        patient_sex="female",
        suspect_drug="sertralina",
        reactions=["Drowsiness"],
        serious=False,
        outcome="recovered",
        narrative="Mucho sueño.",
    )
    result = tools.create_icsr_report(**args)
    saved = json.loads((tmp_path / f"{result['id']}.json").read_text(encoding="utf-8"))
    assert saved["reactions"][0]["meddra_pt"] == "Somnolence"  # LLT mapped to its PT
    assert result["submission_deadline_days"] == 90
    with pytest.raises(ValueError, match="not Preferred Terms"):
        tools.create_icsr_report(**{**args, "reactions": ["sueño"]})
    with pytest.raises(ValueError, match="not identifiable"):
        tools.create_icsr_report(**{**args, "patient_age": None, "patient_sex": "unknown"})
    assert tools.create_icsr_report.requires_confirmation


class _ScriptedLLM:
    """Returns the scripted completions in order and records what it was given."""

    name = "scripted"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def generate(self, messages, max_new_tokens=768, enable_thinking=False, tools=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self.replies.pop(0), 5


def _loop(replies):
    loop = ToolLoop(ToolRegistry().register(tools.naranjo_causality, tools.calculator))
    loop.llm = _ScriptedLLM(replies)
    return loop


def _call(name, **arguments):
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tool_call>"


def test_loop_answers_directly_without_tools():
    loop = _loop(["La farmacovigilancia vigila la seguridad de los medicamentos."])
    response = loop.run("¿Qué es la farmacovigilancia?")
    assert response.tool_calls == [] and response.turns == 1 and response.finished


def test_loop_runs_parallel_calls_then_answers():
    loop = _loop(
        [
            _call("calculator", expression="2*3") + _call("calculator", expression="10/4"),
            "Son 6 y 2,5.",
        ]
    )
    response = loop.run("Calcula 2*3 y 10/4")
    assert [c.result for c in response.tool_calls] == [6.0, 2.5]
    tool_messages = [m for m in loop.llm.calls[1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 2  # both results went back in the same turn
    assert response.turns == 2 and response.answer == "Son 6 y 2,5."


def test_loop_feeds_errors_back_and_lets_the_model_fix_them():
    loop = _loop(
        [
            _call("naranjo_causality", appeared_after_drug="maybe"),
            "<tool_call>{not json}</tool_call>",
            _call("naranjo_causality", appeared_after_drug="yes"),
            "Puntuación 2: posible.",
        ]
    )
    response = loop.run("Naranjo")
    first, second, third = response.tool_calls
    assert not first.ok and "invalid arguments" in first.error
    assert not second.ok and "not valid JSON" in second.error
    assert third.ok and third.result["score"] == 2
    last_tool_message = loop.llm.calls[2]["messages"][-1]
    assert last_tool_message["role"] == "tool" and "error" in last_tool_message["content"]


def test_loop_stops_at_max_turns_and_still_answers():
    loop = _loop([_call("calculator", expression="1+1")] * 2 + ["Resultado parcial: 2."])
    response = loop.run("bucle", max_turns=2)
    assert not response.finished and response.turns == 2 and len(response.tool_calls) == 2
    assert loop.llm.calls[-1]["tools"] is None  # the last generation has no tools


def test_baseline_without_tools_sends_no_definitions():
    loop = _loop(["460 mg"])
    response = loop.run("dosis", use_tools=False)
    assert loop.llm.calls[0]["tools"] is None and response.answer == "460 mg"


def test_bench_scoring_and_answer_checks():
    case = {
        "needs_tool": True,
        "expected_tool": "naranjo_causality",
        "expected_arguments": {"appeared_after_drug": "yes"},
        "answer_checks": [{"contains_number": 8}, {"contains": ["probable"]}],
    }
    calls = [
        {
            "name": "naranjo_causality",
            "arguments": {"appeared_after_drug": "Yes", "dose_related": "no"},
            "ok": True,
        }
    ]
    assert score_case(case, calls) == {"selection_ok": True, "arguments_ok": True, "calls_ok": True}
    assert check_answer(case, "Puntuación 8: causalidad probable.", calls, None)
    assert not check_answer(case, "Puntuación 7.", calls, None)
    assert score_case({"needs_tool": False}, calls)["selection_ok"] is False
    assert contains_number("Hay 5.474 casos", 5474) and contains_number("PRR 4,01", 4.012)
