"""Phase 2 evaluation: does the model pick the right tool with the right arguments, and does
the final answer get better than without tools?

The benchmark (``tool_use/bench/cases.jsonl``, 30 cases) has one JSON object per line::

    {"id": "c07", "query": "¿Cuántas notificaciones de somnolencia con sertralina hay en FAERS?",
     "needs_tool": true, "category": "faers",
     "expected_tool": "faers_reaction_counts",
     "expected_arguments": {"drug": {"any_of": ["sertralina", "sertraline"]},
                            "reaction": "Somnolence"},
     "expected_calls": ["faers_reaction_counts", "disproportionality_prr"],  # multi-step
     "answer_checks": [{"tool_result": {...}}, {"contains_number": 8}, {"contains": ["probable"]}]}

Metrics per case:

* ``selection_ok``: the first call names ``expected_tool``, or there is no call when
  ``needs_tool`` is false.
* ``arguments_ok``: among correct selections, every key of ``expected_arguments`` matches after
  normalisation (case, spaces, 45 == 45.0). Only the listed keys are compared, so a model that
  also passes a default value is not penalised; ``{"any_of": [...]}`` accepts alternatives.
* ``calls_ok``: every name in ``expected_calls`` was called (multi-step chains, parallel calls).
* ``answer_ok``: the final answer passes ``answer_checks``: a number (Spanish or English
  formatting), one of several words, the live value a tool returns (FAERS counts change every
  quarter, so they are computed at evaluation time), or whether an action succeeded.

The same cases run twice, with and without tools, and ``answer_ok`` is compared.

Run::

    uv run python -m tool_use.evaluate --bench tool_use/bench/cases.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_bench(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalise(value: Any) -> Any:
    """Make argument comparison tolerant to case, spacing and numeric formatting."""
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: normalise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalise(v) for v in value]
    return value


def value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict) and "any_of" in expected:
        return any(value_matches(actual, option) for option in expected["any_of"])
    return normalise(actual) == normalise(expected)


def arguments_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(
        key in actual and value_matches(actual[key], value) for key, value in expected.items()
    )


def score_case(case: dict[str, Any], tool_calls: list[dict[str, Any]]) -> dict[str, bool]:
    """Compare the calls the model made against the expectation for one benchmark case."""
    first = tool_calls[0] if tool_calls else None
    if not case.get("needs_tool", True):
        ok = first is None
        return {"selection_ok": ok, "arguments_ok": ok, "calls_ok": ok}
    selection_ok = first is not None and first["name"] == case["expected_tool"]
    expected_args = case.get("expected_arguments", {})
    # Parallel cases: any call to the expected tool may carry the expected arguments.
    candidates = [c for c in tool_calls if c["name"] == case["expected_tool"]]
    arguments_ok = selection_ok and any(
        arguments_match(c["arguments"], expected_args) for c in candidates
    )
    called = [c["name"] for c in tool_calls]
    expected_calls = case.get("expected_calls", [case["expected_tool"]])
    calls_ok = all(called.count(n) >= expected_calls.count(n) for n in set(expected_calls))
    return {"selection_ok": selection_ok, "arguments_ok": arguments_ok, "calls_ok": calls_ok}


_NUMBER = re.compile(r"\d[\d.,\s]*\d|\d")


def numbers_in(text: str) -> list[float]:
    """Every plausible reading of the numbers in a text: '5.474' and '5,474' are 5474 or 5.474,
    '4,01' is 4.01, '1.000.000' is 1e6."""
    values = []
    for token in _NUMBER.findall(text):
        token = token.replace(" ", "")
        readings = {
            token.replace(".", "").replace(",", "."),
            token.replace(",", ""),
            token.replace(".", "").replace(",", ""),
        }
        for reading in readings:
            try:
                values.append(float(reading))
            except ValueError:
                continue
    return values


def contains_number(text: str, target: float, rel: float = 0.01) -> bool:
    return any(abs(v - target) <= max(rel * abs(target), 0.005) for v in numbers_in(text))


def check_answer(case: dict, answer: str, tool_calls: list[dict], registry) -> bool | None:
    """``None`` when the case has no answer checks."""
    checks = case.get("answer_checks")
    if not checks:
        return None
    results = []
    for check in checks:
        if "contains" in check:
            results.append(any(w.lower() in answer.lower() for w in check["contains"]))
        elif "contains_number" in check:
            results.append(contains_number(answer, check["contains_number"]))
        elif "tool_result" in check:
            spec = check["tool_result"]
            record = registry.call(spec["tool"], spec["arguments"])  # live ground truth
            results.append(record.ok and contains_number(answer, record.result[spec["field"]]))
        elif "action_ok" in check:
            results.append(any(c["name"] == check["action_ok"] and c["ok"] for c in tool_calls))
        elif "no_action" in check:
            results.append(not any(c["name"] == check["no_action"] and c["ok"] for c in tool_calls))
    return any(results) if case.get("checks_mode") == "any" else all(results)


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(len(rows), 1)
    checked = [r for r in rows if r.get("answer_ok") is not None]

    def rate(key: str) -> float | None:  # None: not applicable (the run without tools)
        values = [r[key] for r in rows if r.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        "tool_selection_accuracy": rate("selection_ok"),
        "argument_accuracy": rate("arguments_ok"),
        "expected_calls_made": rate("calls_ok"),
        "answer_accuracy": sum(r["answer_ok"] for r in checked) / len(checked) if checked else None,
        "n_answer_checked": len(checked),
        "mean_calls": sum(len(r["tool_calls"]) for r in rows) / n,
        "mean_tokens": sum(r["tokens"] for r in rows) / n,
        "n_cases": len(rows),
    }


def run_bench(cases: list[dict], max_turns: int, use_tools: bool) -> list[dict]:
    from tool_use.executor import build_tool_loop

    loop = build_tool_loop()
    rows = []
    for case in cases:
        response = loop.run(case["query"], max_turns=max_turns, use_tools=use_tools)
        calls = [c.model_dump() for c in response.tool_calls]
        row = {
            **case,
            "tool_calls": calls,
            "answer": response.answer,
            "turns": response.turns,
            "tokens": response.tokens_generated,
            "finished": response.finished,
        }
        row.update(
            score_case(case, calls)
            if use_tools
            else {"selection_ok": None, "arguments_ok": None, "calls_ok": None}
        )
        row["answer_ok"] = check_answer(case, response.answer, calls, loop.registry)
        rows.append(row)
        mark = "✓" if row["selection_ok"] else "✗"
        print(
            f"{mark} {case['id']} [{case['category']}] calls={[c['name'] for c in calls]} "
            f"args_ok={row['arguments_ok']} answer_ok={row['answer_ok']}"
        )
    return rows


def by_category(rows: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    return {name: summarise(group) for name, group in sorted(groups.items())}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bench", default="tool_use/bench/cases.jsonl")
    parser.add_argument("--out", default="reports/phase2_eval.json")
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--no-baseline", action="store_true", help="skip the run without tools")
    args = parser.parse_args()

    cases = load_bench(args.bench)
    report: dict[str, Any] = {}
    print("== with tools")
    rows = run_bench(cases, args.max_turns, use_tools=True)
    report["with_tools"] = {
        "summary": summarise(rows),
        "by_category": by_category(rows),
        "rows": rows,
    }
    if not args.no_baseline:
        print("== without tools")
        base = run_bench(cases, args.max_turns, use_tools=False)
        report["without_tools"] = {
            "summary": summarise(base),
            "by_category": by_category(base),
            "rows": base,
        }
    for name in ("with_tools", "without_tools"):
        if name in report:
            print(name, json.dumps(report[name]["summary"], indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
