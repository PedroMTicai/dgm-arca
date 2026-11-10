"""Phase 4 evaluation: success rate, steps and tokens on multi-tool tasks, per brain.

Each line of the benchmark (``agent/bench/tasks.jsonl``, 20 tasks) has the task, the tools
that must be involved and one or more automatic checks of the final answer::

    {"id": "a02", "task": "¿La somnolencia con sertralina es una señal ...? Calcula el PRR.",
     "required_tools": ["faers_reaction_counts", "disproportionality_prr"],
     "check": {"type": "number", "value": 2.62, "tolerance": 0.05}}

Check types (a list means all of them must pass):

* ``contains`` (one string or a list of alternatives), ``regex``;
* ``number``: some number in the answer is within ``tolerance`` of ``value`` (Spanish and
  English formatting: '1.680', '2,62');
* ``live``: the number a live tool call returns (FAERS counts change every quarter) must be
  in the answer, within ``tolerance_rel``;
* ``pending_action``: the side-effect tool was queued for human confirmation, not executed.

Run every brain::

    uv run python -m agent.evaluate --bench agent/bench/tasks.jsonl --brains base thinking

and put the resulting table in the report next to the JSON-vs-code-agent comparison.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from tool_use.evaluate import numbers_in


def _check_one(check: dict, final_answer: str, response: dict | None, registry) -> bool:
    kind, value = check.get("type"), check.get("value")
    if kind == "contains":
        options = value if isinstance(value, list) else [value]
        return any(str(o).lower() in final_answer.lower() for o in options)
    if kind == "regex":
        return re.search(value, final_answer) is not None
    if kind == "number":
        tolerance = check.get("tolerance", 0)
        return any(abs(n - float(value)) <= tolerance + 1e-9 for n in numbers_in(final_answer))
    if kind == "live":
        record = registry.call(check["tool"], check["arguments"])
        if not record.ok:
            return False
        target = float(record.result[check["field"]])
        rel = check.get("tolerance_rel", 0.0)
        return any(abs(n - target) <= rel * abs(target) for n in numbers_in(final_answer))
    if kind == "pending_action":
        return response is not None and any(
            a["name"] == value for a in response.get("pending_actions", [])
        )
    return False


def check_answer(
    task: dict, final_answer: str, response: dict | None = None, registry=None
) -> bool:
    """Automatic check of the final answer, according to the task's ``check`` field."""
    checks = task.get("check", {})
    checks = checks if isinstance(checks, list) else [checks]
    return all(_check_one(c, final_answer, response, registry) for c in checks)


def tools_used(steps: list[dict]) -> set[str]:
    return {s["action"]["name"] for s in steps if s.get("action")}


def summarise(rows: list[dict]) -> dict:
    n = max(len(rows), 1)
    return {
        "success_rate": sum(r["correct"] for r in rows) / n,
        "required_tools_rate": sum(r["required_tools_used"] for r in rows) / n,
        "followed_protocol": sum(r["stop_reason"] == "final_answer" for r in rows) / n,
        "mean_steps": sum(r["n_steps"] for r in rows) / n,
        "mean_tokens": sum(r["tokens_used"] for r in rows) / n,
        "stop_reasons": dict(Counter(r["stop_reason"] for r in rows)),
        "n_tasks": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bench", default="agent/bench/tasks.jsonl")
    parser.add_argument("--brains", nargs="+", default=["base", "thinking"])
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--out", default="reports/phase4_eval.json")
    args = parser.parse_args()

    from agent.react_agent import ReActAgent

    tasks = [
        json.loads(line)
        for line in Path(args.bench).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Brains not in this run keep their previous results, so brains can be run one at a time.
    previous = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    results = {k: v for k, v in previous.items() if k not in args.brains}

    def save() -> None:
        out.write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=list), encoding="utf-8"
        )

    agent = ReActAgent.from_env()
    for brain in args.brains:
        rows = []
        for task in tasks:
            try:
                response = agent.run(
                    task["task"], max_steps=args.max_steps, brain=brain
                ).model_dump()
            except Exception as exc:  # noqa: BLE001 - one crash (CUDA OOM) must not lose the run
                response = {
                    "final_answer": "",
                    "steps": [],
                    "n_steps": 0,
                    "succeeded": False,
                    "tokens_used": 0,
                    "stop_reason": f"error: {type(exc).__name__}",
                    "pending_actions": [],
                }
            ok = response["succeeded"] and check_answer(
                task, response["final_answer"], response, agent.registry
            )
            used = tools_used(response["steps"])
            rows.append(
                {
                    **task,
                    **response,
                    "correct": ok,
                    "tools_used": sorted(used),
                    "required_tools_used": set(task.get("required_tools", [])) <= used,
                }
            )
            print(
                f"  {brain} {task['id']} [{task['category']}] ok={ok} "
                f"stop={response['stop_reason']} steps={response['n_steps']} "
                f"tools={sorted(used)}",
                flush=True,
            )
            results[brain] = {"summary": summarise(rows), "rows": rows}
            save()  # after every task: a later crash keeps what was already measured
        print(f"{brain:>9}: {json.dumps(results[brain]['summary'])}", flush=True)
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
