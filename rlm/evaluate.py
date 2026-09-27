"""Phase 1 evaluation: pass@1 of base vs SFT vs GRPO on a held-out set, plus training curves.

Run::

    uv run python -m rlm.evaluate --data rlm/data/test.jsonl \
        --adapters base=none sft=rlm/weights/sft_lora grpo=rlm/weights/final_rlm_lora

It writes ``reports/phase1_eval.json`` with per-example verdicts (so you can do the failure
analysis) and ``reports/phase1_pass1.png`` with the bar chart. ``--history`` plots the
reward curves from the JSON history that the training scripts save::

    uv run python -m rlm.evaluate --history rlm/weights/final_rlm_lora/trainer_state.json

For the MedDRA task every row also carries the problem family (``single``, ``multi``,
``diagnosis``, ``death``, ``lack_of_effect``), the F1 between predicted and expected PT sets,
and the verifier's detail ("missing: …", "extra: …", "not in MedDRA: …"). The summary
reports pass@1 per family, which is where the coding rules show up: a model can be good at
single events and still code every symptom of a diagnosis.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rlm.meddra import term_f1
from rlm.rewards import has_valid_format, thinking_length
from rlm.verifier import MedDRAVerifier, Verifier, build_verifier


def evaluate_model(
    base_model: str,
    adapter: str | None,
    dataset,
    verifier: Verifier,
    max_new_tokens: int,
    batch_size: int = 16,
    temperature: float = 0.0,
) -> list[dict]:
    """Greedy (``temperature=0``) generation for every example, one verdict each.

    Return one dict per example with ``question``, ``expected``, ``raw``, ``predicted``,
    ``is_correct``, ``has_valid_format`` and ``n_tokens``, plus ``family``, ``f1``,
    ``thinking_words`` and ``detail`` for the failure analysis.
    """
    from rlm.inference import ReasoningModel

    model = ReasoningModel(base_model, adapter, verifier_name=verifier.name)
    model.load()
    rows: list[dict] = []
    for start in range(0, len(dataset), batch_size):
        batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
        questions = [example["prompt"][-1]["content"] for example in batch]
        outputs = model.generate_batch(questions, max_new_tokens, temperature)
        for example, question, (raw, n_tokens) in zip(batch, questions, outputs, strict=True):
            result = verifier.verify(raw, example["answer"])
            row = {
                "question": question,
                "expected": example["answer"],
                "raw": raw,
                "predicted": result.predicted,
                "is_correct": result.is_correct,
                "has_valid_format": has_valid_format(raw),
                "n_tokens": n_tokens,
                "thinking_words": thinking_length(raw),
                "family": (example.get("branches") or {}).get("family", "all"),
                "detail": result.detail,
            }
            if isinstance(verifier, MedDRAVerifier):
                row["f1"] = term_f1(result.predicted, example["answer"], verifier.dictionary)
            rows.append(row)
        print(f"  [{len(rows)}/{len(dataset)}] pass@1 so far = {pass_at_1(rows):.3f}")
    return rows


def pass_at_1(rows: list[dict]) -> float:
    return sum(r["is_correct"] for r in rows) / max(len(rows), 1)


def summarize(rows: list[dict]) -> dict:
    """Aggregate metrics for one model: pass@1 overall and per family, format rate, lengths."""
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    n = max(len(rows), 1)
    summary = {
        "pass@1": pass_at_1(rows),
        "format_rate": sum(r["has_valid_format"] for r in rows) / n,
        "mean_tokens": sum(r["n_tokens"] for r in rows) / n,
        "mean_thinking_words": sum(r["thinking_words"] for r in rows) / n,
        "pass@1_by_family": {f: pass_at_1(rs) for f, rs in sorted(by_family.items())},
        "n_by_family": {f: len(rs) for f, rs in sorted(by_family.items())},
    }
    if rows and "f1" in rows[0]:
        summary["mean_f1"] = sum(r["f1"] for r in rows) / n
    errors: dict[str, int] = defaultdict(int)
    for row in rows:
        if not row["is_correct"]:
            kind = "no answer" if row["predicted"] is None else row["detail"].split(":")[0]
            errors[kind or "wrong answer"] += 1
    summary["error_types"] = dict(sorted(errors.items(), key=lambda kv: -kv[1]))
    return summary


def plot_pass1(results: dict, path: Path) -> None:
    """Bar chart: pass@1 per model, overall and per family."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    families = sorted({f for r in results.values() for f in r["summary"]["pass@1_by_family"]})
    groups = ["overall", *families]
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(1.6 * len(groups) + 2, 4))
    for i, name in enumerate(names):
        summary = results[name]["summary"]
        values = [summary["pass@1"]] + [summary["pass@1_by_family"].get(f, 0.0) for f in families]
        xs = [g + (i - (len(names) - 1) / 2) * width for g in range(len(groups))]
        bars = ax.bar(xs, values, width, label=name)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("pass@1")
    ax.set_title("MedDRA coding: pass@1 per model and problem family")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(state_path: Path, out: Path) -> None:
    """Training curves from a ``trainer_state.json``: rewards, completion length, KL, loss."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = json.loads(state_path.read_text(encoding="utf-8"))["log_history"]
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for entry in history:
        step = entry.get("step")
        for key, value in entry.items():
            if isinstance(value, int | float) and key not in ("step", "epoch"):
                series[key].append((step, value))

    panels = {
        "rewards": [k for k in series if k.startswith("rewards/") and k.endswith("/mean")]
        or [k for k in series if k == "reward"],
        "total reward": [k for k in ("reward", "reward_std") if k in series],
        "completion length": [k for k in series if k.startswith("completions/mean_length")],
        # Share of groups where all G completions got the same reward (zero advantage, no
        # gradient) and share of completions cut at max_completion_length.
        "GRPO signal": [
            k for k in ("frac_reward_zero_std", "completions/clipped_ratio") if k in series
        ],
        "KL / clipping": [k for k in series if k == "kl" or k.startswith("clip_ratio/region")],
        "loss": [k for k in ("loss", "eval_loss") if k in series],
    }
    panels = {title: keys for title, keys in panels.items() if keys}
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.6 * len(panels)), sharex=True)
    axes = [axes] if len(panels) == 1 else axes
    for ax, (title, keys) in zip(axes, panels.items(), strict=True):
        for key in keys:
            steps, values = zip(*series[key], strict=True)
            ax.plot(steps, values, label=key.replace("rewards/", "").replace("/mean", ""))
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="rlm/data/test.jsonl", help="'gsm8k' or a JSONL")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=None,
        help="name=path pairs; use 'none' for the bare base model (default: base=none)",
    )
    parser.add_argument("--verifier", default="meddra", help="see rlm/verifier.py")
    parser.add_argument("--n-examples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    parser.add_argument("--out", default="reports/phase1_eval.json")
    parser.add_argument(
        "--history",
        nargs="+",
        default=None,
        help="trainer_state.json files to plot as reports/<parent folder>_curves.png",
    )
    args = parser.parse_args()

    if args.history:
        for state in args.history:
            state_path = Path(state)
            out = Path("reports") / f"{state_path.parent.name}_curves.png"
            plot_history(state_path, out)
            print(f"curves -> {out}")
        if args.adapters is None:  # only curves were asked for
            return
    adapters = args.adapters or ["base=none"]

    from rlm.data import load_domain_dataset, load_gsm8k

    if args.data == "gsm8k":
        dataset = load_gsm8k("test", n_examples=args.n_examples)
        verifier = build_verifier("numeric")
    else:
        dataset = load_domain_dataset(args.data)
        dataset = dataset.select(range(min(args.n_examples, len(dataset))))
        verifier = build_verifier(args.verifier)

    results = {}
    for pair in adapters:
        name, path = pair.split("=", 1)
        print(f"== {name} ({path})")
        rows = evaluate_model(
            args.model,
            None if path == "none" else path,
            dataset,
            verifier,
            args.max_new_tokens,
            batch_size=args.batch_size,
            temperature=args.temperature,
        )
        summary = summarize(rows)
        results[name] = {"pass@1": summary["pass@1"], "summary": summary, "rows": rows}
        print(f"{name:>8}: pass@1 = {summary['pass@1']:.3f} on {len(rows)} problems")
        print(json.dumps({k: v for k, v in summary.items() if k != "pass@1"}, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    chart = out.with_name(out.stem.replace("_eval", "") + "_pass1.png")
    plot_pass1(results, chart)
    print(f"details -> {out}\nchart -> {chart}")


if __name__ == "__main__":
    main()
