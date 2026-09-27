"""Phase 1, step 1b: generate reasoning traces with a teacher model and keep the verified ones.

This is what Sky-T1, OpenThoughts and DeepSeek's cold start have in common: a strong
model writes solutions with visible reasoning, a verifier throws away the wrong ones,
and what survives becomes SFT data. Here the teacher is any model that can think in the
``<think>…</think><answer>…</answer>`` format (Qwen3 in thinking mode works well; a
DeepSeek-R1 distilled model too).

Run::

    uv run python -m rlm.distill --data rlm/data/train.jsonl --teacher Qwen/Qwen3-4B \
        --samples 4 --output rlm/data/sft_traces.jsonl

Output: one JSON line per generated trace with ``question``, ``answer``, ``trace``,
``verified`` and ``teacher``. Report in EXPERIMENTS.md the acceptance rate: it is your
first measurement of how hard your domain is. The script also prints the acceptance rate per
problem family (``branches.family`` in our MedDRA dataset) and writes it next to the output
as ``<output>.stats.json``.

Traces are canonicalised before verification: whatever the teacher writes after ``</think>``
is reduced to one ``<answer>`` block (taken from an ``<answer>`` tag, a ``\\boxed{}`` or a
final "Answer:" line), so the SFT targets always have the exact format the rewards check.
Traces whose reasoning is shorter than ``--min-think-words`` are rejected even if the answer
is right: a correct label with no reasoning teaches nothing and is often a lucky guess.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from rlm.rewards import extract_answer
from rlm.verifier import Verifier, build_verifier

_THINK = re.compile(r"<think>(?P<think>.*?)</think>", re.DOTALL)
_FINAL_LINE = re.compile(
    r"^\s*(?:\*\*)?(?:final answer|answer|respuesta(?: final)?)(?:\*\*)?\s*:\s*(?:\*\*)?",
    re.IGNORECASE,
)


def canonicalize_trace(raw: str) -> str | None:
    """Rewrite a teacher completion as ``<think>…</think>\\n<answer>…</answer>``.

    Returns ``None`` when there is no complete reasoning block (the teacher ran out of tokens)
    or no recognisable final answer.
    """
    text = raw.strip()
    if "</think>" in text and "<think>" not in text:
        # Some chat templates put the opening tag in the prompt, not in the completion.
        text = "<think>" + text
    match = _THINK.search(text)
    if match is None:
        return None
    think = match.group("think").strip()
    tail = text[match.end() :]
    answer = extract_answer(tail)
    if answer is None:
        lines = [line.strip() for line in tail.strip().splitlines() if line.strip()]
        for line in reversed(lines):
            if _FINAL_LINE.match(line):
                answer = _FINAL_LINE.sub("", line).strip().strip("*").strip()
                break
        else:
            if len(lines) == 1 and len(lines[0]) <= 200:
                answer = lines[0]
    if not think or not answer:
        return None
    return f"<think>\n{think}\n</think>\n<answer>{answer}</answer>"


def think_words(trace: str) -> int:
    match = _THINK.search(trace)
    return len(match.group("think").split()) if match else 0


def generate_traces(
    dataset,
    teacher: str,
    samples: int,
    max_new_tokens: int,
    verifier: Verifier,
    batch_size: int = 8,
    temperature: float = 0.7,
    min_think_words: int = 15,
) -> list[dict]:
    """Sample ``samples`` completions per problem from the teacher and verify each one.

    Prompts are the same chat messages used for SFT, GRPO and inference (R1-Zero system prompt
    plus the problem), so the teacher reasons over exactly what the student will see.
    Generation is batched (``batch_size`` problems × ``samples`` sequences per call) with left
    padding. Every trace is returned, verified or not, so the acceptance rate can be computed.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(teacher, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        teacher,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    ).eval()

    def render(messages: list[dict]) -> str:
        try:  # Qwen3: ask for its thinking mode explicitly
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    rows: list[dict] = []
    for start in range(0, len(dataset), batch_size):
        batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
        texts = [render(example["prompt"]) for example in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                num_return_sequences=samples,
                pad_token_id=tokenizer.pad_token_id,
            )
        completions = tokenizer.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for i, example in enumerate(batch):
            question = example["prompt"][-1]["content"]
            family = (example.get("branches") or {}).get("family", "all")
            for raw in completions[i * samples : (i + 1) * samples]:
                trace = canonicalize_trace(raw)
                verified = False
                if trace is None:
                    reason = "unparseable (no </think> or no final answer)"
                elif think_words(trace) < min_think_words:
                    reason = f"reasoning shorter than {min_think_words} words"
                else:
                    result = verifier.verify(trace, example["answer"])
                    verified = result.is_correct
                    reason = "" if verified else result.detail or "wrong answer"
                rows.append(
                    {
                        "question": question,
                        "answer": example["answer"],
                        "trace": trace if trace is not None else raw,
                        "verified": verified,
                        "rejection": reason,
                        "family": family,
                        "teacher": teacher,
                    }
                )
        kept = sum(r["verified"] for r in rows)
        done = min(start + batch_size, len(dataset))
        print(f"[{done}/{len(dataset)}] {kept}/{len(rows)} traces verified")
    return rows


def acceptance_stats(rows: list[dict]) -> dict:
    """Acceptance rate overall and per family, coverage of problems, and rejection reasons.

    Rejection reasons are grouped by their first clause ("missing", "extra", "not in MedDRA",
    "unparseable"...), which is already a first error analysis of the teacher.
    """
    by_family: dict[str, list[bool]] = defaultdict(list)
    reasons: dict[str, int] = defaultdict(int)
    solved: dict[str, bool] = {}
    for row in rows:
        by_family[row["family"]].append(row["verified"])
        solved[row["question"]] = solved.get(row["question"], False) or row["verified"]
        if not row["verified"]:
            reasons[row["rejection"].split(":")[0]] += 1
    n_verified = sum(r["verified"] for r in rows)
    return {
        "n_traces": len(rows),
        "n_verified": n_verified,
        "acceptance": n_verified / max(len(rows), 1),
        "problems_with_a_verified_trace": sum(solved.values()) / max(len(solved), 1),
        "acceptance_by_family": {f: sum(v) / len(v) for f, v in sorted(by_family.items())},
        "rejection_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


def main() -> None:
    from rlm.data import load_domain_dataset

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, help="domain JSONL with question / answer")
    parser.add_argument("--teacher", default="Qwen/Qwen3-4B")
    parser.add_argument("--samples", type=int, default=4, help="traces per problem")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8, help="problems per generate call")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--min-think-words", type=int, default=15)
    parser.add_argument("--n-problems", type=int, default=None, help="use only the first N")
    parser.add_argument("--verifier", default="meddra", help="see rlm/verifier.py")
    parser.add_argument("--output", default="rlm/data/sft_traces.jsonl")
    args = parser.parse_args()

    dataset = load_domain_dataset(args.data)
    if args.n_problems:
        dataset = dataset.select(range(min(args.n_problems, len(dataset))))
    traces = generate_traces(
        dataset,
        args.teacher,
        args.samples,
        args.max_new_tokens,
        build_verifier(args.verifier),
        batch_size=args.batch_size,
        temperature=args.temperature,
        min_think_words=args.min_think_words,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = acceptance_stats(traces)
    stats.update({"teacher": args.teacher, "samples": args.samples, "data": args.data})
    stats_path = out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(
        f"{stats['n_verified']}/{stats['n_traces']} traces verified "
        f"({100 * stats['acceptance']:.1f}%) -> {out}"
    )


if __name__ == "__main__":
    main()
