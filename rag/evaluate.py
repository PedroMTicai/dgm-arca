"""Phase 3 evaluation: Recall@k and MRR of each retriever and chunking on the gold set.

The gold set (``rag/gold/gold.jsonl``, format in ``rag/gold.py``) names the document and a
literal quote for each question, so the same questions score every chunking. One run
produces every table of the phase:

1. **Chunking**: number and length of chunks for recursive and semantic chunking.
2. **Retrievers**: Recall@{1,3,5,10} and MRR of BM25, dense and hybrid for both chunkings.
3. **λ sweep** of the hybrid retriever, per chunking. λ is the value with the best MRR
   (ties → better Recall@5), and that is the hybrid reported in table 2.
4. **Query instruction** of the dense retriever (``rag.embed.QUERY_INSTRUCTIONS``).
5. **Context headers**: BM25 and dense with and without ``Chunk.context``.
6. **Per-question ranks** of the relevant chunk, for the failure analysis.

Plus two figures: the λ sweep and the semantic-chunking distances of one document.

Run::

    uv run python -m rag.evaluate                         # everything above
    uv run python -m rag.evaluate --generation            # also run /rag's generator on the
                                                          # gold set: citation checks, abstention

Outputs ``reports/phase3_eval.json``, ``reports/phase3_eval_lambda.png`` and
``reports/phase3_eval_distances.png``.

The development set (``gold.jsonl``) chooses the configuration; the test set
(``gold_test.jsonl``, 100 questions) is measured once with that configuration fixed::

    make rag-test   # --gold rag/gold/gold_test.jsonl --fixed-lambda 0.3 --generation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rag.gold import load_gold, resolve

KS = (1, 3, 5, 10)


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant chunks that appear in the top k."""
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def hit_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1 if at least one relevant chunk is in the top k.

    With quote-based gold a long quote can span two chunks, and either one answers the
    question; hit@k is then the fairer reading of "did we retrieve the answer".
    """
    return float(bool(set(ranked_ids[:k]) & relevant_ids))


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / position of the first relevant chunk (0 when none is retrieved)."""
    for position, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / position
    return 0.0


def score_rankings(rankings: list[list[str]], gold: list[dict]) -> dict[str, float]:
    """Mean metrics over the gold questions, given one ranked id list per question."""
    out: dict[str, float] = {}
    for k in KS:
        out[f"recall@{k}"] = float(
            np.mean(
                [
                    recall_at_k(r, set(g["relevant_chunk_ids"]), k)
                    for r, g in zip(rankings, gold, strict=True)
                ]
            )
        )
        out[f"hit@{k}"] = float(
            np.mean(
                [
                    hit_at_k(r, set(g["relevant_chunk_ids"]), k)
                    for r, g in zip(rankings, gold, strict=True)
                ]
            )
        )
    out["mrr"] = float(
        np.mean(
            [
                reciprocal_rank(r, set(g["relevant_chunk_ids"]))
                for r, g in zip(rankings, gold, strict=True)
            ]
        )
    )
    out["n_questions"] = len(gold)
    return out


def evaluate_retriever(retriever, gold: list[dict], k: int = 10) -> dict[str, float]:
    rankings = [
        [h.chunk.id for h in retriever.search(g["question"], top_k=max(k, 10))] for g in gold
    ]
    return score_rankings(rankings, gold)


def _rank_of_first_relevant(ranked: list[str], relevant: set[str]) -> int | None:
    return next((i for i, cid in enumerate(ranked, start=1) if cid in relevant), None)


def lambda_sweep(bm25, dense, gold: list[dict], lambdas: list[float], pool: int = 20):
    """MRR / Recall@5 of the hybrid for each λ. Scores are computed once per question."""
    from rag.retriever import HybridRetriever

    cached = [(bm25.scores(g["question"]), dense.scores(g["question"])) for g in gold]
    rows = []
    for lam in lambdas:
        hybrid = HybridRetriever(bm25, dense, weight_bm25=lam, pool=pool)
        rankings = [[hybrid.chunks[i].id for i, _ in hybrid.fuse(b, d)[:10]] for b, d in cached]
        rows.append({"lambda": lam, **score_rankings(rankings, gold)})
    best = max(rows, key=lambda r: (r["mrr"], r["recall@5"]))
    return rows, best["lambda"]


def plot_lambda(sweeps: dict[str, list[dict]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name, rows in sweeps.items():
        lams = [r["lambda"] for r in rows]
        ax.plot(lams, [r["mrr"] for r in rows], marker="o", label=f"{name} · MRR")
        ax.plot(
            lams,
            [r["recall@5"] for r in rows],
            marker="s",
            linestyle="--",
            label=f"{name} · Recall@5",
        )
    ax.set_xlabel("λ (weight of BM25; 0 = dense only, 1 = BM25 only)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Hybrid retrieval: λ sweep on the gold set")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_generation(gold_all: list[dict], top_k: int, retriever: str) -> dict:
    """Run ``RagPipeline.answer`` on every gold question: citations and abstention.

    Besides "is there a citation" and "does every cited id exist", two faithfulness checks
    against the gold evidence: whether the relevant chunk was in the prompt at all
    (``relevant_in_context``, a retrieval property) and whether the answer cites it
    (``cites_relevant``). A correct answer that cites a different chunk is the unfaithful
    citation case we want to count.
    """
    from rag.generate import build_pipeline, is_abstention
    from rag.ingest import load_corpus

    pipeline = build_pipeline()
    by_id = {
        g["id"]: set(g["relevant_chunk_ids"])
        for g in resolve(gold_all, pipeline.retrievers["bm25"].chunks, load_corpus())
    }
    rows = []
    for item in gold_all:
        response = pipeline.answer(item["question"], top_k=top_k, retriever=retriever)
        answerable = item.get("type") != "unanswerable"
        relevant = by_id.get(item["id"], set())
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "type": item.get("type"),
                "expected": item.get("answer"),
                "answer": response.answer,
                "citations": response.citations,
                "invalid_citations": response.invalid_citations,
                "abstained": is_abstention(response.answer),
                "answerable": answerable,
                "relevant_in_context": bool(relevant & {c.id for c in response.chunks}),
                "cites_relevant": bool(relevant & set(response.citations)),
            }
        )
        print(
            f"  {item['id']}: {len(response.citations)} citations, "
            f"cites_relevant={rows[-1]['cites_relevant']}, abstained={rows[-1]['abstained']}"
        )
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    in_context = [r for r in answerable if r["relevant_in_context"]]
    summary = {
        "answers_with_citation": np.mean([bool(r["citations"]) for r in answerable]),
        "answers_with_invalid_citation": np.mean([bool(r["invalid_citations"]) for r in rows]),
        "relevant_in_context": np.mean([r["relevant_in_context"] for r in answerable]),
        "cites_relevant": np.mean([r["cites_relevant"] for r in answerable]),
        "cites_relevant_when_in_context": np.mean([r["cites_relevant"] for r in in_context])
        if in_context
        else None,
        "false_abstention_rate": np.mean([r["abstained"] for r in answerable]),
        "abstention_on_unanswerable": np.mean([r["abstained"] for r in unanswerable])
        if unanswerable
        else None,
    }
    return {
        "summary": {k: (float(v) if v is not None else None) for k, v in summary.items()},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gold", default="rag/gold/gold.jsonl")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["recursive", "semantic"],
        help="'recursive', 'semantic' or 'semantic:<threshold>' for a threshold sweep",
    )
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=[round(0.1 * i, 1) for i in range(11)]
    )
    parser.add_argument(
        "--instruction",
        default="pharmacovigilance",
        help="instruction for the main tables (rag.embed.QUERY_INSTRUCTIONS)",
    )
    parser.add_argument(
        "--distance-doc",
        default="fichas/sertralina.md",
        help="document whose sentence distances are plotted",
    )
    parser.add_argument(
        "--generation",
        action="store_true",
        help="also evaluate the generator (needs the chat model)",
    )
    parser.add_argument("--retriever", default="hybrid", help="retriever for --generation")
    parser.add_argument("--top-k", type=int, default=5, help="chunks in the prompt (--generation)")
    parser.add_argument(
        "--fixed-lambda",
        type=float,
        default=None,
        help="use this λ for the hybrid instead of the best one on --gold (test set)",
    )
    parser.add_argument("--out", default="reports/phase3_eval.json")
    args = parser.parse_args()

    from rag.build import SEMANTIC_PARAMS, build_chunks, build_retrievers
    from rag.chunking import describe, plot_sentence_distances
    from rag.embed import DEFAULT_MODEL, QUERY_INSTRUCTIONS, Embedder
    from rag.ingest import load_corpus

    gold_all = load_gold(args.gold)
    documents = load_corpus()
    print(f"{len(gold_all)} gold questions, {len(documents)} documents")
    embedder = Embedder(DEFAULT_MODEL)
    instruction = QUERY_INSTRUCTIONS[args.instruction]
    report: dict = {
        "gold": args.gold,
        "instruction": args.instruction,
        "chunking": {},
        "retrievers": {},
        "lambda_sweep": {},
        "best_lambda": {},
        "per_question": {},
    }

    sweeps = {}
    for strategy in args.strategies:
        print(f"== {strategy} chunking")
        # "semantic:0.75" = semantic chunking with that distance threshold (threshold sweep).
        name, _, threshold = strategy.partition(":")
        params = {"distance_threshold": float(threshold)} if threshold else None
        chunks = build_chunks(name, documents, embedder, params=params)
        report["chunking"][strategy] = describe(chunks)
        gold = resolve(gold_all, chunks, documents)
        report["chunking"][strategy]["mean_relevant_chunks_per_question"] = float(
            np.mean([len(g["relevant_chunk_ids"]) for g in gold])
        )
        retrievers = build_retrievers(chunks, embedder, instruction=instruction)
        rows, best = lambda_sweep(retrievers["bm25"], retrievers["dense"], gold, args.lambdas)
        sweeps[strategy] = rows
        report["lambda_sweep"][strategy] = rows
        report["best_lambda"][strategy] = best
        # On the test set λ is fixed beforehand (--fixed-lambda): the sweep is reported for
        # information but nothing is chosen with the questions we report on.
        if args.fixed_lambda is not None:
            report["best_lambda"][strategy] = args.fixed_lambda
        lam = report["best_lambda"][strategy]
        retrievers["hybrid"].weight_bm25 = lam
        report["retrievers"][strategy] = {}
        for name, retriever in retrievers.items():
            metrics = evaluate_retriever(retriever, gold)
            report["retrievers"][strategy][name] = metrics
            print(
                f"  {name:>6}: recall@5={metrics['recall@5']:.3f} hit@5={metrics['hit@5']:.3f} "
                f"mrr={metrics['mrr']:.3f}" + (f" (λ={lam})" if name == "hybrid" else "")
            )
        report["per_question"][strategy] = [
            {
                "id": g["id"],
                "question": g["question"],
                "relevant": g["relevant_chunk_ids"],
                **{
                    f"rank_{name}": _rank_of_first_relevant(
                        [h.chunk.id for h in r.search(g["question"], top_k=20)],
                        set(g["relevant_chunk_ids"]),
                    )
                    for name, r in retrievers.items()
                },
            }
            for g in gold
        ]

        if strategy == args.strategies[0]:
            # Ablations on the first strategy only: query instruction and context headers.
            report["instructions"] = {}
            for key, text in QUERY_INSTRUCTIONS.items():
                dense = build_retrievers(chunks, embedder, instruction=text)["dense"]
                report["instructions"][key] = evaluate_retriever(dense, gold)
                print(f"  dense, instruction={key}: mrr={report['instructions'][key]['mrr']:.3f}")
            plain = build_retrievers(chunks, embedder, use_context=False, instruction=instruction)
            report["no_context"] = {
                name: evaluate_retriever(plain[name], gold) for name in ("bm25", "dense")
            }
            print(
                "  without context headers: "
                + ", ".join(f"{n} mrr={m['mrr']:.3f}" for n, m in report["no_context"].items())
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_lambda(sweeps, out.with_name(f"{out.stem}_lambda.png"))
    by_source = {d.source: d for d in documents}
    if args.distance_doc in by_source:
        plot_sentence_distances(
            by_source[args.distance_doc],
            embedder.embed_passages,
            SEMANTIC_PARAMS["distance_threshold"],
            out.with_name(f"{out.stem}_distances.png"),
        )
    if args.generation:
        report["generation"] = evaluate_generation(gold_all, args.top_k, args.retriever)
        print(json.dumps(report["generation"]["summary"], indent=2))
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
