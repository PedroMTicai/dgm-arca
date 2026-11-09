"""Corpus → chunks → retrievers, shared by ``RagPipeline`` (the API) and ``rag.evaluate``.

Semantic chunking embeds every sentence of the corpus, which takes minutes on CPU, so its
result is cached as JSON under ``rag/index/`` keyed by the corpus, the embedding model and the
parameters. The dense index is cached the same way by ``DenseRetriever``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from rag.chunking import Chunk, recursive_chunks, semantic_chunks
from rag.ingest import Document

INDEX_DIR = Path(__file__).parent / "index"

# Defaults chosen with rag/evaluate.py on the gold set (see EXPERIMENTS.md, phase 3).
RECURSIVE_PARAMS = {"chunk_size": 800, "chunk_overlap": 100}
# Threshold: consecutive-sentence distances with Qwen3-Embedding-0.6B on the sertraline sheet have
# median 0.54 and 75th percentile 0.65 (EXPERIMENTS.md); 0.25 would make almost every sentence
# its own chunk. The sweep 0.65 / 0.75 / 0.85 on the gold set picked 0.75 (EXPERIMENTS.md).
SEMANTIC_PARAMS = {"distance_threshold": 0.75, "max_chunk_chars": 1500}


def _corpus_key(documents: list[Document], *parts: object) -> str:
    digest = hashlib.sha1()
    for doc in documents:
        digest.update(doc.source.encode())
        digest.update(doc.text.encode("utf-8"))
    for part in parts:
        digest.update(json.dumps(part, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:16]


def build_chunks(
    strategy: str,
    documents: list[Document],
    embedder=None,
    params: dict | None = None,
    cache_dir: Path = INDEX_DIR,
) -> list[Chunk]:
    """``"recursive"`` or ``"semantic"`` chunks of the corpus (semantic ones cached)."""
    if strategy == "recursive":
        return recursive_chunks(documents, **{**RECURSIVE_PARAMS, **(params or {})})
    if strategy != "semantic":
        raise ValueError(f"unknown chunking strategy {strategy!r}")
    if embedder is None:
        raise ValueError("semantic chunking needs an embedder")
    params = {**SEMANTIC_PARAMS, **(params or {})}
    path = cache_dir / f"semantic_{_corpus_key(documents, embedder.model_id, params)}.json"
    if path.exists():
        return [Chunk(**row) for row in json.loads(path.read_text(encoding="utf-8"))]
    chunks = semantic_chunks(documents, embedder.embed_passages, **params)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(c) for c in chunks], ensure_ascii=False), encoding="utf-8")
    return chunks


def build_retrievers(
    chunks: list[Chunk],
    embedder,
    weight_bm25: float = 0.5,
    use_context: bool = True,
    instruction: str | None = None,
    cache_dir: Path = INDEX_DIR,
) -> dict:
    """The three retrievers over the same chunks, keyed by the names ``/rag`` accepts."""
    from rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever

    bm25 = BM25Retriever(chunks, use_context=use_context)
    dense = DenseRetriever(
        chunks, embedder, persist_dir=cache_dir, use_context=use_context, instruction=instruction
    )
    return {"bm25": bm25, "dense": dense, "hybrid": HybridRetriever(bm25, dense, weight_bm25)}
