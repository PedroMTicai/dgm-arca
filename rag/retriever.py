"""Retrieval: similarity search, keyword search (BM25) and the hybrid of both.

All three retrievers return the same ``Hit`` objects so that ``rag/generate.py`` and
``rag/evaluate.py`` do not care which one is used. All three index ``Chunk.indexed_text``
(context line + chunk), see ``rag/chunking.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.chunking import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _top(scores: np.ndarray, chunks: list[Chunk], top_k: int) -> list[Hit]:
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [Hit(chunks[i], float(scores[i])) for i in order]


class BM25Retriever:
    """Keyword ranking: term frequency, inverse document frequency, length normalisation."""

    name = "bm25"

    def __init__(self, chunks: list[Chunk], use_context: bool = True):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        texts = [c.indexed_text if use_context else c.text for c in chunks]
        self.index = BM25Okapi([tokenize(t) for t in texts])

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every chunk, in ``self.chunks`` order."""
        return np.asarray(self.index.get_scores(tokenize(query)), dtype=np.float32)

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        return _top(self.scores(query), self.chunks, top_k)


class DenseRetriever:
    """Embed the chunks once, store them, embed the query and rank by cosine similarity.

    The store is a plain NumPy matrix (one L2-normalised row per chunk) saved under
    ``persist_dir`` as ``dense_<key>.npy``. The key hashes the model id and the indexed texts,
    so a change in the corpus, the chunking or the context lines builds a new index instead of
    silently reusing a stale one. A few thousand chunks fit in memory comfortably and a
    matrix product is exact, so a vector database (Chroma) would add a dependency and an
    approximate search without buying anything at this size.
    """

    name = "dense"

    def __init__(
        self,
        chunks: list[Chunk],
        embedder,
        persist_dir: str | Path = "rag/index",
        use_context: bool = True,
        instruction: str | None = None,
    ):
        from rag.embed import DEFAULT_QUERY_INSTRUCTION

        self.chunks = chunks
        self.embedder = embedder
        # "" is a valid choice (bare query, for the ablation), so only None means default.
        self.instruction = DEFAULT_QUERY_INSTRUCTION if instruction is None else instruction
        texts = [c.indexed_text if use_context else c.text for c in chunks]
        digest = hashlib.sha1(embedder.model_id.encode())
        for text in texts:
            digest.update(text.encode("utf-8"))
        persist_dir = Path(persist_dir)
        path = persist_dir / f"dense_{digest.hexdigest()[:16]}.npy"
        if path.exists():
            self.matrix = np.load(path)
        else:
            self.matrix = np.asarray(embedder.embed_passages(texts), dtype=np.float32)
            persist_dir.mkdir(parents=True, exist_ok=True)
            np.save(path, self.matrix)
            meta = {"model": embedder.model_id, "n_chunks": len(chunks), "first": chunks[0].id}
            path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
        self.index_path = path

    def scores(self, query: str) -> np.ndarray:
        """Cosine similarity of every chunk with the query (embedded with the instruction)."""
        query_vec = self.embedder.embed_queries([query], instruction=self.instruction)[0]
        return self.matrix @ np.asarray(query_vec, dtype=np.float32)

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        return _top(self.scores(query), self.chunks, top_k)


def min_max(values: np.ndarray) -> np.ndarray:
    """Rescale to [0, 1]; a constant vector becomes all zeros (it carries no ranking)."""
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return (values - low) / (high - low)


class HybridRetriever:
    """``s = λ · s_bm25 + (1 - λ) · s_dense`` on scores rescaled to [0, 1].

    The candidates are the union of the top ``pool`` chunks of each retriever. Both scores
    are known for every candidate (BM25 and the dense similarity are computed for the whole
    corpus anyway), so a chunk that only one retriever ranks high still gets its real score
    from the other one instead of a zero. Min-max is applied over the candidates, so λ
    weighs the two rankings on the same scale. λ is chosen on the gold set
    (``rag/evaluate.py``), not by intuition.
    """

    name = "hybrid"

    def __init__(
        self, bm25: BM25Retriever, dense: DenseRetriever, weight_bm25: float = 0.5, pool: int = 20
    ):
        if [c.id for c in bm25.chunks] != [c.id for c in dense.chunks]:
            raise ValueError("BM25 and dense retrievers must index the same chunks")
        self.bm25, self.dense, self.weight_bm25, self.pool = bm25, dense, weight_bm25, pool
        self.chunks = bm25.chunks

    def fuse(self, bm25_scores: np.ndarray, dense_scores: np.ndarray) -> list[tuple[int, float]]:
        """(chunk index, hybrid score) of every candidate, best first."""
        pool = min(self.pool, len(bm25_scores))
        candidates = np.union1d(
            np.argsort(-bm25_scores, kind="stable")[:pool],
            np.argsort(-dense_scores, kind="stable")[:pool],
        )
        weight = self.weight_bm25
        fused = weight * min_max(bm25_scores[candidates]) + (1 - weight) * min_max(
            dense_scores[candidates]
        )
        order = np.argsort(-fused, kind="stable")
        return [(int(candidates[i]), float(fused[i])) for i in order]

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        fused = self.fuse(self.bm25.scores(query), self.dense.scores(query))
        return [Hit(self.chunks[i], score) for i, score in fused[:top_k]]
