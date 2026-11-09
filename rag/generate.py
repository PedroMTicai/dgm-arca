"""Retrieval and generation: augment the prompt with the retrieved chunks and answer with citations.

``RagPipeline.answer`` is what ``POST /rag`` calls:

1. retrieve ``top_k`` chunks with the requested retriever (dense, bm25 or hybrid);
2. build the augmented prompt (``RAG_PROMPT_TEMPLATE``, the one from class, with chunk ids so
   the model can cite them, plus the instruction to answer in the question's language and the
   fixed sentence to use when the passages do not contain the answer);
3. generate the answer with the shared chat model (``common.llm``);
4. extract the citations, check that every cited id exists among the returned chunks
   (``invalid_citations`` lists the ones that do not), and append the safety notice that a
   health-related answer must carry.

Configuration (environment variables, all optional):

* ``ARCA_RAG_CHUNKING``     ``recursive`` (default) or ``semantic``
* ``ARCA_RAG_LAMBDA``       BM25 weight of the hybrid retriever (default in ``DEFAULT_LAMBDA``)
* ``ARCA_RAG_INSTRUCTION``  query instruction key from ``rag.embed.QUERY_INSTRUCTIONS``
* ``ARCA_RAG_MIN_SCORE``    abstain without generating when the best dense similarity is
  below this value (default 0: always generate and let the prompt decide)
* ``ARCA_EMBED_MODEL``      embedding model (default Qwen3-Embedding-0.6B)
* ``ARCA_LLM_MODEL``        generator (default Qwen3-1.7B, see ``common/llm.py``)

If ``rag/corpus/`` is empty (a clean machine), ``load`` runs ``rag.download_corpus`` first.
"""

from __future__ import annotations

import os
import re

from api.schemas import RagResponse, RetrievedChunk
from rag.retriever import Hit

RAG_PROMPT_TEMPLATE = """Use the provided up-to-date information to answer the question.
Cite the passages you rely on by writing their id in square brackets, like [{example_id}].
Every statement must be supported by a cited passage. Answer in the language of the question.
If the information is not enough to answer, say so instead of guessing: answer exactly
"{abstention}"

Retrieved information:
{chunks}

Question: {question}
"""

ABSTENTION = "No encuentro la respuesta en los documentos disponibles."
SAFETY_NOTICE = (
    "Aviso: información orientativa extraída de fichas técnicas y guías de farmacovigilancia; "
    "no sustituye el criterio de un profesional sanitario."
)
# Best MRR on the gold set (recursive chunking, λ sweep in reports/phase3_eval.json).
DEFAULT_LAMBDA = 0.3
CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_-]+)\]")


def format_chunks(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{hit.chunk.id}] (from {hit.chunk.context or hit.chunk.source})\n{hit.chunk.text}"
        for hit in hits
    )


def build_prompt(question: str, hits: list[Hit]) -> str:
    example_id = hits[0].chunk.id if hits else "chunk-id"
    return RAG_PROMPT_TEMPLATE.format(
        example_id=example_id,
        chunks=format_chunks(hits),
        question=question,
        abstention=ABSTENTION,
    )


def extract_citations(answer: str, valid_ids: set[str]) -> list[str]:
    """Ids cited in the answer that actually exist among the retrieved chunks, in order."""
    seen: list[str] = []
    for match in CITATION_PATTERN.findall(answer):
        if match in valid_ids and match not in seen:
            seen.append(match)
    return seen


def invalid_citations(answer: str, valid_ids: set[str]) -> list[str]:
    """Bracketed ids in the answer that are *not* among the retrieved chunks (hallucinated)."""
    bad: list[str] = []
    for match in CITATION_PATTERN.findall(answer):
        if match not in valid_ids and match not in bad:
            bad.append(match)
    return bad


def is_abstention(answer: str) -> bool:
    """The answer *is* the abstention sentence (it starts with it).

    Containing it is not enough: Qwen3-1.7B sometimes writes an uncited, invented answer and
    then appends the sentence ("hedged hallucination"), which must not count as abstaining.
    """
    return answer.strip().lower().startswith(ABSTENTION.lower().rstrip("."))


def guard_abstention(text: str, valid_ids: set[str]) -> str:
    """Hedged hallucination → plain abstention.

    If the model says it cannot find the answer *and* cites no retrieved passage, whatever
    else it wrote is not grounded in the corpus, so only the abstention is returned.
    """
    hedged = ABSTENTION.lower().rstrip(".") in text.lower()
    if hedged and not extract_citations(text, valid_ids):
        return ABSTENTION
    return text


def to_response(answer: str, hits: list[Hit], retriever: str, model: str = "") -> RagResponse:
    chunks = [
        RetrievedChunk(id=h.chunk.id, source=h.chunk.source, score=h.score, text=h.chunk.text)
        for h in hits
    ]
    valid = {c.id for c in chunks}
    return RagResponse(
        answer=answer,
        chunks=chunks,
        citations=extract_citations(answer, valid),
        invalid_citations=invalid_citations(answer, valid),
        retriever=retriever,
        model=model,
    )


class RagPipeline:
    """Everything ``POST /rag`` needs, built once and reused."""

    def __init__(self) -> None:
        self.retrievers: dict = {}
        self.llm = None

    def load(self, load_generator: bool = True) -> None:
        """Ingest → chunk → the three retrievers → the generator model."""
        from rag import download_corpus
        from rag.build import build_chunks, build_retrievers
        from rag.embed import DEFAULT_MODEL, QUERY_INSTRUCTIONS, Embedder
        from rag.ingest import load_corpus

        if os.environ.get("ARCA_RAG_DISABLED"):
            # Used by the test suite: /health would otherwise embed the whole corpus.
            raise NotImplementedError("Phase 3 disabled by ARCA_RAG_DISABLED.")
        documents = load_corpus()
        if not documents:
            download_corpus.main([])
            documents = load_corpus()
        if not documents:
            raise RuntimeError("rag/corpus/ is empty and the download failed")

        self.embedder = Embedder(os.environ.get("ARCA_EMBED_MODEL") or DEFAULT_MODEL)
        self.chunking = os.environ.get("ARCA_RAG_CHUNKING", "recursive")
        chunks = build_chunks(self.chunking, documents, self.embedder)
        instruction_key = os.environ.get("ARCA_RAG_INSTRUCTION", "pharmacovigilance")
        self.retrievers = build_retrievers(
            chunks,
            self.embedder,
            weight_bm25=float(os.environ.get("ARCA_RAG_LAMBDA", DEFAULT_LAMBDA)),
            instruction=QUERY_INSTRUCTIONS[instruction_key],
        )
        self.min_score = float(os.environ.get("ARCA_RAG_MIN_SCORE", "0"))
        if load_generator:
            from common.llm import get_chat_model

            self.llm = get_chat_model()

    def retrieve(self, question: str, top_k: int, retriever: str) -> list[Hit]:
        if retriever not in self.retrievers:
            raise ValueError(f"unknown retriever {retriever!r}; use {sorted(self.retrievers)}")
        return self.retrievers[retriever].search(question, top_k=top_k)

    def answer(self, question: str, top_k: int = 5, retriever: str = "hybrid") -> RagResponse:
        """Retrieve → ``build_prompt`` → generate → ``to_response``."""
        from common.llm import strip_thinking

        hits = self.retrieve(question, top_k, retriever)
        best_dense = float(self.retrievers["dense"].scores(question).max()) if self.min_score else 1
        if best_dense < self.min_score:
            text = ABSTENTION  # nothing in the corpus is close enough: do not let the model guess
        else:
            messages = [{"role": "user", "content": build_prompt(question, hits)}]
            raw, _ = self.llm.generate(messages, max_new_tokens=512)
            text = guard_abstention(strip_thinking(raw), {h.chunk.id for h in hits})
        return to_response(f"{text}\n\n{SAFETY_NOTICE}", hits, retriever, model=self.llm.name)


_pipeline: RagPipeline | None = None


def build_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        pipeline = RagPipeline()
        pipeline.load()
        _pipeline = pipeline
    return _pipeline


def answer_with_rag(question: str, top_k: int = 5, retriever: str = "hybrid") -> RagResponse:
    """Convenience entry point for scripts and the evaluation."""
    return build_pipeline().answer(question, top_k=top_k, retriever=retriever)


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "¿Con qué frecuencia produce somnolencia la sertralina?"
    print(answer_with_rag(question).model_dump_json(indent=2))
