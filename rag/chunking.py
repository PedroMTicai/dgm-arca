"""Data indexing, step 2: chunking.

Two strategies from class. ``recursive_chunks`` is the default of most RAG systems (LangChain's
splitter). ``semantic_chunks`` groups consecutive sentences while their embeddings stay close.
Compare both with ``rag/evaluate.py``: number of chunks, average length, and Recall@k on the
gold set.

Every chunk keeps its character offset in the document (``start``) so that the gold set can
be annotated with a literal quote instead of a chunk id: the same gold set then scores both
strategies (``rag.gold``).

Every chunk also carries a ``context`` line: the document title and the section heading it
falls under. Our corpus has 35 technical sheets with the same structure, and a chunk from the
middle of section 4.8 ("Trastornos del sistema nervioso: Muy frecuentes: mareo, cefalea,
somnolencia") does not say which drug it belongs to. Retrievers index ``context + text``
(``Chunk.indexed_text``), which is what the "contextual chunk headers" idea does; the
evaluation measures it with and without.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from rag.ingest import Document

EmbedFn = Callable[[list[str]], np.ndarray]

_HEADING = re.compile(r"^#{1,3} +(?P<title>.+)$", re.MULTILINE)
# A sentence ends at . ! ? followed by whitespace, or at a blank line.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n\s*\n")


@dataclass
class Chunk:
    id: str
    doc_id: str
    source: str
    text: str
    index: int  # position of the chunk within its document
    start: int = -1  # character offset of ``text`` in the document (-1 if unknown)
    context: str = ""  # document title and section heading, indexed together with the text

    @property
    def end(self) -> int:
        return self.start + len(self.text) if self.start >= 0 else -1

    @property
    def indexed_text(self) -> str:
        """What retrievers see: the context line, then the chunk."""
        return f"{self.context}\n{self.text}" if self.context else self.text


def document_title(doc: Document) -> str:
    """The first Markdown heading, or the file name for PDFs and plain text."""
    match = _HEADING.search(doc.text[:2000])
    if match:
        return match.group("title").strip()
    name = doc.source.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return name.replace("_", " ")


def chunk_context(doc: Document, start: int, title: str | None = None) -> str:
    """``"<document title> · <last section heading before start>"``."""
    title = title or document_title(doc)
    section = None
    for match in _HEADING.finditer(doc.text, 0, max(start, 0) + 1):
        if match.start() <= start:
            section = match.group("title").strip()
    if section and section != title:
        return f"{title} · {section}"
    return title


def recursive_chunks(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", ""),
) -> list[Chunk]:
    """Split by paragraphs, then lines, then sentences... until every chunk fits ``chunk_size``."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        separators=list(separators),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = []
    for doc in documents:
        title = document_title(doc)
        for i, piece in enumerate(splitter.create_documents([doc.text])):
            start = piece.metadata.get("start_index", -1)
            chunks.append(
                Chunk(
                    id=f"{doc.id}-{i:04d}",
                    doc_id=doc.id,
                    source=doc.source,
                    text=piece.page_content,
                    index=i,
                    start=start,
                    context=chunk_context(doc, start, title),
                )
            )
    return chunks


def sentence_spans(text: str, max_sentence_chars: int = 600) -> list[tuple[int, int]]:
    """``(start, end)`` offsets of the sentences of ``text``.

    Sentences end at ``.!?`` or a blank line. A "sentence" longer than ``max_sentence_chars``
    (a table, or a PDF paragraph without full stops) is split again at line breaks, and if a
    single line is still too long, cut at that length.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in [*_SENTENCE_BREAK.finditer(text), None]:
        end = match.start() if match else len(text)
        spans.extend(_split_long(text, cursor, end, max_sentence_chars))
        cursor = match.end() if match else len(text)
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _split_long(text: str, start: int, end: int, limit: int) -> list[tuple[int, int]]:
    if end - start <= limit:
        return [(start, end)]
    pieces, cursor = [], start
    for line in re.finditer(r"[^\n]+", text[start:end]):
        line_start, line_end = start + line.start(), start + line.end()
        if line_end - cursor > limit and line_start > cursor:
            pieces.append((cursor, line_start))
            cursor = line_start
    pieces.append((cursor, end))
    out = []
    for s, e in pieces:
        out.extend((i, min(i + limit, e)) for i in range(s, e, limit))
    return out


def split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter; good enough to feed the semantic chunker."""
    return [text[s:e].strip() for s, e in sentence_spans(text)]


def consecutive_distances(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance ``1 - e_i · e_{i+1}`` between consecutive (L2-normalised) rows."""
    if len(embeddings) < 2:
        return np.zeros(0)
    return 1.0 - np.sum(embeddings[:-1] * embeddings[1:], axis=1)


def semantic_chunks(
    documents: list[Document],
    embed_fn: EmbedFn,
    distance_threshold: float = 0.25,
    max_chunk_chars: int = 1500,
) -> list[Chunk]:
    """Group consecutive sentences while their embeddings stay close (the slides' algorithm).

    1. Split the document into sentences s_1 … s_n (``sentence_spans``).
    2. Embed them: e_1 … e_n (``embed_fn`` returns L2-normalised rows).
    3. Start the first chunk with s_1.
    4. For each consecutive pair, compute the cosine distance 1 - e_i · e_{i+1}.
    5. If the distance is <= threshold, append s_{i+1} to the current chunk.
    6. Otherwise close the chunk and start a new one at s_{i+1}.
    7. Also close a chunk when it would exceed ``max_chunk_chars``.

    A chunk is the contiguous slice of the document from its first sentence to its last, so
    its offsets are exact and the whitespace between sentences is preserved.
    """
    chunks: list[Chunk] = []
    for doc in documents:
        spans = sentence_spans(doc.text)
        if not spans:
            continue
        title = document_title(doc)
        distances = consecutive_distances(embed_fn([doc.text[s:e] for s, e in spans]))
        groups: list[tuple[int, int]] = []
        group_start, group_end = spans[0]
        for (start, end), distance in zip(spans[1:], distances, strict=True):
            if distance <= distance_threshold and end - group_start <= max_chunk_chars:
                group_end = end
            else:
                groups.append((group_start, group_end))
                group_start, group_end = start, end
        groups.append((group_start, group_end))
        for i, (start, end) in enumerate(groups):
            chunks.append(
                Chunk(
                    id=f"{doc.id}-s{i:04d}",
                    doc_id=doc.id,
                    source=doc.source,
                    text=doc.text[start:end],
                    index=i,
                    start=start,
                    context=chunk_context(doc, start, title),
                )
            )
    return chunks


def plot_sentence_distances(
    doc: Document, embed_fn: EmbedFn, threshold: float, out_path, max_sentences: int = 150
) -> None:
    """Distances between consecutive sentences of one document with the threshold line.

    Every peak above the line is a chunk boundary: this is the figure that justifies the
    threshold.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spans = sentence_spans(doc.text)[:max_sentences]
    distances = consecutive_distances(embed_fn([doc.text[s:e] for s, e in spans]))
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(range(1, len(distances) + 1), distances, marker=".", linewidth=1)
    ax.axhline(threshold, color="red", linestyle="--", label=f"threshold = {threshold}")
    ax.set_xlabel("sentence i (distance between i and i+1)")
    ax.set_ylabel("cosine distance")
    ax.set_title(f"Semantic chunking: {document_title(doc)[:70]}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def describe(chunks: list[Chunk]) -> dict[str, float]:
    """Numbers for your chunking comparison table."""
    lengths = sorted(len(c.text) for c in chunks)
    return {
        "n_chunks": len(chunks),
        "mean_chars": sum(lengths) / max(len(lengths), 1),
        "median_chars": lengths[len(lengths) // 2] if lengths else 0,
        "min_chars": min(lengths, default=0),
        "max_chars": max(lengths, default=0),
    }
