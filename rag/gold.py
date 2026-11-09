"""The gold set, annotated with quotes instead of chunk ids.

A chunk id (``<doc_id>-<index>``) only exists for one chunking configuration: change the
chunk size, or switch to semantic chunking, and every id moves. Since we must compare
chunking strategies on the same questions, each gold question names the document and one or
more literal quotes (``evidence``) that answer it::

    {"id": "q01", "question": "¿Con qué frecuencia produce somnolencia la sertralina?",
     "source": "fichas/sertralina.md", "evidence": ["mareo, cefalea * , somnolencia"],
     "answer": "Muy frecuente (≥1/10)", "type": "ficha"}

``relevant_chunk_ids`` resolves those quotes against any list of chunks: a chunk is relevant
when it covers at least half of a quote's characters (or the whole chunk lies inside the
quote). Quotes are matched ignoring case and runs of whitespace, because PDF extraction
breaks lines in arbitrary places.

Questions of ``"type": "unanswerable"`` have no evidence: the corpus cannot answer them. They
are left out of Recall@k and MRR and used to check that ``/rag`` says it does not know.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rag.chunking import Chunk
from rag.ingest import Document


def _normalised_with_map(text: str) -> tuple[str, list[int]]:
    """Casefolded text with whitespace runs collapsed, and the original offset of each char."""
    out, offsets = [], []
    previous_space = False
    for i, char in enumerate(text):
        if char.isspace():
            if previous_space:
                continue
            out.append(" ")
            previous_space = True
        else:
            out.append(char.casefold())
            previous_space = False
        offsets.append(i)
    return "".join(out), offsets


def find_span(text: str, quote: str) -> tuple[int, int] | None:
    """``(start, end)`` of ``quote`` in ``text`` up to case and whitespace, or ``None``."""
    haystack, offsets = _normalised_with_map(text)
    needle = re.sub(r"\s+", " ", quote.strip()).casefold()
    position = haystack.find(needle)
    if position < 0 or not needle:
        return None
    return offsets[position], offsets[position + len(needle) - 1] + 1


def load_gold(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evidence_spans(item: dict, documents: dict[str, Document]) -> list[tuple[str, int, int]]:
    """``(doc_id, start, end)`` of every quote of a gold item. Raises if a quote is not found:
    a gold set that silently drops evidence would inflate nothing and measure nothing."""
    doc = documents.get(item["source"])
    if doc is None:
        raise KeyError(f"{item['id']}: source {item['source']!r} is not in the corpus")
    spans = []
    for quote in item.get("evidence", []):
        span = find_span(doc.text, quote)
        if span is None:
            raise ValueError(f"{item['id']}: quote not found in {item['source']}: {quote!r}")
        spans.append((doc.id, *span))
    return spans


def relevant_chunk_ids(
    item: dict, chunks: list[Chunk], documents: dict[str, Document], min_cover: float = 0.5
) -> set[str]:
    """Ids of the chunks that contain (enough of) the item's evidence."""
    relevant = set()
    for doc_id, start, end in evidence_spans(item, documents):
        length = end - start
        for chunk in chunks:
            if chunk.doc_id != doc_id or chunk.start < 0:
                continue
            overlap = min(end, chunk.end) - max(start, chunk.start)
            inside = chunk.start >= start and chunk.end <= end
            if overlap > 0 and (overlap >= min_cover * length or inside):
                relevant.add(chunk.id)
    return relevant


def resolve(gold: list[dict], chunks: list[Chunk], documents: list[Document]) -> list[dict]:
    """Gold items with evidence, each with ``relevant_chunk_ids`` for this chunking."""
    by_source = {d.source: d for d in documents}
    resolved = []
    for item in gold:
        if item.get("type") == "unanswerable" or not item.get("evidence"):
            continue
        ids = item.get("relevant_chunk_ids") or relevant_chunk_ids(item, chunks, by_source)
        resolved.append({**item, "relevant_chunk_ids": sorted(ids)})
    return resolved


def check(path: str | Path = "rag/gold/gold.jsonl") -> None:
    """Validate every quote of the gold set against the corpus and print a summary."""
    from collections import Counter

    from rag.ingest import load_corpus

    gold = load_gold(path)
    by_source = {d.source: d for d in load_corpus()}
    problems = 0
    for item in gold:
        if item.get("type") == "unanswerable":
            continue
        try:
            evidence_spans(item, by_source)
        except (KeyError, ValueError) as exc:
            problems += 1
            print(f"✗ {exc}")
    print(
        f"{len(gold)} questions, {problems} with problems; by type: "
        f"{dict(Counter(item.get('type', '?') for item in gold))}"
    )


if __name__ == "__main__":
    import sys

    check(sys.argv[1] if len(sys.argv) > 1 else "rag/gold/gold.jsonl")
