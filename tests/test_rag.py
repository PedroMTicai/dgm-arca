"""Phase 3 pieces that can be tested without models or network."""

import numpy as np
import pytest

from rag.chunking import Chunk, chunk_context, semantic_chunks, sentence_spans
from rag.download_corpus import html_to_text
from rag.evaluate import hit_at_k, recall_at_k, reciprocal_rank
from rag.generate import extract_citations, invalid_citations, is_abstention
from rag.gold import find_span, relevant_chunk_ids
from rag.ingest import Document


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=2) == 0.5
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=3) == 1.0
    assert recall_at_k(["x"], {"a"}, k=5) == 0.0


def test_hit_at_k_needs_only_one_relevant_chunk():
    assert hit_at_k(["a", "b", "c"], {"c", "z"}, k=3) == 1.0
    assert hit_at_k(["a", "b", "c"], {"c", "z"}, k=2) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a", "b"], {"a"}) == 0.5
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_extract_citations_keeps_only_valid_ids_in_order():
    answer = "The axolotl regrows limbs [doc1-0001]. See also [doc1-0001] and [ghost-9999]."
    assert extract_citations(answer, {"doc1-0001", "doc1-0002"}) == ["doc1-0001"]


def test_invalid_citations_are_reported():
    answer = "Somnolencia muy frecuente [d1-0003], según [ghost-0001] y [d1-0003]."
    assert invalid_citations(answer, {"d1-0003"}) == ["ghost-0001"]


def test_abstention_is_detected():
    assert is_abstention("No encuentro la respuesta en los documentos disponibles.\n\nAviso: …")
    assert not is_abstention("La somnolencia es muy frecuente [d1-0003].")


def test_recursive_chunking_respects_size_and_offsets():
    pytest.importorskip("langchain_text_splitters")
    from rag.chunking import describe, recursive_chunks

    doc = Document(id="d1", source="notes.md", text=("Sentence one. " * 40 + "\n\n") * 5)
    chunks = recursive_chunks([doc], chunk_size=200, chunk_overlap=20)
    stats = describe(chunks)
    assert stats["n_chunks"] > 5
    assert stats["max_chars"] <= 200
    assert chunks[0].id == "d1-0000"
    assert all(doc.text[c.start : c.end] == c.text for c in chunks)


def test_sentence_spans_split_long_tables_by_line():
    text = "Primera frase. Segunda frase.\n\n" + "\n".join(f"fila {i}: x" * 20 for i in range(5))
    spans = sentence_spans(text, max_sentence_chars=300)
    assert text[spans[0][0] : spans[0][1]] == "Primera frase."
    assert all(e - s <= 300 for s, e in spans)


def test_semantic_chunks_break_where_topics_change():
    text = "Gato uno. Gato dos. Gato tres. Coche uno. Coche dos."
    doc = Document(id="d1", source="t.md", text=text)

    def embed_fn(sentences):  # orthogonal one-hot vectors per topic
        return np.array([[1.0, 0.0] if "Gato" in s else [0.0, 1.0] for s in sentences])

    chunks = semantic_chunks([doc], embed_fn, distance_threshold=0.5)
    assert [c.text for c in chunks] == ["Gato uno. Gato dos. Gato tres.", "Coche uno. Coche dos."]
    assert all(doc.text[c.start : c.end] == c.text for c in chunks)
    # The size limit closes a chunk even inside one topic.
    small = semantic_chunks([doc], embed_fn, distance_threshold=0.5, max_chunk_chars=20)
    assert all(len(c.text) <= 20 for c in small)


def test_chunk_context_uses_title_and_last_heading():
    text = (
        "# Ficha de sertralina\n\n## 4.1. Indicaciones\n\nDepresión.\n\n"
        "## 4.8. Reacciones\n\nSueño."
    )
    doc = Document(id="d1", source="fichas/sertralina.md", text=text)
    assert chunk_context(doc, text.index("Sueño")) == "Ficha de sertralina · 4.8. Reacciones"
    assert chunk_context(doc, text.index("Depresión")) == "Ficha de sertralina · 4.1. Indicaciones"


def test_find_span_ignores_case_and_whitespace():
    text = "Trastornos del sistema nervioso:\n  Muy frecuentes: mareo, SOMNOLENCIA"
    start, end = find_span(text, "muy frecuentes: mareo, somnolencia")
    assert text[start:end] == "Muy frecuentes: mareo, SOMNOLENCIA"
    assert find_span(text, "no está") is None


def test_relevant_chunks_follow_the_quote_across_chunkings():
    text = "AAAA. La somnolencia es muy frecuente. BBBB."
    doc = Document(id="d1", source="f.md", text=text)
    quote_start = text.index("La somnolencia")
    small = [  # the quote is split across two chunks, most of it in the second one
        Chunk("d1-0", "d1", "f.md", text[: quote_start + 5], 0, 0),
        Chunk("d1-1", "d1", "f.md", text[quote_start + 5 :], 1, quote_start + 5),
    ]
    whole = [Chunk("d1-s0", "d1", "f.md", text, 0, 0)]
    item = {"id": "q", "source": "f.md", "evidence": ["la somnolencia es muy frecuente"]}
    assert relevant_chunk_ids(item, small, {"f.md": doc}) == {"d1-1"}
    assert relevant_chunk_ids(item, whole, {"f.md": doc}) == {"d1-s0"}


def test_missing_quote_is_an_error():
    doc = Document(id="d1", source="f.md", text="texto")
    with pytest.raises(ValueError):
        relevant_chunk_ids(
            {"id": "q", "source": "f.md", "evidence": ["otra cosa"]}, [], {"f.md": doc}
        )


class _FixedRetriever:
    def __init__(self, chunks, scores):
        self.chunks, self._scores = chunks, np.asarray(scores, dtype=np.float32)

    def scores(self, query):
        return self._scores


def test_hybrid_weight_moves_between_bm25_and_dense():
    from rag.retriever import HybridRetriever

    chunks = [Chunk(f"c{i}", "d", "s", f"t{i}", i, 0) for i in range(3)]
    bm25 = _FixedRetriever(chunks, [10.0, 0.0, 7.0])  # c0 first by keywords
    dense = _FixedRetriever(chunks, [0.1, 0.9, 0.7])  # c1 first by meaning
    assert HybridRetriever(bm25, dense, weight_bm25=1.0).search("q", 1)[0].chunk.id == "c0"
    assert HybridRetriever(bm25, dense, weight_bm25=0.0).search("q", 1)[0].chunk.id == "c1"
    # c2 is second in both lists: a balanced λ ranks it first.
    assert HybridRetriever(bm25, dense, weight_bm25=0.5).search("q", 1)[0].chunk.id == "c2"
    hits = HybridRetriever(bm25, dense, weight_bm25=0.5).search("q", 3)
    assert all(0.0 <= h.score <= 1.0 for h in hits)


def test_html_tables_keep_frequency_next_to_each_reaction():
    html = (
        "<p>Tabla:</p><table><tr><td>Órgano</td><td>Muy frecuentes (≥1/10)</td>"
        "<td>Frecuentes (≥1/100 a <1/10)</td></tr>"
        "<tr><td>Trastornos del sistema nervioso</td><td><p>mareo,</p><p>somnolencia</p></td>"
        "<td>temblor</td></tr></table>"
    )
    text = html_to_text(html)
    assert (
        "Trastornos del sistema nervioso: Muy frecuentes: mareo, somnolencia; "
        "Frecuentes: temblor" in text
    )
    assert "<1/10" in text  # a literal "<" is not a tag


class _FakeLLM:
    name = "fake-llm"

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def generate(self, messages, max_new_tokens=512):
        self.prompts.append(messages[-1]["content"])
        return self.reply, 10


def test_pipeline_answer_cites_and_flags_invented_ids():
    from rag.generate import SAFETY_NOTICE, RagPipeline
    from rag.retriever import Hit

    chunk = Chunk(
        "d1-0003",
        "d1",
        "fichas/sertralina.md",
        "Muy frecuentes: somnolencia",
        3,
        0,
        "Ficha de sertralina · 4.8. Reacciones adversas",
    )

    class _Retriever:
        def search(self, query, top_k=5):
            return [Hit(chunk, 0.9)]

    pipeline = RagPipeline()
    pipeline.retrievers = {"hybrid": _Retriever(), "dense": _Retriever()}
    pipeline.min_score = 0.0
    pipeline.llm = _FakeLLM("<think>x</think>Es muy frecuente [d1-0003] [made-up-1].")
    response = pipeline.answer("¿Somnolencia con sertralina?", top_k=1, retriever="hybrid")
    assert response.citations == ["d1-0003"]
    assert response.invalid_citations == ["made-up-1"]
    assert "<think>" not in response.answer and response.answer.endswith(SAFETY_NOTICE)
    assert (
        "[d1-0003] (from Ficha de sertralina · 4.8. Reacciones adversas)" in pipeline.llm.prompts[0]
    )


def test_hedged_hallucination_becomes_plain_abstention():
    from rag.generate import ABSTENTION, guard_abstention

    invented = f"La insulina glargina causa náuseas y vómitos.\n\n{ABSTENTION}"
    assert not is_abstention(invented)  # contains the sentence but does not abstain
    assert guard_abstention(invented, {"d1-0001"}) == ABSTENTION
    cited = f"Se notifica en 15 días [d1-0001]. {ABSTENTION}"
    assert guard_abstention(cited, {"d1-0001"}) == cited  # grounded content is kept
