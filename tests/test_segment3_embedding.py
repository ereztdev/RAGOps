"""
Segment 3: Embedding Engine.
Tests for pluggable backend, deterministic fake backend, and run_embedding_pipeline.
No external services; no vector storage.
"""
from __future__ import annotations

import unittest

from core.schema import Chunk, Document, Embedding, chunk_id_from_components
from pipeline.embedding.embedding_engine import (
    FakeEmbeddingBackend,
    run_embedding_pipeline,
    FAKE_EMBEDDING_VECTOR_SIZE,
    FAKE_EMBEDDING_MODEL_ID,
)


# ---------------------------------------------------------------------------
# Fake backend determinism
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_key: str,
    text: str,
    document_id: str = "a" * 64,
    page_number: int = 1,
    source_type: str = "pdf",
    chunk_index: int = 0,
) -> Chunk:
    chunk_id = chunk_id_from_components(document_id, source_type, page_number, chunk_index)
    return Chunk(
        chunk_id=chunk_id,
        chunk_key=chunk_key,
        document_id=document_id,
        source_type=source_type,
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
    )


class TestFakeBackendDeterminism(unittest.TestCase):
    def test_same_chunks_same_vectors(self) -> None:
        """Same chunk list must yield identical embeddings across calls."""
        backend = FakeEmbeddingBackend()
        chunks = [
            _make_chunk("doc1:1", "hello world"),
            _make_chunk("doc1:2", "goodbye"),
        ]
        out1 = backend.embed_chunks(chunks)
        out2 = backend.embed_chunks(chunks)
        self.assertEqual(len(out1), 2)
        self.assertEqual(len(out2), 2)
        for a, b in zip(out1, out2):
            self.assertEqual(a.chunk_key, b.chunk_key)
            self.assertEqual(a.vector, b.vector)

    def test_different_text_different_vector(self) -> None:
        """Different text must yield different vector for same chunk_key."""
        backend = FakeEmbeddingBackend()
        c1 = _make_chunk("x:1", "alpha")
        c2 = _make_chunk("x:1", "beta")
        out1 = backend.embed_chunks([c1])
        out2 = backend.embed_chunks([c2])
        self.assertNotEqual(out1[0].vector, out2[0].vector)

    def test_different_chunk_key_different_vector(self) -> None:
        """Same text, different chunk_key must yield different vector."""
        backend = FakeEmbeddingBackend()
        c1 = _make_chunk("doc:1", "same text")
        c2 = _make_chunk("doc:2", "same text")
        out1 = backend.embed_chunks([c1])
        out2 = backend.embed_chunks([c2])
        self.assertNotEqual(out1[0].vector, out2[0].vector)

    def test_vector_size_fixed(self) -> None:
        """Fake backend must produce fixed-size vectors (8)."""
        backend = FakeEmbeddingBackend()
        chunks = [_make_chunk("k:1", "any")]
        out = backend.embed_chunks(chunks)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0].vector), FAKE_EMBEDDING_VECTOR_SIZE)
        self.assertEqual(len(out[0].vector), 8)

    def test_embedding_chunk_key_preserved(self) -> None:
        """Embedding chunk_key must match input Chunk chunk_key."""
        backend = FakeEmbeddingBackend()
        chunks = [_make_chunk("abc:42", "text")]
        out = backend.embed_chunks(chunks)
        self.assertEqual(out[0].chunk_key, "abc:42")

    def test_backend_model_id(self) -> None:
        """Fake backend must report embedding_model fake_v1."""
        backend = FakeEmbeddingBackend()
        self.assertEqual(backend.embedding_model, FAKE_EMBEDDING_MODEL_ID)
        self.assertEqual(backend.embedding_model, "fake_v1")


# ---------------------------------------------------------------------------
# run_embedding_pipeline
# ---------------------------------------------------------------------------


class TestRunEmbeddingPipeline(unittest.TestCase):
    def test_empty_document(self) -> None:
        """Document with no chunks must yield empty list of embeddings."""
        doc = Document(document_id="d", source_path="/p", chunks=[])
        backend = FakeEmbeddingBackend()
        result = run_embedding_pipeline(doc, backend)
        self.assertEqual(result, [])

    def test_one_chunk_one_embedding(self) -> None:
        """One chunk must yield one embedding in order."""
        chunk = _make_chunk("d:1", "only")
        doc = Document(document_id="d", source_path="/p", chunks=[chunk])
        backend = FakeEmbeddingBackend()
        result = run_embedding_pipeline(doc, backend)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].chunk_key, "d:1")
        self.assertEqual(len(result[0].vector), FAKE_EMBEDDING_VECTOR_SIZE)

    def test_order_preserved(self) -> None:
        """Embeddings must be in same order as document.chunks."""
        chunks = [
            _make_chunk("d:1", "first", page_number=1),
            _make_chunk("d:2", "second", page_number=2),
            _make_chunk("d:3", "third", page_number=3),
        ]
        doc = Document(document_id="d", source_path="/p", chunks=chunks)
        backend = FakeEmbeddingBackend()
        result = run_embedding_pipeline(doc, backend)
        self.assertEqual([r.chunk_key for r in result], ["d:1", "d:2", "d:3"])


# ---------------------------------------------------------------------------
# Embedding type contract
# ---------------------------------------------------------------------------


class TestEmbeddingContract(unittest.TestCase):
    def test_embedding_is_typed(self) -> None:
        """Backend must return list[Embedding] with chunk_key and vector."""
        backend = FakeEmbeddingBackend()
        chunks = [_make_chunk("k:1", "x")]
        out = backend.embed_chunks(chunks)
        self.assertEqual(len(out), 1)
        e = out[0]
        self.assertIsInstance(e, Embedding)
        self.assertIsInstance(e.chunk_key, str)
        self.assertIsInstance(e.vector, tuple)
        self.assertTrue(all(isinstance(x, float) for x in e.vector))
