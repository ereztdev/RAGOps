"""
Segment 4: Persistent vector index storage.
Tests: save/load determinism, identical retrieval after reload, invalid/missing fields fail loudly.
No mocking of storage layer.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from core.schema import Document
from pipeline.embedding.embedding_engine import FakeEmbeddingBackend, run_embedding_pipeline
from pipeline.ingestion.ingestion_pipeline import run_ingestion_pipeline
from pipeline.retrieval.retrieval_engine import (
    build_index_snapshot,
    retrieve,
)
from storage.vector_index_store import load_index, save_index

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEMANTIC_PDF = PROJECT_ROOT / "data" / "test_pdfs" / "ragops_semantic_test_pdf.pdf"


# ---------------------------------------------------------------------------
# save → load → retrieve yields identical RetrievalResult
# ---------------------------------------------------------------------------


class TestSaveLoadRetrieveIdentity(unittest.TestCase):
    """Save index, load it, run retrieval; result must match retrieval on original in-memory index."""

    def test_save_load_retrieve_yields_identical_result(self) -> None:
        doc = run_ingestion_pipeline(str(SEMANTIC_PDF))
        if not doc.chunks:
            self.skipTest("no chunks in fixture")
        backend = FakeEmbeddingBackend()
        embeddings = run_embedding_pipeline(doc, backend)
        index = build_index_snapshot(doc, embeddings, "v1")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index_v1.json"
            save_index(index, path)
            loaded = load_index(path)

        result_original = retrieve("test query", index, backend, top_k=5)
        result_loaded = retrieve("test query", loaded, backend, top_k=5)

        self.assertEqual(
            [h.chunk_id for h in result_original.hits],
            [h.chunk_id for h in result_loaded.hits],
            "chunk_id order must match",
        )
        self.assertEqual(
            [h.similarity_score for h in result_original.hits],
            [h.similarity_score for h in result_loaded.hits],
            "similarity_score order must match",
        )
        self.assertEqual(result_original.index_version, result_loaded.index_version)
        self.assertEqual(result_original.top_k_requested, result_loaded.top_k_requested)
        self.assertEqual(result_original.truncated, result_loaded.truncated)
        for a, b in zip(result_original.hits, result_loaded.hits):
            self.assertEqual(a.chunk_id, b.chunk_id)
            self.assertEqual(a.document_id, b.document_id)
            self.assertEqual(a.raw_text, b.raw_text)
            self.assertEqual(a.embedding_vector_id, b.embedding_vector_id)
            self.assertEqual(a.index_version, b.index_version)
            self.assertEqual(a.similarity_score, b.similarity_score)


# ---------------------------------------------------------------------------
# index file hash identical across two independent builds
# ---------------------------------------------------------------------------


class TestDeterministicFileHash(unittest.TestCase):
    """Same inputs → same index file bytes (byte-identical across runs)."""

    def test_index_file_hash_identical_across_builds(self) -> None:
        doc = run_ingestion_pipeline(str(SEMANTIC_PDF))
        if not doc.chunks:
            self.skipTest("no chunks in fixture")
        backend = FakeEmbeddingBackend()
        embeddings1 = run_embedding_pipeline(doc, backend)
        embeddings2 = run_embedding_pipeline(doc, backend)
        index1 = build_index_snapshot(doc, embeddings1, "v1")
        index2 = build_index_snapshot(doc, embeddings2, "v1")

        with tempfile.TemporaryDirectory() as tmp:
            path1 = Path(tmp) / "index1.json"
            path2 = Path(tmp) / "index2.json"
            save_index(index1, path1)
            save_index(index2, path2)
            raw1 = path1.read_bytes()
            raw2 = path2.read_bytes()

        self.assertEqual(raw1, raw2, "index file must be byte-identical for same inputs")
        h = hashlib.sha256(raw1).hexdigest()
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)


# ---------------------------------------------------------------------------
# loading invalid index_version fails loudly
# ---------------------------------------------------------------------------


class TestInvalidIndexVersionFails(unittest.TestCase):
    """Loading JSON with invalid or missing index_version_id must raise."""

    def test_missing_index_version_id_fails(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write('{"entries": []}')
            path = Path(f.name)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_index(path)
            self.assertIn("index_version_id", str(ctx.exception))
        finally:
            path.unlink(missing_ok=True)

    def test_empty_index_version_id_fails(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write('{"index_version_id": "   ", "entries": []}')
            path = Path(f.name)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_index(path)
            self.assertIn("non-empty", str(ctx.exception))
        finally:
            path.unlink(missing_ok=True)

    def test_index_version_id_not_string_fails(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write('{"index_version_id": 123, "entries": []}')
            path = Path(f.name)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_index(path)
            self.assertIn("str", str(ctx.exception))
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# missing fields fail loudly
# ---------------------------------------------------------------------------


class TestMissingFieldsFail(unittest.TestCase):
    """Loading JSON with missing required fields must raise with clear message."""

    def test_missing_entries_fails(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write('{"index_version_id": "v1"}')
            path = Path(f.name)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_index(path)
            self.assertIn("entries", str(ctx.exception))
        finally:
            path.unlink(missing_ok=True)

    def test_entry_missing_chunk_id_fails(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                '{"index_version_id": "v1", "entries": ['
                '{"document_id": "d1", "raw_text": "x", "embedding_vector_id": "e1", "vector": [0.0]}]}'
            )
            path = Path(f.name)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_index(path)
            self.assertIn("chunk_id", str(ctx.exception))
        finally:
            path.unlink(missing_ok=True)

    def test_entry_missing_vector_fails(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                '{"index_version_id": "v1", "entries": ['
                '{"chunk_id": "c1", "document_id": "d1", "raw_text": "x", "embedding_vector_id": "e1"}]}'
            )
            path = Path(f.name)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_index(path)
            self.assertIn("vector", str(ctx.exception))
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# load from non-existent path fails
# ---------------------------------------------------------------------------


class TestLoadMissingFile(unittest.TestCase):
    """load_index on non-existent path must raise FileNotFoundError."""

    def test_load_nonexistent_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "does_not_exist.json"
            with self.assertRaises(FileNotFoundError) as ctx:
                load_index(path)
            self.assertIn("not found", str(ctx.exception))
