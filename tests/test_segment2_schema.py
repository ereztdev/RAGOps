"""
Segment 2: Canonical Internal Schema & Serialization.
Tests for deterministic IDs, stable chunk_key, and JSON schema.
Runs without external services; uses data/test_pdfs fixtures.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.schema import (
    Chunk,
    Document,
    IndexVersion,
    chunk_id_from_components,
    chunk_key,
    document_id_from_bytes,
)
from core.serialization import (
    ingestion_output_to_dict,
    index_manifest_to_dict,
    write_ingestion_output,
    write_index_manifest,
)
from pipeline.ingestion.ingestion_pipeline import run_ingestion_pipeline


# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEMANTIC_PDF = PROJECT_ROOT / "data" / "test_pdfs" / "ragops_semantic_test_pdf.pdf"
GIBBERISH_PDF = PROJECT_ROOT / "data" / "test_pdfs" / "ragops_gibberish_test_pdf.pdf"


# ---------------------------------------------------------------------------
# Stable identifiers
# ---------------------------------------------------------------------------


class TestStableIdentifiers(unittest.TestCase):
    def test_document_id_deterministic_for_same_pdf(self) -> None:
        """Same PDF bytes must yield the same document_id."""
        raw = SEMANTIC_PDF.read_bytes()
        id1 = document_id_from_bytes(raw)
        id2 = document_id_from_bytes(raw)
        self.assertEqual(id1, id2)
        self.assertEqual(len(id1), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in id1))

    def test_document_id_different_for_different_pdfs(self) -> None:
        """Different PDFs must yield different document_ids."""
        raw_s = SEMANTIC_PDF.read_bytes()
        raw_g = GIBBERISH_PDF.read_bytes()
        self.assertNotEqual(document_id_from_bytes(raw_s), document_id_from_bytes(raw_g))

    def test_chunk_key_format(self) -> None:
        """chunk_key must be document_id:page_number."""
        doc_id = "a" * 64
        self.assertEqual(chunk_key(doc_id, 1), f"{doc_id}:1")
        self.assertEqual(chunk_key(doc_id, 42), f"{doc_id}:42")

    def test_chunk_key_stable(self) -> None:
        """Same document_id and page_number must yield same chunk_key."""
        doc_id = "b" * 64
        self.assertEqual(chunk_key(doc_id, 1), chunk_key(doc_id, 1))

    def test_chunk_id_deterministic(self) -> None:
        """Same (document_id, source_type, page_number, chunk_index) must yield same chunk_id."""
        cid1 = chunk_id_from_components("doc1", "pdf", 1, 0)
        cid2 = chunk_id_from_components("doc1", "pdf", 1, 0)
        self.assertEqual(cid1, cid2)
        self.assertEqual(len(cid1), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in cid1))

    def test_chunk_id_different_for_different_inputs(self) -> None:
        """Different inputs must yield different chunk_ids."""
        base = chunk_id_from_components("d", "pdf", 1, 0)
        self.assertNotEqual(base, chunk_id_from_components("d", "pdf", 2, 0))
        self.assertNotEqual(base, chunk_id_from_components("d", "md", 1, 0))
        self.assertNotEqual(base, chunk_id_from_components("d", "pdf", 1, 1))
        self.assertNotEqual(base, chunk_id_from_components("e", "pdf", 1, 0))


# ---------------------------------------------------------------------------
# Ingestion output schema
# ---------------------------------------------------------------------------


def _ingestion_schema_keys() -> set[str]:
    return {"document_id", "source_path", "chunks"}


def _chunk_schema_keys() -> set[str]:
    return {"chunk_id", "chunk_key", "document_id", "source_type", "page_number", "chunk_index", "text"}


class TestIngestionOutputSchema(unittest.TestCase):
    def test_ingestion_output_json_matches_schema(self) -> None:
        """Serialized ingestion output must have document_id, source_path, ordered chunks."""
        doc = run_ingestion_pipeline(str(SEMANTIC_PDF))
        data = ingestion_output_to_dict(doc)
        self.assertEqual(set(data.keys()), _ingestion_schema_keys())
        self.assertEqual(data["document_id"], doc.document_id)
        self.assertEqual(data["source_path"], doc.source_path)
        for c in data["chunks"]:
            self.assertEqual(set(c.keys()), _chunk_schema_keys())
            self.assertEqual(c["chunk_key"], f"{c['document_id']}:{c['page_number']}")

    def test_semantic_pdf_serializes_correctly(self) -> None:
        """Semantic test PDF must produce valid ingestion_output.json."""
        doc = run_ingestion_pipeline(str(SEMANTIC_PDF))
        data = ingestion_output_to_dict(doc)
        self.assertTrue(data["document_id"])
        self.assertTrue(data["source_path"])
        self.assertIsInstance(data["chunks"], list)
        for chunk in data["chunks"]:
            self.assertTrue(chunk["chunk_key"].endswith(f":{chunk['page_number']}"))
            self.assertIsInstance(chunk["text"], str)

    def test_gibberish_pdf_serializes_correctly(self) -> None:
        """Gibberish test PDF must produce valid ingestion_output.json."""
        doc = run_ingestion_pipeline(str(GIBBERISH_PDF))
        data = ingestion_output_to_dict(doc)
        self.assertTrue(data["document_id"])
        self.assertTrue(data["source_path"])
        self.assertIsInstance(data["chunks"], list)
        for chunk in data["chunks"]:
            self.assertIn(":", chunk["chunk_key"])
            self.assertGreaterEqual(chunk["page_number"], 1)
            self.assertIsInstance(chunk["text"], str)

    def test_ingestion_output_json_roundtrip_stable(self) -> None:
        """Written ingestion_output.json must be valid JSON and match schema."""
        doc = run_ingestion_pipeline(str(SEMANTIC_PDF))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ingestion_output.json"
            write_ingestion_output(doc, path)
            self.assertTrue(path.exists())
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(set(loaded.keys()), _ingestion_schema_keys())
            self.assertEqual(loaded["document_id"], doc.document_id)
            if loaded["chunks"]:
                self.assertEqual(set(loaded["chunks"][0].keys()), _chunk_schema_keys())


# ---------------------------------------------------------------------------
# Index manifest schema (structure only)
# ---------------------------------------------------------------------------


class TestIndexManifestSchema(unittest.TestCase):
    def test_index_manifest_json_matches_schema(self) -> None:
        """index_manifest serialization must have required fields."""
        manifest = IndexVersion(
            index_version_id="v1",
            created_at="2026-01-31T00:00:00Z",
            embedding_model="placeholder",
            chunking_strategy="page",
            document_ids=["doc1", "doc2"],
        )
        data = index_manifest_to_dict(manifest)
        self.assertEqual(data["index_version_id"], "v1")
        self.assertEqual(data["created_at"], "2026-01-31T00:00:00Z")
        self.assertEqual(data["embedding_model"], "placeholder")
        self.assertEqual(data["chunking_strategy"], "page")
        self.assertEqual(data["document_ids"], ["doc1", "doc2"])

    def test_index_manifest_write_read(self) -> None:
        """Written index_manifest.json must be valid JSON."""
        manifest = IndexVersion(
            index_version_id="v1",
            created_at="2026-01-31T00:00:00Z",
            embedding_model="placeholder",
            chunking_strategy="page",
            document_ids=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index_manifest.json"
            write_index_manifest(manifest, path)
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["index_version_id"], "v1")
            self.assertIn("document_ids", loaded)


# ---------------------------------------------------------------------------
# Typed pipeline output
# ---------------------------------------------------------------------------


class TestTypedPipelineOutput(unittest.TestCase):
    def test_ingestion_returns_typed_document(self) -> None:
        """run_ingestion_pipeline must return Document with Chunk list."""
        doc = run_ingestion_pipeline(str(SEMANTIC_PDF))
        self.assertIsInstance(doc, Document)
        self.assertTrue(hasattr(doc, "document_id") and hasattr(doc, "source_path") and hasattr(doc, "chunks"))
        for c in doc.chunks:
            self.assertIsInstance(c, Chunk)
            self.assertEqual(c.chunk_key, chunk_key(doc.document_id, c.page_number))

    def test_ingestion_writes_when_output_path_given(self) -> None:
        """When output_path is set, ingestion_output.json must be written."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out" / "ingestion_output.json"
            run_ingestion_pipeline(str(SEMANTIC_PDF), output_path=out)
            self.assertTrue(out.exists())
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("document_id", data)
            self.assertIn("chunks", data)


if __name__ == "__main__":
    unittest.main()
