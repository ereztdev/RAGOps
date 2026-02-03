"""
Segment 9: End-to-End CLI.
Tests: full flow ingest -> build-index -> evaluate -> promote -> ask;
ask before promote -> refusal; promote with failed evaluation -> error;
repeated runs produce byte-identical artifacts (ingestion, index).
Real filesystem, no mocks, deterministic outputs.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEMANTIC_PDF = PROJECT_ROOT / "data" / "test_pdfs" / "ragops_semantic_test_pdf.pdf"


def _run_cli(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run python -m ragops.cli with given args. cwd defaults to PROJECT_ROOT."""
    cmd = [sys.executable, "-m", "ragops.cli"] + args
    return subprocess.run(
        cmd,
        cwd=cwd or PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# Full flow: ingest -> build-index -> evaluate -> promote -> ask
# ---------------------------------------------------------------------------


class TestFullFlowE2E(unittest.TestCase):
    def test_full_flow_succeeds(self) -> None:
        """ingest -> build-index -> evaluate -> promote -> ask runs successfully."""
        if not SEMANTIC_PDF.exists():
            self.skipTest(f"fixture not found: {SEMANTIC_PDF}")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            indexes_dir = tmp / "indexes"
            indexes_dir.mkdir()
            ingestion_path = tmp / "ingestion.json"
            index_path = tmp / "index.json"
            evals_dir = tmp / "evals"
            queries_path = tmp / "queries.json"

            r = _run_cli(["ingest", "--pdf", str(SEMANTIC_PDF), "--out", str(ingestion_path)])
            self.assertEqual(r.returncode, 0, f"ingest failed: {r.stderr}")
            with open(ingestion_path, encoding="utf-8") as f:
                ingestion_data = json.load(f)
            first_chunk_text = ""
            if ingestion_data.get("chunks"):
                first_chunk_text = (ingestion_data["chunks"][0].get("text") or "")[:100]
            queries = ["what is the main topic?"]
            if first_chunk_text.strip():
                queries.append(first_chunk_text.strip())
            queries_path.write_text(json.dumps(queries), encoding="utf-8")

            r = _run_cli([
                "build-index",
                "--ingestion", str(ingestion_path),
                "--index-version", "v1",
                "--out", str(index_path),
                "--indexes-dir", str(indexes_dir),
            ])
            self.assertEqual(r.returncode, 0, f"build-index failed: {r.stderr}")

            r = _run_cli([
                "evaluate",
                "--index-version", "v1",
                "--queries", str(queries_path),
                "--out", str(evals_dir),
                "--indexes-dir", str(indexes_dir),
                "--top-k", "1",
                "--min-confidence", "0.25",
            ])
            self.assertEqual(r.returncode, 0, f"evaluate failed: {r.stderr}")

            report_files = list(evals_dir.glob("*.json"))
            self.assertGreater(len(report_files), 0, "evaluate must write at least one report")
            report_path = None
            for p in report_files:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("overall_pass"):
                    report_path = p
                    break
            self.assertIsNotNone(report_path, "at least one evaluation report must pass (overall_pass True) for promote")

            r = _run_cli([
                "promote",
                "--index-version", "v1",
                "--evaluation", str(report_path),
                "--indexes-dir", str(indexes_dir),
            ])
            self.assertEqual(r.returncode, 0, f"promote failed: {r.stderr}")

            ask_query = first_chunk_text.strip() if first_chunk_text.strip() else "what is the main topic?"
            r = _run_cli([
                "ask",
                "--query", ask_query,
                "--indexes-dir", str(indexes_dir),
                "--top-k", "1",
                "--min-confidence", "0.25",
            ])
            self.assertEqual(r.returncode, 0, f"ask failed: {r.stderr}")
            self.assertTrue(r.stdout.strip() or "citations" in r.stdout, "ask must print answer or citations")


# ---------------------------------------------------------------------------
# ask before promote -> refusal
# ---------------------------------------------------------------------------


class TestAskBeforePromoteRefusal(unittest.TestCase):
    def test_ask_before_promote_returns_refusal(self) -> None:
        """ask with no active index (before promote) must refuse with non-zero exit."""
        if not SEMANTIC_PDF.exists():
            self.skipTest(f"fixture not found: {SEMANTIC_PDF}")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            indexes_dir = tmp / "indexes"
            indexes_dir.mkdir()
            ingestion_path = tmp / "ingestion.json"
            index_path = tmp / "index.json"
            _run_cli(["ingest", "--pdf", str(SEMANTIC_PDF), "--out", str(ingestion_path)])
            _run_cli([
                "build-index",
                "--ingestion", str(ingestion_path),
                "--index-version", "v1",
                "--out", str(index_path),
                "--indexes-dir", str(indexes_dir),
            ])
            r = _run_cli([
                "ask",
                "--query", "any question",
                "--indexes-dir", str(indexes_dir),
            ])
            self.assertNotEqual(r.returncode, 0, "ask must exit non-zero when no active index")
            self.assertIn("no_active_index", r.stderr or r.stdout)


# ---------------------------------------------------------------------------
# promote with failed evaluation -> error
# ---------------------------------------------------------------------------


class TestPromoteWithFailedEvaluation(unittest.TestCase):
    def test_promote_with_failed_evaluation_errors(self) -> None:
        """promote with evaluation.overall_pass False must exit non-zero."""
        if not SEMANTIC_PDF.exists():
            self.skipTest(f"fixture not found: {SEMANTIC_PDF}")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            indexes_dir = tmp / "indexes"
            indexes_dir.mkdir()
            ingestion_path = tmp / "ingestion.json"
            index_path = tmp / "index.json"
            _run_cli(["ingest", "--pdf", str(SEMANTIC_PDF), "--out", str(ingestion_path)])
            _run_cli([
                "build-index",
                "--ingestion", str(ingestion_path),
                "--index-version", "v1",
                "--out", str(index_path),
                "--indexes-dir", str(indexes_dir),
            ])
            failed_report_path = tmp / "failed_eval.json"
            failed_report_path.write_text(
                json.dumps({
                    "evaluation_id": "e1",
                    "index_version": "v1",
                    "query_hash": "qh1",
                    "top_k_requested": 5,
                    "hit_count": 0,
                    "signals": {"empty_retrieval": {"passed": False, "details": {"hit_count": 0}}},
                    "overall_pass": False,
                    "created_at": "2026-02-02T12:00:00.000000Z",
                }, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            r = _run_cli([
                "promote",
                "--index-version", "v1",
                "--evaluation", str(failed_report_path),
                "--indexes-dir", str(indexes_dir),
            ])
            self.assertNotEqual(r.returncode, 0, "promote must refuse when evaluation did not pass")
            self.assertTrue("overall_pass" in (r.stderr or "") or "refused" in (r.stderr or "").lower() or "error" in (r.stderr or "").lower())


# ---------------------------------------------------------------------------
# build-index: BGE default; reject "fake" unless --use-fake
# ---------------------------------------------------------------------------


class TestBuildIndexBgeDefaultFakeOptIn(unittest.TestCase):
    def test_build_index_rejects_fake_version_without_use_fake(self) -> None:
        """index_version starting with 'fake' must be rejected unless --use-fake."""
        if not SEMANTIC_PDF.exists():
            self.skipTest(f"fixture not found: {SEMANTIC_PDF}")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            indexes_dir = tmp / "indexes"
            indexes_dir.mkdir()
            ingestion_path = tmp / "ingestion.json"
            index_path = tmp / "index.json"
            _run_cli(["ingest", "--pdf", str(SEMANTIC_PDF), "--out", str(ingestion_path)])
            r = _run_cli([
                "build-index",
                "--ingestion", str(ingestion_path),
                "--index-version", "fake_v1",
                "--out", str(index_path),
                "--indexes-dir", str(indexes_dir),
            ])
            self.assertNotEqual(r.returncode, 0, "build-index must reject fake version without --use-fake")
            self.assertIn("fake", (r.stderr or "").lower())

    def test_build_index_accepts_fake_version_with_use_fake(self) -> None:
        """With --use-fake, index_version may start with 'fake' (test-only)."""
        if not SEMANTIC_PDF.exists():
            self.skipTest(f"fixture not found: {SEMANTIC_PDF}")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            indexes_dir = tmp / "indexes"
            indexes_dir.mkdir()
            ingestion_path = tmp / "ingestion.json"
            index_path = tmp / "index.json"
            _run_cli(["ingest", "--pdf", str(SEMANTIC_PDF), "--out", str(ingestion_path)])
            r = _run_cli([
                "build-index",
                "--ingestion", str(ingestion_path),
                "--index-version", "fake_v1",
                "--out", str(index_path),
                "--indexes-dir", str(indexes_dir),
                "--use-fake",
            ])
            self.assertEqual(r.returncode, 0, f"build-index with --use-fake must succeed: {r.stderr}")
            self.assertTrue(index_path.exists())


# ---------------------------------------------------------------------------
# Repeated runs produce byte-identical artifacts
# ---------------------------------------------------------------------------


class TestDeterministicArtifacts(unittest.TestCase):
    def test_ingest_twice_same_pdf_byte_identical_output(self) -> None:
        """Same PDF ingested twice produces byte-identical ingestion_output.json."""
        if not SEMANTIC_PDF.exists():
            self.skipTest(f"fixture not found: {SEMANTIC_PDF}")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out1 = tmp / "ingestion1.json"
            out2 = tmp / "ingestion2.json"
            _run_cli(["ingest", "--pdf", str(SEMANTIC_PDF), "--out", str(out1)])
            _run_cli(["ingest", "--pdf", str(SEMANTIC_PDF), "--out", str(out2)])
            self.assertTrue(out1.exists())
            self.assertTrue(out2.exists())
            self.assertEqual(out1.read_bytes(), out2.read_bytes(), "ingestion output must be byte-identical for same PDF")

    def test_build_index_twice_same_ingestion_byte_identical_index(self) -> None:
        """Same ingestion built twice produces byte-identical index files."""
        if not SEMANTIC_PDF.exists():
            self.skipTest(f"fixture not found: {SEMANTIC_PDF}")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            indexes1 = tmp / "indexes1"
            indexes2 = tmp / "indexes2"
            indexes1.mkdir()
            indexes2.mkdir()
            ingestion_path = tmp / "ingestion.json"
            _run_cli(["ingest", "--pdf", str(SEMANTIC_PDF), "--out", str(ingestion_path)])
            index1 = tmp / "index1.json"
            index2 = tmp / "index2.json"
            _run_cli([
                "build-index",
                "--ingestion", str(ingestion_path),
                "--index-version", "v1",
                "--out", str(index1),
                "--indexes-dir", str(indexes1),
            ])
            _run_cli([
                "build-index",
                "--ingestion", str(ingestion_path),
                "--index-version", "v1",
                "--out", str(index2),
                "--indexes-dir", str(indexes2),
            ])
            self.assertEqual(
                hashlib.sha256(index1.read_bytes()).hexdigest(),
                hashlib.sha256(index2.read_bytes()).hexdigest(),
                "index files from same ingestion and version must be byte-identical",
            )


if __name__ == "__main__":
    unittest.main()
