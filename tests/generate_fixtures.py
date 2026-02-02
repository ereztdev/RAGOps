"""
Generate golden fixtures for Segment 10 regression tests.
Run from repo root: python tests/generate_fixtures.py
Uses known test input: data/test_pdfs/ragops_semantic_test_pdf.pdf
Writes deterministic JSON to tests/fixtures/.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
SEMANTIC_PDF = PROJECT_ROOT / "data" / "test_pdfs" / "ragops_semantic_test_pdf.pdf"
CANONICAL_SOURCE_PATH = "data/test_pdfs/ragops_semantic_test_pdf.pdf"


def _stable_json_dump(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=True, indent=2, ensure_ascii=False)


def main() -> None:
    if not SEMANTIC_PDF.is_file():
        raise FileNotFoundError(f"Test input not found: {SEMANTIC_PDF}")

    from pipeline.ingestion.ingestion_pipeline import run_ingestion_pipeline
    from core.serialization import ingestion_output_to_dict
    from pipeline.embedding.embedding_engine import FakeEmbeddingBackend, run_embedding_pipeline
    from pipeline.retrieval.retrieval_engine import build_index_snapshot, retrieve
    from pipeline.evaluation.evaluation_engine import evaluate_retrieval
    from pipeline.inference.inference_runner import (
        FakeInferenceBackend,
        run_grounded_inference,
    )
    from core.schema import EvaluationConfig
    from storage.vector_index_store import save_index

    # 1) Ingestion output (Document serialization)
    doc = run_ingestion_pipeline(str(SEMANTIC_PDF))
    ingestion_dict = ingestion_output_to_dict(doc)
    ingestion_dict["source_path"] = CANONICAL_SOURCE_PATH
    _stable_json_dump(ingestion_dict, FIXTURES_DIR / "ingestion_output.json")

    # 2) Index snapshot (IndexSnapshot serialization)
    backend = FakeEmbeddingBackend()
    embeddings = run_embedding_pipeline(doc, backend)
    index_version_id = "regression_v1"
    index_snapshot = build_index_snapshot(doc, embeddings, index_version_id)
    index_path = FIXTURES_DIR / "index_snapshot.json"
    save_index(index_snapshot, index_path)

    # 3) Retrieval result (same query -> same top-k, scores, metadata)
    retrieval_query = "regression test query"
    top_k = 5
    retrieval_result = retrieve(retrieval_query, index_snapshot, backend, top_k)
    _stable_json_dump(retrieval_result.to_serializable(), FIXTURES_DIR / "retrieval_result.json")

    # 4) Inference output structure (FakeInferenceBackend is deterministic)
    eval_config = EvaluationConfig(
        min_confidence_score=0.0,
        min_alphabetic_ratio=0.0,
        max_entropy=10.0,
    )
    evaluation_report = evaluate_retrieval(retrieval_result, retrieval_query, eval_config)
    llm = FakeInferenceBackend()
    inference_output = run_grounded_inference(
        retrieval_query, retrieval_result, evaluation_report, llm
    )
    _stable_json_dump(inference_output.to_serializable(), FIXTURES_DIR / "inference_output.json")

    print("Fixtures written to", FIXTURES_DIR)


if __name__ == "__main__":
    main()
