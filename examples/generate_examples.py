#!/usr/bin/env python3
"""
Generate example outputs for RAGOps documentation.
Run from repo root: python examples/generate_examples.py

Creates:
- examples/output/ingestion.json
- examples/output/index_snapshot.json  
- examples/output/evaluation_report.json
- examples/output/retrieval_result.json
- examples/output/answer.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "examples" / "output"
SEMANTIC_PDF = REPO_ROOT / "data" / "test_pdfs" / "ragops_semantic_test_pdf.pdf"


def _write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
    print(f"Wrote: {path}")


def main() -> int:
    if not SEMANTIC_PDF.exists():
        print(f"Test PDF not found: {SEMANTIC_PDF}", file=sys.stderr)
        return 1

    from core.schema import EvaluationConfig
    from core.serialization import ingestion_output_to_dict
    from pipeline.embedding.embedding_engine import BgeEmbeddingBackend, run_embedding_pipeline
    from pipeline.evaluation.evaluation_engine import evaluate_retrieval
    from pipeline.ingestion.ingestion_pipeline import run_ingestion_pipeline
    from pipeline.retrieval.retrieval_engine import build_index_snapshot, retrieve

    # 1. Ingestion
    print("Running ingestion...")
    doc = run_ingestion_pipeline(str(SEMANTIC_PDF))
    ingestion_data = ingestion_output_to_dict(doc)
    _write_json(ingestion_data, OUTPUT_DIR / "ingestion.json")

    # 2. Embedding + Index
    print("Building index with BGE embeddings...")
    backend = BgeEmbeddingBackend()
    embeddings = run_embedding_pipeline(doc, backend)
    index = build_index_snapshot(doc, embeddings, "example_v1")
    _write_json(index.to_serializable(), OUTPUT_DIR / "index_snapshot.json")

    # 3. Retrieval
    query = "What is GLARB-GLARB?"
    print(f"Running retrieval for: {query}")
    result = retrieve(query, index, backend, top_k=3)
    _write_json(result.to_serializable(), OUTPUT_DIR / "retrieval_result.json")

    # 4. Evaluation
    print("Running evaluation...")
    config = EvaluationConfig(
        min_confidence_score=0.3,
        min_alphabetic_ratio=0.5,
        max_entropy=5.0,
    )
    report = evaluate_retrieval(result, query, config, evaluations_dir=None)
    _write_json(report.to_serializable(), OUTPUT_DIR / "evaluation_report.json")

    # 5. Inference (with FakeBackend for deterministic output)
    print("Running inference...")
    from pipeline.inference.inference_runner import FakeInferenceBackend, run_grounded_inference

    llm = FakeInferenceBackend()
    answer = run_grounded_inference(query, result, report, llm)
    _write_json(answer.to_serializable(), OUTPUT_DIR / "answer.json")

    print(f"\nDone. Example outputs in: {OUTPUT_DIR}")
    print(f"\nAnswer: {answer.answer_text}")
    print(f"Found: {answer.found}")
    print(f"Citations: {answer.citation_chunk_ids}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
