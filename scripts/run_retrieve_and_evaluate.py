#!/usr/bin/env python3
"""
Temporary script: load index -> retrieve -> evaluate -> print overall_pass, signals fired, hit count.
Usage: python scripts/run_retrieve_and_evaluate.py [--query "your query"] [--top-k 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schema import EvaluationConfig
from pipeline.embedding.embedding_engine import BgeEmbeddingBackend
from pipeline.evaluation.evaluation_engine import evaluate_retrieval
from pipeline.retrieval.retrieval_engine import retrieve
from storage.vector_index_store import load_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve + evaluate on bge_base_v1 index")
    parser.add_argument("--query", default="semantic content", help="Query string")
    parser.add_argument("--top-k", type=int, default=5, dest="top_k", help="Top-k")
    args = parser.parse_args()

    index_path = Path(__file__).resolve().parent.parent / "indexes" / "bge_base_v1.json"
    if not index_path.is_file():
        print(f"Index file not found: {index_path}", file=sys.stderr)
        return 1

    index = load_index(index_path)
    backend = BgeEmbeddingBackend()
    result = retrieve(args.query, index, backend, args.top_k)
    config = EvaluationConfig(
        min_confidence_score=0.5,
        min_alphabetic_ratio=0.5,
        max_entropy=5.0,
    )
    report = evaluate_retrieval(result, args.query, config, evaluations_dir=None)

    print("overall_pass:", report.overall_pass)
    print("signals fired:")
    for name, sr in report.signals.items():
        status = "PASS" if sr.passed else "FAIL"
        print(f"  {name}: {status}  {sr.details}")
    print("hit_count:", report.hit_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
