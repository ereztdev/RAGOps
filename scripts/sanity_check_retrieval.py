#!/usr/bin/env python3
"""
Sanity-check retrieval using BGE-backed index.
Run after building an index with BGE (e.g. python -m pipeline.embedding.embedding_engine <pdf> --out-index indexes/bge_base_v1.json --index-version bge_base_v1).
Usage: python scripts/sanity_check_retrieval.py --index indexes/bge_base_v1.json --query "your query" [--top-k 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.embedding.embedding_engine import BgeEmbeddingBackend
from pipeline.retrieval.retrieval_engine import retrieve
from storage.vector_index_store import load_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity-check retrieval with BGE-backed index")
    parser.add_argument("--index", required=True, help="Path to index JSON file")
    parser.add_argument("--query", required=True, help="Query string")
    parser.add_argument("--top-k", type=int, default=5, dest="top_k", help="Top-k (default: 5)")
    args = parser.parse_args()

    index_path = Path(args.index)
    if not index_path.is_file():
        print(f"Index file not found: {index_path}", file=sys.stderr)
        return 1

    index = load_index(index_path)
    backend = BgeEmbeddingBackend()
    result = retrieve(args.query, index, backend, args.top_k)
    print(json.dumps(result.to_serializable(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
