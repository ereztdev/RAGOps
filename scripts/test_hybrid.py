#!/usr/bin/env python3
"""Quick sanity check for hybrid retrieval."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.schema import HybridRetrievalConfig
from pipeline.embedding.embedding_engine import BgeEmbeddingBackend, run_embedding_pipeline
from pipeline.ingestion.ingestion_pipeline import run_ingestion_pipeline
from pipeline.retrieval.retrieval_engine import build_index_snapshot, retrieve

pdf_path = "data/test_pdfs/software_development_agreement_enterprise_rag.pdf"
doc = run_ingestion_pipeline(pdf_path)
backend = BgeEmbeddingBackend()
embeddings = run_embedding_pipeline(doc, backend)
index = build_index_snapshot(doc, embeddings, "hybrid_test_v1")

query = "What support is included after delivery?"

# BGE-only (Phase 1)
result_bge = retrieve(query, index, backend, top_k=3)
print("BGE-only top 3 chunk_ids:", [h.chunk_id for h in result_bge.hits])
print("BGE-only top 3 scores:", [h.similarity_score for h in result_bge.hits])

# Hybrid (Phase 2): RRF merge
config = HybridRetrievalConfig(rrf_k=60)
result_hybrid = retrieve(query, index, backend, top_k=3, hybrid_config=config)
print("\nHybrid top 3 chunk_ids:", [h.chunk_id for h in result_hybrid.hits])
print("Hybrid top 3 scores:", [h.similarity_score for h in result_hybrid.hits])

# Expect: hybrid may rank cross-page chunks higher if they contain "support" keyword
