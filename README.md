# RAGOps

Deterministic RAG pipeline for enterprise documents. Retrieval is evaluation-gated, answers are cited or refused, and the same query returns the same result every time.

## Why This Exists

Most RAG systems hallucinate when they don't know something. This one refuses instead.

- Chunks are addressable and traceable back to source pages
- Retrieval quality is scored before answers are generated
- No answer ships without citations
- Index versions are immutable once promoted
- Runs fully on-premises (no API calls to OpenAI/Pinecone)

## Current Status

Phase 2 complete. Hybrid retrieval (BGE embeddings + BM25 keyword search) is working. Targeting 90%+ accuracy on enterprise document QA for seed demo.

**What works:**
- PDF ingestion with sliding-window chunking (cross-page context preserved)
- Local embeddings via BGE (sentence-transformers)
- Hybrid retrieval with configurable semantic/keyword weighting
- Evaluation engine with quality signals (confidence, gibberish detection, known-answer checks)
- Promotion gate (only evaluated indexes can serve inference)
- CLI for full pipeline: ingest, build-index, evaluate, promote, ask

**What's next:**
- Real LLM backend (Ollama + Llama 3.1 8B)
- Benchmark harness (50+ questions across multiple PDFs)
- Accuracy tuning to hit 90%+

## Quick Start

```bash
# Clone and install
git clone https://github.com/ereztdev/RAGOps.git
cd RAGOps
pip install -r requirements.txt

# Ingest a PDF
python -m ragops.cli ingest --pdf data/test_pdfs/ragops_semantic_test_pdf.pdf --out ingestion.json

# Build index
mkdir -p indexes
python -m ragops.cli build-index --ingestion ingestion.json --index-version v1 --out indexes/v1.json --indexes-dir indexes

# Evaluate retrieval quality
echo '["What is GLARB-GLARB?", "What does NEBULITE describe?"]' > queries.json
python -m ragops.cli evaluate --index-version v1 --queries queries.json --out evaluations --indexes-dir indexes --min-confidence 0.3

# Promote index (requires passing evaluation)
python -m ragops.cli promote --index-version v1 --evaluation evaluations/<evaluation_id>.json --indexes-dir indexes

# Ask questions
python -m ragops.cli ask --query "What is GLARB-GLARB?" --indexes-dir indexes
```

## Architecture

```
PDF file
  → hash + extract text
  → sliding-window chunking (400 tokens, 25% overlap)
  → embed via BGE
  → store in versioned index
  → evaluate retrieval quality
  → promote if evaluation passes

Query
  → resolve active index from registry
  → embed query
  → hybrid retrieval (semantic + keyword)
  → assemble context from top-k chunks
  → generate answer with citations (or refuse)
```

## Project Structure

```
ragops/cli/         CLI entrypoint
pipeline/
  ingestion/        PDF parsing, chunking
  embedding/        BGE and pluggable backends
  retrieval/        Hybrid search (BGE + BM25)
  evaluation/       Quality signals, gating
  inference/        Grounded answer generation
  promotion/        Index registry, versioning
storage/            Persistent index store
core/               Schema, serialization
```

## Key Design Decisions

**Deterministic retrieval:** Same query, same index, same top-k returns identical results. No randomness, no ANN approximation (brute-force cosine for now).

**Evaluation before inference:** An index cannot serve queries until it passes evaluation. Signals include: empty retrieval, low confidence, gibberish detection, hit count validation.

**Explicit refusal:** When context doesn't support an answer, the system returns `found: false` with a `refusal_reason`. No invented text.

**Immutable index versions:** Once an index is promoted, it's not modified. New documents require a new index version.

## CLI Reference

| Command | Purpose |
|---------|---------|
| `ingest` | Parse PDF into chunks |
| `build-index` | Embed chunks and create index |
| `evaluate` | Score retrieval quality |
| `promote` | Mark index as active (requires passing eval) |
| `ask` | Query the active index |

Run `python -m ragops.cli <command> --help` for options.

## Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific segment
python -m pytest tests/test_segment7_evaluation.py -v
```

## Requirements

- Python 3.10+
- sentence-transformers (for BGE embeddings)
- pypdf (for PDF extraction)

## License

MIT
