# Example Output

This shows what a successful RAGOps pipeline run looks like.

## 1. Ingest

```bash
$ python -m ragops.cli ingest --pdf data/test_pdfs/ragops_semantic_test_pdf.pdf --out ingestion.json
```

Output file `ingestion.json`:
```json
{
  "document_id": "c7e600d99f24cd7b20ed1063fe0fe549d9791c37def73c0daf84d4f9e34e47a1",
  "source_path": "data/test_pdfs/ragops_semantic_test_pdf.pdf",
  "chunks": [
    {
      "chunk_id": "6536e635...",
      "chunk_key": "c7e600d9...:1",
      "document_id": "c7e600d9...",
      "source_type": "pdf",
      "page_number": 1,
      "chunk_index": 0,
      "text": "RAGOps Semantic Test Corpus Purpose: semantic retrieval testing..."
    }
  ]
}
```

## 2. Build Index

```bash
$ python -m ragops.cli build-index \
    --ingestion ingestion.json \
    --index-version v1 \
    --out indexes/v1.json \
    --indexes-dir indexes
```

Creates `indexes/v1.json` (index snapshot) and registers in `indexes/registry.json`.

## 3. Evaluate

```bash
$ python -m ragops.cli evaluate \
    --index-version v1 \
    --queries queries.json \
    --out evaluations \
    --indexes-dir indexes \
    --min-confidence 0.3
```

Output in `evaluations/<evaluation_id>.json`:
```json
{
  "evaluation_id": "a1b2c3...",
  "index_version": "v1",
  "overall_pass": true,
  "hit_count": 1,
  "signals": {
    "empty_retrieval": {"passed": true, "details": {"hit_count": 1}},
    "low_confidence": {"passed": true, "details": {"top_score": 0.72, "threshold": 0.3}},
    "gibberish_detected": {"passed": true, "details": {"gibberish_chunk_ids": []}},
    "hit_count_insufficient": {"passed": true, "details": {"hit_count": 1, "top_k_requested": 5}}
  }
}
```

## 4. Promote

```bash
$ python -m ragops.cli promote \
    --index-version v1 \
    --evaluation evaluations/a1b2c3....json \
    --indexes-dir indexes
```

Updates `indexes/active_index.json` to point to v1.

## 5. Ask

```bash
$ python -m ragops.cli ask \
    --query "What is GLARB-GLARB?" \
    --indexes-dir indexes
```

Output:
```
A ceremonial farming tool used by the ancient settlers of planet Jiro to cultivate frozen soil.
citations: 6536e63580255b97b8b6d5f30de3239e98698339a2a8dc0084a0227c3ce52eee
```

## Refusal Example

When the query can't be answered from the corpus:

```bash
$ python -m ragops.cli ask \
    --query "What is the population of France?" \
    --indexes-dir indexes
```

Output (stderr):
```
refusal: evaluation_failed: low_confidence
```

The system refuses rather than hallucinating an answer.
