# RAGOps Benchmark Corpus

Benchmark harness for evaluating RAGOps retrieval and inference against the TLD 285 maintenance manual.

**Benchmark: 73% accuracy (11/15)** — ceiling is chunk density, not model. Direct lookup and refusal at 100%. Remaining failures trace to multi-topic chunks diluting retrieval signal.

## What It Is

A corpus of 15 questions across 5 categories, with expected answers or refusal behavior. Designed to measure accuracy on direct lookups, procedural steps, cross-subsystem reasoning, refusal handling, and multi-hop questions.

## How to Run

```bash
python tests/benchmark/run_benchmark.py
```

**Prerequisites:** Index the TLD_285_Maintenance-Manual.pdf first (`ragops run --pdf path/to/TLD_285_Maintenance-Manual.pdf` or ingest + build-index + evaluate + promote). The benchmark shells out to `ragops ask --query <question>` for each question.

## Categories

| Category       | Description                                      |
|----------------|--------------------------------------------------|
| direct_lookup  | Single-fact extraction (specs, thresholds)       |
| procedural     | Step-by-step instructions                        |
| cross_subsystem| Answers spanning engine + generator, etc.        |
| refusal        | Questions the manual does not support             |
| multi_hop      | Multiple facts combined in one answer            |

## Scoring Rules

- **Non-refusal questions:** Pass if the answer contains all `expected_contains` terms (case-insensitive).
- **Refusal questions:** Pass if the answer indicates refusal (contains "refusal" or "refused", or `ragops ask` exits non-zero).

## Output

- Pass/fail table printed to stdout.
- Results saved to `tests/benchmark/results/<timestamp>.json`.
