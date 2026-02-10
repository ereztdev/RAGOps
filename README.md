# RAGOps

**AI-powered troubleshooting for ground support equipment.**

RAGOps is a deterministic Retrieval-Augmented Generation pipeline built for aviation GSE maintenance. It ingests OEM technical manuals, indexes them locally, and answers troubleshooting questions with cited, verifiable answers — or explicitly refuses when the documentation doesn't support a claim.

No cloud. No API keys. No data leaving your infrastructure.

---

## The Problem

A technician is on the ramp. A GPU won't build voltage. The paper manual is 170 pages. The experienced tech who knows this unit retired last year. The new tech googles the error, gets a forum post from 2011, and spends two hours chasing the wrong wire.

Multiply that by every unit in your fleet, every shift, every new hire.

Current options are bad: paper manuals nobody reads fast enough, generic AI chatbots that confidently invent torque specs, or expensive OEM support contracts with hold times measured in days.

## What RAGOps Does

RAGOps takes your OEM maintenance manuals — Hobart, TLD, Guinault, ITW, whatever your fleet runs — and turns them into a private, searchable troubleshooting system that gives technicians the exact answer from the exact manual, with page-level citations.

When the manual doesn't contain the answer, the system says so. It does not guess.

---

## Proof It Works

We tested RAGOps against a real Hobart JETEX 4D (Spec. 7003B) operations and maintenance manual — 170+ pages of troubleshooting procedures, wiring diagrams, and component specifications. 79 chunks, ~40,000 tokens.

### Test 1: Precise Component Identification

**Question:** *The generator will not build up any voltage when I operate the push-to-build-up voltage switch. What specific resistance reading should I look for when testing the generator revolving field, and which two color-coded wires must I disconnect?*

**RAGOps answered:** 10 to 11 ohms. Disconnect the yellow and red-orange wires that go down to the generator.

**Verified against the manual:** Correct. The system located the exact troubleshooting procedure in Chapter 3-2, extracted the resistance specification and wire color codes, and returned them with chunk-level citations. No hallucination. No approximation.

### Test 2: Cross-Procedure Symptom Matching

**Question:** *The engine starts properly but shuts down when the operator releases the engine switch from START to RUN. To test the engine overtemperature protection specifically, where should the technician place a clip-lead jumper wire, and what temperature does the switch trip at?*

This is a harder question. It requires the system to identify the correct troubleshooting procedure by matching the described symptom (engine starts then stops on switch release) against multiple procedures in the manual, then extract the diagnostic steps for the overtemperature circuit specifically.

**RAGOps answered:** Place a clip-lead jumper wire on the water temperature switch terminals. If the engine starts properly after jumpering, the water temperature switch (S402) is defective and should be replaced. The system correctly identified and refused to guess the trip temperature, stating the specific value was not present in the retrieved context.

**Why this matters:** The system didn't just keyword-match "overtemperature" — it matched the symptom description to the correct procedure (item 4 in Engine and Controls troubleshooting), distinguished it from generator troubleshooting procedures that share similar terminology, and extracted the actionable diagnostic step. When part of the answer required data from a different section of the manual, it refused rather than fabricating a number.

That refusal is a feature. In aviation maintenance, a wrong temperature spec is worse than no answer at all.

---

## Why On-Premises Matters

GPU maintenance manuals from Hobart, TLD, and most major OEMs are public domain or freely distributed. But that's not the whole picture.

Fleet operators maintain internal documentation that isn't public: modified wiring diagrams for specific units, fleet-specific maintenance schedules, compliance records, incident reports, and operational procedures tailored to their airport and contracts. Sending that through OpenAI's API or any cloud RAG service means your proprietary operational data is transiting infrastructure you don't control.

RAGOps runs entirely on your hardware. The LLM (Llama 3.1) runs locally through Ollama. Embeddings are computed locally via BGE. There is no outbound network dependency. You can air-gap this system and run it in a hangar with no internet — which, on many ramps, is the reality anyway.

For operators handling military contracts, government airport authorities, or airlines with strict data governance, this isn't a nice-to-have. It's a requirement.

---

## How It Works

RAGOps is not a wrapper around ChatGPT. It's a pipeline with gates at every stage.

**Ingest** — Drop in a PDF. The system extracts text, chunks it into overlapping windows (preserving cross-page context), and generates embeddings locally. Each chunk is traceable to its source document and page.

**Index** — Chunks are stored in a versioned, immutable index. Index versions are never modified after promotion — new documents mean a new index version. This gives you a reproducible, auditable state for every query.

**Evaluate** — Before an index can serve answers, it must pass quality evaluation. The system checks for: empty retrieval, low confidence scores, gibberish detection, and hit count validation. If the index doesn't meet the bar, it doesn't get promoted.

**Retrieve** — When a question comes in, RAGOps runs hybrid retrieval: semantic search (BGE embeddings) combined with keyword matching (BM25), merged through Reciprocal Rank Fusion. A phrase-match bonus rewards chunks that contain exact multi-word sequences from the query — so "water temperature switch" as a phrase scores higher than chunks that happen to contain "water" and "switch" separately. Domain detection identifies whether the query is about engine, generator, or troubleshooting systems, and boosts chunks from the matching domain.

**Answer** — The top-ranked chunks go to the LLM with a grounding instruction: answer only from the provided context, match the symptom to the correct procedure before responding, and refuse if the context doesn't support the answer. Every response includes chunk-level citations.

**Log** — Every query produces a provenance record: which domains were detected, which chunks were retrieved (with confidence scores and boost indicators), and which chunks grounded the final answer. This is the audit trail.

---

## Built With GSE Domain Knowledge

This project is built by an engineer with direct experience in the ground support equipment industry — understanding not just the software architecture but the operational reality: what technicians actually need on the ramp, how troubleshooting flows work across OEM manuals, and why generic AI solutions fail in this environment.

The domain-specific retrieval features (troubleshooting procedure matching, symptom-to-procedure alignment, multi-OEM support) exist because they solve real problems that came from working with real GSE documentation, not because they looked good in a demo.

---

## Getting Started

### Docker (Recommended)

```bash
docker compose up -d --build
docker compose exec ragops bash

# Inside container — ingest a manual and ask a question
ragops run --pdf /app/data/test_pdfs/JETEX4D7003B.pdf
```

First Ollama request takes ~1–2 min (model load). Subsequent queries run in ~15–30s depending on hardware.

### Local Install

```bash
git clone https://github.com/ereztdev/RAGOps.git
cd RAGOps
pip install -e .

# End-to-end: ingest, build index, ask
ragops run --pdf path/to/manual.pdf

# Or step by step
ragops ingest --pdf path/to/manual.pdf
ragops build-index --index-version v1 --overwrite
ragops ask --query "Your troubleshooting question"
```

---

## Technical Details

For engineers evaluating the codebase.

### Architecture

```
PDF → hash + extract → sliding-window chunking (400 tokens, 25% overlap)
    → BGE embeddings (local) → versioned index → evaluation gate → promotion

Query → resolve active index → embed query
      → hybrid retrieval (BGE cosine + BM25 TF-IDF, RRF merge)
      → phrase-match bonus → domain boost → top-10 ranked chunks
      → grounded LLM inference (Ollama) → cited answer or refusal
```

### Design Constraints

- **Deterministic:** Same query + same index + same top-k = identical results. No randomness, no approximate nearest neighbors.
- **Evaluation-gated:** Indexes must pass quality checks before serving inference. No silent degradation.
- **Immutable indexes:** Promoted index versions are never modified. Reproducible state for every query.
- **Explicit refusal:** `found: false` with `refusal_reason` when context doesn't support an answer. No hallucinated text.
- **Pluggable LLM:** Ollama (Llama 3.1 8B default), configurable model. Adapter pattern supports multiple backends.

### Retrieval Scoring

Hybrid retrieval combines BGE semantic similarity with BM25 keyword matching via Reciprocal Rank Fusion (rrf_k=60). Post-RRF, a phrase-match bonus (0.15 per matching 2–4 word phrase, capped at 0.45) boosts symptom-specific chunks. Optional domain detection multiplies scores for chunks matching query domains (1.3x default). Final top-10 returned to LLM with normalized confidence scores.

### Stack

- **Embeddings:** BGE-base-en-v1.5 via sentence-transformers (local, ~768-dim)
- **LLM:** Llama 3.1 8B via Ollama (local, temperature=0, num_ctx=8192)
- **Search:** Brute-force cosine + BM25 TF-IDF, no external vector DB
- **Storage:** JSON indexes, deterministic serialization, one file per index version
- **Tests:** 119 passing (ingestion, schema, embedding, storage, promotion, evaluation, inference, CLI, retrieval tracing, regression fixtures)

### CLI Reference

| Command | Purpose |
|---------|---------|
| `ingest` | Parse PDF into chunks |
| `build-index` | Embed chunks and create index |
| `evaluate` | Score retrieval quality |
| `promote` | Mark index as active (requires passing eval) |
| `ask` | Query the active index |
| `run` | E2E: ingest → build → ask |

Run `ragops <command> --help` for options.

### Requirements

- Python 3.10+
- sentence-transformers
- pypdf
- Ollama

## License

MIT