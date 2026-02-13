# RAGOps

![RAGOps splash — GPU diagnostics and data visualization on the tarmac](gpu_splash.png)

Deterministic troubleshooting system for aviation ground support equipment.

RAGOps is a Retrieval Augmented Generation pipeline built for GSE maintenance teams. It ingests OEM technical manuals, indexes them locally, and answers troubleshooting questions using cited source text. If the documentation does not support a claim, the system refuses.

No cloud dependency. No API keys. No data leaves your infrastructure.

---

### Progress towards main objective achieving >90% Accuracy on multi facted chunked targets
- As of now, the whole E2E and retrieval pipeline is complete (hybrid search, phrase-match bonus, domain boosting[topic labeling => a *"closed envelope"])
- Initial manual testing: correct procedure identification on JETEX 4D queries
- Benchmark harness designed; corpus build in progress (50-100 Q&A pairs from GSE manuals)
- Youtube video where I'm going over:
  - result discussions, distinctions and embedding strategies now and the what's coming up (training a local SLM? Yes.)
  - why another RAG pipeline? What provides distinction with this approach (beyong the hybrid approach and phrase matching)
  - walk through architecture stages E2E and discuss a bit why I chose "this over that"
  - Demo full pipeline E2E
  - Talk a bit about my background why GSE, why GPUs and *"closed envelope"

## The Problem

A technician is on the ramp. A GPU does not build voltage. The manual is 170 pages. The experienced technician who knew the unit retired. The new technician searches online and finds a forum post from 2011.

Two hours later the issue remains unresolved.

This repeats across fleets, shifts, and new hires.

Current options:

* Paper manuals that are slow to navigate
* Generic AI systems that invent specifications
* OEM support contracts with long response times

In aviation maintenance, an incorrect specification is worse than no answer.

---

## What RAGOps Does

RAGOps converts OEM maintenance manuals into a private troubleshooting system.

It provides:

* Page citations
* Exact specifications extracted from source text
* Symptom to procedure alignment
* Explicit refusal when context is insufficient

It does not guess.

Supported documentation can include Hobart, TLD, Guinault, ITW, and internal fleet manuals.

---

## Validation

Tested against the Hobart JETEX 4D Spec 7003B operations and maintenance manual.

* 170+ pages
* 79 indexed chunks
* Approximately 40,000 tokens

### Example 1

Question:

The generator does not build voltage when operating the push to build up switch. What resistance reading should be measured on the revolving field, and which wires must be disconnected?

Result:

* 10 to 11 ohms
* Disconnect yellow and red orange wires to the generator

Verified against Chapter 3-2. Exact match. Cited.

### Example 2

Question:

The engine starts but shuts down when switching from START to RUN. To test overtemperature protection, where is the jumper placed and what temperature does the switch trip?

Result:

* Place jumper on water temperature switch terminals
* Correctly identified S402 procedure
* Refused to provide trip temperature because it was not present in retrieved context

If a value is not present in the source material, it is not generated.

---

## Why Local Deployment Matters

Fleet operators maintain internal documentation:

* Modified wiring diagrams
* Maintenance schedules
* Incident reports
* Compliance records
* Airport specific procedures

Sending this data to external APIs introduces governance risk.

RAGOps runs fully on local hardware:

* Llama 3.1 via Ollama
* BGE embeddings locally
* No outbound network requirement
* Can operate without internet access

For military, government, or regulated operators, this is required.

---

## System Flow

### Ingest

* Parse PDF
* Sliding window chunking at 400 tokens with 25 percent overlap
* Generate embeddings locally
* Maintain page traceability

### Index

* Create versioned index files
* No modification after promotion
* Reproducible retrieval state

### Evaluate

Before activation:

* Empty retrieval checks
* Confidence threshold checks
* Hit count validation
* Noise detection

Failing indexes are not promoted.

### Retrieve

Hybrid retrieval:

* Semantic similarity using BGE
* Keyword scoring using BM25
* Reciprocal Rank Fusion merge
* Phrase bonus for multi word matches
* Domain score boost

Top ranked chunks are passed to inference.

### Answer

Model instructions enforce:

* Use only provided context
* Match symptoms to correct procedure
* Refuse unsupported claims

Responses include chunk citations.

### Log

Each query records:

* Retrieved chunks
* Confidence scores
* Domain detection
* Grounding sources

Full traceability.

---

## Architecture

```mermaid
flowchart TB
    subgraph INGESTION["Ingestion Pipeline"]
        PDF[PDF File]
        HASH[Hash + Extract Text]
        CHUNK[Sliding-Window Chunking<br/>400 tokens, 25% overlap]
        EMBED_I[Embedding Engine<br/>BGE / Fake]
        STORE[Vector Index Store]
        EVAL[Evaluation Engine]
        PROMOTE[Promotion<br/>active_index.json]
        PDF --> HASH --> CHUNK --> EMBED_I --> STORE --> EVAL --> PROMOTE
    end

    subgraph INFERENCE["Inference Pipeline"]
        Q[Question]
        RESOLVE[Resolve Active Index<br/>registry + active_index.json]
        LOAD[Load Index]
        EMBED_Q[Embed Query]
        RETRIEVE[Retrieve<br/>BGE + BM25 RRF + phrase bonus]
        HYBRID[Domain Boost<br/>optional]
        CONTEXT[Assemble Context]
        LLM[LLM Backend<br/>Ollama / Fake]
        ANSWER[Answer + Citations]
        Q --> RESOLVE --> LOAD --> EMBED_Q --> RETRIEVE --> HYBRID --> CONTEXT --> LLM --> ANSWER
    end

    subgraph STORAGE["Persistence"]
        REGISTRY[(Registry)]
        ACTIVE[(Active Index Pointer)]
        INDEX[(Index Files<br/>v1.json, etc.)]
        EVALS[(Evaluations)]
    end

    PROMOTE --> REGISTRY
    PROMOTE --> ACTIVE
    STORE --> INDEX
    EVAL --> EVALS
    RESOLVE --> REGISTRY
    RESOLVE --> ACTIVE
    LOAD --> INDEX
```

---

## Design Constraints

* Deterministic retrieval
* Immutable index versions
* Evaluation required before activation
* Explicit refusal when unsupported
* Configurable local model backend

No approximate nearest neighbor search. No stochastic retrieval layer.

---

## Stack

* Embeddings: BGE base en v1.5
* Model: Llama 3.1 8B via Ollama
* Search: Cosine similarity plus BM25
* Storage: JSON versioned indexes
* Tests: 119 passing

---

## Prerequisites

This project is **NVIDIA-only**. RAGOps requires an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Validate your setup with:

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

If this command fails, install the toolkit before running the stack.

---

## Getting Started

### Docker

```
docker compose up -d --build
# First time only, pulls ~4GB model
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ragops bash
ragops run --pdf /app/data/test_pdfs/ragops_semantic_test_pdf.pdf
```

First model load may take one to two minutes. Later queries typically complete within 15 to 30 seconds depending on hardware.

### Local Install

```
git clone https://github.com/ereztdev/RAGOps.git
cd RAGOps
pip install -e .

ragops run --pdf path/to/manual.pdf
```

Step by step:

```
ragops ingest --pdf path/to/manual.pdf
ragops build-index --index-version v1 --overwrite
ragops ask --query "Your troubleshooting question"
```

---

## CLI Commands

| Command     | Purpose                  |
| ----------- | ------------------------ |
| ingest      | Parse PDF into chunks    |
| build-index | Create embedding index   |
| evaluate    | Run retrieval checks     |
| promote     | Activate evaluated index |
| ask         | Query active index       |
| run         | End to end workflow      |

---

## Requirements

* Python 3.10 or higher
* sentence transformers
* pypdf
* Ollama

---

## License

AGPL 3.0
