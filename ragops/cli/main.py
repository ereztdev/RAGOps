"""
Segment 9: End-to-End CLI. Thin orchestration over existing pipeline modules.
Commands: ingest, build-index, evaluate, promote, ask. Deterministic filesystem outputs.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

_inference_logger = logging.getLogger("ragops.inference")

from core.schema import EvaluationConfig
from core.serialization import ingestion_output_to_dict, load_ingestion_output
from pipeline.embedding.embedding_engine import (
    BgeEmbeddingBackend,
    FakeEmbeddingBackend,
    is_test_only_index_version,
    run_embedding_pipeline,
)
from pipeline.evaluation.evaluation_engine import (
    evaluate_retrieval,
    load_evaluation_report,
)
from pipeline.ingestion.ingestion_pipeline import run_ingestion_pipeline
from pipeline.inference.backends import (
    FakeInferenceBackend,
    OllamaInferenceBackend,
)
from pipeline.inference.inference_runner import run_inference_using_active_index
from pipeline.promotion.index_registry import (
    REGISTRY_STATUS_EVALUATED,
    get_entry,
    load_registry,
    register_index,
    register_or_replace_index,
    resolve_active_index,
    update_entry_status,
)
from pipeline.promotion.promoter import (
    EvaluationFailedError,
    promote_index,
)
from core.schema import HybridRetrievalConfig
from pipeline.retrieval.retrieval_engine import build_index_snapshot, retrieve
from storage.vector_index_store import load_index, save_index

try:
    from ragops.logging import append_run_log as ragops_append_run_log
except ImportError:
    ragops_append_run_log = None

DEFAULT_INDEXES_DIR = "indexes"
DEFAULT_EVALUATIONS_DIR = "evaluations"
DEFAULT_RUN_LOG = "run_log.md"
DEFAULT_TOP_K = 5
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_MIN_ALPHABETIC_RATIO = 0.5
DEFAULT_MAX_ENTROPY = 5.0

# Terminal colors (only when stdout is a TTY)
def _color(code: str) -> str:
    return f"\033[{code}m" if sys.stdout.isatty() else ""


C_RESET = _color("0")
C_GREEN = _color("32")
C_RED = _color("31")
C_YELLOW = _color("33")
C_CYAN = _color("36")
C_DIM = _color("2")


def _eval_config_from_args(args: argparse.Namespace) -> EvaluationConfig:
    """Build EvaluationConfig from CLI args (optional overrides)."""
    def _float_opt(name: str, default: float) -> float:
        v = getattr(args, name, None)
        return v if v is not None else default
    return EvaluationConfig(
        min_confidence_score=_float_opt("min_confidence", DEFAULT_MIN_CONFIDENCE),
        min_alphabetic_ratio=_float_opt("min_alphabetic_ratio", DEFAULT_MIN_ALPHABETIC_RATIO),
        max_entropy=_float_opt("max_entropy", DEFAULT_MAX_ENTROPY),
    )


def _created_at_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _relative_path_from_cwd(file_path: str | Path) -> Path:
    """Return path relative to cwd if under cwd, else absolute."""
    p = Path(file_path).resolve()
    cwd = Path.cwd()
    try:
        return p.relative_to(cwd)
    except ValueError:
        return p


def _default_ingestion_path(indexes_dir: str | Path) -> Path:
    """Default path for ingestion output under indexes_dir (idempotent default)."""
    return Path(indexes_dir) / "ingestion.json"


def cmd_ingest(args: argparse.Namespace) -> int:
    """ingest --pdf <path> [--out <ingestion_output.json>]. Default --out is indexes_dir/ingestion.json for idempotent pipeline."""
    out_path = args.out if args.out else _default_ingestion_path(args.indexes_dir)
    try:
        doc = run_ingestion_pipeline(args.pdf, output_path=str(out_path))
    except (FileNotFoundError, RuntimeError) as e:
        print(f"ingest error: {e}", file=sys.stderr)
        return 1
    if not doc.chunks:
        print("ingest error: PDF produced no extractable text (empty extract)", file=sys.stderr)
        return 1

    result = ingestion_output_to_dict(doc)
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))

    out_rel = _relative_path_from_cwd(out_path)
    print(f"\nOutput written to (relative to cwd): {out_rel}")
    return 0


def _default_index_path(indexes_dir: str | Path, index_version: str) -> Path:
    """Default path for index file under indexes_dir (idempotent default)."""
    return Path(indexes_dir) / f"{index_version}.json"


def _approx_tokens_from_chunks(chunks: list) -> int:
    """Rough token count from chunk texts (whitespace words * 1.3). No hashes."""
    total_words = sum(len((getattr(c, "text", None) or "").split()) for c in chunks)
    return int(total_words * 1.3)


def _append_run_log(
    log_path: Path,
    pdf_name: str,
    pdf_size_kb: float,
    pages: int,
    chunks: int,
    tokens_approx: int,
    part1_sec: float,
    part2_sec: float,
    index_version: str = "v1",
    question: str = "",
    model: str = "",
    answer: str = "",
) -> None:
    """Append one run entry to run_log.md. Repo-friendly: no long hashes."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    question_line = f"- **Question:** {question.replace(chr(10), ' ')}\n" if question else ""
    model_line = f"- **Model:** {model}\n" if model else ""
    answer_escaped = (answer or "(no answer recorded)").replace(chr(10), " ").strip()
    answer_line = f"- **Answer:** {answer_escaped}\n"
    block = f"""
## {ts} — {pdf_name}
- **PDF:** {pdf_name} · {pdf_size_kb:.0f} KB · **Pages:** {pages}
- **Chunks:** {chunks} · **Tokens (approx):** {tokens_approx:,}
- **Index version:** {index_version}
{model_line}{question_line}{answer_line}- **Part 1 (ingest+build):** {part1_sec:.1f} s
- **Part 2 (eval+ask):** {part2_sec:.1f} s
"""
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("# RAGOps run log\n\n*Appended by `ragops run`. Part 1 = ingest+build; Part 2 = evaluate+promote+ask.*\n")
        f.write(block)


def cmd_build_index(args: argparse.Namespace) -> int:
    """build-index --ingestion <path> --index-version <id> [--out <index_path>]. Default --out is indexes_dir/<index_version>.json. Use --overwrite to replace existing. Default: BGE backend. Use --use-fake for test-only."""
    indexes_dir = Path(args.indexes_dir)
    ingestion_path = args.ingestion if args.ingestion is not None else str(indexes_dir / "ingestion.json")
    out_path = Path(args.out) if args.out is not None else _default_index_path(indexes_dir, args.index_version)
    if out_path.exists() and not getattr(args, "overwrite", False):
        print(f"build-index error: index file already exists: {out_path} (use --overwrite to replace)", file=sys.stderr)
        return 1
    registry_path = indexes_dir / "registry.json"
    try:
        document = load_ingestion_output(ingestion_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"build-index error: {e}", file=sys.stderr)
        return 1
    if not document.chunks:
        print("build-index error: ingestion has no chunks", file=sys.stderr)
        return 1
    use_fake = getattr(args, "use_fake", False)
    if args.index_version.strip().lower().startswith("fake") and not use_fake:
        print(
            "build-index error: index_version_id must not start with 'fake' unless --use-fake is set (test-only)",
            file=sys.stderr,
        )
        return 1
    backend = FakeEmbeddingBackend() if use_fake else BgeEmbeddingBackend()
    embeddings = run_embedding_pipeline(document, backend)
    snapshot = build_index_snapshot(document, embeddings, args.index_version)
    save_index(snapshot, out_path)  # out_path is Path
    created_at = _created_at_utc()
    try:
        register_index(
            registry_path,
            index_version_id=args.index_version,
            index_path=str(out_path.resolve()),
            created_at=created_at,
        )
    except ValueError as e:
        print(f"build-index error: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """evaluate --index-version <id> --queries <queries.json> [--out <evaluation_dir>]. Default --out is evaluations/ for idempotent pipeline."""
    indexes_dir = Path(args.indexes_dir)
    registry_path = indexes_dir / "registry.json"
    out_dir = Path(args.out)  # default set in parser
    if not registry_path.is_file():
        print(f"evaluate error: registry not found: {registry_path}", file=sys.stderr)
        return 1
    entries = load_registry(registry_path)
    entry = get_entry(entries, args.index_version)
    if entry is None:
        print(f"evaluate error: index version not in registry: {args.index_version}", file=sys.stderr)
        return 1
    index_path = Path(entry.index_path)
    if not index_path.is_absolute():
        index_path = indexes_dir / index_path
    if not index_path.is_file():
        print(f"evaluate error: index file not found: {index_path}", file=sys.stderr)
        return 1
    try:
        with open(args.queries, encoding="utf-8") as f:
            queries_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"evaluate error: {e}", file=sys.stderr)
        return 1
    if isinstance(queries_data, list):
        queries = [str(q) for q in queries_data]
    elif isinstance(queries_data, dict) and "queries" in queries_data:
        queries = [str(q) for q in queries_data["queries"]]
    else:
        print("evaluate error: queries file must be JSON array of strings or {queries: [...]}", file=sys.stderr)
        return 1
    if not queries:
        print("evaluate error: no queries in queries file", file=sys.stderr)
        return 1
    index = load_index(index_path)
    backend = FakeEmbeddingBackend() if is_test_only_index_version(index.index_version_id) else BgeEmbeddingBackend()
    config = _eval_config_from_args(args)
    top_k = getattr(args, "top_k", DEFAULT_TOP_K) or DEFAULT_TOP_K
    first_report_id: str | None = None
    out_dir.mkdir(parents=True, exist_ok=True)
    for query in queries:
        retrieval_result = retrieve(query, index, backend, top_k)
        report = evaluate_retrieval(
            retrieval_result,
            query,
            config,
            evaluations_dir=out_dir,
        )
        if first_report_id is None:
            first_report_id = report.evaluation_id
    if first_report_id is None:
        return 1
    try:
        update_entry_status(
            registry_path,
            args.index_version,
            REGISTRY_STATUS_EVALUATED,
            evaluation_report_id=first_report_id,
        )
    except ValueError as e:
        print(f"evaluate error: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    """promote --index-version <id> --evaluation <evaluation_path>"""
    indexes_dir = Path(args.indexes_dir)
    registry_path = indexes_dir / "registry.json"
    active_index_path = indexes_dir / "active_index.json"
    try:
        report = load_evaluation_report(args.evaluation)
    except (FileNotFoundError, ValueError) as e:
        print(f"promote error: {e}", file=sys.stderr)
        return 1
    if not report.overall_pass:
        print("promote error: evaluation did not pass (overall_pass is False); promotion refused", file=sys.stderr)
        return 1
    try:
        promote_index(
            args.index_version,
            report,
            registry_path,
            active_index_path,
        )
    except EvaluationFailedError:
        print("promote error: evaluation did not pass; promotion refused", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"promote error: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """ask --query "<question>" [--llm fake|ollama] [--model <ollama_model>]. Backend matches active index provenance (BGE or test-only)."""
    indexes_dir = Path(args.indexes_dir)
    try:
        _index_version_id, index_path = resolve_active_index(
            indexes_dir / "registry.json", indexes_dir / "active_index.json"
        )
    except (FileNotFoundError, ValueError):
        _index_version_id = None
    if _index_version_id is not None:
        path = Path(index_path)
        if not path.is_absolute():
            path = indexes_dir / path
        if path.is_file():
            index = load_index(path)
            backend = FakeEmbeddingBackend() if is_test_only_index_version(index.index_version_id) else BgeEmbeddingBackend()
        else:
            backend = BgeEmbeddingBackend()
    else:
        backend = BgeEmbeddingBackend()
    top_k = getattr(args, "top_k", DEFAULT_TOP_K) or DEFAULT_TOP_K

    llm_name = getattr(args, "llm", "ollama") or "ollama"
    if llm_name == "fake":
        llm = FakeInferenceBackend()
        _inference_logger.info("backend_selected backend=fake model_id=%s", llm.model_id)
    elif llm_name == "ollama":
        model = getattr(args, "model", "llama3.1:8b") or "llama3.1:8b"
        try:
            llm = OllamaInferenceBackend(model=model)
            _inference_logger.info("backend_selected backend=ollama model_id=%s", llm.model_id)
        except ImportError as e:
            print(f"ask error: {e}", file=sys.stderr)
            return 1
    else:
        print(f"ask error: unsupported --llm {llm_name!r}", file=sys.stderr)
        return 1

    result, _retrieval_result, _detected_domains = run_inference_using_active_index(
        args.query,
        indexes_dir,
        backend,
        top_k,
        llm,
        _eval_config_from_args(args),
        evaluations_dir=None,
    )
    if result.found:
        print(result.answer_text or "")
        if result.citation_chunk_ids:
            print("citations:", ",".join(result.citation_chunk_ids))
    else:
        print("refusal:", result.refusal_reason or "unknown", file=sys.stderr)
        return 1
    return 0


def _spinner_while(stop_event: threading.Event, prefix: str = "Asking Ollama... ") -> None:
    """Print a spinning indicator until stop_event is set. Runs in thread."""
    chars = "|/-\\"
    i = 0
    while not stop_event.wait(timeout=0.4):
        sys.stdout.write(f"\r  {prefix}{chars[i % 4]} ")
        sys.stdout.flush()
        i += 1
    sys.stdout.write("\r" + " " * (len(prefix) + 6) + "\r")
    sys.stdout.flush()


def cmd_run(args: argparse.Namespace) -> int:
    """E2E run: ingest + build index in one step, then one question → evaluate, promote, ask (Ollama). Colored terminal output."""
    # Auto-name: indexes/<pdf_stem>/ so each PDF gets its own subfolder under indexes/
    base_indexes = Path(args.indexes_dir)
    pdf_stem = Path(args.pdf).stem
    indexes_dir = base_indexes / pdf_stem
    indexes_dir.mkdir(parents=True, exist_ok=True)
    evaluations_dir = Path(getattr(args, "evaluations_dir", None) or DEFAULT_EVALUATIONS_DIR)
    index_version = getattr(args, "index_version", "v1") or "v1"
    ingestion_path = indexes_dir / "ingestion.json"
    index_path = indexes_dir / f"{index_version}.json"
    registry_path = indexes_dir / "registry.json"
    active_index_path = indexes_dir / "active_index.json"

    part1_start = time.perf_counter()
    # Step 1: Ingest
    try:
        doc = run_ingestion_pipeline(args.pdf, output_path=str(ingestion_path))
    except (FileNotFoundError, RuntimeError) as e:
        print(f"{C_RED}ingest error: {e}{C_RESET}", file=sys.stderr)
        return 1
    if not doc.chunks:
        print(f"{C_RED}ingest error: PDF produced no extractable text (empty extract){C_RESET}", file=sys.stderr)
        return 1

    # Step 1: Build index (overwrite if exists for idempotent run)
    try:
        document = load_ingestion_output(str(ingestion_path))
    except (FileNotFoundError, ValueError) as e:
        print(f"{C_RED}build-index error: {e}{C_RESET}", file=sys.stderr)
        return 1
    use_fake = getattr(args, "use_fake", False)
    if index_version.strip().lower().startswith("fake") and not use_fake:
        print(
            f"{C_RED}build-index error: index_version must not start with 'fake' unless --use-fake{C_RESET}",
            file=sys.stderr,
        )
        return 1
    backend_emb = FakeEmbeddingBackend() if use_fake else BgeEmbeddingBackend()
    embeddings = run_embedding_pipeline(document, backend_emb)
    snapshot = build_index_snapshot(document, embeddings, index_version)
    save_index(snapshot, index_path)
    created_at = _created_at_utc()
    try:
        register_or_replace_index(
            registry_path,
            index_version_id=index_version,
            index_path=str(index_path.resolve()),
            created_at=created_at,
        )
    except ValueError as e:
        print(f"{C_RED}build-index error: {e}{C_RESET}", file=sys.stderr)
        return 1

    part1_sec = time.perf_counter() - part1_start
    pdf_path = Path(args.pdf)
    pdf_name = pdf_path.name
    pdf_size_kb = pdf_path.stat().st_size / 1024.0
    n_chunks = len(document.chunks)
    n_pages = max((c.page_number for c in document.chunks), default=0)
    tokens_approx = _approx_tokens_from_chunks(document.chunks)

    print(f"{C_GREEN}Index ready.{C_RESET} index_version={index_version!r} path={_relative_path_from_cwd(index_path)} ({part1_sec:.1f} s)")

    # Step 2: One question
    try:
        prompt_text = f"{C_YELLOW}Ask a question (or Enter to exit):{C_RESET} "
        question = input(prompt_text).strip()
    except EOFError:
        return 0
    if not question:
        return 0

    part2_start = time.perf_counter()
    top_k = getattr(args, "top_k", DEFAULT_TOP_K) or DEFAULT_TOP_K
    config = _eval_config_from_args(args)
    index = load_index(index_path)
    try:
        from ragops.config import METADATA_BOOST_ENABLED
        from ragops.retrieval.hybrid_retrieval import hybrid_retrieve
        if METADATA_BOOST_ENABLED and hybrid_retrieve is not None:
            retrieval_result, _ = hybrid_retrieve(
                question, index, backend_emb,
                final_top_k=top_k,
                hybrid_config=HybridRetrievalConfig(),
            )
        else:
            retrieval_result = retrieve(
                question, index, backend_emb, top_k,
                hybrid_config=HybridRetrievalConfig(),
            )
    except ImportError:
        retrieval_result = retrieve(
            question, index, backend_emb, top_k,
            hybrid_config=HybridRetrievalConfig(),
        )
    print(f"  {C_DIM}[{time.perf_counter() - part2_start:.1f} s] Retrieved context{C_RESET}", file=sys.stderr)
    report = evaluate_retrieval(retrieval_result, question, config, evaluations_dir=evaluations_dir)
    print(f"  {C_DIM}[{time.perf_counter() - part2_start:.1f} s] Evaluated{C_RESET}", file=sys.stderr)

    if not report.overall_pass:
        failed = [name for name, sr in report.signals.items() if not sr.passed]
        print(f"{C_RED}Evaluation did not pass for this query. Refusing to promote or ask.{C_RESET}", file=sys.stderr)
        for name in failed:
            sr = report.signals[name]
            details = sr.details if isinstance(sr.details, dict) else {}
            if name == "low_confidence":
                ts = details.get("top_score")
                th = details.get("threshold")
                if ts is not None and th is not None:
                    print(f"{C_DIM}  • low_confidence: top similarity {ts:.3f} < {th} (try --min-confidence {max(0.0, ts - 0.05):.2f}){C_RESET}", file=sys.stderr)
                else:
                    print(f"{C_DIM}  • {name}: {details}{C_RESET}", file=sys.stderr)
            elif name == "gibberish_detected":
                n_bad = len(details.get("gibberish_chunk_ids", []))
                n_checked = details.get("checked", 0)
                print(f"{C_DIM}  • gibberish_detected: {n_bad} of {n_checked} retrieved chunks flagged as low-signal (try --min-alphabetic-ratio 0.3 or --max-entropy 6.0){C_RESET}", file=sys.stderr)
            else:
                print(f"{C_DIM}  • {name}: {details}{C_RESET}", file=sys.stderr)
        return 1

    try:
        update_entry_status(
            registry_path,
            index_version,
            REGISTRY_STATUS_EVALUATED,
            evaluation_report_id=report.evaluation_id,
        )
    except ValueError as e:
        print(f"{C_RED}evaluate error: {e}{C_RESET}", file=sys.stderr)
        return 1

    try:
        promote_index(index_version, report, registry_path, active_index_path)
    except (EvaluationFailedError, Exception) as e:
        print(f"{C_RED}promote error: {e}{C_RESET}", file=sys.stderr)
        return 1
    print(f"  {C_DIM}[{time.perf_counter() - part2_start:.1f} s] Promoted{C_RESET}", file=sys.stderr)

    llm_name = getattr(args, "llm", "ollama") or "ollama"
    model_used = ""
    if llm_name == "fake":
        llm = FakeInferenceBackend()
        model_used = "fake"
    elif llm_name == "ollama":
        model_used = getattr(args, "model", "llama3.1:8b") or "llama3.1:8b"
        try:
            llm = OllamaInferenceBackend(model=model_used)
        except ImportError as e:
            print(f"{C_RED}ask error: {e}{C_RESET}", file=sys.stderr)
            return 1
    else:
        print(f"{C_RED}ask error: unsupported --llm {llm_name!r}{C_RESET}", file=sys.stderr)
        return 1

    # Progress callback: log each phase of "ask" (resolve → load → retrieve → evaluate → LLM)
    ask_phase_start = time.perf_counter()
    def _progress(phase: str, elapsed: float) -> None:
        print(f"  {C_DIM}[{elapsed:.1f} s] {phase}{C_RESET}", file=sys.stderr)
    progress_cb = _progress if sys.stderr.isatty() else None

    # Spinner while Ollama runs (long wait with no UX otherwise)
    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(target=_spinner_while, args=(stop_spinner,), daemon=True)
    if sys.stdout.isatty():
        spinner_thread.start()
    try:
        result, retrieval_result, detected_domains = run_inference_using_active_index(
            question,
            indexes_dir,
            backend_emb,
            top_k,
            llm,
            config,
            evaluations_dir=None,
            progress_callback=progress_cb,
        )
    finally:
        stop_spinner.set()
        if sys.stdout.isatty():
            spinner_thread.join(timeout=1.0)

    if progress_cb:
        print(f"  {C_DIM}[{time.perf_counter() - ask_phase_start:.1f} s] ask done (LLM responded){C_RESET}", file=sys.stderr)
    part2_sec = time.perf_counter() - part2_start

    run_log_path = getattr(args, "run_log", None)
    if run_log_path and str(run_log_path).lower() not in ("", "none", "no"):
        answer_for_log = (result.answer_text or "").strip() if result.found else (f"(refused: {result.refusal_reason or 'unknown'})")
        if not answer_for_log:
            answer_for_log = "(no answer recorded)"
        try:
            if ragops_append_run_log is not None:
                ragops_append_run_log(
                    Path(run_log_path),
                    pdf_name=pdf_name,
                    pdf_size_kb=pdf_size_kb,
                    pages=n_pages,
                    chunks=n_chunks,
                    tokens_approx=tokens_approx,
                    part1_sec=part1_sec,
                    part2_sec=part2_sec,
                    index_version=index_version,
                    question=question,
                    model=model_used,
                    answer=answer_for_log,
                    retrieval_result=retrieval_result,
                    detected_domains=detected_domains,
                )
            else:
                _append_run_log(
                    Path(run_log_path),
                    pdf_name=pdf_name,
                    pdf_size_kb=pdf_size_kb,
                    pages=n_pages,
                    chunks=n_chunks,
                    tokens_approx=tokens_approx,
                    part1_sec=part1_sec,
                    part2_sec=part2_sec,
                    index_version=index_version,
                    question=question,
                    model=model_used,
                    answer=answer_for_log,
                )
        except OSError as e:
            print(f"{C_DIM}(run log not written: {e}){C_RESET}", file=sys.stderr)

    if result.found:
        print(f"{C_GREEN}{result.answer_text or ''}{C_RESET}")
        if result.citation_chunk_ids:
            print(f"{C_DIM}citations: {','.join(result.citation_chunk_ids)}{C_RESET}")
        print(f"{C_DIM}(Part 2: {part2_sec:.1f} s){C_RESET}")
        return 0
    else:
        print(f"{C_RED}refusal: {result.refusal_reason or 'unknown'}{C_RESET}", file=sys.stderr)
        print(f"{C_DIM}(Part 2: {part2_sec:.1f} s){C_RESET}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="ragops", description="RAGOps pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_indexes_dir(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--indexes-dir",
            default=DEFAULT_INDEXES_DIR,
            help=f"Directory for registry and active index (default: {DEFAULT_INDEXES_DIR})",
        )

    # ingest (default --out under indexes-dir for idempotent pipeline)
    p_ingest = subparsers.add_parser("ingest", help="Ingest PDF into typed Document + Chunks")
    _add_indexes_dir(p_ingest)
    p_ingest.add_argument("--pdf", required=True, help="Path to PDF file")
    p_ingest.add_argument("--out", default=None, help="Path to write ingestion_output.json (default: indexes_dir/ingestion.json)")
    p_ingest.set_defaults(func=cmd_ingest)

    # build-index (default --ingestion and --out under indexes-dir for idempotent pipeline)
    p_build = subparsers.add_parser("build-index", help="Build index from ingestion output and register (default: BGE backend)")
    _add_indexes_dir(p_build)
    p_build.add_argument("--ingestion", default=None, help="Path to ingestion_output.json (default: indexes_dir/ingestion.json)")
    p_build.add_argument("--index-version", required=True, dest="index_version", help="Index version id (must not start with 'fake' unless --use-fake)")
    p_build.add_argument("--out", default=None, help="Path to write index file (default: indexes_dir/<index_version>.json)")
    p_build.add_argument("--overwrite", action="store_true", help="Overwrite existing index file (idempotent re-run)")
    p_build.add_argument("--use-fake", action="store_true", dest="use_fake", help="Use test-only fake embedding backend (index_version may start with 'fake')")
    p_build.set_defaults(func=cmd_build_index)

    # evaluate (default --out evaluations/ for idempotent pipeline)
    p_eval = subparsers.add_parser("evaluate", help="Evaluate retrieval and persist reports")
    _add_indexes_dir(p_eval)
    p_eval.add_argument("--index-version", required=True, dest="index_version", help="Index version id")
    p_eval.add_argument("--queries", required=True, help="Path to queries.json (array of query strings)")
    p_eval.add_argument("--out", default=DEFAULT_EVALUATIONS_DIR, help=f"Directory to write evaluation report(s) (default: {DEFAULT_EVALUATIONS_DIR})")
    p_eval.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, dest="top_k", help=f"Top-k for retrieval (default: {DEFAULT_TOP_K})")
    p_eval.add_argument("--min-confidence", type=float, default=None, dest="min_confidence", help="Min top similarity score for evaluation (default: 0.5)")
    p_eval.add_argument("--min-alphabetic-ratio", type=float, default=None, dest="min_alphabetic_ratio", help="Min alphabetic ratio for gibberish (default: 0.5)")
    p_eval.add_argument("--max-entropy", type=float, default=None, dest="max_entropy", help="Max character entropy for gibberish (default: 5.0)")
    p_eval.set_defaults(func=cmd_evaluate)

    # promote
    p_promote = subparsers.add_parser("promote", help="Promote index to active (requires passing evaluation)")
    _add_indexes_dir(p_promote)
    p_promote.add_argument("--index-version", required=True, dest="index_version", help="Index version id")
    p_promote.add_argument("--evaluation", required=True, help="Path to evaluation report JSON")
    p_promote.set_defaults(func=cmd_promote)

    # ask
    p_ask = subparsers.add_parser("ask", help="Answer question using active index only")
    _add_indexes_dir(p_ask)
    p_ask.add_argument("--query", required=True, help="Question to answer")
    p_ask.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, dest="top_k", help=f"Top-k for retrieval (default: {DEFAULT_TOP_K})")
    p_ask.add_argument("--llm", choices=["fake", "ollama"], default="ollama", help="Inference backend (default: ollama)")
    p_ask.add_argument("--model", default="llama3.1:8b", help="Ollama model name when --llm ollama (default: llama3.1:8b)")
    p_ask.add_argument("--min-confidence", type=float, default=None, dest="min_confidence", help="Min top similarity score (default: 0.5)")
    p_ask.add_argument("--min-alphabetic-ratio", type=float, default=None, dest="min_alphabetic_ratio", help="Min alphabetic ratio for gibberish (default: 0.5)")
    p_ask.add_argument("--max-entropy", type=float, default=None, dest="max_entropy", help="Max character entropy for gibberish (default: 5.0)")
    p_ask.set_defaults(func=cmd_ask)

    # run (E2E: ingest + build, then one question → evaluate, promote, ask; colored output)
    p_run = subparsers.add_parser(
        "run",
        help="E2E: ingest PDF + build index, then one question → evaluate, promote, ask (Ollama). Colored terminal.",
    )
    p_run.add_argument(
        "--indexes-dir",
        default=DEFAULT_INDEXES_DIR,
        help=f"Base directory; run writes to <base>/<pdf_stem>/ (default: {DEFAULT_INDEXES_DIR}), e.g. indexes/my_document/",
    )
    p_run.add_argument("--pdf", required=True, help="Path to PDF file")
    p_run.add_argument("--evaluations-dir", default=DEFAULT_EVALUATIONS_DIR, dest="evaluations_dir", help=f"Directory for evaluation reports (default: {DEFAULT_EVALUATIONS_DIR})")
    p_run.add_argument("--index-version", default="v1", dest="index_version", help="Index version id (default: v1)")
    p_run.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, dest="top_k", help=f"Top-k for retrieval (default: {DEFAULT_TOP_K})")
    p_run.add_argument("--llm", choices=["fake", "ollama"], default="ollama", help="Inference backend (default: ollama)")
    p_run.add_argument("--model", default="llama3.1:8b", help="Ollama model when --llm ollama (default: llama3.1:8b)")
    p_run.add_argument("--use-fake", action="store_true", dest="use_fake", help="Use test-only fake embedding (index_version may start with 'fake')")
    p_run.add_argument("--min-confidence", type=float, default=None, dest="min_confidence", help="Min top similarity score (default: 0.5)")
    p_run.add_argument("--min-alphabetic-ratio", type=float, default=None, dest="min_alphabetic_ratio", help="Min alphabetic ratio for gibberish (default: 0.5)")
    p_run.add_argument("--max-entropy", type=float, default=None, dest="max_entropy", help="Max character entropy for gibberish (default: 5.0)")
    p_run.add_argument("--run-log", default=DEFAULT_RUN_LOG, dest="run_log", help=f"Append run summary to this MD file (default: {DEFAULT_RUN_LOG}); use 'none' to disable")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
