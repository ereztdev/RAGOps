"""
Segment 9: End-to-End CLI. Thin orchestration over existing pipeline modules.
Commands: ingest, build-index, evaluate, promote, ask. Deterministic filesystem outputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from core.schema import EvaluationConfig
from core.serialization import load_ingestion_output
from pipeline.embedding.embedding_engine import (
    BgeEmbeddingBackend,
    FakeEmbeddingBackend,
    is_test_only_index_version,
    run_embedding_pipeline,
)
from pipeline.evaluation.evaluation_engine import (
    load_evaluation_report,
    evaluate_retrieval,
)
from pipeline.ingestion.ingestion_pipeline import run_ingestion_pipeline
from pipeline.inference.inference_runner import (
    FakeInferenceBackend,
    run_inference_using_active_index,
)
from pipeline.promotion.index_registry import (
    REGISTRY_STATUS_EVALUATED,
    get_entry,
    load_registry,
    register_index,
    resolve_active_index,
    update_entry_status,
)
from pipeline.promotion.promoter import (
    EvaluationFailedError,
    promote_index,
)
from pipeline.retrieval.retrieval_engine import build_index_snapshot, retrieve
from storage.vector_index_store import load_index, save_index

DEFAULT_INDEXES_DIR = "indexes"
DEFAULT_EVALUATIONS_DIR = "evaluations"
DEFAULT_TOP_K = 5
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_MIN_ALPHABETIC_RATIO = 0.5
DEFAULT_MAX_ENTROPY = 5.0


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


def cmd_ingest(args: argparse.Namespace) -> int:
    """ingest --pdf <path> [--out <ingestion_output.json>]"""
    try:
        doc = run_ingestion_pipeline(args.pdf, output_path=args.out if args.out else None)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"ingest error: {e}", file=sys.stderr)
        return 1
    if not doc.chunks:
        print("ingest error: PDF produced no extractable text (empty extract)", file=sys.stderr)
        return 1
    return 0


def cmd_build_index(args: argparse.Namespace) -> int:
    """build-index --ingestion <path> --index-version <id> --out <index_path>. Default: BGE backend. Use --use-fake for test-only."""
    out_path = Path(args.out)
    if out_path.exists():
        print(f"build-index error: index file already exists: {out_path}", file=sys.stderr)
        return 1
    indexes_dir = Path(args.indexes_dir)
    registry_path = indexes_dir / "registry.json"
    try:
        document = load_ingestion_output(args.ingestion)
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
    save_index(snapshot, out_path)
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
    """evaluate --index-version <id> --queries <queries.json> --out <evaluation_path>"""
    indexes_dir = Path(args.indexes_dir)
    registry_path = indexes_dir / "registry.json"
    out_dir = Path(args.out)
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
    """ask --query "<question>". Backend matches active index provenance (BGE or test-only)."""
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
    llm = FakeInferenceBackend()
    result = run_inference_using_active_index(
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


def main() -> int:
    parser = argparse.ArgumentParser(prog="ragops", description="RAGOps pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_indexes_dir(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--indexes-dir",
            default=DEFAULT_INDEXES_DIR,
            help=f"Directory for registry and active index (default: {DEFAULT_INDEXES_DIR})",
        )

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest PDF into typed Document + Chunks")
    p_ingest.add_argument("--pdf", required=True, help="Path to PDF file")
    p_ingest.add_argument("--out", default=None, help="Optional path to write ingestion_output.json")
    p_ingest.set_defaults(func=cmd_ingest)

    # build-index
    p_build = subparsers.add_parser("build-index", help="Build index from ingestion output and register (default: BGE backend)")
    _add_indexes_dir(p_build)
    p_build.add_argument("--ingestion", required=True, help="Path to ingestion_output.json")
    p_build.add_argument("--index-version", required=True, dest="index_version", help="Index version id (must not start with 'fake' unless --use-fake)")
    p_build.add_argument("--out", required=True, help="Path to write index file (must not exist)")
    p_build.add_argument("--use-fake", action="store_true", dest="use_fake", help="Use test-only fake embedding backend (index_version may start with 'fake')")
    p_build.set_defaults(func=cmd_build_index)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate retrieval and persist reports")
    _add_indexes_dir(p_eval)
    p_eval.add_argument("--index-version", required=True, dest="index_version", help="Index version id")
    p_eval.add_argument("--queries", required=True, help="Path to queries.json (array of query strings)")
    p_eval.add_argument("--out", required=True, help="Directory to write evaluation report(s)")
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
    p_ask.add_argument("--min-confidence", type=float, default=None, dest="min_confidence", help="Min top similarity score (default: 0.5)")
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
