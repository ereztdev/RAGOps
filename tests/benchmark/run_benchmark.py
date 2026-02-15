#!/usr/bin/env python3
"""
Benchmark script for RAGOps retrieval + inference.
Loads corpus.json, runs ragops ask for each question, checks expected_contains and expected_refusal.
Does NOT run as part of pytest. Run: python tests/benchmark/run_benchmark.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_PATH = Path(__file__).resolve().parent / "corpus.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_corpus() -> dict:
    """Load corpus.json."""
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_ask(query: str, pdf_stem: str | None = None) -> tuple[str, int]:
    """Shell out to ragops ask --query <question>. Returns (stdout, returncode)."""
    cmd = [sys.executable, "-m", "ragops.cli.main", "ask"]
    if pdf_stem:
        cmd += ["--pdf-stem", pdf_stem]
    cmd += ["--min-alphabetic-ratio", "0.25", "--max-entropy", "6.5"]
    cmd += ["--query", query]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    return combined, result.returncode


def check_expected_contains(answer: str, expected: list[str]) -> bool:
    """Check that all expected terms appear in answer (case-insensitive, commas stripped)."""
    answer_lower = answer.lower().replace(",", "")
    return all(term.lower().replace(",", "") in answer_lower for term in expected)


def check_expected_refusal(answer: str, returncode: int) -> bool:
    """Check that answer indicates refusal (contains refusal phrases)."""
    answer_lower = answer.lower()
    refusal_phrases = [
        "refusal", "refused",
        "cannot find", "not found", "not mentioned", "does not mention", "no information",
    ]
    return any(p in answer_lower for p in refusal_phrases) or returncode != 0


def main() -> int:
    corpus = load_corpus()
    questions = corpus.get("questions", [])
    if not questions:
        print("No questions in corpus", file=sys.stderr)
        return 1

    results: list[dict] = []
    category_stats: dict[str, list[bool]] = {}
    pdf_stem = Path(corpus.get("manual", "")).stem or None

    print(f"Running benchmark: {len(questions)} questions from {corpus.get('manual', 'unknown')}")
    print("-" * 60)

    for q in questions:
        qid = q.get("id", "?")
        question = q.get("question", "")
        expected_contains = q.get("expected_contains", [])
        expected_refusal = q.get("expected_refusal", False)
        category = q.get("category", "unknown")

        answer_text, returncode = run_ask(question, pdf_stem=pdf_stem)

        if expected_refusal:
            passed = check_expected_refusal(answer_text, returncode)
        else:
            passed = check_expected_contains(answer_text, expected_contains) if expected_contains else True

        if category not in category_stats:
            category_stats[category] = []
        category_stats[category].append(passed)

        status = "PASS" if passed else "FAIL"
        results.append({
            "id": qid,
            "category": category,
            "question": question,
            "passed": passed,
            "expected_refusal": expected_refusal,
            "expected_contains": expected_contains,
            "answer_snippet": answer_text[:500],
        })
        print(f"  {qid:12} {category:18} {status}")

    # Summary table
    print("-" * 60)
    print("\n| Category       | Questions | Correct | Accuracy |")
    print("|----------------|-----------|---------|----------|")

    total_correct = 0
    total_questions = 0
    for cat in sorted(category_stats.keys()):
        vals = category_stats[cat]
        correct = sum(vals)
        n = len(vals)
        pct = (100 * correct / n) if n else 0
        total_correct += correct
        total_questions += n
        print(f"| {cat:14} | {n:9} | {correct:7} | {pct:6.0f}% |")

    overall_pct = (100 * total_correct / total_questions) if total_questions else 0
    print(f"| {'Overall':14} | {total_questions:9} | {total_correct:7} | {overall_pct:6.0f}% |")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{timestamp}.json"
    out_data = {
        "timestamp": timestamp,
        "manual": corpus.get("manual", ""),
        "total": total_questions,
        "correct": total_correct,
        "accuracy_pct": overall_pct,
        "by_category": {
            cat: {"correct": sum(vals), "total": len(vals)}
            for cat, vals in category_stats.items()
        },
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    return 0 if total_correct == total_questions else 1


if __name__ == "__main__":
    sys.exit(main())
