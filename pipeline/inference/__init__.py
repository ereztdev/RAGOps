"""Grounded inference: answer only when supported by evaluated retrieval."""

from pipeline.inference.inference_runner import (
    FakeInferenceBackend,
    REFUSAL_EVALUATION_FAILED_PREFIX,
    REFUSAL_NO_RETRIEVAL_HITS,
    REFUSAL_UNSUPPORTED_BY_CONTEXT,
    run_grounded_inference,
)

__all__ = [
    "FakeInferenceBackend",
    "REFUSAL_EVALUATION_FAILED_PREFIX",
    "REFUSAL_NO_RETRIEVAL_HITS",
    "REFUSAL_UNSUPPORTED_BY_CONTEXT",
    "run_grounded_inference",
]
