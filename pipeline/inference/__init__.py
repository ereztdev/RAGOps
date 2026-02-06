"""Grounded inference: answer only when supported by evaluated retrieval."""

from pipeline.inference.backends import FakeInferenceBackend, InferenceBackend
from pipeline.inference.inference_runner import (
    REFUSAL_EVALUATION_FAILED_PREFIX,
    REFUSAL_NO_ACTIVE_INDEX,
    REFUSAL_NO_RETRIEVAL_HITS,
    REFUSAL_UNSUPPORTED_BY_CONTEXT,
    run_grounded_inference,
    run_inference_using_active_index,
)

__all__ = [
    "FakeInferenceBackend",
    "InferenceBackend",
    "REFUSAL_EVALUATION_FAILED_PREFIX",
    "REFUSAL_NO_ACTIVE_INDEX",
    "REFUSAL_NO_RETRIEVAL_HITS",
    "REFUSAL_UNSUPPORTED_BY_CONTEXT",
    "run_grounded_inference",
    "run_inference_using_active_index",
]
