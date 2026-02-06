"""Pluggable inference backends: protocol, fake (tests), and Ollama."""

from pipeline.inference.backends.base import (
    FakeInferenceBackend,
    InferenceBackend,
)
from pipeline.inference.backends.ollama_backend import OllamaInferenceBackend

__all__ = [
    "FakeInferenceBackend",
    "InferenceBackend",
    "OllamaInferenceBackend",
]
