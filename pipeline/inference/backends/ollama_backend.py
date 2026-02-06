"""
Ollama inference backend: real LLM via local Ollama (Llama 3.1 8B default).
Optional dependency: clear error on instantiation if ollama package is missing.
Deterministic when temperature=0.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger("ragops.inference")

_OLLAMA_AVAILABLE = False
_ollama_client_module = None

try:
    from ollama import Client as OllamaClient
    _OLLAMA_AVAILABLE = True
    _ollama_client_module = "ollama"
except ImportError:
    pass


SYSTEM_PROMPT = """You are a grounded question-answering assistant. Answer ONLY using the provided context. Do not invent or assume facts. If the context does not contain enough information to answer, say so briefly. When you use specific facts from the context, your answer should be supported by that text. Keep answers concise. Do not repeat the question."""


class OllamaInferenceBackend:
    """
    Inference backend using local Ollama API. Uses temperature=0 for deterministic output.
    Retries once on timeout or connection error; then returns a refusal message and logs.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ) -> None:
        if not _OLLAMA_AVAILABLE:
            raise ImportError(
                "The 'ollama' package is required for OllamaInferenceBackend. "
                "Install it with: pip install ollama"
            ) from None
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = OllamaClient(host=base_url)

    @property
    def model_id(self) -> str:
        return self._model

    def generate(self, query: str, context: str) -> str:
        if not context.strip():
            return ""
        prompt = f"""Context (use only this to answer):\n\n{context}\n\nQuestion: {query}\n\nAnswer (based only on the context above):"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        options = {"temperature": 0}
        t0 = time.perf_counter()
        logger.info(
            "request_sent backend=ollama model=%s query_len=%d context_len=%d",
            self._model,
            len(query),
            len(context),
        )
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = self._client.chat(
                    model=self._model,
                    messages=messages,
                    options=options,
                )
                elapsed = time.perf_counter() - t0
                answer = (response.message.content or "").strip()
                logger.info(
                    "response_received backend=ollama model=%s elapsed_sec=%.3f response_len=%d",
                    self._model,
                    elapsed,
                    len(answer),
                )
                return answer
            except (TimeoutError, OSError, ConnectionError) as e:
                last_error = e
                logger.warning(
                    "retry_triggered backend=ollama model=%s attempt=%d error=%s",
                    self._model,
                    attempt + 1,
                    type(e).__name__,
                )
                if attempt == 0:
                    continue
                break
        logger.error(
            "final_failure backend=ollama model=%s error=%s",
            self._model,
            last_error,
        )
        return ""
