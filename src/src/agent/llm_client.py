"""LLM client wrapper (Layer 3) — the graph's gateway to the external model API.

Wraps the LLM in a single retry-and-timeout boundary so that token timeouts,
rate limits, or transient API failures never crash the LangGraph execution.
All nodes call :class:`LLMClient` rather than the provider SDK directly, keeping
the provider swappable (OpenAI-compatible ``base_url`` by default).

The client is optional at construction time: when no API key is configured it
operates in a deterministic "offline" mode that returns canned, well-formed
judgements, so the graph can be unit-tested without network access.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional

from ..common.config import AppConfig
from ..common.logging_setup import get_logger

_logger: logging.Logger = get_logger("agent.llm_client")

# Load a project .env (if present) into os.environ so OPENAI_API_KEY /
# OPENAI_BASE_URL persist without manual export. No-op when no .env exists.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional
    pass


class LLMError(RuntimeError):
    """Raised when an LLM call fails terminally after all retries."""


class LLMClient:
    """Retry-protected gateway to the LLM, with an offline fallback."""

    def __init__(
        self,
        config: AppConfig,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Any = None,
    ) -> None:
        self._config = config
        self._model = config.agent.llm.model
        self._temperature = config.agent.llm.temperature
        self._timeout = config.agent.llm.request_timeout
        self._max_retries = config.agent.llm.max_retries
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._client = client  # injectable for tests

    @property
    def offline(self) -> bool:
        """True when no client/API key is available (offline fallback mode)."""
        return self._client is None and not self._api_key

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's text response, or an offline fallback string.

        Retries up to ``max_retries`` on exception with a small backoff. Raises
        :class:`LLMError` only if every attempt fails AND offline mode is off.
        """
        if self.offline and self._client is None:
            _logger.debug("LLM offline mode: returning canned response.")
            return _offline_response(system_prompt, user_prompt)

        client = self._client or self._build_client()
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 2):  # initial + retries
            try:
                return self._call(client, system_prompt, user_prompt)
            except Exception as exc:  # noqa: BLE001 - any failure is retried
                last_exc = exc
                _logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt, self._max_retries + 1, exc,
                )
                if attempt <= self._max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        _logger.error("LLM call failed terminally after retries: %s", last_exc)
        raise LLMError(f"LLM call failed after {self._max_retries + 1} attempts: {last_exc}")

    def invoke_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Invoke and parse a JSON object from the response.

        Tolerates markdown fences and leading/trailing prose around the JSON.
        """
        raw = self.invoke(system_prompt, user_prompt)
        return _extract_json(raw)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_client(self) -> Any:
        from openai import OpenAI

        return OpenAI(api_key=self._api_key, base_url=self._base_url, timeout=self._timeout)

    def _call(self, client: Any, system_prompt: str, user_prompt: str) -> str:
        response = client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""


def _extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object in ``text``, tolerating fences/prose."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {text[:200]}")
    return json.loads(match.group(0))


def _offline_response(system_prompt: str, user_prompt: str) -> str:
    """Deterministic canned response used when no API key is configured.

    Inspects the system prompt to emit a well-formed judgement matching the
    expected node schema. This is a test/dev fallback, not production logic.
    """
    prompt_lower = system_prompt.lower()
    if "document grader" in prompt_lower or "relevance" in prompt_lower:
        return json.dumps({"score": 0.9, "decision": "relevant"})
    if "hallucination" in prompt_lower or "grounded" in prompt_lower or "faithfulness" in prompt_lower:
        return json.dumps({"grounded_fraction": 1.0, "decision": "grounded", "unsupported_claims": []})
    if "query analy" in prompt_lower or "router" in prompt_lower or "rewrite" in prompt_lower:
        # Extract the original question line from the user prompt for the query.
        question = _extract_field(user_prompt, "user question", "original user question")
        return json.dumps({"route": "default", "processed_query": question})
    # Default: echo a short answer derived from the user prompt.
    return f"Offline response: {_extract_field(user_prompt, 'user question', 'user question')}"


def _extract_field(text: str, *labels: str) -> str:
    """Pull the value after a 'Label:' marker in ``text``; fall back to text."""
    for label in labels:
        marker = f"{label}:"
        low = text.lower()
        idx = low.find(marker)
        if idx != -1:
            rest = text[idx + len(marker):].strip()
            return rest.split("\n", 1)[0].strip()
    return text.strip()
