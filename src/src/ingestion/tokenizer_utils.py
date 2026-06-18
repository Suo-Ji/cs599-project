"""Token-counting utilities for the ingestion pipeline.

The chunker targets a token window (not a character window), so it needs a fast,
deterministic token counter. tiktoken's ``cl100k_base`` is used as the default
proxy for embedding-model token length. It is swappable via :func:`get_token_counter`.
"""

from __future__ import annotations

from typing import Callable, Protocol

_TIKTOKEN_AVAILABLE: bool
_TIKTOKEN_AVAILABLE = True
try:
    import tiktoken  # type: ignore

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken is optional at runtime
    _TIKTOKEN_AVAILABLE = False
    _ENCODER = None  # type: ignore[assignment]


class TokenCounter(Protocol):
    def __call__(self, text: str) -> int: ...


def tiktoken_counter(text: str) -> int:
    """Count tokens using tiktoken cl100k_base encoding."""
    assert _ENCODER is not None
    return len(_ENCODER.encode(text))


def whitespace_counter(text: str) -> int:
    """Fallback counter: approximates tokens as whitespace-split words.

    Used only when tiktoken is unavailable so the pipeline degrades gracefully.
    """
    return len(text.split()) if text.strip() else 0


def get_token_counter() -> TokenCounter:
    """Return the best available token counter."""
    if _TIKTOKEN_AVAILABLE and _ENCODER is not None:
        return tiktoken_counter
    return whitespace_counter


def count_tokens(text: str, counter: TokenCounter | None = None) -> int:
    """Count tokens in ``text`` using ``counter`` (default: best available)."""
    fn: TokenCounter = counter or get_token_counter()
    return fn(text)


def sanitize_text(text: str) -> str:
    """Remove characters that break UTF-8 encoding (lone surrogate halves).

    PDF text extractors (pypdf) occasionally emit unpaired surrogate code units
    for math-symbol ligatures; Python str can hold them but ``.encode('utf-8')``
    raises. Stripping them keeps hashing, logging, and embedding safe.
    """
    if not text:
        return text
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")
