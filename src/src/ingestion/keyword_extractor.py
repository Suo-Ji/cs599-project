"""Keyword extraction for chunk metadata injection (Layer 1).

Scans chunk text for domain terms from the configured lexicon (statistical
metrics and model abbreviations) plus general ALL-CAPS acronyms. Extracted
keywords become scalar-filterable metadata in the vector store.
"""

from __future__ import annotations

import re

from ..common.config import AppConfig

# Generic acronym pattern: 2-6 uppercase letters/digits, optionally plural.
# Negative lookahead on a trailing hyphen avoids matching a prefix of a
# hyphenated compound (e.g. "MC" inside "MC-Dropout").
_ACRONYM_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{1,5})s?\b(?!\-)")


def extract_keywords(text: str, config: AppConfig) -> list[str]:
    """Return the de-duplicated, ordered list of keywords found in ``text``.

    Combines:
      * exact matches from the configured metrics and models lexicons
      * generic ALL-CAPS acronyms (bounded length) as a fallback
    """
    found: list[str] = []
    seen: set[str] = set()

    for category in ("metrics", "models"):
        for term in config.ingestion.keyword_lexicon.get(category, []):
            if term.lower() in text.lower() and term not in seen:
                seen.add(term)
                found.append(term)

    for match in _ACRONYM_PATTERN.findall(text):
        candidate = match
        if candidate not in seen and len(candidate) >= 2 and candidate.isupper():
            seen.add(candidate)
            found.append(candidate)

    return found
