"""Deterministic metric scorers (Layer 4 evaluation).

Heuristic implementations of the four tracked metrics. They are used directly in
offline mode and as fallbacks when the LLM-as-a-judge is unavailable. All scorers
are pure functions returning floats in [0, 1].

Metrics:
  * faithfulness        — fraction of answer sentences whose key terms appear in
                          the retrieved context (proxy for grounded generation)
  * context_recall      — fraction of ground-truth key terms present in context
  * answer_relevance    — term overlap between the question and the answer
  * context_precision   — fraction of retrieved chunks that share key terms with
                          the ground truth (relevance of the retrieved set)
"""

from __future__ import annotations

import re
from functools import lru_cache

from ..common.schemas import DocumentChunk

# Significant-term extraction: decimal numbers, acronyms, and non-stopword words
# longer than 4 characters. These carry the factual load of scientific answers.
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,6}\b")
_STOPWORDS: frozenset[str] = frozenset(
    {
        "about", "above", "after", "again", "against", "because", "below",
        "between", "both", "during", "each", "from", "have", "into", "other",
        "over", "should", "some", "that", "their", "there", "these", "this",
        "those", "under", "what", "which", "with", "would",
    }
)


@lru_cache(maxsize=4096)
def extract_key_terms(text: str) -> frozenset[str]:
    """Return the set of significant terms in ``text`` (case-folded words)."""
    if not text:
        return frozenset()
    numbers = {m.group(0).lower() for m in _NUMBER.finditer(text)}
    acronyms = {m.group(0).lower() for m in _ACRONYM.finditer(text)}
    words = {
        w.lower()
        for w in re.findall(r"[A-Za-z][A-Za-z\-]+", text)
        if len(w) > 4 and w.lower() not in _STOPWORDS
    }
    return frozenset(numbers | acronyms | words)


def _context_text(contexts: list[str] | list[DocumentChunk]) -> str:
    if not contexts:
        return ""
    if isinstance(contexts[0], DocumentChunk):
        return " ".join(c.content for c in contexts)  # type: ignore[union-attr]
    return " ".join(contexts)  # type: ignore[arg-type]


# ------------------------------------------------------------------
# Metric scorers
# ------------------------------------------------------------------


def score_faithfulness(answer: str, contexts: list[str] | list[DocumentChunk]) -> float:
    """Fraction of answer sentences whose key terms are supported by context."""
    if not answer.strip():
        return 0.0
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return 0.0
    context_terms = extract_key_terms(_context_text(contexts))
    if not context_terms:
        return 0.0
    supported = 0
    for sentence in sentences:
        sent_terms = extract_key_terms(sentence)
        # A sentence is supported if at least half of its key terms are in context.
        if not sent_terms:
            continue
        overlap = len(sent_terms & context_terms)
        if overlap >= max(1, len(sent_terms) / 2):
            supported += 1
    return supported / len(sentences)


def score_context_recall(
    ground_truth: str, contexts: list[str] | list[DocumentChunk]
) -> float:
    """Fraction of ground-truth key terms found in the retrieved context."""
    truth_terms = extract_key_terms(ground_truth)
    if not truth_terms:
        return 0.0
    context_terms = extract_key_terms(_context_text(contexts))
    if not truth_terms:
        return 0.0
    found = sum(1 for term in truth_terms if term in context_terms)
    return found / len(truth_terms)


def score_answer_relevance(question: str, answer: str) -> float:
    """Term-overlap relevance between the question and the answer."""
    q_terms = extract_key_terms(question)
    a_terms = extract_key_terms(answer)
    if not q_terms or not a_terms:
        return 0.0
    overlap = len(q_terms & a_terms)
    return overlap / len(q_terms)


def score_context_precision(
    retrieved: list[DocumentChunk], ground_truth: str
) -> float:
    """Fraction of retrieved chunks that share key terms with the ground truth."""
    if not retrieved:
        return 0.0
    truth_terms = extract_key_terms(ground_truth)
    if not truth_terms:
        return 0.0
    relevant = 0
    for chunk in retrieved:
        chunk_terms = extract_key_terms(chunk.content)
        if chunk_terms & truth_terms:
            relevant += 1
    return relevant / len(retrieved)
