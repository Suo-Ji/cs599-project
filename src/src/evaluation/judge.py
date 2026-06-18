"""LLM-as-a-judge evaluator (Layer 4 evaluation).

Scores the four tracked metrics. When the LLM client is online, each metric is
scored by a focused judge prompt that returns calibrated JSON. When offline (no
API key) or on failure, the deterministic heuristic scorers in
:mod:`src.evaluation.metrics` are used so the evaluation pipeline always runs.

Judge prompts follow the same objective, deterministic style as the agent
prompts: no figurative language, explicit decision rules.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..agent.llm_client import LLMClient
from ..common.config import AppConfig
from ..common.logging_setup import get_logger
from ..common.schemas import DocumentChunk
from . import metrics

_logger: logging.Logger = get_logger("evaluation.judge")


# ------------------------------------------------------------------
# Judge prompts
# ------------------------------------------------------------------

FAITHFULNESS_JUDGE = """\
You are an evaluator scoring answer faithfulness for an academic-literature RAG system.

Task: determine what fraction of the factual claims in the answer are directly
supported by the retrieved context.

Decision rule:
  - A claim is "supported" only if the context states the same fact (same metric
    value, model name, or definition).
  - Numeric values must match exactly to count as supported.

Compute faithfulness = supported_claims / total_claims (0 if there are no claims).

Output ONLY a JSON object: {"faithfulness": <float in [0,1]>}"""

CONTEXT_RECALL_JUDGE = """\
You are an evaluator scoring context recall for an academic-literature RAG system.

Task: determine what fraction of the ground-truth facts are attributable to the
retrieved context.

Decision rule:
  - Split the ground truth into atomic facts. A fact is "attributable" if the
    context contains the information needed to state it.

Compute context_recall = attributable_facts / total_facts.

Output ONLY a JSON object: {"context_recall": <float in [0,1]>}"""

ANSWER_RELEVANCE_JUDGE = """\
You are an evaluator scoring answer relevance for an academic-literature RAG system.

Task: score how directly the answer addresses the question, on a 0-1 scale.

Decision rule:
  - 1.0 = the answer fully resolves the question.
  - 0.0 = the answer does not address the question at all.

Output ONLY a JSON object: {"answer_relevancy": <float in [0,1]>}"""


# ------------------------------------------------------------------
# Judge
# ------------------------------------------------------------------


class LLMJudge:
    """Scores metrics via LLM with deterministic heuristic fallback."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        llm: Optional[LLMClient] = None,
    ) -> None:
        self._config = config or _default_config()
        self._llm = llm or LLMClient(self._config)

    @property
    def offline(self) -> bool:
        return self._llm.offline

    # ------------------------------------------------------------------
    # Metric entry points
    # ------------------------------------------------------------------

    def faithfulness(
        self, answer: str, contexts: list[str] | list[DocumentChunk]
    ) -> float:
        user_prompt = (
            f"Answer:\n{answer}\n\nRetrieved context:\n{_format(contexts)}"
        )
        result = self._safe_score(FAITHFULNESS_JUDGE, user_prompt, "faithfulness")
        if result is not None:
            return _clip(result)
        return metrics.score_faithfulness(answer, contexts)

    def context_recall(
        self, ground_truth: str, contexts: list[str] | list[DocumentChunk]
    ) -> float:
        user_prompt = (
            f"Ground truth:\n{ground_truth}\n\nRetrieved context:\n{_format(contexts)}"
        )
        result = self._safe_score(CONTEXT_RECALL_JUDGE, user_prompt, "context_recall")
        if result is not None:
            return _clip(result)
        return metrics.score_context_recall(ground_truth, contexts)

    def answer_relevance(self, question: str, answer: str) -> float:
        user_prompt = f"Question:\n{question}\n\nAnswer:\n{answer}"
        result = self._safe_score(ANSWER_RELEVANCE_JUDGE, user_prompt, "answer_relevancy")
        if result is not None:
            return _clip(result)
        return metrics.score_answer_relevance(question, answer)

    def context_precision(
        self, retrieved: list[DocumentChunk], ground_truth: str
    ) -> float:
        # Context precision is deterministic (no LLM needed): fraction of
        # retrieved chunks sharing key terms with the ground truth.
        return metrics.score_context_precision(retrieved, ground_truth)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _safe_score(self, system_prompt: str, user_prompt: str, key: str) -> Optional[float]:
        try:
            data = self._llm.invoke_json(system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001 - fall back to heuristic
            _logger.debug("LLM judge failed (%s); using heuristic: %s", key, exc)
            return None
        value = data.get(key)
        if not isinstance(value, (int, float)):
            _logger.warning("Judge response missing numeric '%s'; using heuristic.", key)
            return None
        return float(value)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _format(contexts: list[str] | list[DocumentChunk]) -> str:
    if not contexts:
        return "(no context)"
    if isinstance(contexts[0], DocumentChunk):
        return "\n\n".join(
            f"[{c.section_title}] {c.content}" for c in contexts  # type: ignore[union-attr]
        )
    return "\n\n".join(contexts)  # type: ignore[arg-type]


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _default_config() -> AppConfig:
    from ..common.config import get_config

    return get_config()
