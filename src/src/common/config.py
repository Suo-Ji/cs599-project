"""Typed configuration loader.

Reads ``config/config.yaml`` once, validates the known sections, and exposes a
singleton ``AppConfig`` object. All modules import settings from here instead of
hard-coding values or re-parsing YAML.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# Project root = two levels up from this file (src/common/ -> project root).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"


class _BaseSection(BaseModel):
    """Base for nested config sections; forbids silent typos."""

    model_config = {"extra": "forbid"}


class PathsSection(_BaseSection):
    pdf_dir: str
    vectorstore_dir: str
    collection_name: str
    eval_dataset: str
    eval_results: str


class IngestionSection(_BaseSection):
    embedding_model: str
    embedding_dim: int
    chunk_min_tokens: int
    chunk_max_tokens: int
    overlap_ratio: float
    min_chunk_tokens: int
    keyword_lexicon: dict[str, list[str]]


class VectorstoreSection(_BaseSection):
    distance_metric: str
    allow_reset: bool
    hnsw: dict[str, Any]


class RetrievalSection(_BaseSection):
    dense_top_k: int
    sparse_top_k: int
    rrf_k: int
    rerank_model: str
    rerank_candidates: int
    final_top_n: int
    grade_score_threshold: float


class LLMSection(_BaseSection):
    model: str
    temperature: float
    request_timeout: int
    max_retries: int


class AgentSection(_BaseSection):
    llm: LLMSection
    max_rewrite_retries: int
    grading_binary: bool
    hallucination_binary: bool
    grounded_claim_threshold: float


class EvalSection(_BaseSection):
    metrics: list[str]
    llm_as_judge_model: str
    fail_on: dict[str, float]


class LoggingSection(_BaseSection):
    level: str
    rich_console: bool


class AppConfig(_BaseSection):
    paths: PathsSection
    ingestion: IngestionSection
    vectorstore: VectorstoreSection
    retrieval: RetrievalSection
    agent: AgentSection
    evaluation: EvalSection
    logging: LoggingSection

    @property
    def pdf_dir(self) -> Path:
        return PROJECT_ROOT / self.paths.pdf_dir

    @property
    def vectorstore_dir(self) -> Path:
        return PROJECT_ROOT / self.paths.vectorstore_dir

    @property
    def eval_dataset_path(self) -> Path:
        return PROJECT_ROOT / self.paths.eval_dataset

    @property
    def eval_results_path(self) -> Path:
        return PROJECT_ROOT / self.paths.eval_results


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}.")
    return data


@lru_cache(maxsize=1)
def get_config(config_path: str | os.PathLike[str] | None = None) -> AppConfig:
    """Load and validate the application config (cached singleton)."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw = _load_yaml(path)
    return AppConfig(**raw)
