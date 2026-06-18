"""Evaluation dataset loading (Layer 4).

Loads and validates the mock benchmark dataset
(``tests/data/eval_queries.json``). The dataset pairs each query with a
ground-truth answer and the key terms/section that define a correct retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from ..common.config import AppConfig


class EvalQuery(BaseModel):
    """One benchmark entry: query + ground truth + retrieval expectations."""

    id: str
    category: str
    query: str
    ground_truth: str
    key_terms: list[str] = Field(default_factory=list)
    relevant_section: str = "Results"
    target_arxiv_id: Optional[str] = None


class EvalDataset(BaseModel):
    """The full benchmark dataset."""

    description: str = ""
    source_paper: str = ""
    queries: list[EvalQuery]


def load_dataset(path: Optional[str | Path] = None, config: Optional[AppConfig] = None) -> EvalDataset:
    """Load and validate the benchmark dataset from JSON."""
    from ..common.config import get_config

    cfg = config or get_config()
    dataset_path = Path(path) if path else cfg.eval_dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    try:
        return EvalDataset(**raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid evaluation dataset at {dataset_path}: {exc}") from exc
