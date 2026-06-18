"""Application entry point.

Phase 1 status: structure + configuration + schemas are initialized. This module
currently validates that the config loads and the schemas import correctly. Later
phases wire in ingestion, retrieval, the LangGraph agent, and evaluation here.
"""

from __future__ import annotations

import sys

from src.common import get_logger, setup_logging
from src.common.config import get_config


def check_environment() -> int:
    """Verify the configuration loads and reports the loaded baseline.

    Returns a process exit code: 0 on success, 1 on failure.
    """
    logger = setup_logging()

    try:
        config = get_config()
    except Exception as exc:  # noqa: BLE001 - surface any config error as a failure
        logger.error("Configuration load failed: %s", exc, exc_info=True)
        return 1

    logger.info("Configuration loaded.")
    logger.info("Embedding model: %s", config.ingestion.embedding_model)
    logger.info("Rerank model:    %s", config.retrieval.rerank_model)
    logger.info(
        "Chunk target:   %d-%d tokens, %d%% overlap",
        config.ingestion.chunk_min_tokens,
        config.ingestion.chunk_max_tokens,
        int(config.ingestion.overlap_ratio * 100),
    )
    logger.info(
        "Retrieval:      dense_top_k=%d sparse_top_k=%d rrf_k=%d final_top_n=%d",
        config.retrieval.dense_top_k,
        config.retrieval.sparse_top_k,
        config.retrieval.rrf_k,
        config.retrieval.final_top_n,
    )
    logger.info("Max rewrite retries: %d", config.agent.max_rewrite_retries)
    logger.info("Phase 1 baseline verified.")
    return 0


def main() -> int:
    return check_environment()


if __name__ == "__main__":
    sys.exit(main())
