"""Agentic RAG system for academic literature analysis.

Top-level package. Subpackages:
  * common    — config, logging, Pydantic schemas
  * ingestion — PDF parsing + semantic chunking (Layer 1)
  * retrieval — hybrid dense/BM25 + RRF + cross-encoder reranking (Layer 2)
  * agent     — LangGraph control flow with self-correction (Layer 3)
"""

__version__ = "0.1.0"
