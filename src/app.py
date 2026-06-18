"""Chainlit frontend for the Academic-Literature Agentic RAG system.

A standard chat interface that drives the LangGraph agent (src/agent/graph.py)
and surfaces the intermediate reasoning as collapsed Chainlit Steps:

    * Query_Analyzer / Query_Rewriter  -> the (re)written query
    * Retrieve_&_Rerank               -> the retrieved candidate chunks
    * Document_Grader                  -> relevance verdict
    * Hallucination_Checker           -> grounding verdict

The final answer is streamed token-by-token, with source citations
(source_file + page_number) appended from the retrieved documents' metadata.

Run with:
    chainlit run app.py

The agent runs in deterministic offline mode when OPENAI_API_KEY is unset; set
the key (and optionally OPENAI_BASE_URL) for real LLM generation.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

import chainlit as cl
from chainlit import user_session

from src.agent.graph import build_graph
from src.agent.llm_client import LLMClient
from src.common.config import get_config
from src.common.logging_setup import setup_logging
from src.common.schemas import DocumentChunk
from src.ingestion.parser import parse_document
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.splitter import SemanticSplitter
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid import HybridRetriever

_logger: logging.Logger = setup_logging()

# Whether to render the agent's intermediate reasoning (query analysis, retrieved
# chunks, grading, hallucination check) as collapsed Chainlit Steps. Set False to
# show only the final streamed answer + source citations.
SHOW_INTERMEDIATE_STEPS: bool = False

# Fixture corpus used to build a BM25 retriever when no vector store is indexed.
# Replace with a real VectorIndex (bge-large-zh embeddings) for production.
_FIXTURE_CORPUS = Path(__file__).resolve().parent / "tests" / "data" / "sample_paper.txt"


# ------------------------------------------------------------------
# Retriever bootstrap (shared across the session)
# ------------------------------------------------------------------


def _build_retriever() -> HybridRetriever:
    """Build a BM25-backed hybrid retriever over the real PDF corpus in data/pdfs/.

    Indexes every PDF in the configured pdf_dir (default data/pdfs/). Falls back
    to the synthetic fixture only when no real PDFs are present, so the frontend
    always has something to retrieve over.
    """
    config = get_config()
    splitter = SemanticSplitter(config)
    chunks: list[DocumentChunk] = []

    pdf_dir = config.pdf_dir
    pdfs = sorted(p for p in pdf_dir.glob("*.pdf") if p.stat().st_size > 0) if pdf_dir.exists() else []
    for pdf in pdfs:
        try:
            document = parse_document(pdf)
            chunks.extend(splitter.split_document(document))
        except Exception as exc:  # noqa: BLE001 - one bad PDF must not break startup
            _logger.error("Failed to parse %s: %s", pdf.name, exc, exc_info=True)

    if not chunks:
        # No real PDFs (or all failed) -> fall back to the synthetic fixture.
        _logger.warning("No real PDFs in %s; falling back to fixture corpus.", pdf_dir)
        if _FIXTURE_CORPUS.exists():
            _document, chunks = IngestionPipeline(config=config).parse_and_split(_FIXTURE_CORPUS)
        else:
            _logger.warning("Fixture corpus also missing; retriever will return no documents.")
            return HybridRetriever(config=config, bm25=BM25Retriever(), vector_index=None)
    else:
        _logger.info("Indexed %d chunks from %d real PDFs in %s.", len(chunks), len(pdfs), pdf_dir)

    bm25 = BM25Retriever()
    bm25.add_chunks(chunks)
    return HybridRetriever(config=config, bm25=bm25, vector_index=None, reranker=None)


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize the compiled graph, LLM client, and retriever for the session."""
    config = get_config()
    try:
        retriever = _build_retriever()
        llm = LLMClient(config=config)
        graph = build_graph(config=config, retriever=retriever, llm=llm)
    except Exception as exc:  # noqa: BLE001 - surface init failure to the user
        _logger.error("Agent initialization failed: %s", exc, exc_info=True)
        await cl.Message(
            content=f"Agent 初始化失败: `{exc}`。请检查依赖与配置。"
        ).send()
        return

    user_session.set("graph", graph)
    user_session.set("llm", llm)
    mode = "在线 LLM" if not llm.offline else "离线确定性模式(未配置 OPENAI_API_KEY)"
    await cl.Message(
        content=(
            "## 学术文献 Agentic RAG 系统\n"
            f"运行模式:**{mode}**\n\n"
        )
    ).send()


# ------------------------------------------------------------------
# Step rendering helpers
# ------------------------------------------------------------------


async def _show_retrieval_step(chunks: list[DocumentChunk]) -> None:
    """Render retrieved candidate chunks as a collapsed Step."""
    async with cl.Step(name="Retrieve_&_Rerank", type="run") as step:
        if not chunks:
            step.output = "未召回任何候选文档。"
            return
        lines = [f"召回 {len(chunks)} 个候选文档块:\n"]
        for i, chunk in enumerate(chunks, start=1):
            lines.append(
                f"**[{i}]** `{chunk.source_file}` :: {chunk.section_title} "
                f"(p{chunk.page_number})\n{chunk.content[:300]}{'...' if len(chunk.content) > 300 else ''}\n"
            )
        step.output = "\n".join(lines)


async def _show_verdict_step(name: str, label: str, details: dict[str, Any]) -> None:
    """Render a grader/hallucination verdict step."""
    async with cl.Step(name=name, type="run") as step:
        lines = [f"**{label}**"]
        for key, value in details.items():
            lines.append(f"- `{key}`: {value}")
        step.output = "\n".join(lines)


def _format_sources(chunks: list[DocumentChunk]) -> str:
    """Format deduplicated source citations from retrieved chunk metadata."""
    if not chunks:
        return ""
    seen: dict[tuple[str, int], str] = {}
    for chunk in chunks:
        key = (chunk.source_file, chunk.page_number)
        # Keep the first occurrence's section as a hint.
        seen.setdefault(key, chunk.section_title)
    lines = ["\n\n---\n**参考来源(Sources):**"]
    for (source_file, page_number), section in seen.items():
        lines.append(f"- {source_file} — 第 {page_number} 页 ({section})")
    return "\n".join(lines)


def _format_final_sources_from_state(state: dict[str, Any]) -> str:
    """Pull source_file + page_number from the final graph state's documents."""
    docs = state.get("retrieved_documents", []) or []
    chunks: list[DocumentChunk] = [d for d in docs if isinstance(d, DocumentChunk)]
    return _format_sources(chunks)


# ------------------------------------------------------------------
# Main message handler
# ------------------------------------------------------------------


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Drive the agent graph, render steps, stream the answer, attach sources."""
    graph = user_session.get("graph")
    if graph is None:
        await cl.Message(content="Agent 尚未初始化,无法处理消息。").send()
        return

    question = message.content.strip()
    if not question:
        return

    final_state: dict[str, Any] = {"question": question}
    final_answer: str = ""
    answer_msg: Optional[cl.Message] = None
    retrieval_shown: bool = False

    # Stream the graph node-by-node. Intermediate reasoning is rendered as Steps
    # only when SHOW_INTERMEDIATE_STEPS is enabled; otherwise we just track state
    # (for source citations) and capture the final answer.
    async for update in graph.astream({"question": question}, stream_mode="updates"):
        for node_name, delta in update.items():
            delta = delta or {}
            if SHOW_INTERMEDIATE_STEPS:
                if node_name in ("query_analyzer", "query_rewriter"):
                    processed = delta.get("processed_query", "")
                    label = "Query_Analyzer/Router" if node_name == "query_analyzer" else "Query_Rewriter"
                    async with cl.Step(name=label, type="run") as step:
                        step.output = (
                            f"**{('分析后查询' if node_name == 'query_analyzer' else '重写后查询')}:**\n```\n{processed}\n```"
                        )
                elif node_name == "retrieve":
                    retrieved = delta.get("retrieved_documents", []) or []
                    await _show_retrieval_step(retrieved)
                    retrieval_shown = True
                elif node_name == "document_grader":
                    await _show_verdict_step(
                        "Document_Grader", "文档相关性评估",
                        {
                            "decision": delta.get("grade_decision", "-"),
                            "retry_count": delta.get("retry_count", "-"),
                        },
                    )
                elif node_name == "hallucination_checker":
                    await _show_verdict_step(
                        "Hallucination_Checker", "忠实度核验",
                        {"decision": delta.get("hallucination_decision", "-")},
                    )

            if node_name == "generate":
                final_answer = delta.get("generation", "") or ""

            # Track the latest accumulated state for citation extraction.
            for key, value in delta.items():
                if isinstance(value, list) and isinstance(final_state.get(key), list):
                    final_state[key] = final_state[key] + value
                else:
                    final_state[key] = value

    if SHOW_INTERMEDIATE_STEPS and not retrieval_shown:
        # Degenerate path (e.g. retrieve failed) — only noted when steps are shown.
        async with cl.Step(name="Retrieve_&_Rerank", type="run") as step:
            step.output = "本回合未召回候选文档。"

    # Stream the final answer token-by-token, then append sources.
    sources = _format_final_sources_from_state(final_state)
    full_text = final_answer + sources

    answer_msg = cl.Message(content="")
    await answer_msg.send()

    if full_text:
        await _stream_text(answer_msg, full_text)


async def _stream_text(msg: cl.Message, text: str, chunk_size: int = 4) -> None:
    """Render ``text`` into ``msg`` token-by-token.

    Streams in small word-slices to emulate token-by-token output while keeping
    interactive latency low. ``chunk_size`` controls granularity (words per slice).
    """
    tokens = text.split(" ")
    buffer = ""
    for i, token in enumerate(tokens):
        piece = token if i == 0 else " " + token
        buffer += piece
        # Flush every chunk_size words or on the final token.
        if (i + 1) % chunk_size == 0 or i == len(tokens) - 1:
            await msg.stream_token(buffer)
            buffer = ""
            await asyncio.sleep(0.01)
    if not text.strip():
        await msg.update()


if __name__ == "__main__":
    # Allow `python app.py` to print a usage hint; Chainlit is launched via the CLI.
    print("启动 Chainlit 前端:\n    chainlit run app.py")
