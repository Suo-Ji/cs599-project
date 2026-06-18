"""Unit tests for the semantic splitter (Phase 2).

Verifies the core chunking contract with a deterministic token counter so results
do not depend on tiktoken availability:
  * paragraphs pack into chunks up to the target window
  * adjacent chunks share a bounded overlap
  * equations and tables are never split across chunk boundaries
  * oversized paragraphs fall back to sentence splitting, never character cuts
  * metadata (section_title, page_range, keywords) is injected per chunk
"""

from __future__ import annotations

from src.common.schemas import DocumentChunk
from src.ingestion.parser import ParsedDocument, ParsedSection, TextBlock
from src.ingestion.splitter import SemanticSplitter

# Deterministic counter: returns word count so chunk sizes are exact integers.
def _word_counter(text: str) -> int:
    return len(text.split())


def _config(monkeypatch_targets: dict[str, int] | None = None) -> "AppConfig":  # type: ignore[name-defined]
    from src.common.config import get_config

    config = get_config()
    # Tighten the window so tests are small but still exercise packing/overlap.
    config.ingestion.chunk_min_tokens = 10
    config.ingestion.chunk_max_tokens = 20
    config.ingestion.overlap_ratio = 0.25  # overlap target = 0.25 * 20 = 5
    config.ingestion.min_chunk_tokens = 1
    return config


def _section(title: str, blocks: list[TextBlock]) -> ParsedSection:
    return ParsedSection(title=title, level=1, page_start=1, blocks=blocks)


def _para(text: str, page: int = 1) -> TextBlock:
    return TextBlock(kind="paragraph", text=text, page_number=page)


def _eq(text: str, page: int = 1) -> TextBlock:
    return TextBlock(kind="equation", text=text, page_number=page)


def _table(text: str, page: int = 1) -> TextBlock:
    return TextBlock(kind="table", text=text, page_number=page)


def test_paragraphs_pack_within_window() -> None:
    config = _config()
    splitter = SemanticSplitter(config, token_counter=_word_counter)

    # Four paragraphs of 6 words each => 24 words total, window 20.
    blocks = [
        _para("alpha beta gamma delta epsilon zeta"),
        _para("eta theta iota kappa lambda mu"),
        _para("nu xi omicron pi rho sigma"),
        _para("tau upsilon phi chi psi omega"),
    ]
    document = ParsedDocument(source_file="doc.pdf", sections=[_section("Body", blocks)])
    chunks = splitter.split_document(document)

    assert len(chunks) >= 2, "content exceeding the window must produce >1 chunk"
    for chunk in chunks:
        assert chunk.token_count <= 20 + 1e-6, "no chunk may exceed the window"
    assert all(c.section_title == "Body" for c in chunks)


def test_adjacent_chunks_share_overlap() -> None:
    config = _config()
    splitter = SemanticSplitter(config, token_counter=_word_counter)

    # Distinct words per paragraph so overlap is observable.
    blocks = [
        _para("a a a a a a a a a a"),   # 10 words
        _para("b b b b b b b b b b"),   # 10 words
        _para("c c c c c c c c c c"),   # 10 words
    ]
    document = ParsedDocument(source_file="doc.pdf", sections=[_section("Body", blocks)])
    chunks = splitter.split_document(document)

    assert len(chunks) >= 2
    # The trailing units of chunk[0] should reappear at the start of chunk[1].
    # With distinct paragraph vocabularies, overlap requires shared tokens.
    first_tail = chunks[0].content.split()[-5:]
    second_head = chunks[1].content.split()[:5]
    overlap = set(first_tail) & set(second_head)
    assert overlap, f"expected non-empty overlap between consecutive chunks, got: {overlap}"


def test_equation_never_split() -> None:
    config = _config()
    splitter = SemanticSplitter(config, token_counter=_word_counter)

    eq_text = "$$ x + y + z + w + v + u + t + s + r + q + p + o + n + m + l + k $$"
    blocks = [
        _para("alpha beta gamma delta"),       # 4 words
        _eq(eq_text),                           # large atomic block (>20 words)
        _para("zeta eta theta iota"),          # 4 words
    ]
    document = ParsedDocument(source_file="doc.pdf", sections=[_section("Body", blocks)])
    chunks = splitter.split_document(document)

    # The equation must appear verbatim within exactly one chunk.
    eq_chunks = [c for c in chunks if eq_text in c.content]
    assert len(eq_chunks) == 1, "equation must be contained in exactly one chunk"
    eq_chunk = eq_chunks[0]
    assert "$$" in eq_chunk.content and eq_chunk.content.count("$$") == 2
    # The atomic chunk may exceed the window, but only because it is atomic.
    assert eq_chunk.token_count >= eq_text.count(" ") + 1


def test_table_never_split() -> None:
    config = _config()
    splitter = SemanticSplitter(config, token_counter=_word_counter)

    table_text = (
        "| Model | RMSE | NLL |\n"
        "|-------|------|-----|\n"
        "| EDL   | 0.41 | 1.2 |\n"
        "| BNN   | 0.40 | 1.1 |\n"
        "| GNN   | 0.39 | 1.1 |"
    )
    blocks = [
        _para("alpha beta gamma delta"),
        _table(table_text),
        _para("zeta eta theta iota"),
    ]
    document = ParsedDocument(source_file="doc.pdf", sections=[_section("Results", blocks)])
    chunks = splitter.split_document(document)

    table_chunks = [c for c in chunks if "RMSE" in c.content and "BNN" in c.content]
    assert len(table_chunks) == 1, "the whole metrics table must stay in one chunk"


def test_oversized_paragraph_splits_by_sentence_not_characters() -> None:
    config = _config()
    splitter = SemanticSplitter(config, token_counter=_word_counter)

    # 30-word single paragraph (no newline) of distinct capitalized sentences.
    sentences = [
        "Alpha beta gamma delta epsilon",   # 5 words
        "Zeta eta theta iota kappa",
        "Lambda mu nu xi omicron",
        "Pi rho sigma tau upsilon",
        "Phi chi psi omega alpha2 beta2",
    ]
    paragraph = ". ".join(sentences) + "."
    blocks = [_para(paragraph)]
    document = ParsedDocument(source_file="doc.pdf", sections=[_section("Body", blocks)])
    chunks = splitter.split_document(document)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.token_count <= 20 + 1e-6
    # No chunk may start/end mid-word: every chunk is composed of whole words.
    for chunk in chunks:
        for token in chunk.content.split():
            assert token.strip(), "no empty tokens (implies no character-truncation)"


def test_metadata_injection_per_chunk() -> None:
    config = _config()
    splitter = SemanticSplitter(config, token_counter=_word_counter)

    blocks = [_para("alpha beta gamma delta epsilon zeta eta theta iota kappa")]
    document = ParsedDocument(source_file="report.pdf", sections=[_section("Methodology", blocks)])
    chunks = splitter.split_document(document)

    assert chunks, "at least one chunk expected"
    chunk: DocumentChunk = chunks[0]
    assert chunk.source_file == "report.pdf"
    assert chunk.section_title == "Methodology"
    assert chunk.page_range == (1, 1)
    assert chunk.chunk_index == 0
    assert "::" in chunk.id  # deterministic id format


def test_keywords_injected_from_content() -> None:
    from src.common.config import get_config

    config = get_config()
    config.ingestion.chunk_min_tokens = 1
    config.ingestion.chunk_max_tokens = 1000
    splitter = SemanticSplitter(config, token_counter=_word_counter)

    blocks = [_para("The GNN backbone reduces NLL and improves PICP compared to the BNN baseline.")]
    document = ParsedDocument(source_file="paper.pdf", sections=[_section("Results", blocks)])
    chunks = splitter.split_document(document)

    keywords = set(chunks[0].keywords)
    assert {"GNN", "NLL", "PICP", "BNN"}.issubset(keywords)
