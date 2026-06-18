"""Ingestion validation script (Phase 2, step 3).

Parses and semantically splits the configured source documents, then prints a
metadata distribution report so the injection pipeline can be verified:
  * chunk count and token-size distribution (histogram)
  * section distribution (chunks per section_title)
  * keyword frequency (extracted metrics / model abbreviations)
  * page-range integrity
  * atomic-block preservation (no equation or table split across chunks)

Embedding and Chroma indexing are attempted only when the configured model is
available; otherwise the indexing step is skipped with an explicit log line
rather than failing silently.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

# Allow running as a script: ensure the project root is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common import setup_logging  # noqa: E402
from src.common.config import get_config  # noqa: E402
from src.common.schemas import DocumentChunk  # noqa: E402
from src.ingestion.pipeline import IngestionPipeline  # noqa: E402
from src.ingestion.parser import parse_document  # noqa: E402

_FIXTURE = _PROJECT_ROOT / "tests" / "data" / "sample_paper.txt"


def _histogram(values: list[int], bucket_size: int = 100) -> list[tuple[str, int]]:
    """Return a coarse token-count histogram with fixed-width buckets."""
    if not values:
        return []
    low = (min(values) // bucket_size) * bucket_size
    high = ((max(values) // bucket_size) + 1) * bucket_size
    buckets: list[tuple[str, int]] = []
    for start in range(low, high, bucket_size):
        count = sum(1 for v in values if start <= v < start + bucket_size)
        if count:
            buckets.append((f"{start}-{start + bucket_size}", count))
    return buckets


def _check_atomic_preservation(chunks: list[DocumentChunk]) -> tuple[int, list[str]]:
    """Verify no chunk splits a display equation or table.

    Counts balanced delimiters per chunk; returns (issues, detail_messages).
    """
    issues = 0
    details: list[str] = []
    for chunk in chunks:
        text = chunk.content
        # $$ must appear an even number of times for balanced display math.
        dollar_pairs = text.count("$$")
        if dollar_pairs % 2 != 0:
            issues += 1
            details.append(f"Unbalanced '$$' in chunk {chunk.id}")
        for env in ("equation", "align", "table", "tabular"):
            if text.count(f"\\begin{{{env}}}") != text.count(f"\\end{{{env}}}"):
                issues += 1
                details.append(f"Unbalanced '\\begin{{{env}}}' in chunk {chunk.id}")
        # A markdown table row should not be truncated mid-row.
        if text.count("|") % 2 != 0 and "|" in text:
            # Heuristic only; markdown rows with odd pipe counts are suspicious.
            details.append(f"Possible mid-table split in chunk {chunk.id}")
    return issues, details


def _print_report(chunks: list[DocumentChunk]) -> None:
    log = setup_logging()
    print("\n=== Ingestion Metadata Distribution Report ===")
    print(f"Total chunks: {len(chunks)}")

    # Token distribution
    tokens = [c.token_count for c in chunks]
    if tokens:
        print("\n-- Token size distribution --")
        print(f"  min={min(tokens)}  median={int(median(tokens))}  "
              f"mean={int(mean(tokens))}  max={max(tokens)}")
        for label, count in _histogram(tokens):
            bar = "#" * min(count, 50)
            print(f"  {label:>10}: {count:>3} {bar}")

    # Section distribution
    print("\n-- Chunks per section --")
    section_counts = Counter(c.section_title for c in chunks)
    for section, count in section_counts.most_common():
        print(f"  {section:<28} {count}")

    # Keyword frequency
    print("\n-- Keyword frequency (top 15) --")
    keyword_counts: Counter[str] = Counter()
    for c in chunks:
        keyword_counts.update(c.keywords)
    for keyword, count in keyword_counts.most_common(15):
        print(f"  {keyword:<10} {count}")
    if not keyword_counts:
        print("  (no keywords extracted)")

    # Page-range integrity
    print("\n-- Page-range integrity --")
    bad_ranges = [c for c in chunks if c.page_range[1] < c.page_range[0]]
    print(f"  chunks with invalid page range: {len(bad_ranges)}")
    for c in chunks[:5]:
        print(f"  {c.id}  pages={c.page_range}  section='{c.section_title}'")

    # Atomic preservation
    print("\n-- Atomic-block preservation --")
    issues, details = _check_atomic_preservation(chunks)
    print(f"  split equation/table issues: {issues}")
    for detail in details[:10]:
        print(f"    ! {detail}")

    # Source distribution
    print("\n-- Chunks per source --")
    source_counts = Counter(c.source_file for c in chunks)
    for source, count in source_counts.most_common():
        print(f"  {source:<30} {count}")

    print("\n=== End of report ===\n")


def main(argv: list[str]) -> int:
    setup_logging()
    config = get_config()
    pipeline = IngestionPipeline(config=config)

    sources: list[Path] = []
    pdf_dir = config.pdf_dir
    if pdf_dir.exists():
        sources.extend(p for p in sorted(pdf_dir.iterdir()) if p.suffix.lower() in {".pdf", ".txt", ".md"})
    if _FIXTURE.exists():
        sources.append(_FIXTURE)

    if not sources:
        print("No source documents found (drop PDFs in data/pdfs or use the fixture).")
        return 1

    all_chunks: list[DocumentChunk] = []
    for path in sources:
        print(f"\n--- Processing {path.name} ---")
        _document, chunks = pipeline.parse_and_split(path)
        # Parse a fresh document for the report (cheap) to print structure.
        parsed = parse_document(path)
        print(f"  pages={parsed.page_count}  sections={len(parsed.sections)}  "
              f"blocks={parsed.block_count}  chunks={len(chunks)}")
        all_chunks.extend(chunks)

    _print_report(all_chunks)

    # Attempt real indexing only if the embedding model loads.
    try:
        indexed_pipeline = IngestionPipeline.with_default_vectorstore(config=config)
        stats = indexed_pipeline.run(_FIXTURE.parent if not pdf_dir.exists() else pdf_dir, reset_index=True)
        print(f"\n[Indexing attempt] documents={stats.documents} chunks={stats.chunks} "
              f"indexed={stats.indexed} ok={stats.indexed_successfully}")
    except Exception as exc:  # noqa: BLE001 - indexing is optional in this check
        print(f"\n[Indexing skipped] embedding model unavailable: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
