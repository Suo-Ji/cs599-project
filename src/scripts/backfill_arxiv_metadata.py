"""Backfill arXiv metadata for the 7 already-downloaded papers.

Queries the arXiv API by id_list (a single batched call) and updates
data/arxiv_testset.json so every entry carries full title/abstract/authors/
categories, replacing any 'orphaned PDF' placeholder records. Resilient to
arXiv throttling via the same retry logic as the fetcher.

Usage:
    python scripts/backfill_arxiv_metadata.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.fetch_arxiv_testset import _http_get, _parse_feed, _normalize_ws  # noqa: E402
from src.common.logging_setup import setup_logging  # noqa: E402

_logger = setup_logging()

ARXIV_API = "http://export.arxiv.org/api/query"
MANIFEST = _PROJECT_ROOT / "data" / "arxiv_testset.json"
PDF_DIR = _PROJECT_ROOT / "data" / "pdfs"


def query_by_ids(arxiv_ids: list[str]) -> dict[str, dict]:
    """Batch-query arXiv for a list of ids; return {id: entry}."""
    import urllib.parse

    if not arxiv_ids:
        return {}
    id_list = ",".join(arxiv_ids)
    params = urllib.parse.urlencode({"id_list": id_list, "max_results": len(arxiv_ids)})
    url = f"{ARXIV_API}?{params}"
    _logger.info("arXiv id_list query: %s", url)
    try:
        raw = _http_get(url, retries=5)
    except Exception as exc:  # noqa: BLE001
        _logger.error("arXiv id_list query failed: %s", exc)
        return {}
    return {entry["arxiv_id"]: entry for entry in _parse_feed(raw)}


def backfill() -> int:
    if not MANIFEST.exists():
        _logger.error("Manifest not found: %s", MANIFEST)
        return 1

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records: list[dict] = manifest.get("papers", [])
    # Determine which ids need real metadata (orphans or missing abstracts).
    needs_meta = [
        r["arxiv_id"] for r in records
        if r.get("status") == "cached_orphan" or not r.get("abstract")
    ]
    # Also cover any PDFs on disk not yet in the manifest.
    on_disk = {p.stem for p in PDF_DIR.glob("*.pdf") if p.stat().st_size > 0}
    known = {r["arxiv_id"] for r in records}
    needs_meta = sorted(set(needs_meta) | (on_disk - known))

    _logger.info("Backfilling metadata for %d ids: %s", len(needs_meta), needs_meta)
    if not needs_meta:
        _summary(records)
        return 0

    metadata = query_by_ids(needs_meta)
    time.sleep(3.0)  # arXiv courtesy

    # Merge fetched metadata into existing records (by id), preserving local_pdf/status.
    by_id = {r["arxiv_id"]: r for r in records}
    for arxiv_id in needs_meta:
        meta = metadata.get(arxiv_id)
        local_pdf = PDF_DIR / f"{arxiv_id}.pdf"
        if meta:
            meta = dict(meta)
            meta["local_pdf"] = str(local_pdf.relative_to(_PROJECT_ROOT))
            meta["status"] = "cached" if (local_pdf.exists() and local_pdf.stat().st_size > 0) else meta.get("status", "metadata_only")
            meta["pdf_bytes"] = local_pdf.stat().st_size if local_pdf.exists() else 0
            by_id[arxiv_id] = meta
            _logger.info("Backfilled %s: %s", arxiv_id, _normalize_ws(meta.get("title", ""))[:60])
        else:
            _logger.warning("No metadata returned for %s", arxiv_id)

    records = [by_id[i] for i in by_id]
    manifest["papers"] = records
    manifest["count"] = len(records)
    manifest["source"] = "arXiv API (export.arxiv.org/api/query) — metadata backfilled"
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _logger.info("Manifest updated: %s", MANIFEST)
    _summary(records)
    return 0


def _summary(records: list[dict]) -> None:
    have_abstract = sum(1 for r in records if r.get("abstract"))
    have_pdf = sum(1 for r in records if r.get("status") in ("cached", "downloaded", "cached_orphan"))
    print("\n=== arXiv Test Set (after backfill) ===")
    print(f"Total papers: {len(records)}  (with abstract: {have_abstract}, with PDF: {have_pdf})")
    cats = {}
    for r in records:
        cats[r.get("primary_category", "?")] = cats.get(r.get("primary_category", "?"), 0) + 1
    print("By primary category:")
    for cat, n in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<18} {n}")
    print("\nPapers:")
    for r in records:
        title = _normalize_ws(r.get("title", ""))[:60] or "(no title)"
        print(f"  {r['arxiv_id']}  [{r.get('primary_category','?')}]  {title}")


if __name__ == "__main__":
    sys.exit(backfill())
