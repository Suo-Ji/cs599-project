"""Fetch a real academic-literature test set from the arXiv API.

Downloads N papers (default 20) as PDFs plus a metadata manifest, so the RAG
ingestion pipeline can immediately parse and index them. Uses only the Python
standard library (urllib + xml.etree) — no extra dependencies.

Defaults honor "any domain" (领域不限) by sampling across five arXiv domains:
    cs.LG            Machine Learning          (Computer Science)
    cond-mat.mtrl-sci Condensed Matter           (Physics)
    q-bio.QM         Quantitative Methods       (Quantitative Biology)
    math.OC          Optimization & Control     (Mathematics)
    stat.ML          Statistics - ML            (Statistics)

All knobs are overridable via CLI flags. The run is idempotent: already-downloaded
PDFs are skipped, so re-running resumes without re-fetching.

arXiv usage policy: sequential requests with a >=3s gap between API calls.

Usage:
    python scripts/fetch_arxiv_testset.py
    python scripts/fetch_arxiv_testset.py --total 20 --delay 3.0
    python scripts/fetch_arxiv_testset.py --categories "cs.LG:10,cs.CL:10"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.logging_setup import setup_logging  # noqa: E402

_logger = logging.getLogger("rag.fetch_arxiv") if not True else None  # placeholder
_logger = setup_logging()

# --- arXiv Atom namespaces ---
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

ARXIV_API = "http://export.arxiv.org/api/query"
USER_AGENT = "rag-arxiv-testset/1.0 (academic RAG benchmark builder; +mailto:none@example.com)"

# Default cross-domain distribution totaling 20. Categories chosen for
# reliability (consistent arXiv results) while spanning multiple fields.
DEFAULT_CATEGORIES = {
    "cs.LG": 4,        # Computer Science — Machine Learning
    "cs.CL": 4,        # Computer Science — Computation & Language (NLP)
    "stat.ML": 4,      # Statistics — Machine Learning
    "math.OC": 4,      # Mathematics — Optimization & Control
    "q-bio.QM": 4,     # Quantitative Biology — Quantitative Methods
}


# ------------------------------------------------------------------
# HTTP
# ------------------------------------------------------------------


def _http_get(url: str, timeout: int = 60, retries: int = 4) -> bytes:
    """Perform a polite GET with exponential-backoff retries.

    Retries on transient errors (timeouts, connection resets, and arXiv's
    429/503 throttling responses) so a flaky moment does not abort the run.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            # Throttling / server errors are retryable; 4xx (except 429) are not.
            if exc.code not in (429, 500, 502, 503, 504):
                raise
            _logger.warning("HTTP %s on %s (attempt %d/%d); backing off.", exc.code, url, attempt, retries)
        except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError) as exc:
            last_exc = exc
            _logger.warning("Network error on %s (attempt %d/%d): %s", url, attempt, retries, exc)
        # Exponential backoff with jitter-free spacing.
        time.sleep(min(2 ** attempt, 16))
    raise last_exc if last_exc else RuntimeError("HTTP GET failed after retries")


def _download_file(url: str, dest: Path, timeout: int = 60) -> int:
    """Stream a URL to ``dest``; return bytes written."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response, dest.open("wb") as out:
        data = response.read()
        out.write(data)
        return len(data)


# ------------------------------------------------------------------
# arXiv query + parse
# ------------------------------------------------------------------


def query_arxiv(search_query: str, max_results: int, start: int = 0) -> list[dict]:
    """Query the arXiv API and return parsed entry dicts.

    Retries the query (with a backoff) when arXiv returns an empty result set
    for a non-empty category, which is a hallmark of transient throttling.
    """
    params = urllib.parse.urlencode(
        {
            "search_query": search_query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    url = f"{ARXIV_API}?{params}"

    for attempt in range(1, 4):
        _logger.info("arXiv query (attempt %d/3): %s", attempt, url)
        try:
            raw = _http_get(url)
        except urllib.error.HTTPError as exc:
            _logger.error("arXiv HTTP error %s for %s", exc.code, url)
            return []
        except Exception as exc:  # noqa: BLE001
            _logger.warning("arXiv query failed (attempt %d/3): %s", attempt, exc)
            time.sleep(min(2 ** attempt, 16))
            continue

        entries = _parse_feed(raw)
        if entries:
            return entries
        # Empty result may indicate throttling; back off and retry.
        _logger.warning("Empty result set for %s (attempt %d/3); retrying.", search_query, attempt)
        time.sleep(min(2 ** attempt, 16))

    _logger.error("arXiv query yielded no entries after retries: %s", search_query)
    return []


def _parse_feed(raw_xml: bytes) -> list[dict]:
    """Parse an arXiv Atom feed into a list of entry metadata dicts.

    Returns an empty list if the response is not valid Atom XML (e.g. an HTML
    throttling page), so callers can treat it as "no entries" and retry.
    """
    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError as exc:
        _logger.warning("arXiv response was not valid XML (%s); treating as empty.", exc)
        return []
    entries: list[dict] = []
    for entry in root.findall("atom:entry", _NS):
        parsed = _parse_entry(entry)
        if parsed:
            entries.append(parsed)
    return entries


def _text(entry: ET.Element, path: str) -> str:
    node = entry.find(path, _NS)
    return (node.text or "").strip() if node is not None else ""


def _parse_entry(entry: ET.Element) -> Optional[dict]:
    """Extract structured metadata from one <entry> element."""
    id_url = _text(entry, "atom:id")
    arxiv_id = _id_from_url(id_url)
    if not arxiv_id:
        return None

    title = _normalize_ws(_text(entry, "atom:title"))
    summary = _normalize_ws(_text(entry, "atom:summary"))

    authors = [_normalize_ws(a.text or "") for a in entry.findall("atom:author/atom:name", _NS)]
    authors = [a for a in authors if a]

    # PDF link: prefer the <link title="pdf">, else construct it.
    pdf_url = ""
    for link in entry.findall("atom:link", _NS):
        if link.get("title") == "pdf" or link.get("rel") == "related" and "pdf" in (link.get("href") or ""):
            pdf_url = link.get("href", "")
            break
    if not pdf_url:
        pdf_url = f"http://arxiv.org/pdf/{arxiv_id}.pdf"

    primary_cat_node = entry.find("arxiv:primary_category", _NS)
    primary_category = primary_cat_node.get("term", "") if primary_cat_node is not None else ""
    categories = [c.get("term", "") for c in entry.findall("atom:category", _NS) if c.get("term")]

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "abstract": summary,
        "primary_category": primary_category,
        "categories": categories,
        "published": _text(entry, "atom:published"),
        "updated": _text(entry, "atom:updated"),
        "abs_url": id_url,
        "pdf_url": pdf_url,
        "comment": _text(entry, "arxiv:comment"),
        "journal_ref": _text(entry, "arxiv:journal_ref"),
    }


def _id_from_url(id_url: str) -> str:
    """Extract the arXiv id from an abs URL, stripping the version suffix."""
    # e.g. http://arxiv.org/abs/2401.01234v2 -> 2401.01234
    last = id_url.rstrip("/").split("/")[-1]
    return re.sub(r"v\d+$", "", last)


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------


def parse_categories_arg(spec: Optional[str]) -> dict[str, int]:
    """Parse a 'cat:count,cat:count' spec into a dict; fall back to defaults."""
    if not spec:
        return dict(DEFAULT_CATEGORIES)
    result: dict[str, int] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise argparse.ArgumentTypeError(f"Invalid category spec '{token}', expected 'cat:count'.")
        cat, count = token.rsplit(":", 1)
        result[cat.strip()] = int(count)
    return result


def fetch_testset(
    categories: dict[str, int],
    pdf_dir: Path,
    manifest_path: Path,
    delay: float,
    total: int,
) -> list[dict]:
    """Fetch papers across categories until ``total`` unique papers are gathered.

    The manifest is persisted after every category so an interruption (timeout,
    throttling) never loses already-downloaded papers. On startup, any PDFs that
    already exist on disk are reconciled into the manifest.
    """
    pdf_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    records, seen = _reconcile_existing(pdf_dir, manifest_path)
    _logger.info("Resumed with %d existing papers.", len(records))
    _write_manifest(records, manifest_path)

    for category, count in categories.items():
        if len(records) >= total:
            break
        remaining = total - len(records)
        want = min(count, remaining)
        entries = query_arxiv(f"cat:{category}", max_results=want + 3)
        time.sleep(delay)  # arXiv rate-limit courtesy

        added = 0
        for entry in entries:
            if added >= want or len(records) >= total:
                break
            if entry["arxiv_id"] in seen:
                continue
            seen.add(entry["arxiv_id"])

            local_pdf = pdf_dir / f"{entry['arxiv_id']}.pdf"
            record = dict(entry)
            record["local_pdf"] = str(local_pdf.relative_to(_PROJECT_ROOT))

            if local_pdf.exists() and local_pdf.stat().st_size > 0:
                record["status"] = "cached"
                record["pdf_bytes"] = local_pdf.stat().st_size
                _logger.info("Cached PDF for %s (%s).", entry["arxiv_id"], entry["title"][:50])
            else:
                record = _download_entry_pdf(record, local_pdf)
                time.sleep(delay)

            records.append(record)
            added += 1
        _logger.info("Category %s: added %d papers.", category, added)
        _write_manifest(records, manifest_path)  # persist progress after each category

    _write_manifest(records, manifest_path)
    _summary(records, manifest_path)
    return records


def _reconcile_existing(pdf_dir: Path, manifest_path: Path) -> tuple[list[dict], set[str]]:
    """Rebuild partial state from a prior manifest + PDFs already on disk.

    Loads any saved manifest, then promotes orphaned PDFs (present on disk but
    missing from the manifest) into minimal records so nothing downloaded is lost.
    """
    records: list[dict] = []
    seen: set[str] = set()

    if manifest_path.exists():
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            for paper in prior.get("papers", []):
                arxiv_id = paper.get("arxiv_id")
                if arxiv_id and arxiv_id not in seen:
                    records.append(paper)
                    seen.add(arxiv_id)
        except Exception as exc:  # noqa: BLE001 - corrupt manifest is non-fatal
            _logger.warning("Could not read prior manifest (%s); starting fresh.", exc)

    # Promote orphaned PDFs on disk that are not in the manifest.
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        if pdf.stat().st_size == 0:
            continue
        arxiv_id = pdf.stem
        if arxiv_id in seen:
            continue
        records.append({
            "arxiv_id": arxiv_id,
            "title": f"(orphaned PDF; metadata not captured)",
            "authors": [],
            "abstract": "",
            "primary_category": "",
            "categories": [],
            "published": "",
            "updated": "",
            "abs_url": f"http://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"http://arxiv.org/pdf/{arxiv_id}.pdf",
            "local_pdf": str(pdf.relative_to(_PROJECT_ROOT)),
            "status": "cached_orphan",
            "pdf_bytes": pdf.stat().st_size,
        })
        seen.add(arxiv_id)
        _logger.info("Reconciled orphaned PDF %s from disk.", arxiv_id)

    return records, seen


def _download_entry_pdf(record: dict, dest: Path) -> dict:
    """Download one PDF; update the record with status/size on success or failure."""
    try:
        size = _download_file(record["pdf_url"], dest)
        record["status"] = "downloaded"
        record["pdf_bytes"] = size
        _logger.info("Downloaded %s (%d bytes): %s", record["arxiv_id"], size, record["title"][:50])
    except Exception as exc:  # noqa: BLE001 - per-paper resilience
        record["status"] = "failed"
        record["error"] = str(exc)
        _logger.error("PDF download failed for %s: %s", record["arxiv_id"], exc)
    return record


def _write_manifest(records: list[dict], manifest_path: Path) -> None:
    payload = {
        "source": "arXiv API (export.arxiv.org/api/query)",
        "count": len(records),
        "fetched_at_note": "timestamp omitted to keep this build deterministic",
        "papers": records,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _logger.info("Manifest written to %s", manifest_path)


def _summary(records: list[dict], manifest_path: Path) -> None:
    ok = [r for r in records if r.get("status") in ("downloaded", "cached")]
    failed = [r for r in records if r.get("status") == "failed"]
    cats = {}
    for r in records:
        cats[r.get("primary_category", "?")] = cats.get(r.get("primary_category", "?"), 0) + 1
    total_bytes = sum(r.get("pdf_bytes", 0) for r in ok)
    print("\n=== arXiv Test Set ===")
    print(f"Total papers: {len(records)}  (downloaded/cached: {len(ok)}, failed: {len(failed)})")
    print(f"PDF storage : ~{total_bytes / 1e6:.1f} MB across {len(ok)} files")
    print("By primary category:")
    for cat, n in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<20} {n}")
    print(f"Manifest    : {manifest_path}")
    print(f"PDF dir     : {(_PROJECT_ROOT / 'data' / 'pdfs')}")
    if failed:
        print("Failed downloads:")
        for r in failed:
            print(f"  - {r['arxiv_id']}: {r.get('error', 'unknown')}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Fetch a real arXiv test set.")
    parser.add_argument("--total", type=int, default=20, help="Total papers to fetch (default 20).")
    parser.add_argument(
        "--categories", type=str, default=None,
        help="Category distribution as 'cat:count,cat:count'. Overrides defaults.",
    )
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between requests (arXiv courtesy).")
    parser.add_argument("--pdf-dir", type=str, default="data/pdfs", help="Output PDF directory.")
    parser.add_argument("--manifest", type=str, default="data/arxiv_testset.json", help="Manifest JSON path.")
    args = parser.parse_args(argv)

    categories = parse_categories_arg(args.categories)
    pdf_dir = _PROJECT_ROOT / args.pdf_dir
    manifest_path = _PROJECT_ROOT / args.manifest

    records = fetch_testset(
        categories=categories,
        pdf_dir=pdf_dir,
        manifest_path=manifest_path,
        delay=args.delay,
        total=args.total,
    )

    # Exit non-zero if too few papers were actually obtained with PDFs.
    have_pdf = sum(1 for r in records if r.get("status") in ("downloaded", "cached"))
    return 0 if have_pdf > 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
