#!/usr/bin/env python3
"""Deterministic migration of baseline entity mentions and verified claims onto
the re-chunked v2 corpus (config bge_T512_O64_snapON, chunk namespace
"chunk-rebuild-v2").

No model inference, no network. Pure Python (stdlib) plus the deterministic
quote locator ``work/quote_locate.py``.

What it does
------------
Every baseline candidate row is an exact verbatim span inside one old chunk:
    old_chunk.text[cand.start:cand.end] == cand.text
and every old chunk is a verified single-page slice:
    old_chunk.text == page.text[old_chunk.char_start:old_chunk.char_end]

The migration maps those spans onto the new chunks of the same document:

* Candidates: the new chunk whose text fully contains the page-relative span
  is located by exact text search (context-window probes disambiguate
  overlapping/repeated text).  Neighbour chunks are considered, so a span that
  straddles a new chunk boundary still migrates when the O=64 sentence overlap
  repeats it inside a neighbour.  A span that no new chunk fully contains is a
  migration failure (queued).
* Retained repaired claims (repaired_claims.jsonl, retained == true): the quote
  is relocated with ``locate_quote`` -- one exact pass over all candidate new
  chunks (doc order), then one normalized pass.  ``sentence_window`` is
  disabled (align_threshold > 1) because migration must stay within the
  verified exact/normalized contract.  Quotes shorter than the locator gate
  (min_chars=40, min_words=5) fail by construction and are queued.
* Eligibility (G3 in .audit/claim_repair_20261019/METHOD_FIX_quote_first.md):
  a migrated claim keeps only candidate_ids that (a) migrated successfully and
  (b) whose new candidate span lies inside the claim's new span (same new
  chunk).  ``graph_retrieval_eligible`` = any remain.

DROP list: documents from the listed source_ids / document_uids are skipped
entirely -- not migrated, not queued.

Inputs (defaults)
-----------------
* --baseline   /Users/pika/Downloads/MP_KG_RAG_v2/semantic_graph_source
* --repaired   .audit/claim_repair_20261019/repaired_claims.jsonl
* --new-corpus .../rebuild_v2/semantic_graph_source_v2   (does not exist yet)
* --quote-locate .../work/quote_locate.py

New-corpus layout assumptions
-----------------------------
pages.json rows: document_uid, page, text (same schema as the baseline pages).
chunks.json rows: chunk_id, document_uid, text, and enough geometry to place
the chunk inside its document:
  1. explicit ``doc_char_start``/``doc_char_end`` document offsets, or
  2. ``page`` + page-relative ``char_start``/``char_end`` (single-page chunks,
     text == page slice), or
  3. ``page`` / ``page_first`` / ``page_last`` hints + exact text search inside
     the document text (pages joined with --page-sep, default "\\n"), or
  4. normalized text search (whitespace/typography tolerant) as a last resort.
Unresolvable chunks are reported in migration_stats.json and simply cannot own
records.  Same-page chunks that keep ``text == page slice`` (the documented
builder property) resolve through strategy 2; cross-page chunks through 1/3/4.
If the real builder writes ``rebuild_v2/build_report.md`` with a different
cross-page joining convention, pass the matching --page-sep or add the
document offsets; the two-pass resolver verifies every placement against the
actual chunk text, so a wrong convention fails loudly (all counts reported)
instead of silently misplacing spans.

Outputs (written to --out)
--------------------------
* candidates_v2.json     migrated candidate rows (baseline schema, new
                         chunk_id/start/end; text unchanged; slice re-verified)
* extractions_v2.json    one row per migrated claim, mirroring the baseline
                         extraction row schema + provenance:
                         {chunk_id, chunk_sha256, page, document_uid,
                          request_id: "migrated:<chunk>#<claim>", source_id,
                          claims: [{candidate_ids, claim, span_start,
                                    span_end, span_method, source,
                                    source_claim_id}]}
* relocation_failures.json
                         {old_chunk_id: {document_uid, reason,
                                         claims_failed: [...],
                                         candidates_failed: [...]}}  (queue;
                         drop-list documents are never queued)
* migration_stats.json   counters, rates, failure histograms, per-document
                         table, summary text

Usage (fixture):
    python3 migrate_candidates_claims_v2.py \\
        --baseline <fixture>/inputs --repaired <fixture>/inputs/repaired_claims.jsonl \\
        --new-corpus <fixture>/new_corpus --out <fixture>/outputs

Usage (real corpus, when built):
    python3 migrate_candidates_claims_v2.py --run \\
        --new-corpus /Users/pika/Downloads/MP_KG_RAG_v2/source_refresh/rebuild_v2/semantic_graph_source_v2
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_BASELINE = Path("/Users/pika/Downloads/MP_KG_RAG_v2/semantic_graph_source")
DEFAULT_REPAIRED = Path(
    "/Users/pika/Downloads/Hate_Speech_Counter_Narrative_RAG_Project/.audit/"
    "claim_repair_20261019/repaired_claims.jsonl"
)
DEFAULT_NEW_CORPUS = Path(
    "/Users/pika/Downloads/MP_KG_RAG_v2/source_refresh/rebuild_v2/semantic_graph_source_v2"
)
DEFAULT_QUOTE_LOCATE = Path("/Users/pika/Downloads/MP_KG_RAG_v2/work/quote_locate.py")

DROP_SOURCE_IDS = frozenset(
    {"SRC094", "SRC057", "SRC035", "SRC021", "SRC014", "SRC008", "SRC002", "SRC043"}
)
DROP_DOCUMENT_UIDS = frozenset(
    {
        "a533552e15d8eaad5f1a",
        "e73ce899d0d017b75b3f",
        "e5cf65fda072f471c943",
        "5870aa0b966171a20605",
        "9840d21271d20ff290b3",
        "e9175600c16216faf150",
    }
)

MIN_CHARS = 40
MIN_WORDS = 5
QUOTE_WINDOW_PAD = 200  # chars of slack around the quote when picking new chunks


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_rows(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("records", data.get("rows", data.get("chunks", data)))
    return data if isinstance(data, list) else []


def read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _int(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def find_all(haystack: str, needle: str) -> list[int]:
    if not needle:
        return []
    positions = []
    pos = haystack.find(needle)
    while pos != -1:
        positions.append(pos)
        pos = haystack.find(needle, pos + 1)
    return positions


def load_quote_locate(path: Path):
    spec = importlib.util.spec_from_file_location("_migrate_quote_locate", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pages(path: Path):
    docs: dict[str, list[dict]] = defaultdict(list)
    for row in read_rows(path):
        uid = row.get("document_uid")
        if uid is None:
            continue
        docs[str(uid)].append(row)
    for uid in docs:
        docs[uid].sort(key=lambda row: (int(row.get("page", 0)), str(row.get("page", ""))))
    return docs


class DocLayout:
    """Document text assembled from pages plus per-page offset bookkeeping."""

    def __init__(self, pages: list[dict], sep: str):
        self.pages = pages
        self.sep = sep
        self.text_parts = []
        self.page_ranges = []  # (page_number, start, end)
        offset = 0
        for rec in pages:
            text = str(rec.get("text") or "")
            self.text_parts.append(text)
            self.page_ranges.append((rec.get("page"), offset, offset + len(text)))
            offset += len(text) + len(sep)
        self.text = sep.join(self.text_parts)

    def page_start(self, page) -> int | None:
        for number, start, end in self.page_ranges:
            if number == page:
                return start
        return None


class NewChunk:
    __slots__ = ("row", "chunk_id", "document_uid", "source_id", "page", "text",
                 "d0", "d1", "pages", "chunk_sha256")

    def __init__(self, row, chunk_id, document_uid, source_id, page, text, d0, d1, pages, chunk_sha256):
        self.row = row
        self.chunk_id = chunk_id
        self.document_uid = document_uid
        self.source_id = source_id
        self.page = page
        self.text = text
        self.d0 = d0
        self.d1 = d1
        self.pages = pages
        self.chunk_sha256 = chunk_sha256


def resolve_new_chunks(new_rows, new_pages, sep, normalize_with_index_map):
    """Place every new chunk inside its document; drop what cannot be placed."""
    layouts: dict[str, DocLayout] = {}
    resolved: dict[str, list[NewChunk]] = defaultdict(list)
    stats = Counter()
    page_text_by_key = {}
    for uid, recs in new_pages.items():
        layouts[uid] = DocLayout(recs, sep)
        for rec in recs:
            page_text_by_key[(uid, rec.get("page"))] = str(rec.get("text") or "")

    for row in new_rows:
        uid = row.get("document_uid")
        text = row.get("text")
        chunk_id = row.get("chunk_id")
        if uid is None or not isinstance(text, str) or not text or not chunk_id:
            stats["unresolved_missing_fields"] += 1
            continue
        uid = str(uid)
        layout = layouts.get(uid)
        if layout is None:
            stats["unresolved_document_not_in_new_pages"] += 1
            continue

        d0 = d1 = None
        doc_text = layout.text

        # 1. explicit document offsets
        a, b = _int(row.get("doc_char_start")), _int(row.get("doc_char_end"))
        if a is not None and b is not None and a >= 0 and b >= a and doc_text[a:b] == text:
            d0, d1 = a, b
            stats["by_doc_offsets"] += 1
        # 2. page-relative offsets on a single page (text == page slice)
        if d0 is None:
            page = row.get("page")
            cs, ce = _int(row.get("char_start")), _int(row.get("char_end"))
            if page is not None and cs is not None and ce is not None:
                page_text = page_text_by_key.get((uid, page))
                start = layout.page_start(page)
                if page_text is not None and start is not None and page_text[cs:ce] == text:
                    d0, d1 = start + cs, start + ce
                    stats["by_page_offsets"] += 1
        # 3. exact search in the document text (page hints filter duplicates)
        if d0 is None:
            positions = find_all(doc_text, text)
            hints = []
            for key in ("page_first", "page_last", "page"):
                value = row.get(key)
                if value is not None:
                    hints.append(value)
            if len(positions) > 1 and hints:
                lo = layout.page_start(hints[0])
                hi_start = layout.page_start(hints[-1])
                if lo is not None and hi_start is not None:
                    hi_end = next(end for number, start, end in layout.page_ranges if number == hints[-1])
                    filtered = [p for p in positions if lo <= p < hi_end]
                    if len(filtered) == 1:
                        positions = filtered
            if len(positions) == 1:
                d0 = positions[0]
                stats["by_text_search"] += 1
            elif len(positions) > 1:
                d0 = positions[0]
                stats["by_text_search_ambiguous_first"] += 1
        # 4. normalized search (typography/whitespace tolerant)
        if d0 is None:
            norm_doc, index_map = normalize_with_index_map(doc_text)
            norm_text, _ = normalize_with_index_map(text)
            if norm_text:
                occurrences = find_all(norm_doc, norm_text)
                if occurrences:
                    d0 = index_map[occurrences[0]]
                    d1 = index_map[occurrences[0] + len(norm_text)]
                    if doc_text[d0:d1] != text:
                        stats["by_normalized_search_text_differs"] += 1
                    else:
                        stats["by_normalized_search"] += 1

        if d0 is None:
            stats["unresolved_not_found"] += 1
            continue
        if d1 is None:
            d1 = d0 + len(text)
        if doc_text[d0:d1] != text:
            stats["unresolved_slice_verify_failed"] += 1
            continue

        pages_covered = [
            number
            for number, start, end in layout.page_ranges
            if min(end, d1) > max(start, d0)
        ]
        canonical = sha256_text(text)
        provided = row.get("chunk_sha256")
        if provided is not None and str(provided).casefold() != canonical:
            stats["provided_sha256_mismatch"] += 1
        page = row.get("page")
        if page is None and pages_covered:
            page = pages_covered[0]
        resolved[uid].append(
            NewChunk(
                row=row,
                chunk_id=str(chunk_id),
                document_uid=uid,
                source_id=row.get("source_id"),
                page=page,
                text=text,
                d0=d0,
                d1=d1,
                pages=pages_covered,
                chunk_sha256=canonical,
            )
        )
    for uid in resolved:
        resolved[uid].sort(key=lambda chunk: (chunk.d0, chunk.d1))
    return resolved, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--repaired", type=Path, default=DEFAULT_REPAIRED)
    parser.add_argument("--new-corpus", type=Path, default=DEFAULT_NEW_CORPUS)
    parser.add_argument("--quote-locate", type=Path, default=DEFAULT_QUOTE_LOCATE)
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: <new-corpus>/../migration_v2_out)")
    parser.add_argument("--page-sep", default="\n", help="separator joining page texts into a document")
    parser.add_argument("--run", action="store_true",
                        help="confirm a real-corpus migration when --new-corpus is the default path")
    args = parser.parse_args()

    if args.new_corpus == DEFAULT_NEW_CORPUS and not args.run:
        print("refusing to read the default real corpus without --run (fixture runs pass --new-corpus explicitly)")
        return 2
    if not (args.new_corpus / "chunks.json").is_file() or not (args.new_corpus / "pages.json").is_file():
        print(f"new corpus not found: {args.new_corpus} (chunks.json + pages.json required)")
        return 2
    out_dir = args.out or (args.new_corpus.parent / "migration_v2_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    quote_locate = load_quote_locate(args.quote_locate)
    normalize_with_index_map = quote_locate.normalize_with_index_map

    # ---------------------------------------------------------------- baseline
    old_pages = load_pages(args.baseline / "pages.json")
    old_page_text = {
        (uid, rec.get("page")): str(rec.get("text") or "")
        for uid, recs in old_pages.items()
        for rec in recs
    }
    old_chunks = read_rows(args.baseline / "chunks.json")
    old_by_id = {str(row.get("chunk_id")): row for row in old_chunks}
    candidates = read_rows(args.baseline / "candidates.json")
    extractions = read_rows(args.baseline / "semantic_extractions.json")
    extraction_by_chunk = {str(row.get("chunk_id")): row for row in extractions}
    repaired = read_jsonl(args.repaired)
    retained_claims = [row for row in repaired if row.get("retained") is True]

    # ------------------------------------------------------------- new corpus
    new_pages = load_pages(args.new_corpus / "pages.json")
    new_layouts = {uid: DocLayout(recs, args.page_sep) for uid, recs in new_pages.items()}
    new_rows = read_rows(args.new_corpus / "chunks.json")
    resolved_new, resolver_stats = resolve_new_chunks(
        new_rows, new_pages, args.page_sep, normalize_with_index_map
    )

    stats = Counter()
    failure_reasons_claims = Counter()
    failure_reasons_candidates = Counter()
    queue: dict[str, dict] = {}
    doc_stats: dict[str, Counter] = defaultdict(Counter)

    def queue_record(old_chunk_id, document_uid, reason, claim_id=None, candidate_id=None):
        entry = queue.setdefault(
            str(old_chunk_id),
            {"document_uid": str(document_uid), "reason": reason,
             "claims_failed": [], "candidates_failed": []},
        )
        if claim_id is not None and claim_id not in entry["claims_failed"]:
            entry["claims_failed"].append(claim_id)
        if candidate_id is not None and candidate_id not in entry["candidates_failed"]:
            entry["candidates_failed"].append(candidate_id)

    def is_dropped(source_id, document_uid) -> bool:
        return str(source_id) in DROP_SOURCE_IDS or str(document_uid) in DROP_DOCUMENT_UIDS

    def chunks_covering_page(uid: str, page):
        return [
            chunk for chunk in resolved_new.get(uid, [])
            if page is not None and page in chunk.pages
        ]

    # ------------------------------------------------------- candidate mapping
    migrated_candidates: dict[tuple[str, str], dict] = {}  # (candidate_id, old_chunk) -> row
    migrated_candidate_rows: list[dict] = []
    candidate_slice_ok = 0
    stats["baseline_chunks"] = len(old_chunks)
    stats["new_chunks_raw"] = len(new_rows)
    stats["new_chunks_resolved"] = sum(len(v) for v in resolved_new.values())

    for cand in candidates:
        stats["candidates_total"] += 1
        old_chunk = old_by_id.get(str(cand.get("chunk_id")))
        if old_chunk is None:
            stats["candidates_skipped_old_chunk_missing"] += 1
            failure_reasons_candidates["old_chunk_missing"] += 1
            continue
        uid = str(old_chunk.get("document_uid"))
        source_id = old_chunk.get("source_id")
        if is_dropped(source_id, uid):
            stats["candidates_skipped_drop_list"] += 1
            continue
        stats["candidates_attempted"] += 1
        doc_stats[uid]["candidates_attempted"] += 1

        page = old_chunk.get("page")
        cs = _int(old_chunk.get("char_start"))
        start, end = _int(cand.get("start")), _int(cand.get("end"))
        page_text = old_page_text.get((uid, page))
        if page_text is None or cs is None or start is None or end is None:
            stats["candidates_failed"] += 1
            failure_reasons_candidates["old_geometry_missing"] += 1
            queue_record(old_chunk.get("chunk_id"), uid, "old_geometry_missing",
                         candidate_id=cand.get("candidate_id"))
            continue

        a, b = cs + start, cs + end
        expected = cand.get("text")
        target = page_text[a:b]
        if not isinstance(expected, str) or target != expected:
            stats["candidates_failed"] += 1
            failure_reasons_candidates["old_slice_mismatch"] += 1
            queue_record(old_chunk.get("chunk_id"), uid, "old_slice_mismatch",
                         candidate_id=cand.get("candidate_id"))
            continue

        owner_chunks = chunks_covering_page(uid, page)
        if not owner_chunks:
            stats["candidates_failed"] += 1
            failure_reasons_candidates["no_new_chunk_covers_page"] += 1
            queue_record(old_chunk.get("chunk_id"), uid, "no_new_chunk_covers_page",
                         candidate_id=cand.get("candidate_id"))
            continue

        new_page_start = new_layouts[uid].page_start(page) if uid in new_layouts else None
        if new_page_start is None:
            stats["candidates_failed"] += 1
            failure_reasons_candidates["no_new_chunk_covers_page"] += 1
            queue_record(old_chunk.get("chunk_id"), uid, "no_new_chunk_covers_page",
                         candidate_id=cand.get("candidate_id"))
            continue

        # exact relocation: retained documents keep page texts byte-identical,
        # so the span's true document position is new_page_start + page-local
        # offset. It must be fully contained by a chunk; a span crossing a new
        # chunk boundary is a genuine split and must not be rerouted onto a
        # look-alike occurrence of the same text elsewhere in the document.
        true_start = new_page_start + a
        true_end = true_start + (b - a)
        containing = [
            chunk for chunk in owner_chunks
            if chunk.d0 <= true_start and true_end <= chunk.d1
        ]
        if not containing:
            stats["candidates_failed"] += 1
            failure_reasons_candidates["span_split_across_new_chunks"] += 1
            queue_record(old_chunk.get("chunk_id"), uid, "span_split_across_new_chunks",
                         candidate_id=cand.get("candidate_id"))
            continue

        # chunks overlap; a span inside an overlap band is contained by both
        # chunks at the same document offset -- keep the earliest (birth) chunk
        chunk = min(containing, key=lambda item: (item.d0, item.d1))
        new_start, new_end = true_start - chunk.d0, true_end - chunk.d0
        if chunk.text[new_start:new_end] != expected:
            stats["candidates_failed"] += 1
            failure_reasons_candidates["slice_verify_failed"] += 1
            queue_record(old_chunk.get("chunk_id"), uid, "slice_verify_failed",
                         candidate_id=cand.get("candidate_id"))
            continue

        row = dict(cand)
        row["chunk_id"] = chunk.chunk_id
        row["start"] = new_start
        row["end"] = new_end
        migrated_candidate_rows.append(row)
        migrated_candidates[(str(cand.get("candidate_id")), str(old_chunk.get("chunk_id")))] = {
            "chunk_id": chunk.chunk_id, "start": new_start, "end": new_end,
        }
        stats["candidates_migrated"] += 1
        candidate_slice_ok += 1
        doc_stats[uid]["candidates_migrated"] += 1

    # ----------------------------------------------------------- claim mapping
    migrated_claims: list[dict] = []
    stats["retained_claims"] = len(retained_claims)
    claim_slice_exact_ok = 0
    claim_slice_normalized_ok = 0

    for claim in retained_claims:
        claim_id = str(claim.get("claim_id"))
        old_chunk = old_by_id.get(str(claim.get("chunk_id")))
        if old_chunk is None:
            stats["claims_skipped_old_chunk_missing"] += 1
            failure_reasons_claims["old_chunk_missing"] += 1
            continue
        uid = str(old_chunk.get("document_uid"))
        source_id = old_chunk.get("source_id")
        if is_dropped(source_id, uid):
            stats["claims_skipped_drop_list"] += 1
            continue
        stats["claims_attempted"] += 1
        doc_stats[uid]["claims_attempted"] += 1

        quote = claim.get("quote")
        quote_start, quote_end = _int(claim.get("quote_start")), _int(claim.get("quote_end"))
        page = old_chunk.get("page")
        cs = _int(old_chunk.get("char_start"))
        if not isinstance(quote, str) or quote_start is None or quote_end is None or cs is None:
            stats["claims_failed"] += 1
            failure_reasons_claims["missing_old_offsets"] += 1
            queue_record(old_chunk.get("chunk_id"), uid, "missing_old_offsets", claim_id=claim_id)
            continue
        if old_chunk["text"][quote_start:quote_end] != quote:
            stats["claims_failed"] += 1
            failure_reasons_claims["old_quote_slice_mismatch"] += 1
            queue_record(old_chunk.get("chunk_id"), uid, "old_quote_slice_mismatch", claim_id=claim_id)
            continue
        if len(quote.strip()) < MIN_CHARS or len(quote.split()) < MIN_WORDS:
            stats["claims_failed"] += 1
            failure_reasons_claims["below_min_length_gate"] += 1
            queue_record(old_chunk.get("chunk_id"), uid, "below_min_length_gate", claim_id=claim_id)
            continue

        owner_chunks = chunks_covering_page(uid, page)
        if not owner_chunks:
            stats["claims_failed"] += 1
            failure_reasons_claims["no_new_chunk_covers_page"] += 1
            queue_record(old_chunk.get("chunk_id"), uid, "no_new_chunk_covers_page", claim_id=claim_id)
            continue

        a, b = cs + quote_start, cs + quote_end
        new_page_start = new_layouts[uid].page_start(page) if uid in new_layouts else None
        if new_page_start is None:
            stats["claims_failed"] += 1
            failure_reasons_claims["no_new_chunk_covers_page"] += 1
            queue_record(old_chunk.get("chunk_id"), uid, "no_new_chunk_covers_page", claim_id=claim_id)
            continue

        # exact relocation: the quote must land on its true document position
        # (page texts are byte-identical for retained documents); a quote that
        # wraps a new chunk boundary is a genuine miss.
        true_start = new_page_start + a
        true_end = new_page_start + b
        containing = [
            chunk for chunk in owner_chunks
            if chunk.d0 <= true_start and true_end <= chunk.d1
        ]

        located = None
        method = None
        duplicates = 0
        for chunk in containing:  # pass 1: exact, at the true document position
            occurrences = find_all(chunk.text, quote)
            if not occurrences:
                continue
            if len(occurrences) > 1:
                duplicates += 1
            start = next((p for p in occurrences if chunk.d0 + p == true_start), None)
            if start is None:
                continue
            end = start + len(quote)
            if chunk.text[start:end] == quote:
                located, method = (chunk, start, end), "exact"
                break
        if located is None and containing:  # pass 2: normalized, nearest chunk to the true position
            for chunk in sorted(containing, key=lambda c: (
                abs(true_start - (c.d0 + c.d1) // 2), c.d0,
            )):
                hint = true_start - chunk.d0
                result = quote_locate.locate_quote(
                    chunk.text, quote, min_chars=MIN_CHARS, min_words=MIN_WORDS,
                    hint=hint, align_threshold=1.01,
                )
                if result is not None and result["method"] == "normalized":
                    located, method = (chunk, result["start"], result["end"]), "normalized"
                    break

        if located is None:
            stats["claims_failed"] += 1
            failure_reasons_claims["not_located"] += 1
            queue_record(old_chunk.get("chunk_id"), uid, "not_located", claim_id=claim_id)
            continue

        chunk, new_start, new_end = located
        if method == "exact" and chunk.text[new_start:new_end] != quote:
            stats["claims_failed"] += 1
            failure_reasons_claims["slice_verify_failed"] += 1
            queue_record(old_chunk.get("chunk_id"), uid, "slice_verify_failed", claim_id=claim_id)
            continue

        extraction = extraction_by_chunk.get(str(claim.get("chunk_id")))
        old_ids: list[str] = []
        claim_index = claim.get("claim_index")
        if extraction is not None and isinstance(claim_index, int):
            claims_list = extraction.get("claims")
            if isinstance(claims_list, list) and 0 <= claim_index < len(claims_list):
                entry = claims_list[claim_index]
                if isinstance(entry, dict):
                    old_ids = [value for value in entry.get("candidate_ids") or [] if isinstance(value, str)]

        kept_ids: list[str] = []
        for candidate_id in dict.fromkeys(old_ids):
            span = migrated_candidates.get((candidate_id, str(claim.get("chunk_id"))))
            if span is None:
                stats["candidate_ids_dropped_unmigrated"] += 1
                continue
            if span["chunk_id"] != chunk.chunk_id:
                stats["candidate_ids_dropped_foreign_chunk"] += 1
                continue
            if span["start"] >= new_start and span["end"] <= new_end:
                kept_ids.append(candidate_id)
            else:
                stats["candidate_ids_dropped_outside_span"] += 1

        record = {
            "claim_id": claim_id,
            "old_chunk_id": str(claim.get("chunk_id")),
            "new_chunk_id": chunk.chunk_id,
            "document_uid": uid,
            "source_id": source_id,
            "page": chunk.page,
            "chunk_sha256": chunk.chunk_sha256,
            "span_start": new_start,
            "span_end": new_end,
            "span_method": method,
            "quote": quote,
            "claim_index": claim_index if isinstance(claim_index, int) else None,
            "candidate_ids": kept_ids,
            "graph_retrieval_eligible": bool(kept_ids),
            "duplicate_occurrences_in_chunk": duplicates,
        }
        migrated_claims.append(record)
        stats["claims_relocated"] += 1
        stats[f"claims_relocated_{method}"] += 1
        if method == "exact":
            claim_slice_exact_ok += 1
        else:
            claim_slice_normalized_ok += 1
        if kept_ids:
            stats["claims_eligible"] += 1
        doc_stats[uid]["claims_relocated"] += 1
        doc_stats[uid]["eligible"] += 1 if kept_ids else 0

    # ------------------------------------------------------------------ output
    def sort_claims(record):
        d0 = next((line.d0 for line in resolved_new.get(record["document_uid"], [])
                   if line.chunk_id == record["new_chunk_id"]), 10**12)
        return (record["document_uid"], d0, record["claim_index"] if record["claim_index"] is not None else -1)

    migrated_claims.sort(key=sort_claims)
    migrated_candidate_rows.sort(key=lambda row: (str(row.get("chunk_id")), row.get("start", 0), row.get("end", 0)))

    extraction_rows = []
    for record in migrated_claims:
        extraction_rows.append(
            {
                "chunk_id": record["new_chunk_id"],
                "chunk_sha256": record["chunk_sha256"],
                "claims": [
                    {
                        "candidate_ids": record["candidate_ids"],
                        "claim": record["quote"],
                        "span_start": record["span_start"],
                        "span_end": record["span_end"],
                        "span_method": record["span_method"],
                        "source": "migrated",
                        "source_claim_id": record["claim_id"],
                    }
                ],
                "document_uid": record["document_uid"],
                "page": record["page"],
                "request_id": f"migrated:{record['new_chunk_id']}#{record['claim_id']}",
                "source_id": record["source_id"],
            }
        )

    def write_json(path: Path, payload):
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    write_json(out_dir / "candidates_v2.json", migrated_candidate_rows)
    write_json(out_dir / "extractions_v2.json", extraction_rows)
    write_json(out_dir / "relocation_failures.json", queue)

    attempted_candidates = stats["candidates_attempted"]
    attempted_claims = stats["claims_attempted"]
    candidate_pct = round(100.0 * stats["candidates_migrated"] / attempted_candidates, 3) if attempted_candidates else 0.0
    claim_pct = round(100.0 * stats["claims_relocated"] / attempted_claims, 3) if attempted_claims else 0.0

    per_doc = []
    for uid in sorted(doc_stats):
        row = dict(doc_stats[uid])
        row["document_uid"] = uid
        per_doc.append(row)

    summary = (
        f"candidates: {stats['candidates_migrated']}/{attempted_candidates} migrated "
        f"({candidate_pct}%); claims: {stats['claims_relocated']}/{attempted_claims} relocated "
        f"({claim_pct}%; exact {stats['claims_relocated_exact']}, normalized {stats['claims_relocated_normalized']}); "
        f"eligible claims: {stats['claims_eligible']}; "
        f"failed candidates: {stats['candidates_failed']}, failed claims: {stats['claims_failed']} "
        f"(queue chunks: {len(queue)})"
    )

    stats_payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "baseline": str(args.baseline),
            "repaired": str(args.repaired),
            "new_corpus": str(args.new_corpus),
            "quote_locate": str(args.quote_locate),
            "page_sep": args.page_sep.replace("\n", "\\n"),
        },
        "config": {
            "chunk_config": "bge_T512_O64_snapON",
            "new_chunk_namespace": "chunk-rebuild-v2",
            "min_chars": MIN_CHARS,
            "min_words": MIN_WORDS,
            "drop_source_ids": sorted(DROP_SOURCE_IDS),
            "drop_document_uids": sorted(DROP_DOCUMENT_UIDS),
        },
        "totals": {
            **{key: stats[key] for key in sorted(stats)},
            "candidate_migration_pct": candidate_pct,
            "claim_relocation_pct": claim_pct,
        },
        "new_chunk_resolution": dict(sorted(resolver_stats.items())),
        "slice_checks": {
            "candidates_verified": candidate_slice_ok,
            "candidates_failed_after_mapping": stats["candidates_failed"],
            "claims_exact_verified": claim_slice_exact_ok,
            "claims_normalized_verified": claim_slice_normalized_ok,
        },
        "candidate_failure_reasons": dict(sorted(failure_reasons_candidates.items())),
        "claim_failure_reasons": dict(sorted(failure_reasons_claims.items())),
        "per_document": per_doc,
        "summary": summary,
        "seconds": round(time.time() - start_time, 1),
    }
    write_json(out_dir / "migration_stats.json", stats_payload)

    print(summary)
    print(f"outputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
