#!/usr/bin/env python3
"""rebuild_v2 corpus acceptance gate suite (CPU only, no model inference).

Validates a ``semantic_graph_source``-style corpus directory (``chunks.json``,
``pages.json``, ``evidence_ledger.json``, ``manifest.json`` and optional
``candidates.json`` / ``semantic_extractions.json``) against the audited
baseline corpus.

Checks
------
C1  Files parse; chunk rows carry the baseline required key set.
C2  Text-slice invariant for sampled chunks (same-page rule; both cross-page
    conventions are tested when ``build_report.md`` declares none).
C3  Empirical hash consistency: chunk_sha256 == sha256(text); page text hashes;
    ledger rows join 1:1 by chunk_id (text agreement reported separately).
C4  chunk_id uniqueness; duplicate (document_uid, text hash) pairs.
C5  candidates.json span slices; semantic_extractions.json chunk joins and
    exact claim spans.
C6  Band statistics: BGE-M3 token counts on a 10% sample (500-900 band,
    median, mid-sentence rate; same endset rule as the rechunk study).
C7  Manifest artifact hashes when the manifest carries them.
C8  Document counts by source_type vs swap plan / acquisition manifests.

The only model artifact touched is the cached BGE-M3 tokenizer.json (pure
tokenization; the rechunk study already uses it).  No forward passes.

Usage:
  python3 accept_corpus_v2.py --corpus DIR [--sample-frac 0.1] [--full] [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASELINE_CHUNKS = Path("/Users/pika/Downloads/MP_KG_RAG_v2/semantic_graph_source/chunks.json")
BGE_TOKENIZER_JSON = Path(
    "/Users/pika/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/"
    "5617a9f61b028005a4858fdac845db406aefb181/tokenizer.json"
)
BUILD_REPORT = Path("/Users/pika/Downloads/MP_KG_RAG_v2/source_refresh/rebuild_v2/build_report.md")
SWAP_PLAN = Path(
    "/Users/pika/Downloads/Hate_Speech_Counter_Narrative_RAG_Project/.audit/"
    "source_refresh_20261019/swap_plan.md"
)
V3_MANIFEST = Path("/Users/pika/Downloads/MP_KG_RAG_v2/source_refresh/acquisition_v3_manifest.json")
DEFAULT_OUT_DIR = Path(
    "/Users/pika/Downloads/Hate_Speech_Counter_Narrative_RAG_Project/.audit/rebuild_v2_acceptance_20261019"
)

ENDSET = set('.!?\u0964"\'\u201d\u2019')
SOURCE_TYPES = ("peer_reviewed", "ngo_report", "official", "un_agency", "court", "education", "government", "survey")
POST_REBUILD_MIN, POST_REBUILD_MAX = 104, 115
BASELINE_DOCS = 90
DUP_TOLERANCE = 0.005

REQUIRED_FILES = ("chunks.json", "pages.json", "evidence_ledger.json", "manifest.json")
OPTIONAL_FILES = ("candidates.json", "semantic_extractions.json")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def page_key(value):
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return value


def load_rows(path: Path):
    """Load a JSON list, or rows from a dict carrying records/rows/chunks."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)], None
    if isinstance(data, dict):
        for key in ("records", "rows", "chunks"):
            rows = data.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)], data
        return [], data
    return [], data


def uid_of(row) -> str:
    for key in ("document_uid", "document_id", "id"):
        value = row.get(key)
        if value is not None and str(value):
            return str(value)
    return ""


def first_int(row, keys):
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
            return int(value.strip())
    return None


def check(check_id, title, status, summary, details=None, examples=None, hard=True):
    return {
        "id": check_id,
        "title": title,
        "status": status,
        "summary": summary,
        "details": details or {},
        "examples": (examples or [])[:5],
        "hard": hard,
    }


def sample_rows(rows, sample_frac, full):
    if full or sample_frac >= 1.0:
        return list(rows), 1.0
    step = max(1, int(round(1.0 / sample_frac)))
    sampled = [row for index, row in enumerate(rows) if index % step == 0]
    return sampled, (len(sampled) / len(rows) if rows else 0.0)


class Corpus:
    def __init__(self, directory: Path, sample_frac: float, full: bool):
        self.dir = directory
        self.sample_frac = sample_frac
        self.full = full
        self.files = {}
        self.rows = {}
        self.docs = {}
        self.parse_errors = {}
        for name in REQUIRED_FILES + OPTIONAL_FILES:
            path = directory / name
            self.files[name] = path.exists()
            if not path.exists():
                continue
            try:
                rows, doc = load_rows(path)
                self.rows[name] = rows
                self.docs[name] = doc
            except (OSError, ValueError, TypeError) as exc:
                self.parse_errors[name] = repr(exc)
        self.chunks = self.rows.get("chunks.json") or []
        self.pages = self.rows.get("pages.json") or []
        self.ledger = self.rows.get("evidence_ledger.json") or []
        self.manifest = self.docs.get("manifest.json") or {}
        if not self.manifest and isinstance(self.rows.get("manifest.json"), list):
            self.manifest = {}
        self.sampled, self.effective_frac = sample_rows(self.chunks, sample_frac, full)
        self.page_text = {}
        for page in self.pages:
            self.page_text[(uid_of(page), page_key(page.get("page")))] = str(page.get("text") or "")
        self.chunk_text = {str(row.get("chunk_id")): str(row.get("text") or "") for row in self.chunks}
        self.baseline_keys = set()

    def load_baseline_keys(self):
        rows, _ = load_rows(BASELINE_CHUNKS)
        for row in rows:
            self.baseline_keys.update(row.keys())
        return len(rows)


def check_c1(corpus: Corpus):
    details = {
        "required_present": {name: corpus.files[name] for name in REQUIRED_FILES},
        "optional_present": {name: corpus.files[name] for name in OPTIONAL_FILES},
        "parse_errors": corpus.parse_errors,
        "row_counts": {name: len(corpus.rows.get(name) or []) for name in corpus.rows},
    }
    missing_files = [name for name in REQUIRED_FILES if not corpus.files[name]]
    if missing_files:
        return check("C1", "files parse + baseline key set", "FAIL",
                     f"missing required files: {', '.join(missing_files)}", details)
    if corpus.parse_errors:
        return check("C1", "files parse + baseline key set", "FAIL",
                     f"unparseable files: {', '.join(corpus.parse_errors)}", details)

    baseline_rows = corpus.load_baseline_keys()
    details["baseline_rows"] = baseline_rows
    details["required_key_count"] = len(corpus.baseline_keys)
    if not corpus.baseline_keys:
        return check("C1", "files parse + baseline key set", "FAIL",
                     f"baseline key set unavailable at {BASELINE_CHUNKS}", details)

    missing_counter = Counter()
    rows_missing_any = 0
    examples = []
    for row in corpus.chunks:
        missing = sorted(corpus.baseline_keys - set(row.keys()))
        if missing:
            rows_missing_any += 1
            missing_counter.update(missing)
            if len(examples) < 5:
                examples.append({"chunk_id": str(row.get("chunk_id"))[:16], "missing_keys": missing})
    extra_keys = sorted({key for row in corpus.chunks for key in row} - corpus.baseline_keys)
    details["rows_missing_required_keys"] = rows_missing_any
    details["missing_key_counts"] = dict(missing_counter)
    details["extra_keys_vs_baseline"] = extra_keys
    if rows_missing_any:
        return check("C1", "files parse + baseline key set", "FAIL",
                     f"{rows_missing_any}/{len(corpus.chunks)} rows miss baseline keys", details, examples)
    return check("C1", "files parse + baseline key set", "PASS",
                 f"all {len(corpus.chunks)} chunk rows carry the {len(corpus.baseline_keys)} baseline keys", details)


CROSS_VARIANTS = ("middle_newline_only", "all_segments_newline")


def cross_page_text(first_text, middle_texts, last_text, a, b, variant):
    if variant == "middle_newline_only":
        return first_text[a:] + "\n".join(middle_texts) + last_text[:b]
    return "\n".join([first_text[a:]] + list(middle_texts) + [last_text[:b]])


def check_c2(corpus: Corpus):
    details = {
        "sampled": len(corpus.sampled),
        "effective_sample_frac": round(corpus.effective_frac, 4),
        "convention_declared": None,
        "same_page_ok": 0,
        "same_page_bad": 0,
        "cross_page_rows": 0,
        "cross_variant_matches": {name: 0 for name in CROSS_VARIANTS},
        "cross_page_unmatched": 0,
        "pages_missing": 0,
    }
    if BUILD_REPORT.exists():
        details["convention_declared"] = "build_report.md present (see file for declared rule)"
    examples = []
    for row in corpus.sampled:
        uid = uid_of(row)
        a = first_int(row, ("char_start", "start_char", "span_start"))
        b = first_int(row, ("char_end", "end_char", "span_end"))
        text = str(row.get("text") or "")
        page = page_key(row.get("page"))
        page_first = page_key(row.get("page_first")) if row.get("page_first") is not None else page
        page_last = page_key(row.get("page_last")) if row.get("page_last") is not None else page
        if a is None or b is None:
            details["pages_missing"] += 1
            if len(examples) < 5:
                examples.append({"chunk_id": str(row.get("chunk_id"))[:16], "problem": "char offsets missing"})
            continue
        first_text = corpus.page_text.get((uid, page_first))
        last_text = corpus.page_text.get((uid, page_last))
        if first_text is None or last_text is None:
            details["pages_missing"] += 1
            if len(examples) < 5:
                examples.append({"chunk_id": str(row.get("chunk_id"))[:16], "problem": f"missing page text doc={uid} first={page_first} last={page_last}"})
            continue
        if page_first == page_last:
            if text == first_text[a:b]:
                details["same_page_ok"] += 1
            else:
                details["same_page_bad"] += 1
                if len(examples) < 5:
                    examples.append({
                        "chunk_id": str(row.get("chunk_id"))[:16], "rule": "same_page",
                        "text_head": text[:60], "page_slice_head": first_text[a:b][:60],
                    })
            continue
        details["cross_page_rows"] += 1
        middle_texts = []
        middle_missing = False
        if isinstance(page_first, int) and isinstance(page_last, int):
            for number in range(page_first + 1, page_last):
                middle = corpus.page_text.get((uid, number))
                if middle is None:
                    middle_missing = True
                    break
                middle_texts.append(middle)
        else:
            middle_missing = True
        if middle_missing:
            details["cross_page_unmatched"] += 1
            if len(examples) < 5:
                examples.append({"chunk_id": str(row.get("chunk_id"))[:16], "rule": "cross_page", "problem": "middle pages unavailable"})
            continue
        matched = [name for name in CROSS_VARIANTS if text == cross_page_text(first_text, middle_texts, last_text, a, b, name)]
        if matched:
            for name in matched:
                details["cross_variant_matches"][name] += 1
        else:
            details["cross_page_unmatched"] += 1
            if len(examples) < 5:
                examples.append({
                    "chunk_id": str(row.get("chunk_id"))[:16], "rule": "cross_page",
                    "pages": [page_first, page_last], "text_head": text[:60],
                    "variant_a_head": cross_page_text(first_text, middle_texts, last_text, a, b, CROSS_VARIANTS[0])[:60],
                })

    hard_bad = details["same_page_bad"] + details["cross_page_unmatched"] + details["pages_missing"]
    if hard_bad:
        return check("C2", "text-slice invariant", "FAIL",
                     f"{hard_bad} sampled chunks violate the slice rule", details, examples)
    if details["cross_page_rows"]:
        winners = [name for name, count in details["cross_variant_matches"].items() if count == details["cross_page_rows"]]
        summary = f"all {len(corpus.sampled)} sampled chunks match; cross-page winner: {winners[0] if winners else 'mixed/underdetermined'}"
    else:
        summary = f"all {len(corpus.sampled)} sampled chunks match the same-page rule; no cross-page rows in sample"
        if not corpus.full and corpus.effective_frac < 1.0:
            summary += " (sampled; run --full to exercise cross-page rows)"
    return check("C2", "text-slice invariant", "PASS", summary, details)


def check_c3(corpus: Corpus):
    details = {}
    examples = []
    chunk_bad = 0
    for row in corpus.chunks:
        text = str(row.get("text") or "")
        if row.get("chunk_sha256") is not None and str(row["chunk_sha256"]).casefold() != sha256_text(text):
            chunk_bad += 1
            if len(examples) < 5:
                examples.append({"chunk_id": str(row.get("chunk_id"))[:16], "problem": "chunk_sha256 != sha256(text)"})
    details["chunk_sha256_rows_checked"] = len(corpus.chunks)
    details["chunk_sha256_mismatches"] = chunk_bad

    page_checked = page_ok = page_bad = 0
    for row in corpus.chunks:
        supplied = row.get("page_sha256") if row.get("page_sha256") is not None else row.get("page_text_sha256")
        if supplied is None:
            continue
        page_checked += 1
        text = corpus.page_text.get((uid_of(row), page_key(row.get("page"))))
        if text is not None and str(supplied).casefold() == sha256_text(text):
            page_ok += 1
        else:
            page_bad += 1
            if len(examples) < 5:
                examples.append({"chunk_id": str(row.get("chunk_id"))[:16], "problem": "page hash != sha256(page.text)"})
    details["page_hashes_checked"] = page_checked
    details["page_hashes_ok"] = page_ok
    details["page_hashes_mismatched"] = page_bad

    chunk_ids = {str(row.get("chunk_id")) for row in corpus.chunks}
    ledger_ids = [str(row.get("chunk_id")) for row in corpus.ledger]
    ledger_counts = Counter(ledger_ids)
    ledger_dupes = sum(1 for count in ledger_counts.values() if count > 1)
    missing_ledger = len(chunk_ids - set(ledger_ids))
    extra_ledger = len(set(ledger_ids) - chunk_ids)
    details["ledger_rows"] = len(corpus.ledger)
    details["ledger_duplicate_chunk_ids"] = ledger_dupes
    details["chunks_without_ledger_row"] = missing_ledger
    details["ledger_rows_without_chunk"] = extra_ledger

    text_mismatch = sha_mismatch = span_mismatch = 0
    text_examples = []
    ledger_by_id = {str(row.get("chunk_id")): row for row in corpus.ledger}
    for row in corpus.chunks:
        ledger_row = ledger_by_id.get(str(row.get("chunk_id")))
        if ledger_row is None:
            continue
        chunk_text = str(row.get("text") or "")
        ledger_text = ledger_row.get("text")
        if ledger_text is None:
            ledger_text = ledger_row.get("displayed_text")
        if str(ledger_text or "") != chunk_text:
            text_mismatch += 1
            if len(text_examples) < 5:
                text_examples.append({"chunk_id": str(row.get("chunk_id"))[:16], "ledger_chars": len(str(ledger_text or "")), "chunk_chars": len(chunk_text)})
        supplied_sha = ledger_row.get("text_sha256")
        if supplied_sha is not None and str(supplied_sha).casefold() != sha256_text(chunk_text):
            sha_mismatch += 1
        for chunk_key, ledger_key in (("span_start", "span_start"), ("span_end", "span_end")):
            chunk_value = row.get(chunk_key)
            ledger_value = ledger_row.get(ledger_key)
            if chunk_value is not None and ledger_value is not None and chunk_value != ledger_value:
                span_mismatch += 1
                break
    details["ledger_text_mismatches"] = text_mismatch
    details["ledger_text_sha256_mismatches"] = sha_mismatch
    details["ledger_span_mismatches"] = span_mismatch

    if chunk_bad or page_bad or ledger_dupes or missing_ledger or extra_ledger:
        return check("C3", "hash consistency + ledger join", "FAIL",
                     "hash or ledger join failures", details, examples + text_examples)
    status = "PASS"
    notes = [f"{len(corpus.chunks)} chunk hashes consistent"]
    if page_checked:
        notes.append(f"{page_ok}/{page_checked} page hashes consistent")
    else:
        notes.append("page hashes absent")
    notes.append("ledger joins 1:1 by chunk_id")
    if text_mismatch:
        status = "WARN"
        notes.append(f"ledger text disagrees on {text_mismatch} rows (stale-ledger quirk; index build warns stale_evidence_ledger)")
    return check("C3", "hash consistency + ledger join", status, "; ".join(notes), details, text_examples)


def declaration_present(*texts) -> bool:
    joined = "\n".join(text for text in texts if text).casefold()
    return any(key in joined for key in ("declared overlap", "overlap convention", "duplicate chunk", "alternate_format_duplicate", "declared duplicates"))


def check_c4(corpus: Corpus):
    details = {}
    examples = []
    ids = [str(row.get("chunk_id")) for row in corpus.chunks]
    id_dupes = Counter(ids)
    duplicate_ids = {key: value for key, value in id_dupes.items() if value > 1}
    details["chunk_rows"] = len(ids)
    details["unique_chunk_ids"] = len(set(ids))
    details["duplicate_chunk_id_rows"] = sum(value - 1 for value in duplicate_ids.values())
    for key in list(duplicate_ids)[:5]:
        examples.append({"chunk_id": key[:16], "problem": "duplicate chunk_id", "count": duplicate_ids[key]})

    text_hashes = Counter()
    for row in corpus.chunks:
        text_hashes[(uid_of(row), sha256_text(str(row.get("text") or "")))] += 1
    duplicates = {key: value for key, value in text_hashes.items() if value > 1}
    extra_rows = sum(value - 1 for value in duplicates.values())
    details["duplicate_doc_text_hash_pairs"] = len(duplicates)
    details["extra_rows_from_duplicates"] = extra_rows

    build_report_text = BUILD_REPORT.read_text(encoding="utf-8", errors="ignore") if BUILD_REPORT.exists() else ""
    manifest_text = json.dumps(corpus.manifest, default=str)
    declared = declaration_present(build_report_text, manifest_text)
    details["overlap_declared"] = declared
    share = (extra_rows / len(corpus.chunks)) if corpus.chunks else 0.0
    details["duplicate_share"] = round(share, 6)
    details["duplicate_tolerance"] = DUP_TOLERANCE
    for (uid, digest), count in list(duplicates.items())[:5]:
        examples.append({"document_uid": uid[:16], "text_sha256": digest[:16], "count": count})

    if duplicate_ids:
        return check("C4", "chunk uniqueness", "FAIL",
                     f"{details['duplicate_chunk_id_rows']} duplicate chunk_id rows", details, examples)
    if extra_rows == 0:
        return check("C4", "chunk uniqueness", "PASS",
                     f"{len(ids)} unique chunk_ids; no duplicate (document_uid, text-hash) pairs", details)
    if declared:
        return check("C4", "chunk uniqueness", "WARN",
                     f"{extra_rows} duplicate rows within declared overlap/duplicate convention", details, examples)
    if share <= DUP_TOLERANCE:
        return check("C4", "chunk uniqueness", "WARN",
                     f"{extra_rows} duplicate (doc, text-hash) extra rows ({share:.4%} <= {DUP_TOLERANCE:.2%} tolerance), undeclared", details, examples)
    return check("C4", "chunk uniqueness", "FAIL",
                 f"{extra_rows} duplicate (doc, text-hash) extra rows ({share:.4%}) exceed undeclared tolerance", details, examples)


def check_c5(corpus: Corpus):
    details = {}
    examples = []
    hard_failure = False

    candidates = corpus.rows.get("candidates.json") or []
    details["candidates_present"] = corpus.files.get("candidates.json", False)
    details["candidates_rows"] = len(candidates)
    cand_ok = cand_bad = cand_unknown = 0
    for row in candidates:
        chunk_id = str(row.get("chunk_id"))
        text = corpus.chunk_text.get(chunk_id)
        start, end = first_int(row, ("start",)), first_int(row, ("end",))
        if text is None:
            cand_unknown += 1
            if len(examples) < 5:
                examples.append({"chunk_id": chunk_id[:16], "problem": "candidate chunk_id not in chunks.json"})
            continue
        surface = row.get("text")
        if surface is None:
            surface = row.get("label")
        if start is None or end is None or not isinstance(surface, str) or text[start:end] != surface:
            cand_bad += 1
            if len(examples) < 5:
                examples.append({
                    "chunk_id": chunk_id[:16], "start": start, "end": end,
                    "candidate_head": str(surface)[:40], "chunk_slice_head": text[start:end][:40] if start is not None and end is not None else None,
                })
        else:
            cand_ok += 1
    details["candidate_spans_ok"] = cand_ok
    details["candidate_spans_bad"] = cand_bad
    details["candidate_chunks_unknown"] = cand_unknown
    if corpus.files.get("candidates.json") and (cand_bad or cand_unknown):
        hard_failure = True

    extractions = corpus.rows.get("semantic_extractions.json") or []
    details["extractions_present"] = corpus.files.get("semantic_extractions.json", False)
    details["extractions_rows"] = len(extractions)
    join_missing = 0
    claims_total = 0
    claims_exact_ok = 0
    claims_exact_bad = 0
    claims_other_methods = 0
    claim_examples = []
    for row in extractions:
        chunk_id = str(row.get("chunk_id"))
        text = corpus.chunk_text.get(chunk_id)
        if text is None:
            join_missing += 1
            continue
        claims = row.get("claims")
        if isinstance(claims, str):
            try:
                claims = json.loads(claims)
            except ValueError:
                claims = []
        if not isinstance(claims, list):
            continue
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            claims_total += 1
            method = claim.get("span_method")
            claim_text = claim.get("claim")
            if claim_text is None:
                claim_text = claim.get("quote")
            if method not in (None, "exact"):
                claims_other_methods += 1
                continue
            start = claim.get("span_start")
            end = claim.get("span_end")
            if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start <= end <= len(text)) or text[start:end] != claim_text:
                claims_exact_bad += 1
                if len(claim_examples) < 5:
                    claim_examples.append({
                        "chunk_id": chunk_id[:16], "span_start": start, "span_end": end,
                        "claim_head": str(claim_text)[:40],
                        "chunk_slice_head": text[start:end][:40] if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(text) else None,
                    })
            else:
                claims_exact_ok += 1
    details["extraction_chunk_join_missing"] = join_missing
    details["claims_total"] = claims_total
    details["claims_exact_ok"] = claims_exact_ok
    details["claims_exact_mismatch"] = claims_exact_bad
    details["claims_other_span_methods"] = claims_other_methods
    instructions = "span_method absent treated as exact"
    if not corpus.files.get("candidates.json"):
        details["candidates_present"] = False
    if corpus.files.get("semantic_extractions.json") and (join_missing or claims_exact_bad):
        hard_failure = True

    if hard_failure and not (corpus.files.get("semantic_extractions.json")):
        return check("C5", "candidate + claim spans", "FAIL",
                     f"candidate span failures: bad={cand_bad} unknown_chunk={cand_unknown}", details, examples)
    if hard_failure:
        return check("C5", "candidate + claim spans", "FAIL",
                     f"span failures: candidates bad={cand_bad}/unknown={cand_unknown}; claims exact ok={claims_exact_ok}/{claims_total} bad={claims_exact_bad}; join_missing={join_missing} ({instructions})",
                     details, (claim_examples or examples))
    return check("C5", "candidate + claim spans", "PASS",
                 f"candidates {cand_ok}/{cand_ok + cand_bad} exact; claims exact {claims_exact_ok}/{claims_total}; joins clean", details)


def check_c6(corpus: Corpus):
    sample_texts = [str(row.get("text") or "") for row in corpus.sampled]
    details = {
        "sampled": len(sample_texts),
        "effective_sample_frac": round(corpus.effective_frac, 4),
        "tokenizer": str(BGE_TOKENIZER_JSON),
    }
    if not BGE_TOKENIZER_JSON.exists():
        return check("C6", "band statistics (BGE-M3 token counts)", "WARN",
                     f"BGE tokenizer not found at {BGE_TOKENIZER_JSON}; band stats skipped", details, hard=False)
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(BGE_TOKENIZER_JSON))
    started = time.time()
    encodings = tokenizer.encode_batch(sample_texts, add_special_tokens=False)
    counts = [len(encoding.ids) for encoding in encodings]
    details["tokenize_seconds"] = round(time.time() - started, 2)
    in_band = sum(1 for value in counts if 500 <= value <= 900)
    loose = sum(1 for value in counts if 460 <= value <= 960)
    mid = sum(1 for text in sample_texts if text.rstrip() and text.rstrip()[-1] not in ENDSET)
    details.update({
        "pct_500_900": round(100.0 * in_band / len(counts), 2),
        "pct_460_960": round(100.0 * loose / len(counts), 2),
        "median_tokens": statistics.median(counts),
        "mean_tokens": round(statistics.mean(counts), 1),
        "min_tokens": min(counts),
        "max_tokens": max(counts),
        "mid_sentence_pct": round(100.0 * mid / len(counts), 2),
    })
    field_counts = [row.get("token_count") for row in corpus.sampled]
    if all(isinstance(value, int) for value in field_counts):
        details["token_count_field_median"] = statistics.median(field_counts)
        details["token_count_field_equals_words"] = all(
            row.get("token_count") == len(str(row.get("text") or "").split()) for row in corpus.sampled
        )
    summary = (f"{details['pct_500_900']}% in 500-900 (n={len(counts)}), median={details['median_tokens']}, "
               f"mid-sentence={details['mid_sentence_pct']}%")
    status = "PASS" if details["pct_500_900"] >= 90.0 else "INFO"
    return check("C6", "band statistics (BGE-M3 token counts)", status, summary, details, hard=False)


def check_c7(corpus: Corpus):
    details = {"artifacts_present": "artifacts" in corpus.manifest, "hashed_entries": 0, "verified": 0, "mismatched": 0, "unresolved": []}
    examples = []
    artifacts = corpus.manifest.get("artifacts")
    if artifacts is None:
        details["note"] = "manifest has no artifacts section"
        return check("C7", "manifest artifact hashes", "NA", "manifest.artifacts absent", details, hard=False)
    if isinstance(artifacts, dict):
        iterable = artifacts.items()
    elif isinstance(artifacts, list):
        iterable = ((str(item.get("name") or item.get("id") or index), item) for index, item in enumerate(artifacts) if isinstance(item, dict))
    else:
        return check("C7", "manifest artifact hashes", "NA", "manifest.artifacts has no recognizable shape", details, hard=False)
    hard_failure = False
    for name, entry in iterable:
        if isinstance(entry, dict):
            digest = entry.get("sha256") or entry.get("hash")
            rel = entry.get("path") or entry.get("file") or entry.get("name")
            if digest and rel:
                details["hashed_entries"] += 1
                path = (corpus.dir / str(rel))
                if not path.is_file():
                    details["unresolved"].append({"name": name, "path": str(rel), "problem": "missing"})
                    hard_failure = True
                    continue
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual.casefold() == str(digest).casefold():
                    details["verified"] += 1
                else:
                    details["mismatched"] += 1
                    hard_failure = True
                    if len(examples) < 5:
                        examples.append({"name": name, "path": str(rel), "expected": str(digest)[:16], "actual": actual[:16]})
            else:
                details.setdefault("entries_without_hash", []).append(str(name))
        elif isinstance(entry, str):
            if not (corpus.dir / entry).is_file():
                details["unresolved"].append({"name": name, "path": entry, "problem": "missing (no hash contract)"})
    source_hashes = corpus.manifest.get("source_hashes")
    if isinstance(source_hashes, dict) and source_hashes:
        details["source_hashes_checked"] = 0
        for name, digest in source_hashes.items():
            mapping = {
                "chunks": "chunks.json", "evidence_ledger": "evidence_ledger.json",
                "semantic_extractions": "semantic_extractions.json", "candidates": "candidates.json",
                "manifest": "manifest.json", "pages": "pages.json",
            }
            filename = mapping.get(str(name), str(name))
            path = corpus.dir / filename
            if path.is_file():
                details["source_hashes_checked"] += 1
                if hashlib.sha256(path.read_bytes()).hexdigest().casefold() != str(digest).casefold():
                    details.setdefault("source_hash_mismatches", []).append(str(name))
                    hard_failure = True
    if hard_failure:
        return check("C7", "manifest artifact hashes", "FAIL", "artifact hash verification failed", details, examples)
    if details["hashed_entries"]:
        return check("C7", "manifest artifact hashes", "PASS",
                     f"{details['verified']}/{details['hashed_entries']} hashed artifacts verified", details)
    return check("C7", "manifest artifact hashes", "NA",
                 "manifest.artifacts carries plain paths (no hashes) - nothing to verify", details, hard=False)


def parse_swap_plan():
    plan = {"projected_by_type": {}, "post_drop": None, "table_present": False}
    if not SWAP_PLAN.exists():
        return plan
    text = SWAP_PLAN.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(r"^\|\s*(\w+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*$", re.M)
    for match in pattern.finditer(text):
        source_type, current, after_drops, after_adds = match.groups()
        if source_type in SOURCE_TYPES:
            plan["projected_by_type"][source_type] = {"current": int(current), "after_drops": int(after_drops), "after_adds": int(after_adds)}
            plan["table_present"] = True
    match = re.search(r"Post-drop corpus:\s*(\d+)\s*docs", text)
    if match:
        plan["post_drop"] = int(match.group(1))
    return plan


def check_c8(corpus: Corpus):
    plan = parse_swap_plan()
    actual_by_type = Counter()
    docs = {}
    for row in corpus.chunks:
        uid = uid_of(row)
        if uid not in docs:
            docs[uid] = str(row.get("source_type") or "unknown")
            actual_by_type[docs[uid]] += 1
    total = len(docs)
    details = {
        "total_docs": total,
        "docs_by_source_type": dict(sorted(actual_by_type.items())),
        "plan_projected_by_type": plan["projected_by_type"],
        "plan_post_drop": plan["post_drop"],
        "expected_window": [POST_REBUILD_MIN, POST_REBUILD_MAX],
        "baseline_docs": BASELINE_DOCS,
    }
    if corpus.pages:
        page_docs = {uid_of(row) for row in corpus.pages}
        details["pages_docs"] = len(page_docs)
    v3 = None
    if V3_MANIFEST.exists():
        try:
            v3_data = json.loads(V3_MANIFEST.read_text(encoding="utf-8"))
            v3 = {
                "new_downloads": len(v3_data.get("new_downloads") or []),
                "verification_only": len(v3_data.get("verification_raw_v2") or []),
                "not_found": len(v3_data.get("not_found") or []),
            }
        except (OSError, ValueError):
            v3 = None
    details["acquisition_v3_manifest"] = v3
    diffs = {}
    for source_type, projected in plan["projected_by_type"].items():
        actual = actual_by_type.get(source_type, 0)
        if actual != projected["after_adds"]:
            diffs[source_type] = {"actual": actual, "projected": projected["after_adds"], "delta": actual - projected["after_adds"]}
    details["source_type_deltas_vs_projection"] = diffs
    if total == BASELINE_DOCS:
        return check("C8", "document counts vs plan", "WARN",
                     f"{total} docs = pre-drop baseline (expected {POST_REBUILD_MIN}-{POST_REBUILD_MAX} post-rebuild)", details, hard=False)
    if POST_REBUILD_MIN <= total <= POST_REBUILD_MAX:
        status = "PASS"
        summary = f"{total} docs inside post-rebuild window [{POST_REBUILD_MIN},{POST_REBUILD_MAX}]"
        if diffs:
            summary += f"; {len(diffs)} source_type deltas vs projection (see details)"
            if any(abs(item["delta"]) > 2 for item in diffs.values()):
                status = "WARN"
        return check("C8", "document counts vs plan", status, summary, details, hard=False)
    return check("C8", "document counts vs plan", "FAIL",
                 f"{total} docs outside post-rebuild window [{POST_REBUILD_MIN},{POST_REBUILD_MAX}] and != baseline {BASELINE_DOCS}", details)


def render_markdown(report):
    lines = [
        "# rebuild_v2 corpus acceptance report",
        "",
        f"- corpus: `{report['meta']['corpus']}`",
        f"- generated_at: {report['meta']['generated_at']}",
        f"- tool: `{report['meta']['tool']}`",
        f"- sample: {report['meta']['sampled_chunks']} chunks (effective fraction {report['meta']['effective_sample_frac']}, full={report['meta']['full']})",
        f"- exit code: {report['summary']['exit_code']}",
        "",
        "## Summary",
        "",
        "| check | status | summary |",
        "| --- | --- | --- |",
    ]
    for item in report["checks"]:
        summary = str(item["summary"]).replace("|", "/")
        lines.append(f"| C{item['id'][1]} {item['title']} | {item['status']} | {summary} |")
    lines.append("")
    lines.append(f"Counts: PASS={report['summary']['pass']} WARN={report['summary']['warn']} "
                 f"FAIL={report['summary']['fail']} NA/INFO={report['summary']['na_info']}")
    lines.append("")
    for item in report["checks"]:
        lines.append(f"## {item['id']} - {item['title']}: {item['status']}")
        lines.append("")
        lines.append(f"{item['summary']}")
        lines.append("")
        if item["details"]:
            lines.append("```json")
            lines.append(json.dumps(item["details"], indent=1, ensure_ascii=False, default=str))
            lines.append("```")
            lines.append("")
        if item["examples"]:
            lines.append("Failure examples (cap 5):")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(item["examples"], indent=1, ensure_ascii=False, default=str))
            lines.append("```")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--sample-frac", type=float, default=0.1)
    parser.add_argument("--full", action="store_true", help="disable sampling (covers the whole corpus)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    corpus_dir = args.corpus.expanduser().resolve()
    if not corpus_dir.is_dir():
        print(f"corpus directory not found: {corpus_dir}", file=sys.stderr)
        return 2
    out_dir = args.out.expanduser() if args.out else None
    if out_dir is None or not out_dir.is_absolute():
        out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    corpus = Corpus(corpus_dir, args.sample_frac, args.full)
    checks = []
    for runner in (check_c1, check_c2, check_c3, check_c4, check_c5, check_c6, check_c7, check_c8):
        t0 = time.time()
        try:
            item = runner(corpus)
        except Exception as exc:  # a crashing check is a hard failure
            item = check(runner.__name__.replace("check_", "C").upper(), runner.__doc__ or runner.__name__, "FAIL",
                         f"check crashed: {exc!r}")
        item["seconds"] = round(time.time() - t0, 2)
        checks.append(item)

    hard_failures = [item["id"] for item in checks if item["status"] == "FAIL" and item.get("hard", True)]
    exit_code = 1 if hard_failures else 0
    report = {
        "meta": {
            "tool": "rebuild_v2/accept_corpus_v2.py",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "corpus": str(corpus_dir),
            "corpus_name": corpus_dir.name,
            "sample_frac_requested": args.sample_frac,
            "full": bool(args.full),
            "sampled_chunks": len(corpus.sampled),
            "effective_sample_frac": round(corpus.effective_frac, 4),
            "total_chunks": len(corpus.chunks),
            "baseline_chunks": str(BASELINE_CHUNKS),
            "bge_tokenizer": str(BGE_TOKENIZER_JSON),
            "seconds": round(time.time() - started, 2),
        },
        "checks": checks,
        "summary": {
            "pass": sum(1 for item in checks if item["status"] == "PASS"),
            "warn": sum(1 for item in checks if item["status"] == "WARN"),
            "fail": sum(1 for item in checks if item["status"] == "FAIL"),
            "na_info": sum(1 for item in checks if item["status"] in {"NA", "INFO"}),
            "hard_failures": hard_failures,
            "exit_code": exit_code,
        },
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"accept_{corpus_dir.name}_{stamp}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    for item in checks:
        print(f"[{item['status']:4}] {item['id']} {item['title']}: {item['summary']}")
    print(f"exit_code={exit_code} reports: {json_path.name}, {md_path.name}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
