#!/usr/bin/env python3
"""Build the rebuild_v2 corpus (semantic_graph_source_v2).

Applies the study-recommended configuration ``bge_T512_O64_snapON``
(min=500 / target=512 / max=900 BGE-M3-token units, sentence snap ON,
overlap=64, tail repair) to the post-repair document set:

  90 baseline docs - 14 drops + 35 adds = 111 docs
  adds = 9 staged (SRC101-109) + 17 raw_v2 (SRC110-126) + 9 raw_v3 (SRC127-135)

The chunker is a literal transcription of
``.audit/rechunk_study_20261019/simulate_rechunk.py::snap_on_chunks``
(plus ``build_doc``/``make_chunk``/``repair_tail``); a per-document
verification pass compares every chunk against the imported simulation
module before rows are emitted (disable with --no-verify).

Emitted conventions (documented in build_report.md):
  * chunk text   = raw slice of the document text (pages joined with "\\n");
                   identical to the simulation's sentence-join for most chunks
    span_start/span_end       = document-global offsets
    char_*/character_*/start_char/end_char = page-local offsets (first/last page)
    token_start = BGE token count of the first page's text before the chunk
    token_end   = token_start + token_count, token_count = chunk units
    page/page_number = page_first (int)

No model inference: tokenization only (cached BGE-M3 tokenizer.json).
Read-only inputs: baseline semantic_graph_source/, source_refresh/ staging data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path("/Users/pika/Downloads/MP_KG_RAG_v2")
REBUILD = PROJECT / "source_refresh" / "rebuild_v2"
BASELINE = PROJECT / "semantic_graph_source"
STAGING = PROJECT / "source_refresh" / "staging_project" / "semantic_graph_source"
SIM_PATH = Path(
    "/Users/pika/Downloads/Hate_Speech_Counter_Narrative_RAG_Project/.audit/"
    "rechunk_study_20261019/simulate_rechunk.py"
)
BGE_TOKENIZER_JSON = Path(
    "/Users/pika/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/"
    "5617a9f61b028005a4858fdac845db406aefb181/tokenizer.json"
)
DEFAULT_OUT = REBUILD / "semantic_graph_source_v2"

TARGET = 512
OVERLAP = 64
MIN_UNITS = 500
MAX_UNITS = 900
UNIT = "token"  # bge space
CONFIG_NAME = "bge_T512_O64_snapON"

SENT_SPLIT = re.compile(r'(?<=[.!?\u0964\u2026])(?:["\'\)\]]*)\s+')
WORD_RE = re.compile(r"\S+")

DROP_UIDS = {
    "a533552e15d8eaad5f1a",
    "e73ce899d0d017b75b3f",
    "e5cf65fda072f471c943",
    "5870aa0b966171a20605",
    "9840d21271d20ff290b3",
    "e9175600c16216faf150",
    "49677fd1dc33dc2136f1",  # SRC008
    "5a2c409735f50d18d1f8",  # SRC002
    "f7acb60dab5c6650cd64",  # SRC043
}
DROP_SRC = {"SRC094", "SRC057", "SRC035", "SRC021", "SRC014"}
ICD11_DOCX_ID = "ALT-ICD11-CH17-2019-DOCX"

AUTHORITY_BY_TYPE = {
    "official": 1.0,
    "court": 0.95,
    "peer_reviewed": 0.95,
    "un_agency": 0.92,
    "government": 0.92,
    "ngo_report": 0.75,
    "survey": 0.75,
    "education": 0.72,
}

BASELINE_KEYS = [
    # exact key order of baseline chunks.json rows
    "authority_score", "char_end", "char_start", "character_end",
    "character_start", "chunk_id", "chunk_index", "chunk_sha256",
    "content_sha256", "document_hash", "document_sha256", "document_uid",
    "end_char", "end_token", "locator", "page", "page_chunk_index",
    "page_id", "page_number", "page_sha256", "page_text_sha256",
    "relative_path", "source_id", "source_id_sha256", "source_locator",
    "source_sha256", "source_text_key", "source_text_sha256", "source_type",
    "span_end", "span_start", "start_char", "start_token", "status", "text",
    "text_sha256", "token_count", "token_end", "token_start",
]

HEX64 = re.compile(r"[0-9a-f]{64}")


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def uid_for(manifest_id: str, file_sha: str) -> str:
    return sha256_str(manifest_id + "\x00" + file_sha)[:20]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# chunker: literal transcription of simulate_rechunk.py (token space)
# ---------------------------------------------------------------------------

def build_doc_t(text: str, page: int, source_type) -> list:
    """Sentences of one page with raw page-local char extents."""
    sentences = []
    cursor = 0
    for sent in SENT_SPLIT.split(text):
        s = sent.strip()
        if not s:
            continue
        pos = text.find(s, cursor)
        if pos < 0:
            pos = cursor if cursor <= len(text) else 0
        start = pos
        end = pos + len(s)
        cursor = end
        sentences.append(
            {
                "text": s,
                "words": len(WORD_RE.findall(s)),
                "tokens": None,
                "last_char": s.rstrip()[-1] if s.rstrip() else "",
                "page": page,
                "source_type": source_type,
                "start_char": start,
                "end_char": end,
                "piece": False,
            }
        )
    return sentences


def set_sentence_tokens(sentences, tokenizer):
    batch = [s["text"] for s in sentences]
    for i in range(0, len(batch), 4096):
        encs = tokenizer.encode_batch(batch[i : i + 4096], add_special_tokens=False)
        for s, enc in zip(sentences[i : i + 4096], encs):
            s["tokens"] = len(enc.ids)


def unit_of(seg, unit=UNIT):
    if seg.get("piece"):
        return seg["units"]
    return seg["words"] if unit == "word" else seg["tokens"]


def make_chunk(cur, unit=UNIT):
    return {
        "units": sum(unit_of(x, unit) for x in cur),
        "last_char": cur[-1]["last_char"],
        "page_first": cur[0]["page"],
        "page_last": cur[-1]["page"],
        "hard_split": False,
        "source_type": cur[0]["source_type"],
        "text": " ".join(x["text"] for x in cur),
        "i0": cur[0]["idx"],
        "i1": cur[-1]["idx"],
        "sents": list(cur),
        "content": list(cur),
    }


def _remove_last_identity(lst, obj):
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] is obj:
            lst.pop(i)
            return


def repair_tail(chunks, unit=UNIT):
    if len(chunks) < 2:
        return
    last = chunks[-1]
    prev = chunks[-2]
    if last["units"] >= MIN_UNITS:
        return
    if prev["units"] + last["units"] <= MAX_UNITS:
        prev["sents"].extend(last["sents"])
        prev["content"].extend(last["content"])
        prev["units"] += last["units"]
        prev["text"] = prev["text"] + " " + last["text"]
        prev["last_char"] = last["last_char"]
        prev["page_last"] = last["page_last"]
        prev["merged_tail"] = True
        chunks.pop()
        return
    covered = {s["idx"] for s in last["sents"]}
    guard = 0
    while last["units"] < MIN_UNITS and len(prev["sents"]) > 1 and guard < 512:
        guard += 1
        s = prev["sents"][-1]
        if prev["units"] - unit_of(s, unit) < MIN_UNITS:
            break
        prev["sents"].pop()
        _remove_last_identity(prev["content"], s)
        prev["units"] -= unit_of(s, unit)
        prev["text"] = " ".join(x["text"] for x in prev["sents"])
        prev["last_char"] = prev["sents"][-1]["last_char"]
        prev["page_last"] = prev["sents"][-1]["page"]
        prev["i1"] = prev["sents"][-1]["idx"]
        if s["idx"] in covered:
            continue
        last["sents"].insert(0, s)
        last["content"].insert(0, s)
        last["units"] += unit_of(s, unit)
        last["text"] = s["text"] + " " + last["text"]
        last["page_first"] = s["page"]
        last["i0"] = s["idx"]
        covered.add(s["idx"])


def _word_pieces(sentence, tokenizer):
    """Hard-split pieces of a too-long sentence with raw char extents."""
    spans = [(m.start(), m.end()) for m in WORD_RE.finditer(sentence["text"])]
    pieces = []
    cut = 0
    n = len(spans)
    while cut < n:
        take = spans[cut : cut + MAX_UNITS]
        cut += MAX_UNITS
        c0, c1 = take[0][0], take[-1][1]
        text = sentence["text"][c0:c1]
        pieces.append(
            {
                "text": text,
                "page": sentence["page"],
                "source_type": sentence["source_type"],
                "start_char": sentence["start_char"] + c0,
                "end_char": sentence["start_char"] + c1,
                "last_char": text[-1] if text else "",
                "units": len(take),
                "piece": True,
                "sentence": sentence,
                "sim_text": " ".join(text.split()),
            }
        )
    return pieces


def _token_pieces(sentence, tokenizer):
    enc = tokenizer.encode(sentence["text"], add_special_tokens=False)
    offsets = enc.offsets
    ids = enc.ids
    pieces = []
    cut = 0
    while cut < len(ids):
        end = min(cut + MAX_UNITS, len(ids))
        end_off = offsets[end - 1][1] or offsets[end - 1][0] + 1
        start_off = offsets[cut][0]
        if start_off >= end_off:
            start_off = end_off - 1
        if start_off < 0:
            start_off = 0
        text = sentence["text"][start_off:end_off]
        if not text:
            text = sentence["text"][end_off - 1 : end_off] or "x"
        pieces.append(
            {
                "text": text,
                "page": sentence["page"],
                "source_type": sentence["source_type"],
                "start_char": sentence["start_char"] + start_off,
                "end_char": sentence["start_char"] + end_off,
                "last_char": sentence["text"][end_off - 1],
                "units": end - cut,
                "piece": True,
                "sentence": sentence,
                "sim_text": tokenizer.decode(ids[cut:end]),
            }
        )
        cut = end
    return pieces


def snap_on_chunks(sentences, unit=UNIT, target=TARGET, overlap=OVERLAP, tokenizer=None):
    chunks = []
    carry = []
    i = 0
    n = len(sentences)
    while i < n:
        cur = list(carry)
        n_carry = len(cur)
        carry = []
        cur_units = sum(unit_of(s, unit) for s in cur)
        while i < n:
            s = sentences[i]
            u = unit_of(s, unit)
            if u > MAX_UNITS:
                if unit == "word":
                    pieces = _word_pieces(s, tokenizer)
                else:
                    pieces = _token_pieces(s, tokenizer)
                for p in pieces:
                    chunks.append(
                        {
                            "units": p["units"],
                            "last_char": p["last_char"],
                            "page_first": s["page"],
                            "page_last": s["page"],
                            "hard_split": True,
                            "source_type": s["source_type"],
                            "text": p["sim_text"] if "sim_text" in p else p["text"],
                            "i0": s["idx"],
                            "i1": s["idx"],
                            "sents": [],
                            "content": [p],
                        }
                    )
                i += 1
                continue
            if cur_units >= MIN_UNITS and cur_units + u > target:
                break
            if cur_units + u > MAX_UNITS:
                if cur_units < MIN_UNITS and n_carry > 0:
                    cur = cur[n_carry:]
                    n_carry = 0
                    cur_units = sum(unit_of(x, unit) for x in cur)
                    if cur_units == 0 or cur_units + u <= MAX_UNITS:
                        cur.append(s)
                        cur_units += u
                        i += 1
                        continue
                break
            cur.append(s)
            cur_units += u
            i += 1
        if not cur:
            i += 1
            continue
        chunks.append(make_chunk(cur, unit))
        if overlap > 0 and len(cur) > 1:
            acc = 0
            k = 0
            while k < len(cur) - 1:
                u = unit_of(cur[len(cur) - 1 - k], unit)
                if acc + u > overlap:
                    break
                acc += u
                k += 1
            carry = cur[len(cur) - k :] if k else []
    if chunks and chunks[-1]["units"] is not None:
        repair_tail(chunks, unit)
    return chunks


# ---------------------------------------------------------------------------
# input loading
# ---------------------------------------------------------------------------

def find_raw_file(name: str):
    for cand in (
        PROJECT / "source_refresh" / "raw" / name,
        PROJECT / "source_refresh" / "staging_project" / "raw" / name,
        PROJECT / "source_refresh" / "raw_v2" / name,
        PROJECT / "source_refresh" / "raw_v3" / name,
    ):
        if cand.is_file():
            return cand
    return None


def fill_page_gaps(pages):
    """Insert empty page records for missing page numbers inside [min, max].

    The baseline pages.json omits empty pages, so consecutive extracted pages
    can be non-contiguous (e.g. 1..100 with 97 rows).  Cross-page chunk text is
    a raw slice of the newline-joined page texts; the acceptance gate (C2) also
    expects the interior page numbers to exist in pages.json, so gaps are
    filled with empty-text page records.
    """
    pages = sorted(pages, key=lambda r: r["page"])
    if not pages:
        return pages
    by_num = {r["page"]: r for r in pages}
    return [by_num.get(p, {"page": p, "text": ""}) for p in range(pages[0]["page"], pages[-1]["page"] + 1)]


def build_doc_table(problems, warnings):
    baseline_pages = load_json(BASELINE / "pages.json")["records"]
    baseline_chunks = load_json(BASELINE / "chunks.json")
    base_by_uid = defaultdict(list)
    for rec in baseline_pages:
        base_by_uid[rec["document_uid"]].append(rec)

    n_base_uids = len(base_by_uid)
    docs = []
    for uid, recs in base_by_uid.items():
        sid = recs[0]["source_id"]
        if uid in DROP_UIDS or sid in DROP_SRC:
            continue
        recs.sort(key=lambda r: r["page"])
        pages_f = fill_page_gaps([{"page": int(r["page"]), "text": r["text"]} for r in recs])
        page_rows = [
            {
                "authority_score": recs[0]["authority_score"],
                "document_sha256": recs[0]["document_sha256"],
                "document_uid": uid,
                "page": pg["page"],
                "relative_path": recs[0]["relative_path"],
                "source_id": sid,
                "source_type": recs[0]["source_type"],
                "text": pg["text"],
            }
            for pg in pages_f
        ]
        docs.append(
            {
                "source_id": sid,
                "uid": uid,
                "source_type": recs[0]["source_type"],
                "relative_path": recs[0]["relative_path"],
                "document_sha256": recs[0]["document_sha256"],
                "authority_score": recs[0]["authority_score"],
                "pages": pages_f,
                "is_add": False,
                "meta": {},
                "page_rows": page_rows,
            }
        )
    kept = len(docs)

    # --- adds: batch-1 (staged) -------------------------------------------
    staged = load_json(STAGING / "chunks.json")
    if isinstance(staged, dict):
        staged = staged.get("records") or staged.get("rows")
    staged_by_uid = {}
    for row in staged:
        staged_by_uid.setdefault(row["document_uid"], row)

    extracted = load_jsonl(PROJECT / "source_refresh" / "extracted" / "pages.jsonl")
    ex_by_id = defaultdict(list)
    for r in extracted:
        ex_by_id[r["id"]].append(r)

    staged_meta = {r["document_uid"]: r for r in staged_by_uid.values()}
    used_files = set()
    for uid, row in sorted(staged_meta.items()):
        if uid in base_by_uid:
            continue
        file = Path(row["relative_path"]).name
        cand = None
        for did, recs in ex_by_id.items():
            if recs[0]["file"] == file:
                cand = (did, recs)
                break
        if cand is None:
            problems.append(f"batch-1 doc for uid {uid} ({file}) not found in extracted pages")
            continue
        did, recs = cand
        recs = sorted(recs, key=lambda r: (r["page"] if r["page"] is not None else 1))
        pages = fill_page_gaps(
            [{"page": int(r["page"] if r["page"] is not None else 1), "text": r["text"]} for r in recs]
        )
        raw = find_raw_file(file)
        file_sha = recs[0]["sha256"]
        if raw is not None:
            actual = sha256_bytes(raw.read_bytes())
            if actual != file_sha:
                problems.append(f"{uid}: raw sha mismatch {actual[:12]} != {file_sha[:12]}")
        else:
            warnings.append(f"{uid}: raw file not found for {file} (trusting row sha)")
        expect_uid = uid_for(did, file_sha)
        if expect_uid != uid:
            problems.append(f"{uid}: uid formula mismatch (expected {expect_uid})")
        docs.append(
            {
                "source_id": row["source_id"],
                "uid": uid,
                "source_type": row["source_type"],
                "relative_path": row["relative_path"],
                "document_sha256": file_sha,
                "authority_score": AUTHORITY_BY_TYPE.get(row["source_type"], row.get("authority_score", 0.72)),
                "pages": pages,
                "is_add": True,
                "meta": {
                    "language": row.get("language") or recs[0].get("language"),
                    "topic": row.get("topic") or recs[0].get("topic"),
                    "publisher": row.get("publisher") or recs[0].get("publisher"),
                    "publication_year": row.get("publication_year") or recs[0].get("year"),
                    "source_url": row.get("source_url") or recs[0].get("url"),
                    "source_title": row.get("source_title") or recs[0].get("title"),
                    "region": row.get("region"),
                    "country": row.get("country"),
                    "extraction_status": row.get("extraction_status") or "text_extracted_unreviewed",
                    "claim_extraction_status": row.get("claim_extraction_status") or "not_run",
                    "entity_linking_status": row.get("entity_linking_status") or "not_run",
                    "page_extraction_method": row.get("page_extraction_method") or recs[0].get("extraction_method"),
                },
                "page_rows": [],
            }
        )
        used_files.add(file)

    # --- adds: raw_v2 + raw_v3 --------------------------------------------
    pv2 = load_jsonl(REBUILD / "pages_v2.jsonl")
    pv2_by_id = defaultdict(list)
    for r in pv2:
        pv2_by_id[r["id"]].append(r)
    v3_manifest = load_json(PROJECT / "source_refresh" / "acquisition_v3_manifest.json")
    v3_ids = {d["id"] for d in v3_manifest["new_downloads"]} - {ICD11_DOCX_ID}
    all_ids = set(pv2_by_id)
    docx_ids = {i for i in all_ids if any(r["file"].endswith(".docx") for r in pv2_by_id[i])}
    if docx_ids != {ICD11_DOCX_ID}:
        problems.append(f"docx id set unexpected: {sorted(docx_ids)}")
    v2_ids = all_ids - v3_ids - docx_ids
    if len(v2_ids) != 17 or len(v3_ids) != 9:
        problems.append(f"raw add split unexpected: v2={len(v2_ids)} v3={len(v3_ids)}")
    for batch, ids, first_src in (("v2", v2_ids, 110), ("v3", v3_ids, 127)):
        for idx, did in enumerate(sorted(ids)):
            recs = sorted(pv2_by_id[did], key=lambda r: int(r["page"]))
            sid = f"SRC{first_src + idx:03d}"
            file = recs[0]["file"]
            file_sha = recs[0]["sha256"]
            raw = find_raw_file(file)
            if raw is not None:
                actual = sha256_bytes(raw.read_bytes())
                if actual != file_sha:
                    problems.append(f"{sid} ({did}): raw sha mismatch {actual[:12]} != {file_sha[:12]}")
            else:
                warnings.append(f"{sid} ({did}): raw file not found for {file} (trusting row sha)")
            expect_uid = uid_for(did, file_sha)
            pages_f = fill_page_gaps([{"page": int(r["page"]), "text": r["text"]} for r in recs])
            docs.append(
                {
                    "source_id": sid,
                    "uid": expect_uid,
                    "source_type": recs[0]["source_type"],
                    "relative_path": f"documents/{file}",
                    "document_sha256": file_sha,
                    "authority_score": AUTHORITY_BY_TYPE.get(recs[0]["source_type"], 0.72),
                    "pages": pages_f,
                    "is_add": True,
                    "meta": {
                        "language": recs[0].get("language"),
                        "topic": recs[0].get("topic"),
                        "publisher": recs[0].get("publisher"),
                        "publication_year": recs[0].get("year"),
                        "source_url": recs[0].get("url"),
                        "source_title": recs[0].get("title"),
                        "extraction_status": "text_extracted_unreviewed",
                        "claim_extraction_status": "not_run",
                        "entity_linking_status": "not_run",
                        "page_extraction_method": recs[0].get("extraction_method"),
                    },
                    "page_rows": [],
                }
            )
    docs.sort(key=lambda d: d["source_id"])
    print(f"documents: {n_base_uids} baseline -> {kept} kept, +{len(docs) - kept} adds = {len(docs)}")
    return docs, n_base_uids


# ---------------------------------------------------------------------------
# row emission
# ---------------------------------------------------------------------------

def emit_doc(doc, tokenizer, verify_module=None, verify_tokenizer=None, problems=None):
    pages = doc["pages"]
    parts = []
    psc = {}
    offset = 0
    for rec in pages:
        psc[rec["page"]] = offset
        parts.append(rec["text"])
        offset += len(rec["text"]) + 1
    doc_text = "\n".join(parts)
    page_sha = {rec["page"]: sha256_str(rec["text"]) for rec in pages}

    sentences = []
    for rec in pages:
        sentences.extend(build_doc_t(rec["text"], rec["page"], doc["source_type"]))
    for k, s in enumerate(sentences):
        s["idx"] = k
    set_sentence_tokens(sentences, tokenizer)

    chunks = snap_on_chunks(sentences, UNIT, TARGET, OVERLAP, tokenizer)

    if verify_module is not None:
        sim_sentences = verify_module.build_doc(
            [{"text": r["text"], "page": r["page"], "source_type": doc["source_type"]} for r in pages]
        )["sentences"]
        for k, s in enumerate(sim_sentences):
            s["idx"] = k
        # reuse the token counts computed above (same batch semantics)
        if len(sim_sentences) != len(sentences):
            problems.append(f"{doc['source_id']}: sentence count mismatch")
        else:
            for mine, theirs in zip(sentences, sim_sentences):
                theirs["tokens"] = mine["tokens"]
        sim_chunks = verify_module.snap_on_chunks(
            sim_sentences, "token", TARGET, OVERLAP, None, verify_tokenizer
        )
        if len(sim_chunks) != len(chunks):
            problems.append(
                f"{doc['source_id']}: chunk count mismatch mine={len(chunks)} sim={len(sim_chunks)}"
            )
        else:
            fields = ("units", "page_first", "page_last", "hard_split", "i0", "i1", "last_char", "text")
            for j, (m, s) in enumerate(zip(chunks, sim_chunks)):
                for f in fields:
                    if m[f] != s[f]:
                        problems.append(
                            f"{doc['source_id']} chunk {j}: field {f} mine={str(m[f])[:60]!r} sim={str(s[f])[:60]!r}"
                        )
                if bool(m.get("merged_tail")) != bool(s.get("merged_tail")):
                    problems.append(f"{doc['source_id']} chunk {j}: merged_tail mismatch")
                if len(m["sents"]) != len(s["sents"]):
                    problems.append(f"{doc['source_id']} chunk {j}: sents length mismatch")

    rows = []
    page_counter = {}
    for j, ch in enumerate(chunks):
        c0, cN = ch["content"][0], ch["content"][-1]
        a, b = c0["start_char"], cN["end_char"]
        d0 = psc[c0["page"]] + a
        d1 = psc[cN["page"]] + b
        text = doc_text[d0:d1]
        pf, pl = ch["page_first"], ch["page_last"]
        ph = page_sha[pf]
        page_id = sha256_str(f"{doc['uid']}\x00{pf}\x00{ph}")
        page_text = next(r["text"] for r in pages if r["page"] == pf)
        token_start = len(tokenizer.encode(page_text[:a], add_special_tokens=False).ids)
        token_count = ch["units"]
        chunk_id = f"chunk-rebuild-v2-{doc['uid'][:10]}-{j:04d}"
        csha = sha256_str(text)
        rel = doc["relative_path"]
        loc = f"{rel}#page={pf}"
        sid = doc["source_id"]
        pcindex = page_counter.get(pf, 0)
        page_counter[pf] = pcindex + 1
        row = {
            "authority_score": doc["authority_score"],
            "char_end": b,
            "char_start": a,
            "character_end": b,
            "character_start": a,
            "chunk_id": chunk_id,
            "chunk_index": j,
            "chunk_sha256": csha,
            "content_sha256": doc["document_sha256"],
            "document_hash": doc["document_sha256"],
            "document_sha256": doc["document_sha256"],
            "document_uid": doc["uid"],
            "end_char": b,
            "end_token": token_start + token_count,
            "locator": loc,
            "page": pf,
            "page_chunk_index": pcindex,
            "page_id": page_id,
            "page_number": pf,
            "page_sha256": ph,
            "page_text_sha256": ph,
            "relative_path": rel,
            "source_id": sid,
            "source_id_sha256": sha256_str(sid),
            "source_locator": loc,
            "source_sha256": sha256_str(sid),
            "source_text_key": "source-text-" + page_id[:20],
            "source_text_sha256": ph,
            "source_type": doc["source_type"],
            "span_end": d1,
            "span_start": d0,
            "start_char": a,
            "start_token": token_start,
            "status": "accepted",
            "text": text,
            "text_sha256": csha,
            "token_count": token_count,
            "token_end": token_start + token_count,
            "token_start": token_start,
            "page_first": pf,
            "page_last": pl,
            "doc_char_start": d0,
            "doc_char_end": d1,
            "hard_split": bool(ch["hard_split"]),
            "merged_tail": bool(ch.get("merged_tail")),
            "page_spanning": pf != pl,
        }
        if doc["is_add"]:
            row["retrieval_only_addition"] = True
            row["factual_index_allowed"] = True
            for k, v in doc["meta"].items():
                if v is not None:
                    row[k] = v
        rows.append(row)
    return rows, doc_text, pages, psc


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N docs")
    args = ap.parse_args()
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    problems: list[str] = []
    warnings: list[str] = []

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(BGE_TOKENIZER_JSON))

    verify_module = verify_tokenizer = None
    if not args.no_verify:
        spec = importlib.util.spec_from_file_location("simulate_rechunk", SIM_PATH)
        verify_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verify_module)
        verify_tokenizer = tokenizer

    docs, n_base = build_doc_table(problems, warnings)
    if args.limit:
        docs = docs[: args.limit]

    all_rows = []
    all_page_rows = []
    doc_stats = []
    for doc in docs:
        rows, doc_text, pages, psc = emit_doc(doc, tokenizer, verify_module, verify_tokenizer, problems)
        n_cross = sum(1 for r in rows if r["page_spanning"])
        n_hard = sum(1 for r in rows if r["hard_split"])
        n_merged = sum(1 for r in rows if r["merged_tail"])
        doc_stats.append(
            {
                "source_id": doc["source_id"],
                "document_uid": doc["uid"],
                "source_type": doc["source_type"],
                "file": Path(doc["relative_path"]).name,
                "pages": len(pages),
                "chunks": len(rows),
                "cross_page": n_cross,
                "hard_split": n_hard,
                "merged_tail": n_merged,
            }
        )
        all_rows.extend(rows)
        if doc["is_add"]:
            for rec in pages:
                all_page_rows.append(
                    {
                        "authority_score": doc["authority_score"],
                        "document_sha256": doc["document_sha256"],
                        "document_uid": doc["uid"],
                        "page": rec["page"],
                        "relative_path": doc["relative_path"],
                        "source_id": doc["source_id"],
                        "source_type": doc["source_type"],
                        "text": rec["text"],
                    }
                )
        else:
            for rec in doc["page_rows"]:
                all_page_rows.append(
                    {
                        "authority_score": rec["authority_score"],
                        "document_sha256": rec["document_sha256"],
                        "document_uid": rec["document_uid"],
                        "page": int(rec["page"]),
                        "relative_path": rec["relative_path"],
                        "source_id": rec["source_id"],
                        "source_type": rec["source_type"],
                        "text": rec["text"],
                    }
                )

    # --- internal invariants -------------------------------------------------
    for row in all_rows:
        if not (sha256_str(row["text"]) == row["chunk_sha256"] == row["text_sha256"]):
            problems.append(f"{row['chunk_id']}: chunk hash invariant violated")
        if row["span_end"] <= row["span_start"]:
            problems.append(f"{row['chunk_id']}: span ordering violated")
        if not isinstance(row["text"], str) or not row["text"]:
            problems.append(f"{row['chunk_id']}: empty text")
    for key in BASELINE_KEYS:
        if key not in all_rows[0]:
            problems.append(f"baseline key missing: {key}")

    # --- write -----------------------------------------------------------------
    chunks_path = out_dir / "chunks.json"
    pages_path = out_dir / "pages.json"
    ledger_path = out_dir / "evidence_ledger.json"
    manifest_path = out_dir / "manifest.json"

    chunks_path.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pages_path.write_text(
        json.dumps({"records": all_page_rows, "stage": "rebuild_v2"}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    h_chunks = sha256_bytes(chunks_path.read_bytes())
    h_pages = sha256_bytes(pages_path.read_bytes())
    stage_hash = sha256_str(
        json.dumps({"stage": "rebuild_v2", "chunks.json": h_chunks, "pages.json": h_pages}, sort_keys=True)
    )

    ledger_rows = [
        {
            "authority_score": r["authority_score"],
            "chunk_id": r["chunk_id"],
            "chunk_sha256": r["chunk_sha256"],
            "displayed_text": r["text"],
            "document_uid": r["document_uid"],
            "evidence_id": r["chunk_id"],
            "factual_index_allowed": True,
            "page": r["page"],
            "source_id": r["source_id"],
            "source_type": r["source_type"],
            "span_end": r["span_end"],
            "span_start": r["span_start"],
            "status": "accepted",
            "text": r["text"],
            "text_sha256": r["chunk_sha256"],
        }
        for r in all_rows
    ]
    ledger_path.write_text(
        json.dumps({"records": ledger_rows, "stage_hash": stage_hash}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    h_ledger = sha256_bytes(ledger_path.read_bytes())

    code_hash = sha256_bytes(Path(__file__).read_bytes())
    manifest = {
        "artifacts": {
            "chunks": {"path": "chunks.json", "sha256": h_chunks},
            "pages": {"path": "pages.json", "sha256": h_pages},
            "evidence_ledger": {"path": "evidence_ledger.json", "sha256": h_ledger},
        },
        "source_hashes": {
            "chunks.json": h_chunks,
            "pages.json": h_pages,
            "evidence_ledger.json": h_ledger,
        },
        "complete": True,
        "stage": "rebuild_v2",
        "stage_hash": stage_hash,
        "identity": {
            "payload": {
                "config": CONFIG_NAME,
                "target": TARGET,
                "overlap": OVERLAP,
                "min_units": MIN_UNITS,
                "max_units": MAX_UNITS,
                "space": "bge",
                "documents": len(docs),
                "chunks": len(all_rows),
            },
            "code_hash": code_hash,
            "corpus_identity": stage_hash,
        },
        "documents": doc_stats,
        "config": {
            "unit_space": "bge",
            "min_units": MIN_UNITS,
            "target": TARGET,
            "max_units": MAX_UNITS,
            "overlap": OVERLAP,
            "sentence_snap": "on",
            "chunker": "simulate_rechunk.py::snap_on_chunks (literal transcription)",
            "span_convention": "span_* doc-global; char_*/character_*/start_char/end_char page-local",
            "text_convention": "raw document-text slice [doc_char_start, doc_char_end)",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # --- summary -----------------------------------------------------------------
    units = [r["token_count"] for r in all_rows]
    band = sum(1 for u in units if MIN_UNITS <= u <= MAX_UNITS)
    loose = sum(1 for u in units if 460 <= u <= 960)
    endset = set('.!?\u0964"\'\u201d\u2019')
    mid = sum(1 for r in all_rows if r["text"].rstrip() and r["text"].rstrip()[-1] not in endset)
    by_type = Counter(r["source_type"] for r in all_rows)
    print(f"wrote {out_dir}")
    print(f"  chunks: {len(all_rows)}  docs: {len(docs)}  pages: {len(all_page_rows)}")
    print(f"  band 500-900: {100.0 * band / len(units):.2f}%   loose 460-960: {100.0 * loose / len(units):.2f}%")
    print(f"  median units: {statistics.median(units)}  mid-sentence: {100.0 * mid / len(all_rows):.2f}%")
    print(f"  cross-page: {sum(1 for r in all_rows if r['page_spanning'])}  hard-split: {sum(1 for r in all_rows if r['hard_split'])}  merged: {sum(1 for r in all_rows if r['merged_tail'])}")
    print(f"  source_types: {dict(sorted(by_type.items()))}")
    print(f"  stage_hash: {stage_hash}")
    print(f"  seconds: {time.time() - t0:.1f}")
    if warnings:
        print(f"warnings ({len(warnings)}):")
        for w in warnings[:20]:
            print("  -", w)
    if problems:
        print(f"PROBLEMS ({len(problems)}):")
        for p in problems[:40]:
            print("  -", p)
        return 1
    print("no problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
