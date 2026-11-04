#!/usr/bin/env python3
"""Assemble the index-buildable corpus_v3 from new corpus + migrated + v3 extraction.

Outputs (corpus_v3/): chunks.json, candidates.json, semantic_extractions.json,
manifest.json, evidence_ledger.json.

Rules:
  - chunks.json = new corpus chunks (accepted)
  - candidates.json = queue candidates (all) + migrated candidates for chunks
    not in the queue (queue's migrated copies already included)
  - semantic_extractions.json = migrated claim entries for non-queue chunks
    + v3 claims entries (converted to index schema) for queue chunks
  - manifest.json / evidence_ledger.json copied byte-exact from the new corpus

Refuses to run while extraction is incomplete unless --allow-partial.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

PROJECT = Path("/Users/pika/Downloads/MP_KG_RAG_v2")
REBUILD = PROJECT / "source_refresh" / "rebuild_v2"
NEW = REBUILD / "semantic_graph_source_v2"
QUEUE = REBUILD / "queue_v3"
MIGRATED = REBUILD / "migrated_v2"
EXTRACT_BOTH = REBUILD / "extraction_v3" / "both" / "claims_v3.jsonl"
EXTRACT_CLAIMS = REBUILD / "extraction_v3" / "claims" / "claims_v3.jsonl"
DEFAULT_OUT = REBUILD / "corpus_v3"
AUDIT = Path(
    "/Users/pika/Downloads/Hate_Speech_Counter_Narrative_RAG_Project/.audit/"
    "rebuild_v2_corpus_v3_20261019"
)


def load_json(path: Path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    return data


def load_jsonl(path: Path):
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--report", type=Path, default=AUDIT / "assemble_report.json")
    args = parser.parse_args()
    started = time.time()

    new_chunks = load_json(NEW / "chunks.json")
    chunk_by_id = {str(c["chunk_id"]): c for c in new_chunks}
    accepted = [c for c in new_chunks if c.get("status") == "accepted"]
    print(f"new corpus chunks: {len(new_chunks)} (accepted {len(accepted)})")

    names = ("chunk_ids_both.json", "chunk_ids_claims.json")
    both = json.loads((QUEUE / names[0]).read_text(encoding="utf-8"))["chunk_ids"]
    claims_ids = json.loads((QUEUE / names[1]).read_text(encoding="utf-8"))["chunk_ids"]
    queue_ids = set(map(str, both)) | set(map(str, claims_ids))
    print(f"queue: {len(both)} both + {len(claims_ids)} claims")

    v3_rows: dict[str, dict] = {}
    for path, expected in ((EXTRACT_BOTH, both), (EXTRACT_CLAIMS, claims_ids)):
        rows = load_jsonl(path) if path.is_file() else []
        for row in rows:
            v3_rows[str(row["chunk_id"])] = row
        missing = [c for c in expected if str(c) not in v3_rows]
        dupes = len(rows) - len({str(r["chunk_id"]) for r in rows})
        print(f"  {path.parent.name}: {len(rows)} rows, missing {len(missing)}, dupes {dupes}")
        if (missing or dupes) and not args.allow_partial:
            print(f"extraction incomplete for {path} ({len(missing)} missing, {dupes} dupes)")
            return 1

    queue_candidates = load_json(QUEUE / "queue_candidates.json")
    migrated_candidates = load_json(MIGRATED / "candidates_v2.json")

    def candidate_key(row: dict) -> tuple:
        return (
            str(row["chunk_id"]),
            str(row["candidate_id"]),
            int(row["start"]),
            int(row["end"]),
        )

    problems: list[dict] = []
    candidates_out: list[dict] = []
    identical_dropped = 0
    seen_candidate = set()
    for row in queue_candidates:
        key = candidate_key(row)
        if key in seen_candidate:
            identical_dropped += 1
            continue
        seen_candidate.add(key)
        candidates_out.append(row)
    migrated_kept = 0
    for row in migrated_candidates:
        chunk_id = str(row["chunk_id"])
        if chunk_id in queue_ids:
            continue
        key = candidate_key(row)
        if key in seen_candidate:
            identical_dropped += 1
            continue
        seen_candidate.add(key)
        candidates_out.append(row)
        migrated_kept += 1
    print(f"candidates: {len(queue_candidates)} queue + {migrated_kept} non-queue migrated = {len(candidates_out)} ({identical_dropped} identical dropped)")

    def slice_state(chunk: dict, row: dict) -> str:
        text = str(chunk.get("text") or "")
        start, end = int(row["start"]), int(row["end"])
        if 0 <= start < end <= len(text):
            surface = str(row.get("text") or row.get("normalized_text") or "")
            if text[start:end] == surface:
                return "exact"
            if surface and text[start:end].strip() == surface.strip():
                return "trimmed"
        return "invalid"

    cand_states = Counter()
    for row in candidates_out:
        chunk = chunk_by_id.get(str(row["chunk_id"]))
        if chunk is None:
            problems.append({"kind": "candidate_chunk_missing", "chunk_id": str(row["chunk_id"])})
            continue
        cand_states[slice_state(chunk, row)] += 1
    print(f"candidate slice states: {dict(cand_states)}")
    if cand_states.get("invalid"):
        problems.append({"kind": "invalid_candidate_slices", "count": cand_states["invalid"]})

    migrated_extractions = load_json(MIGRATED / "extractions_v2.json")
    extractions_out: list[dict] = []
    kept_migrated = 0
    for entry in migrated_extractions:
        chunk_id = str(entry["chunk_id"])
        if chunk_id in queue_ids:
            continue
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None:
            problems.append({"kind": "migrated_entry_chunk_missing", "chunk_id": chunk_id})
            continue
        if str(entry.get("chunk_sha256")) != str(chunk.get("chunk_sha256")):
            problems.append({"kind": "migrated_entry_sha_mismatch", "chunk_id": chunk_id})
            continue
        extractions_out.append(entry)
        kept_migrated += 1

    span_states = Counter()
    v3_entries = 0
    v3_claims_total = 0
    for chunk_id in queue_ids:
        row = v3_rows.get(chunk_id)
        if row is None:
            continue
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None:
            problems.append({"kind": "v3_chunk_missing", "chunk_id": chunk_id})
            continue
        text = str(chunk.get("text") or "")
        claims = []
        for claim in row.get("claims") or []:
            quote = str(claim.get("quote") or "")
            start, end = claim.get("span_start"), claim.get("span_end")
            if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
                piece = text[start:end]
                if piece == quote:
                    span_states["exact"] += 1
                elif piece.strip() == quote.strip():
                    span_states["trimmed"] += 1
                else:
                    span_states["mismatch"] += 1
            else:
                span_states["invalid"] += 1
            claims.append({
                "claim": quote,
                "span_start": start,
                "span_end": end,
                "span_method": claim.get("span_method"),
                "candidate_ids": list(claim.get("candidate_ids_valid") or []),
                "graph_retrieval_eligible": bool(claim.get("graph_retrieval_eligible")),
                "source": "extraction_v3",
            })
        v3_claims_total += len(claims)
        extractions_out.append({
            "chunk_id": chunk_id,
            "chunk_sha256": chunk.get("chunk_sha256"),
            "document_uid": chunk.get("document_uid"),
            "page": chunk.get("page"),
            "request_id": row.get("request_id"),
            "source_id": chunk.get("source_id"),
            "claims": claims,
        })
        v3_entries += 1
    print(f"extractions: {kept_migrated} migrated + {v3_entries} v3 entries = {len(extractions_out)}")
    print(f"v3 claim span states: {dict(span_states)}")

    out = args.out
    if out.exists() and any(out.iterdir()):
        problems.append({"kind": "out_dir_not_empty", "out": str(out)})
    if problems:
        print(f"problems ({len(problems)}); not writing:")
        for item in problems[:10]:
            print("  ", item)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    (out / "chunks.json").write_text(
        json.dumps(accepted, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "candidates.json").write_text(
        json.dumps(candidates_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "semantic_extractions.json").write_text(
        json.dumps(extractions_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copy2(NEW / "manifest.json", out / "manifest.json")
    shutil.copy2(NEW / "evidence_ledger.json", out / "evidence_ledger.json")

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        return digest.hexdigest()

    report = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - started, 2),
        "out_dir": str(out),
        "counts": {
            "chunks": len(accepted),
            "candidates": len(candidates_out),
            "candidates_identical_dropped": identical_dropped,
            "migrated_extraction_entries": kept_migrated,
            "v3_extraction_entries": v3_entries,
            "total_claims_v3": v3_claims_total,
        },
        "candidate_slice_states": dict(cand_states),
        "v3_claim_span_states": dict(span_states),
        "sha256": {
            name: sha256_file(out / name)
            for name in ("chunks.json", "candidates.json", "semantic_extractions.json", "manifest.json", "evidence_ledger.json")
        },
    }
    report_json = json.dumps(report, indent=2) + "\n"
    (out / "assemble_report.json").write_text(report_json, encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_json, encoding="utf-8")
    print(f"wrote {out} ({report['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
