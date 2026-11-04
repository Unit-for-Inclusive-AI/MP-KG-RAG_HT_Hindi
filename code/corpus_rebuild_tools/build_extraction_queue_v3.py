#!/usr/bin/env python3
"""Build the v3 extraction queue for the rebuild (semantic_graph_source_v2).

Groups
  A  new-document chunks          (docs with no baseline counterpart)  mode both
  B  failed-claim chunks          (relocation_failures.json queue keys) mode claims
  C  needs-reextraction chunks    (retained=false repaired rows)        mode claims
  D  failed-candidate chunks      (span_split rows that never migrated) mode claims

Old chunk ids map to new chunks by page-local offset arithmetic
(new position = new page start + old page-local offset); every mapped region
is asserted byte-equal to the old chunk text.

Candidates supplied per chunk:
  * A: deterministic_entity_candidates over the new chunk text (reviewed orgs
       reconstructed from the baseline candidates file + the 7 controlled
       terms from config/production.json)
  * B/C: migrated candidate rows (candidates_v2.json) of the mapped chunks
  * D:   migrated rows unioned with a deterministic regeneration; a migrated
         row wins on identical (start, end, normalized_text, type)

Outputs (--out):
  chunk_ids_both.json    {"chunk_ids": [...]}  new-corpus order
  chunk_ids_claims.json  {"chunk_ids": [...]}  new-corpus order
  queue_candidates.json  candidate rows in migrated_v2/candidates_v2.json schema
  queue_modes.json       {chunk_id: {"mode": ..., "groups": [...]}}
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path("/Users/pika/Downloads/MP_KG_RAG_v2")
REBUILD = PROJECT / "source_refresh" / "rebuild_v2"
BASELINE = PROJECT / "semantic_graph_source"
NEW = REBUILD / "semantic_graph_source_v2"
MIGRATED = REBUILD / "migrated_v2"
REPAIRED = Path(
    "/Users/pika/Downloads/Hate_Speech_Counter_Narrative_RAG_Project/.audit/"
    "claim_repair_20261019/repaired_claims.jsonl"
)
CONFIG = PROJECT / "config" / "production.json"
MPC_PATH = PROJECT / ".worktrees/vllm-production-kb25/work/mpkg_production_core.py"
DEFAULT_OUT = REBUILD / "queue_v3"

CANDIDATE_ROW_KEYS = (
    "candidate_id", "candidate_type", "chunk_id", "end", "entity_type",
    "label", "normalized_text", "start", "text", "type",
)

# LGBTQ acronym-family surfaces missing from the 7-term controlled list; the
# governed matcher requires non-alnum boundaries, so bare "LGBTQ" never matches
# "LGBTQIA+". Same mechanism (literal + boundary check), config-level addition.
SUPPLEMENTAL_CONTROLLED_TERMS = (
    "LGBT", "LGBTI", "LGBTQ+", "LGBTQI", "LGBTQI+", "LGBTQIA", "LGBTQIA+",
)


def longest_match_terms(rows: list[dict]) -> list[dict]:
    """Drop controlled-term rows contained in a longer term row (same start)."""
    terms = sorted(
        (r for r in rows if r["type"] == "controlled_term"),
        key=lambda r: (r["start"], -r["end"]),
    )
    kept: list[dict] = []
    last_end = -1
    for row in terms:
        if row["start"] >= last_end:
            kept.append(row)
            last_end = row["end"]
    kept_ids = {id(r) for r in kept}
    return [r for r in rows if r["type"] != "controlled_term" or id(r) in kept_ids]


def load_json(path: Path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    return data


def load_jsonl(path: Path):
    # split on '\n' only: some claim texts contain exotic Unicode separators
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def layout(recs, sep="\n"):
    recs = sorted(recs, key=lambda r: r.get("page", 0))
    parts, ranges, offset = [], {}, 0
    for rec in recs:
        t = str(rec.get("text") or "")
        parts.append(t)
        ranges[rec.get("page")] = (offset, offset + len(t))
        offset += len(t) + len(sep)
    return sep.join(parts), ranges


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_deterministic_fn():
    spec = importlib.util.spec_from_file_location("mpkg_production_core_qb", MPC_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.deterministic_entity_candidates


def reviewed_orgs_from_baseline(old_candidates: list[dict]) -> list[str]:
    groups: dict[str, Counter] = defaultdict(Counter)
    for row in old_candidates:
        if str(row.get("type") or row.get("candidate_type")) != "organization":
            continue
        groups[str(row.get("normalized_text"))][str(row.get("text"))] += 1
    return sorted(counter.most_common(1)[0][0] for counter in groups.values())


def candidate_key(row: dict) -> tuple:
    return (int(row["start"]), int(row["end"]), str(row["normalized_text"]), str(row["type"]))


def slim_candidate(row: dict) -> dict:
    return {key: row[key] for key in CANDIDATE_ROW_KEYS if key != "chunk_id"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    started = time.time()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    new_chunks = load_json(NEW / "chunks.json")
    new_pages = load_json(NEW / "pages.json")
    old_chunks = load_json(BASELINE / "chunks.json")
    old_candidates = load_json(BASELINE / "candidates.json")
    migrated_candidates = load_json(MIGRATED / "candidates_v2.json")
    relocation_failures = json.loads((MIGRATED / "relocation_failures.json").read_text(encoding="utf-8"))
    migration_stats = json.loads((MIGRATED / "migration_stats.json").read_text(encoding="utf-8"))
    repaired = load_jsonl(REPAIRED)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    controlled_terms = list(dict.fromkeys(
        [*cfg["semantic"]["controlled_terms"], *SUPPLEMENTAL_CONTROLLED_TERMS]
    ))
    drop_uids = {str(v) for v in migration_stats["config"]["drop_document_uids"]}
    drop_sids = {str(v) for v in migration_stats["config"]["drop_source_ids"]}
    deterministic_entity_candidates = load_deterministic_fn()

    def dropped(uid, sid) -> bool:
        return str(uid) in drop_uids or str(sid) in drop_sids

    old_by_id = {str(c["chunk_id"]): c for c in old_chunks}
    new_by_id = {str(c["chunk_id"]): c for c in new_chunks}

    new_layout: dict[str, tuple] = {}
    pages_by_doc: dict[str, list] = defaultdict(list)
    for rec in new_pages:
        pages_by_doc[str(rec.get("document_uid"))].append(rec)
    for uid, recs in pages_by_doc.items():
        new_layout[uid] = layout(recs)

    chunks_by_doc: dict[str, list] = defaultdict(list)
    for chunk in new_chunks:
        chunks_by_doc[str(chunk["document_uid"])].append(chunk)
    for rows in chunks_by_doc.values():
        rows.sort(key=lambda row: int(row.get("chunk_index", 0)))

    problems: list[dict] = []
    mapping_stats = Counter()

    def map_old_chunk(old: dict) -> list[dict]:
        uid = str(old["document_uid"])
        text, ranges = new_layout[uid]
        page = old["page"]
        page_start = None
        for number, (s, e) in ranges.items():
            if number == page or str(number) == str(page):
                page_start = s
                break
        if page_start is None:
            problems.append({"kind": "page_missing", "chunk_id": old["chunk_id"], "page": page})
            return []
        start = page_start + int(old["char_start"])
        end = page_start + int(old["char_end"])
        if text[start:end] != str(old["text"]):
            problems.append({"kind": "region_text_mismatch", "chunk_id": old["chunk_id"]})
            return []
        hits = [
            c for c in chunks_by_doc[uid]
            if int(c["doc_char_start"]) < end and int(c["doc_char_end"]) > start
        ]
        if not hits:
            problems.append({"kind": "no_new_overlap", "chunk_id": old["chunk_id"]})
            return []
        mapping_stats["old_chunks_mapped"] += 1
        mapping_stats["new_chunks_touched"] += len(hits)
        return hits

    # ---------------------------------------------------------------- groups
    baseline_uids = {str(c["document_uid"]) for c in old_chunks}
    new_doc_uids = {str(c["document_uid"]) for c in new_chunks} - baseline_uids
    group_a = [str(c["chunk_id"]) for c in new_chunks if str(c["document_uid"]) in new_doc_uids]

    group_b_old = sorted(relocation_failures)
    group_b_claims = sum(len(entry.get("claims_failed", [])) for entry in relocation_failures.values())

    group_c_old_all = sorted({str(r["chunk_id"]) for r in repaired if not r.get("retained", True)})
    group_c_old, group_c_dropped = [], []
    for chunk_id in group_c_old_all:
        old = old_by_id.get(chunk_id)
        if old is None:
            problems.append({"kind": "repaired_chunk_not_in_baseline", "chunk_id": chunk_id})
            continue
        if dropped(old["document_uid"], old.get("source_id")):
            group_c_dropped.append(chunk_id)
        else:
            group_c_old.append(chunk_id)

    group_d_old = sorted(
        chunk_id for chunk_id, entry in relocation_failures.items()
        if entry.get("candidates_failed")
    )
    failed_candidate_rows = sum(
        len(entry.get("candidates_failed") or ()) for entry in relocation_failures.values()
    )

    # -------------------------------------------------- old -> new mapping
    queue: dict[str, dict] = {}

    def add_new_chunk(chunk_id: str, mode: str, group: str) -> None:
        entry = queue.setdefault(chunk_id, {"mode": "claims", "groups": set()})
        if mode == "both":
            entry["mode"] = "both"
        entry["groups"].add(group)

    for chunk_id in group_a:
        add_new_chunk(chunk_id, "both", "A")

    mapped_stats = Counter()
    for group, old_ids in (("B", group_b_old), ("C", group_c_old), ("D", group_d_old)):
        for chunk_id in old_ids:
            old = old_by_id.get(str(chunk_id))
            if old is None:
                problems.append({"kind": f"group_{group}_chunk_not_in_baseline", "chunk_id": str(chunk_id)})
                continue
            if dropped(old["document_uid"], old.get("source_id")):
                problems.append({"kind": f"group_{group}_dropped_doc", "chunk_id": str(chunk_id)})
                continue
            hits = map_old_chunk(old)
            mapped_stats[f"group_{group}_old"] += 1
            if hits:
                mapped_stats[f"group_{group}_new"] += len(hits)
            for hit in hits:
                add_new_chunk(str(hit["chunk_id"]), "claims", group)

    # ---------------------------------------------------- candidate supply
    migrated_by_chunk: dict[str, list[dict]] = defaultdict(list)
    for row in migrated_candidates:
        migrated_by_chunk[str(row["chunk_id"])].append(slim_candidate(row))
    for rows in migrated_by_chunk.values():
        rows.sort(key=lambda row: candidate_key(row))

    chunk_order = {str(c["chunk_id"]): idx for idx, c in enumerate(new_chunks)}
    queue_candidates: list[dict] = []
    supply_stats = Counter()
    deterministic_calls = 0
    reviewed_orgs = reviewed_orgs_from_baseline(old_candidates)

    for chunk_id in sorted(queue, key=lambda cid: chunk_order.get(cid, 1 << 60)):
        groups = queue[chunk_id]["groups"]
        rows: list[dict] = []
        if groups & {"B", "C", "D"}:
            rows.extend(migrated_by_chunk.get(chunk_id, []))
            supply_stats["migrated_rows"] += len(rows)
        if "A" in groups or "D" in groups:
            text = str(new_by_id[chunk_id]["text"])
            generated = longest_match_terms(deterministic_entity_candidates(
                text, reviewed_orgs, controlled_terms, limit=48
            ))
            deterministic_calls += 1
            seen = {candidate_key(row) for row in rows}
            added = 0
            for row in generated:
                if candidate_key(row) in seen:
                    continue
                seen.add(candidate_key(row))
                rows.append(row)
                added += 1
            supply_stats["deterministic_rows_added"] += added
            supply_stats["deterministic_rows_generated"] += len(generated)
        rows.sort(key=candidate_key)
        for row in rows:
            queue_candidates.append({"chunk_id": chunk_id, **slim_candidate(row)})

    both_ids = [cid for cid in sorted(queue, key=lambda c: chunk_order.get(c, 1 << 60)) if queue[cid]["mode"] == "both"]
    claims_ids = [cid for cid in sorted(queue, key=lambda c: chunk_order.get(c, 1 << 60)) if queue[cid]["mode"] != "both"]
    supply_stats["queue_chunks_total"] = len(queue)
    supply_stats["queue_chunks_both"] = len(both_ids)
    supply_stats["queue_chunks_claims"] = len(claims_ids)
    supply_stats["candidates_total_rows"] = len(queue_candidates)

    (out_dir / "chunk_ids_both.json").write_text(
        json.dumps({"chunk_ids": both_ids}, indent=2) + "\n", encoding="utf-8")
    (out_dir / "chunk_ids_claims.json").write_text(
        json.dumps({"chunk_ids": claims_ids}, indent=2) + "\n", encoding="utf-8")
    (out_dir / "queue_candidates.json").write_text(
        json.dumps(queue_candidates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "queue_modes.json").write_text(
        json.dumps({cid: {"mode": queue[cid]["mode"], "groups": sorted(queue[cid]["groups"])}
                    for cid in sorted(queue, key=lambda c: chunk_order.get(c, 1 << 60))},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    seconds = round(time.time() - started, 2)
    report = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": seconds,
        "out_dir": str(out_dir),
        "inputs": {
            "new_corpus": str(NEW), "baseline": str(BASELINE), "migrated": str(MIGRATED),
            "repaired": str(REPAIRED), "config": str(CONFIG),
            "new_chunks_sha256": sha256_file(NEW / "chunks.json"),
            "baseline_chunks_sha256": sha256_file(BASELINE / "chunks.json"),
            "migrated_candidates_sha256": sha256_file(MIGRATED / "candidates_v2.json"),
            "relocation_failures_sha256": sha256_file(MIGRATED / "relocation_failures.json"),
        },
        "reviewed_orgs": reviewed_orgs,
        "controlled_terms": controlled_terms,
        "controlled_terms_supplemental": list(SUPPLEMENTAL_CONTROLLED_TERMS),
        "groups": {
            "A_new_doc_chunks": len(group_a),
            "A_new_docs": len(new_doc_uids),
            "B_old_chunks": len(group_b_old), "B_failed_claims": group_b_claims,
            "C_old_chunks_all": len(group_c_old_all), "C_old_chunks_used": len(group_c_old),
            "C_old_chunks_dropped_docs": len(group_c_dropped),
            "D_old_chunks": len(group_d_old), "D_failed_candidate_rows": failed_candidate_rows,
        },
        "mapping": dict(mapping_stats) | {"deterministic_calls": deterministic_calls},
        "mapped_by_group": dict(mapped_stats),
        "queue": {
            "chunks_total": len(queue),
            "chunks_both": len(both_ids),
            "chunks_claims": len(claims_ids),
            "candidates_total_rows": len(queue_candidates),
            "supply": dict(supply_stats),
        },
        "problems": problems,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"queue: {len(queue)} chunks (both {len(both_ids)}, claims {len(claims_ids)}); "
        f"candidates {len(queue_candidates)} rows; problems {len(problems)}"
    )
    print(f"outputs -> {out_dir}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
