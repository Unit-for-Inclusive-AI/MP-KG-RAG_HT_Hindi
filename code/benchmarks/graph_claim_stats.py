#!/usr/bin/env python3
"""Table-6-style claim span statistics for a graph.json file.

Usage:
    python3 graph_claim_stats.py GRAPH_JSON [--json] [--no-locate]

Reports, for the Claim table: total claims, exact span matches
(chunk[span_start:span_end] == text), normalized matches (not exact but
normalization-equal), span mismatches, out-of-range / missing offsets, and
status distributions. Also reports quote-first locate feasibility using
work/quote_locate.py (exact -> normalized -> sentence/window), which measures
how many legacy claims the deterministic locator could recover.

Read-only. No model inference, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from quote_locate import locate_quote, normalize_with_index_map
except Exception:  # pragma: no cover - keep the script usable standalone

    def normalize_with_index_map(text: str):
        return text, []

    def locate_quote(chunk_text, quote, min_chars=40, min_words=5, hint=None, align_threshold=0.90):
        return None


def _rows(table):
    if isinstance(table, dict):
        return list(table.values())
    if isinstance(table, list):
        return table
    return []


def compute_stats(graph_path: str, run_locate: bool = True) -> dict:
    with open(graph_path, encoding="utf-8") as handle:
        graph = json.load(handle)

    tables = graph.get("tables") if isinstance(graph.get("tables"), dict) else graph
    claims = _rows(tables.get("Claim"))
    chunks = _rows(tables.get("EvidenceChunk"))

    chunk_text_by_id: dict[str, str] = {}
    for chunk in chunks:
        text = chunk.get("text")
        if not isinstance(text, str):
            continue
        for key in ("chunk_id", "evidence_chunk_id"):
            value = chunk.get(key)
            if isinstance(value, str) and value not in chunk_text_by_id:
                chunk_text_by_id[value] = text

    stats = {
        "graph_path": str(graph_path),
        "generation_id": graph.get("generation_id"),
        "schema_version": tables.get("schema_version") or graph.get("schema_version"),
        "claims": len(claims),
        "chunks": len(chunks),
        "chunk_lookup_missing": 0,
        "exact": 0,
        "normalized": 0,
        "mismatch": 0,
        "out_of_range": 0,
        "out_of_range_none_offsets": 0,
        "span_alignment": Counter(),
        "validation_status": Counter(),
        "review_state": Counter(),
        "graph_retrieval_eligible": Counter(),
        "status": Counter(),
        "span_method": Counter(),
        "locate": Counter() if run_locate else None,
    }

    for claim in claims:
        if not isinstance(claim, dict):
            continue
        text = claim.get("text")
        if not isinstance(text, str):
            text = ""
        chunk_id = claim.get("chunk_id") or claim.get("evidence_chunk_id")
        chunk_text = chunk_text_by_id.get(chunk_id) if isinstance(chunk_id, str) else None
        if chunk_text is None:
            stats["chunk_lookup_missing"] += 1

        start, end = claim.get("span_start"), claim.get("span_end")
        valid_span = (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and start >= 0
            and end > start
            and chunk_text is not None
            and end <= len(chunk_text)
        )
        if not valid_span:
            stats["out_of_range"] += 1
            if start is None or end is None:
                stats["out_of_range_none_offsets"] += 1
        else:
            slice_text = chunk_text[start:end]
            if slice_text == text:
                stats["exact"] += 1
            else:
                norm_slice, _ = normalize_with_index_map(slice_text)
                norm_text, _ = normalize_with_index_map(text)
                if norm_slice and norm_slice == norm_text:
                    stats["normalized"] += 1
                else:
                    stats["mismatch"] += 1

        for field in ("span_alignment", "validation_status", "review_state", "status", "span_method"):
            value = claim.get(field)
            if value is not None:
                stats[field][str(value)] += 1
        stats["graph_retrieval_eligible"][str(bool(claim.get("graph_retrieval_eligible")))] += 1

        if run_locate and chunk_text is not None and text:
            quote = text.strip()
            if len(quote) < 40 or len(quote.split()) < 5:
                stats["locate"]["below_min_gate"] += 1
            else:
                result = locate_quote(chunk_text, quote)
                stats["locate"][result["method"] if result else "not_locatable"] += 1

    for key in ("span_alignment", "validation_status", "review_state", "graph_retrieval_eligible", "status", "span_method", "locate"):
        value = stats.get(key)
        if isinstance(value, Counter):
            stats[key] = dict(sorted(value.items()))
    return stats


def format_text(stats: dict) -> str:
    total = stats["claims"] or 1
    lines = []
    lines.append("== claim span stats ==")
    lines.append(f"graph: {stats['graph_path']}")
    lines.append(f"generation_id: {stats['generation_id']}")
    lines.append(f"schema_version: {stats['schema_version']}")
    lines.append(f"claims: {stats['claims']}  chunks: {stats['chunks']}  chunk_lookup_missing: {stats['chunk_lookup_missing']}")
    lines.append(f"exact span matches (chunk[span]==text): {stats['exact']} ({100.0 * stats['exact'] / total:.2f}%)")
    lines.append(f"normalized matches (not exact): {stats['normalized']} ({100.0 * stats['normalized'] / total:.2f}%)")
    lines.append(f"span mismatches: {stats['mismatch']} ({100.0 * stats['mismatch'] / total:.2f}%)")
    lines.append(
        f"out-of-range offsets: {stats['out_of_range']} ({100.0 * stats['out_of_range'] / total:.2f}%)"
        f" [of which null offsets: {stats['out_of_range_none_offsets']}]"
    )
    for key in ("span_alignment", "validation_status", "review_state", "graph_retrieval_eligible", "status", "span_method"):
        if stats.get(key):
            lines.append(f"{key}: {stats[key]}")
    if stats.get("locate") is not None:
        locate = stats["locate"]
        locatable = sum(v for k, v in locate.items() if k in {"exact", "normalized", "sentence_window"})
        lines.append("== quote-first locate feasibility (quote_locate, >=40 chars / >=5 words) ==")
        lines.append(f"locatable total: {locatable} ({100.0 * locatable / total:.2f}%)")
        for key in ("exact", "normalized", "sentence_window", "below_min_gate", "not_locatable"):
            if key in locate:
                lines.append(f"  {key}: {locate[key]}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_json", help="path to graph.json")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--no-locate", action="store_true", help="skip the quote-first locate feasibility pass")
    args = parser.parse_args()

    stats = compute_stats(args.graph_json, run_locate=not args.no_locate)
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_text(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
