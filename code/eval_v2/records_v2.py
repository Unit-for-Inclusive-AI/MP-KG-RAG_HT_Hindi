"""Audit-record assembly for the MP-arm v2 driver.

Records keep score_corrected.py-compatible top-level keys (ID/Text/output/
error/reference) alongside the full retrieval, ledger, and prompt audit trail.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

CANDIDATE_KEYS = (
    "evidence_chunk_id",
    "chunk_id",
    "document_uid",
    "source_id",
    "page",
    "span_start",
    "span_end",
    "authority_score",
    "status",
    "rrf_score",
    "graph_paths",
    "relations",
    "stances",
    "rerank_logit",
    "rerank_probability",
)


def compact_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    compact = {key: row.get(key) for key in CANDIDATE_KEYS}
    compact["branch_traces"] = [
        {
            "branch": trace.get("branch"),
            "rank": trace.get("rank"),
            "raw_score": trace.get("raw_score"),
            "weighted_contribution": trace.get("weighted_contribution"),
        }
        for trace in row.get("branch_traces", [])
    ]
    compact["text_preview"] = str(row.get("text", ""))[:200]
    return compact


def compact_fit(fit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": fit.get("status"),
        "prompt_tokens": fit.get("prompt_tokens"),
        "available_input_tokens": fit.get("available_input_tokens"),
        "payload_index": fit.get("payload_index"),
        "selected_evidence_ids": list(fit.get("selected_evidence_ids", [])),
        "dropped_evidence_ids": list(fit.get("dropped_evidence_ids", [])),
        "attempt_count": len(fit.get("attempts", [])),
        "reason": fit.get("reason"),
    }


def _ledger_public(row: Mapping[str, Any]) -> dict[str, Any]:
    public = dict(row)
    public.pop("source_text", None)
    return public


def build_record(
    source_record: Mapping[str, Any],
    *,
    generation_id: str,
    signature: Mapping[str, Any],
    signature_source: str,
    retrieval: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
    perspectives: Sequence[Mapping[str, Any]],
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    selected_ids = [row["evidence_id"] for row in ledger]
    abstention = retrieval.get("abstention_reason")
    if abstention == "no_linked_entities":
        generation_status = "skipped_gate_abstention"
    elif not ledger:
        generation_status = "skipped_no_selected_evidence"
    else:
        generation_status = "pending_generation"
    record: dict[str, Any] = {
        "ID": source_record.get("ID"),
        "Category": source_record.get("Category"),
        "Target": source_record.get("Target"),
        "Text": source_record.get("Text"),
        "reference": source_record.get("Counter Narrative"),
        "variant": "mp_kg_rag",
        "model": "qwen35_mp_kg_rag_v2",
        "generation_id": generation_id,
        "signature_source": signature_source,
        "query_signature": {k: v for k, v in signature.items() if k != "target_anchor"},
        "target_anchor": signature.get("target_anchor"),
        "abstained": bool(abstention),
        "abstention_reason": abstention,
        "graph_enabled": retrieval.get("graph_enabled"),
        "graph_hit_count": retrieval.get("graph_hit_count"),
        "retrieval_trace": retrieval.get("retrieval_trace"),
        "candidate_evidence_ids": retrieval.get("candidate_evidence_ids", []),
        "candidates": [compact_candidate(row) for row in retrieval.get("candidates", [])],
        "selected_evidence_ids": selected_ids,
        "selection": {
            "minimum_rerank_probability": retrieval.get("selection_threshold"),
            "selected_count": len(selected_ids),
            "rejected_count": len(retrieval.get("candidates", [])) - len(selected_ids),
        },
        "evidence": [
            {**{k: v for k, v in row.items() if k != "source_text"}} for row in ledger
        ],
        "ledger": [_ledger_public(row) for row in ledger],
        "prompts": {
            "perspectives": [
                {
                    "perspective": entry.get("perspective"),
                    "goal": entry.get("goal"),
                    "messages": entry.get("messages"),
                    "rendered": entry.get("rendered"),
                    "fit": compact_fit(entry.get("fit", {})),
                    "rendered_sha256": hashlib.sha256(
                        str(entry.get("rendered", "")).encode("utf-8")
                    ).hexdigest(),
                }
                for entry in perspectives
            ]
        },
        "plan": None,
        "output": None,
        "error": None,
        "generation_status": generation_status,
        "elapsed": dict(timing),
    }
    for extra in ("query_en", "seed", "revision"):
        if extra in source_record:
            record[extra] = source_record[extra]
    return record
