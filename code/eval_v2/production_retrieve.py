"""Production mirror of the runtime's ``retrieve_evidence`` path.

The function below is a faithful transcription of
``outputs/remote_vm_qwen35_mpkg_rag_runtime.py`` lines 5493-5499 (gate,
production query strings, RRF weights 1.1/1.0/0.8, constant 60.0, payload
merge, full-candidate reranking, and selection at 0.55 with
``max_evidence = rerank_top_k``), operating on a :class:`ProductionIndex`.
"""

from __future__ import annotations

from typing import Any, Mapping

import mpkg_rag_core as core

from .config_v2 import PRODUCTION_RETRIEVAL, PRODUCTION_RRF_WEIGHTS
from .production_index import ProductionIndex

SIGNATURE_KEYS = ("entity_ids", "predicates", "polarities", "modalities", "desired_stances")


def coerce_signature(raw: Mapping[str, Any] | None) -> dict[str, list[str]]:
    signature: dict[str, list[str]] = {}
    source = dict(raw or {})
    for key in SIGNATURE_KEYS:
        value = source.get(key)
        if value is None:
            signature[key] = []
        elif isinstance(value, (list, tuple)):
            signature[key] = [str(item) for item in value]
        else:
            signature[key] = [str(value)]
    return signature


def target_anchor_signature(index: ProductionIndex, target: str) -> dict[str, Any]:
    anchor = index.resolve_target_anchor_entities(target)
    signature = coerce_signature({"entity_ids": anchor["entity_ids"]})
    signature["target_anchor"] = {
        "match_mode": anchor["match_mode"],
        "canonical_names": anchor["canonical_names"],
    }
    return signature


def production_retrieve(
    index: ProductionIndex,
    post: str,
    target: str,
    category: str,
    query_signature: Mapping[str, Any],
    graph_enabled: bool = True,
) -> dict[str, Any]:
    signature = dict(query_signature or {})
    if not signature.get("entity_ids"):
        return {
            "evidence": [],
            "candidate_evidence_ids": [],
            "candidates": [],
            "selection": None,
            "graph_enabled": graph_enabled,
            "graph_hit_count": 0,
            "retrieval_trace": {"dense_hits": 0, "bm25_hits": 0, "graph_hits": 0},
            "selection_threshold": float(index.retrieval["minimum_rerank_probability"]),
            "abstention_reason": "no_linked_entities",
        }
    dense_hits = index.dense_search(
        f"evidence addressing harmful or misleading claims about {target}",
        int(index.retrieval["dense_top_k"]),
    )
    bm25_hits = index.bm25_search(f"{target} {post}", int(index.retrieval["bm25_top_k"]))
    graph_hits = index.graph_search(
        target,
        int(index.retrieval["graph_top_k"]),
        post,
        category,
        signature,
        graph_enabled,
    )
    fused = core.reciprocal_rank_fusion(
        {"dense": dense_hits, "bm25": bm25_hits, "graph": graph_hits},
        dict(PRODUCTION_RRF_WEIGHTS),
        float(PRODUCTION_RETRIEVAL["rrf_constant"]),
    )
    for item in fused:
        item.update(index.evidence_by_id.get(item["evidence_chunk_id"], {}))
    trace = {
        "dense_hits": len(dense_hits),
        "bm25_hits": len(bm25_hits),
        "graph_hits": len(graph_hits),
    }
    if not fused:
        return {
            "evidence": [],
            "candidate_evidence_ids": [],
            "candidates": [],
            "selection": None,
            "graph_enabled": graph_enabled,
            "graph_hit_count": len(graph_hits),
            "retrieval_trace": trace,
            "selection_threshold": float(index.retrieval["minimum_rerank_probability"]),
            "abstention_reason": "no_positive_candidates",
        }
    logits = index.rerank(post, [str(row.get("text", "")) for row in fused])
    for row, logit in zip(fused, logits):
        row["rerank_logit"] = logit
        row["rerank_probability"] = core._stable_sigmoid(logit)
    selection = core.select_evidence(
        fused,
        {row["evidence_chunk_id"]: logit for row, logit in zip(fused, logits)},
        {
            "minimum_rerank_probability": index.retrieval["minimum_rerank_probability"],
            "max_evidence": int(index.retrieval["rerank_top_k"]),
        },
    )
    return {
        "evidence": selection["selected"],
        "candidate_evidence_ids": [row["evidence_chunk_id"] for row in fused],
        "candidates": fused,
        "selection": selection,
        "graph_enabled": graph_enabled,
        "graph_hit_count": len(graph_hits),
        "retrieval_trace": trace,
        "selection_threshold": float(index.retrieval["minimum_rerank_probability"]),
        "abstention_reason": selection["reason"],
    }
