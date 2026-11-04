"""Frozen production constants and generation-manifest loading for the eval driver.

Every value here is transcribed from the released production runtime
(``outputs/remote_vm_qwen35_mpkg_rag_runtime.py``, CONFIG at lines 4783-4785)
or read from a generation's ``index_manifest.json`` contract.  The driver must
fail loudly instead of guessing when a generation contradicts these anchors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

RUNTIME_SOURCE_SHA256 = "2fef5083c0cbb72c0f88a3055460ee6ffee43dc82be5441968243e7bc60f8c40"
RAG_CORE_SHA256 = "14271e34f998c2a5448094ba72f1f27dce4f9225c7e239fb9f5bc213d5f6051c"
EVAL_CORE_SHA256 = "f8055e11a959bcd414f3e0878b918e778176889822e0f87bdf3fece93e2c0cf3"

PRODUCTION_RETRIEVAL = {
    "dense_top_k": 12,
    "bm25_top_k": 12,
    "graph_top_k": 8,
    "max_hops": 2,
    "minimum_authority": 0.60,
    "minimum_dense_score": 0.25,
    "minimum_graph_score": 0.0,
    "minimum_rerank_probability": 0.55,
    "rerank_top_k": 5,
    "rrf_constant": 60.0,
}

PRODUCTION_RRF_WEIGHTS = {"dense": 1.1, "bm25": 1.0, "graph": 0.8}

PRODUCTION_PROMPT = {
    "max_seq_length": 4096,
    "perspective_max_new_tokens": 768,
    "plan_max_new_tokens": 768,
    "answer_max_new_tokens": 768,
    "thinking_enabled": True,
    "minimum_parse_rate": 0.98,
    "evidence_char_budget": 9000,
    "generator_model": "unsloth/Qwen3.5-4B",
}

RETRIEVAL_ANCHOR_KEYS = tuple(key for key in PRODUCTION_RETRIEVAL if key != "rrf_constant")

GENERATION_FILES = (
    "index_manifest.json",
    "embeddings.npy",
    "id_map.json",
    "graph.json",
    "evidence.sqlite3",
)

CORPUS_FILES = {
    "chunks": "chunks.json",
    "evidence_ledger": "evidence_ledger.json",
    "semantic_extractions": "semantic_extractions.json",
    "candidates": "candidates.json",
    "manifest": "manifest.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_generation_id(index_root: Path, generation_id: str | None) -> str:
    if generation_id:
        if Path(generation_id).name != generation_id:
            raise ValueError("generation_id must be a plain directory name")
        return generation_id
    pointer = Path(index_root) / "CURRENT"
    value = pointer.read_text(encoding="utf-8").strip()
    if not value or Path(value).name != value:
        raise RuntimeError(f"invalid CURRENT pointer under {index_root}")
    return value


def load_generation(index_root: Path, generation_id: str | None = None) -> dict[str, Any]:
    root = Path(index_root).expanduser().resolve()
    gid = resolve_generation_id(root, generation_id)
    generation_dir = root / "generations" / gid
    if not generation_dir.is_dir():
        raise FileNotFoundError(f"generation directory does not exist: {generation_dir}")
    missing = [name for name in GENERATION_FILES if not (generation_dir / name).exists()]
    if missing:
        raise FileNotFoundError("generation is missing files: " + ", ".join(missing))
    manifest = json.loads((generation_dir / "index_manifest.json").read_text(encoding="utf-8"))
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("generation manifest is missing an identity object")
    retrieval = identity.get("retrieval")
    if not isinstance(retrieval, Mapping):
        raise RuntimeError("generation identity is missing the retrieval contract")
    missing_retrieval = [key for key in RETRIEVAL_ANCHOR_KEYS if key not in retrieval]
    if missing_retrieval:
        raise RuntimeError(
            "generation retrieval contract missing keys: " + ", ".join(sorted(missing_retrieval))
        )
    return {
        "index_root": root,
        "generation_id": gid,
        "generation_dir": generation_dir,
        "manifest": manifest,
        "identity": dict(identity),
        "retrieval": dict(retrieval),
        "source_hashes": dict(identity.get("source_hashes") or {}),
        "row_counts": dict(manifest.get("row_counts") or {}),
    }


def retrieval_divergences(generation: Mapping[str, Any]) -> dict[str, Any]:
    retrieval = generation.get("retrieval")
    if not isinstance(retrieval, Mapping):
        raise TypeError("generation.retrieval must be a mapping")
    differing = {
        key: {"generation": retrieval.get(key), "production_runtime": expected}
        for key, expected in PRODUCTION_RETRIEVAL.items()
        if key != "rrf_constant" and retrieval.get(key) != expected
    }
    weights = retrieval.get("rrf_weights")
    if isinstance(weights, Mapping):
        weight_diffs = {
            key: {"generation": weights.get(key), "production_runtime": value}
            for key, value in PRODUCTION_RRF_WEIGHTS.items()
            if weights.get(key) != value
        }
    else:
        weight_diffs = {"rrf_weights": {"generation": weights, "production_runtime": PRODUCTION_RRF_WEIGHTS}}
    return {"retrieval": differing, "rrf_weights": weight_diffs}


def verify_corpus_binding(generation: Mapping[str, Any], corpus_dir: Path) -> dict[str, Any]:
    corpus_root = Path(corpus_dir).expanduser().resolve()
    expected = generation.get("source_hashes") or {}
    results: dict[str, Any] = {}
    for name, filename in CORPUS_FILES.items():
        expected_hash = expected.get(name)
        path = corpus_root / filename
        if expected_hash is None:
            results[name] = {"status": "not_in_identity"}
            continue
        if not path.exists():
            results[name] = {"status": "missing", "expected": expected_hash}
            continue
        actual = sha256_file(path)
        results[name] = {
            "status": "match" if actual == expected_hash else "mismatch",
            "expected": expected_hash,
            "actual": actual,
        }
    results["all_match"] = all(
        entry.get("status") in {"match", "not_in_identity"} for entry in results.values()
    )
    return results
