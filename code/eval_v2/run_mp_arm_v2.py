#!/usr/bin/env python3
"""Driver: re-run the production MP-arm pipeline over Test IDs against a
given index generation, without ever calling the demo's build/validate path.

Modes:
  (default) --no-generate: deterministic gate -> dense/bm25/graph retrieval ->
  RRF -> rerank -> select -> ledger -> 5 perspective prompt assemblies.
  The generation stage is intentionally out of scope here (MLX runtime is not
  invoked); records carry `generation_status: pending_generation` plus the
  fully assembled prompts for a later generation pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "work"))
sys.path.insert(0, str(REPO_ROOT / ".worktrees" / "local-mpkg-gradio"))

DEFAULT_INDEX_ROOT = REPO_ROOT / "local_mpkg_demo_index"
DEFAULT_IDS_FILE = REPO_ROOT / "outputs" / "2026-10-16-local-mlx-test155" / "test_homophobic.jsonl"
DEFAULT_CORPUS_DIR = REPO_ROOT / "semantic_graph_source"

CODE_HASH_TARGETS = {
    "eval_v2/config_v2.py": REPO_ROOT / "eval_v2" / "config_v2.py",
    "eval_v2/bm25_vendored.py": REPO_ROOT / "eval_v2" / "bm25_vendored.py",
    "eval_v2/production_index.py": REPO_ROOT / "eval_v2" / "production_index.py",
    "eval_v2/production_retrieve.py": REPO_ROOT / "eval_v2" / "production_retrieve.py",
    "eval_v2/production_prompts.py": REPO_ROOT / "eval_v2" / "production_prompts.py",
    "eval_v2/records_v2.py": REPO_ROOT / "eval_v2" / "records_v2.py",
    "work/mpkg_rag_core.py": REPO_ROOT / "work" / "mpkg_rag_core.py",
    "outputs/remote_vm_qwen35_mpkg_rag_runtime.py": REPO_ROOT
    / "outputs"
    / "remote_vm_qwen35_mpkg_rag_runtime.py",
    "local_mpkg_demo/backends.py": REPO_ROOT
    / ".worktrees"
    / "local-mpkg-gradio"
    / "local_mpkg_demo"
    / "backends.py",
    "local_mpkg_demo/config.py": REPO_ROOT
    / ".worktrees"
    / "local-mpkg-gradio"
    / "local_mpkg_demo"
    / "config.py",
    "local_mpkg_demo/index_store.py": REPO_ROOT
    / ".worktrees"
    / "local-mpkg-gradio"
    / "local_mpkg_demo"
    / "index_store.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--generation-id", default=None)
    parser.add_argument("--ids-file", type=Path, default=DEFAULT_IDS_FILE)
    parser.add_argument("--ids", default=None, help="comma-separated ID filter")
    parser.add_argument("--max-ids", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-generate", action="store_true", default=True)
    parser.add_argument("--generate", action="store_true", default=False)
    parser.add_argument("--signatures-file", type=Path, default=None)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--tokenizer", default="auto", help="auto | none | HF id")
    parser.add_argument("--device", default=os.environ.get("MPKG_RETRIEVAL_DEVICE", "cpu"))
    parser.add_argument("--online", action="store_true", default=False)
    parser.add_argument("--graph-enabled", action="store_true", default=True)
    return parser.parse_args()


def load_ids(path: Path, id_filter: set[str] | None, max_ids: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if id_filter is not None:
        rows = [row for row in rows if str(row.get("ID")) in id_filter]
    if max_ids is not None:
        rows = rows[:max_ids]
    return rows


def load_signatures(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    signatures: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        record_id = str(row.get("ID"))
        signature = row.get("query_signature", row)
        signatures[record_id] = signature
    return signatures


def _hf_snapshot_dir(model_id: str) -> Path | None:
    hub = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
    model_dir = hub / ("models--" + model_id.replace("/", "--"))
    snapshots = sorted((model_dir / "snapshots").glob("*")) if (model_dir / "snapshots").exists() else []
    return snapshots[-1] if snapshots else None


def resolve_tokenizer(choice: str, model_cache_dir: str | None):
    if choice == "none":
        return None
    model_id = "mlx-community/Qwen3.5-4B-4bit" if choice == "auto" else choice
    from transformers import AutoTokenizer

    attempts: list[str] = []
    snapshot = _hf_snapshot_dir(model_id)
    if model_cache_dir:
        candidate = _hf_snapshot_dir(model_id)
        if candidate is None:
            model_dir = Path(model_cache_dir) / ("models--" + model_id.replace("/", "--"))
            candidates = sorted((model_dir / "snapshots").glob("*")) if (model_dir / "snapshots").exists() else []
            snapshot = candidates[-1] if candidates else None
    if snapshot is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
            return {"model_id": model_id, "source": "snapshot", "path": str(snapshot), "tokenizer": tokenizer}
        except Exception as exc:  # noqa: BLE001 - recorded in the manifest
            attempts.append(f"snapshot: {type(exc).__name__}: {exc}")
    try:
        kwargs = {"local_files_only": True}
        if model_cache_dir:
            kwargs["cache_dir"] = str(model_cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        return {"model_id": model_id, "source": "hub", "tokenizer": tokenizer}
    except Exception as exc:  # noqa: BLE001 - recorded in the manifest
        attempts.append(f"hub: {type(exc).__name__}: {exc}")
    return {"model_id": model_id, "source": "unavailable", "errors": attempts, "tokenizer": None}


def main() -> int:
    args = parse_args()
    if args.generate:
        print("generation is not implemented in this environment (requires the MLX runtime)", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["MPKG_RETRIEVAL_DEVICE"] = args.device
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from eval_v2.config_v2 import (
        PRODUCTION_PROMPT,
        PRODUCTION_RETRIEVAL,
        PRODUCTION_RRF_WEIGHTS,
        load_generation,
        retrieval_divergences,
        verify_corpus_binding,
    )
    from eval_v2.production_index import ProductionIndex
    from eval_v2.production_prompts import adaptive_perspective_messages, build_evidence_ledger
    from eval_v2.production_retrieve import coerce_signature, production_retrieve, target_anchor_signature
    from eval_v2.records_v2 import build_record

    from dataclasses import replace

    from local_mpkg_demo.backends import load_retrieval_backends
    from local_mpkg_demo.config import DemoConfig

    generation = load_generation(args.index_root, args.generation_id)
    index = ProductionIndex(generation)
    print(f"index: {index.generation_id} | {index.stats()}")

    divergences = retrieval_divergences(generation)
    print(f"retrieval divergences: {divergences}")

    cfg = replace(DemoConfig.load(REPO_ROOT), offline=not args.online)
    embedding, reranker = load_retrieval_backends(cfg)
    index.attach_models(embedding, reranker)

    tokenizer_info = resolve_tokenizer(args.tokenizer, getattr(cfg, "model_cache_dir", None))
    tokenizer = tokenizer_info.get("tokenizer") if tokenizer_info else None
    print(f"tokenizer: {tokenizer_info.get('model_id')} ok={tokenizer is not None}")

    signatures = load_signatures(args.signatures_file)
    id_filter = set(args.ids.split(",")) if args.ids else None
    rows = load_ids(args.ids_file, id_filter, args.max_ids)
    if not rows:
        print("no input rows after filtering", file=sys.stderr)
        return 2

    out_path = args.out / "mp_kg_rag.jsonl"
    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        done_ids = {
            str(json.loads(line).get("ID"))
            for line in out_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    corpus_binding = None
    if args.corpus_dir.exists():
        corpus_binding = verify_corpus_binding(generation, args.corpus_dir)
        print(f"corpus binding: {corpus_binding.get('all_match')}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "index_root": str(args.index_root),
        "generation_id": index.generation_id,
        "generation_manifest_sha256": sha256_file(index.generation_dir / "index_manifest.json"),
        "generation_identity": index.identity,
        "retrieval_contract": index.retrieval,
        "production_retrieval_constants": PRODUCTION_RETRIEVAL,
        "production_rrf_weights": PRODUCTION_RRF_WEIGHTS,
        "production_prompt_constants": PRODUCTION_PROMPT,
        "retrieval_divergences": divergences,
        "corpus_binding": corpus_binding,
        "code_sha256": {
            name: (sha256_file(path) if path.exists() else None)
            for name, path in CODE_HASH_TARGETS.items()
        },
        "device": args.device,
        "tokenizer": {k: v for k, v in (tokenizer_info or {}).items() if k != "tokenizer"},
        "signature_source": "file" if signatures else "target-anchor",
        "ids_file": str(args.ids_file),
        "ids_requested": len(rows),
        "graph_enabled": bool(args.graph_enabled),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    processed = 0
    with out_path.open("a" if args.resume else "w", encoding="utf-8") as sink:
        for row in rows:
            record_id = str(row.get("ID"))
            if record_id in done_ids:
                continue
            started = time.time()
            target = str(row.get("Target", ""))

            if record_id in signatures:
                signature = coerce_signature(signatures[record_id])
                signature_source = "file"
            else:
                signature = target_anchor_signature(index, target)
                signature_source = "target-anchor"

            retrieval_started = time.time()
            retrieval = production_retrieve(
                index,
                str(row.get("Text", "")),
                target,
                str(row.get("Category", "")),
                signature,
                args.graph_enabled,
            )
            retrieval_seconds = time.time() - retrieval_started

            prompt_started = time.time()
            perspectives: list[dict] = []
            ledger: list[dict] = []
            if retrieval["evidence"]:
                perspectives = adaptive_perspective_messages(
                    str(row.get("Text", "")),
                    target,
                    str(row.get("Category", "")),
                    retrieval["evidence"],
                    tokenizer=tokenizer,
                    max_seq_length=int(PRODUCTION_PROMPT["max_seq_length"]),
                    max_new_tokens=int(PRODUCTION_PROMPT["perspective_max_new_tokens"]),
                )
                ledger = build_evidence_ledger(retrieval["evidence"])
            prompt_seconds = time.time() - prompt_started

            record = build_record(
                row,
                generation_id=index.generation_id,
                signature=signature,
                signature_source=signature_source,
                retrieval=retrieval,
                ledger=ledger,
                perspectives=perspectives,
                timing={
                    "retrieval_seconds": round(retrieval_seconds, 3),
                    "prompt_seconds": round(prompt_seconds, 3),
                    "total_seconds": round(time.time() - started, 3),
                },
            )
            sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            sink.flush()
            processed += 1
            max_logit = max(
                (c.get("rerank_logit") for c in retrieval["candidates"] if c.get("rerank_logit") is not None),
                default=None,
            )
            print(
                f"[{processed}] {record_id} mode={signature.get('target_anchor', {}).get('match_mode', 'file')} "
                f"candidates={len(retrieval['candidates'])} selected={len(retrieval['evidence'])} "
                f"max_logit={round(max_logit, 3) if max_logit is not None else None} "
                f"abstain={retrieval['abstention_reason']} {round(time.time() - started, 1)}s"
            )

    print(f"done: {processed} records -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
