#!/usr/bin/env python3
"""Build a local_mpkg_demo index generation from a corpus into a scratch root.

Corpus layout expected (semantic_graph_source-compatible):
  chunks.json, evidence_ledger.json, semantic_extractions.json,
  candidates.json (optional), manifest.json

The generation is assembled by the shipped builder
``local_mpkg_demo.index_store.build_or_validate_index`` (the exports-port copy,
which is the copy the smoke/acceptance tools import) with the pinned BGE-M3
encoder on CPU.  The live index (``MP_KG_RAG_v2/local_mpkg_demo_index``) is
never selected, written, or re-pointed; ``--out-root`` must be outside it.

Subsetting: ``--docs`` / ``--limit`` stage a filtered copy of the corpus under
``<out-root>/_staged_corpus/<utc-stamp>/`` (source corpus stays read-only) with
a regenerated evidence ledger and rewritten manifest source-hashes, so a small
scratch build does not need a pre-existing corpus directory.

Modes:
  build         (default) build, reseal (``--generation-id``), verify
  --dry-run     validate inputs and print the plan; writes nothing
  --verify-only verify/reseal-check an existing generation; writes nothing

Usage:
  python3 build_index_v2.py --corpus DIR --out-root DIR [--docs U1,U2,...]
      [--limit N] [--generation-id ID] [--dry-run] [--verify-only]
      [--allow-download] [--load-check] [--report FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORT_ROOT = PROJECT_ROOT / "exports/MP_KG_RAG_Gradio_UI_Windows_Claude_Port"
LIVE_INDEX_ROOT = PROJECT_ROOT / "local_mpkg_demo_index"
GENERATION_FILES = ("index_manifest.json", "embeddings.npy", "id_map.json", "graph.json", "evidence.sqlite3")
STAGED_CORPUS_FILES = ("chunks.json", "evidence_ledger.json", "semantic_extractions.json", "candidates.json")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _ensure_package() -> tuple[Any, Any]:
    if str(EXPORT_ROOT) not in sys.path:
        sys.path.insert(0, str(EXPORT_ROOT))
    import local_mpkg_demo.index_store as index_store

    from local_mpkg_demo.config import DemoConfig

    return index_store, DemoConfig


def _assert_scratch_root(out_root: Path) -> None:
    out = Path(out_root).expanduser().resolve()
    live = LIVE_INDEX_ROOT.resolve()
    if out == live or live in out.parents or out in live.parents:
        raise SystemExit(f"refusing to use out-root {out}: it overlaps the live index {live}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any, *, sort_keys: bool = True, indent: int | None = None) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=sort_keys, indent=indent, default=str) + "\n", encoding="utf-8")


def _parse_docs(values: list[str] | None) -> set[str]:
    docs: set[str] = set()
    for value in values or []:
        docs.update(part.strip() for part in value.split(",") if part.strip())
    return docs


def _corpus_manifest(corpus: Path) -> dict[str, Any]:
    path = corpus / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"corpus manifest missing: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SystemExit(f"corpus manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SystemExit("corpus manifest must be a JSON object")
    if manifest.get("complete") is False:
        raise SystemExit("corpus manifest is marked incomplete")
    return manifest


def _select_rows(index_store: Any, corpus: Path, docs: set[str], limit: int | None) -> dict[str, Any]:
    chunks_path = corpus / "chunks.json"
    if not chunks_path.is_file():
        raise SystemExit(f"corpus chunks missing: {chunks_path}")
    chunks = index_store._read_rows(chunks_path)
    order = {str(row.get("chunk_id")): position for position, row in enumerate(chunks)}
    accepted = [
        row
        for row in chunks
        if row.get("status") == "accepted" and (not docs or str(row.get("document_uid")) in docs)
    ]
    if limit is not None:
        accepted = sorted(accepted, key=lambda row: str(row.get("chunk_id")))[: max(0, int(limit))]
        accepted = sorted(accepted, key=lambda row: order.get(str(row.get("chunk_id")), 0))
    chunk_ids = {str(row.get("chunk_id")) for row in accepted}
    extractions_path = corpus / "semantic_extractions.json"
    extractions = index_store._read_rows(extractions_path) if extractions_path.is_file() else []
    candidates_path = corpus / "candidates.json"
    candidates = index_store._read_rows(candidates_path) if candidates_path.is_file() else []
    filter_extra = [row for row in extractions if str(row.get("chunk_id")) in chunk_ids]
    filter_candidates = [row for row in candidates if str(row.get("chunk_id")) in chunk_ids]
    return {
        "chunks": accepted,
        "extractions": filter_extra,
        "candidates": filter_candidates,
        "source_counts": {
            "chunks_total": len(chunks),
            "extractions_total": len(extractions),
            "candidates_total": len(candidates),
        },
        "has_candidates_file": candidates_path.is_file(),
    }


def _stage_corpus(
    index_store: Any,
    corpus: Path,
    out_root: Path,
    selection: dict[str, Any],
    manifest: dict[str, Any],
) -> Path:
    staged = out_root / "_staged_corpus" / f"corpus-{_utc_stamp()}"
    staged.mkdir(parents=True)
    chunks = selection["chunks"]
    _write_json(staged / "chunks.json", chunks)
    ledger_rows = []
    for chunk in chunks:
        row = {
            "evidence_id": str(chunk.get("chunk_id")),
            "document_uid": chunk.get("document_uid"),
            "chunk_id": chunk.get("chunk_id"),
            "source_id": chunk.get("source_id"),
            "source_type": chunk.get("source_type"),
            "page": chunk.get("page"),
            "span_start": chunk.get("span_start"),
            "span_end": chunk.get("span_end"),
            "chunk_sha256": chunk.get("chunk_sha256"),
            "text": chunk.get("text"),
            "displayed_text": chunk.get("text"),
            "status": chunk.get("status"),
            "authority_score": chunk.get("authority_score"),
        }
        ledger_rows.append(row)
    stage_hash = manifest.get("stage_hash") or (manifest.get("identity") or {}).get("stage_hash")
    ledger_payload: dict[str, Any] = {"records": ledger_rows}
    if stage_hash:
        ledger_payload["stage_hash"] = stage_hash
    _write_json(staged / "evidence_ledger.json", ledger_payload)
    _write_json(staged / "semantic_extractions.json", selection["extractions"])
    if selection["has_candidates_file"]:
        _write_json(staged / "candidates.json", selection["candidates"])
    source_hashes = {
        name.removesuffix(".json"): _sha256_file(staged / name)
        for name in STAGED_CORPUS_FILES
        if (staged / name).is_file()
    }
    staged_manifest = dict(manifest)
    staged_manifest["source_hashes"] = source_hashes
    _write_json(staged / "manifest.json", staged_manifest)
    return staged


def _build_config(DemoConfig: Any, corpus: Path, out_root: Path, *, allow_download: bool) -> Any:
    config = DemoConfig.load(PROJECT_ROOT)
    return replace(
        config,
        offline=not allow_download,
        chunks_path=corpus / "chunks.json",
        evidence_ledger_path=corpus / "evidence_ledger.json",
        semantic_extractions_path=corpus / "semantic_extractions.json",
        candidates_path=corpus / "candidates.json",
        source_manifest_path=corpus / "manifest.json",
        semantic_graph_source=corpus,
        index_path=out_root,
    )


def _load_encoder(config: Any, *, allow_download: bool) -> Any:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = SentenceTransformer(
            config.embedding_model,
            revision=config.embedding_revision,
            device=device,
            local_files_only=not allow_download,
        )
    except Exception as first:
        if allow_download:
            raise
        snapshot = (
            Path.home()
            / ".cache/huggingface/hub"
            / f"models--{config.embedding_model.replace('/', '--')}"
            / "snapshots"
            / config.embedding_revision
        )
        if not snapshot.is_dir():
            raise RuntimeError(
                f"offline model load failed ({first!r}); pass --allow-download or pre-populate the HF cache"
            ) from first
        model = SentenceTransformer(str(snapshot), device=device, local_files_only=True)
    model.max_seq_length = 512
    return model


class _BuildOnlyReranker:
    name = "build-only:reranker-not-loaded"


def _build(
    index_store: Any,
    DemoConfig: Any,
    corpus: Path,
    out_root: Path,
    *,
    allow_download: bool,
) -> tuple[dict[str, Any], float]:
    out_root.mkdir(parents=True, exist_ok=True)
    config = _build_config(DemoConfig, corpus, out_root, allow_download=allow_download)
    manifest_doc = _corpus_manifest(corpus)
    fixture = (manifest_doc.get("identity") or {}).get("stage_hash") == "tiny"
    dense_encoder = None
    reranker = None
    model_seconds = None
    if not fixture:
        started = time.perf_counter()
        dense_encoder = _load_encoder(config, allow_download=allow_download)
        model_seconds = round(time.perf_counter() - started, 1)
        reranker = _BuildOnlyReranker()
    started = time.perf_counter()
    result = index_store.build_or_validate_index(
        config,
        progress=lambda message, fraction: print(f"[{fraction:5.1%}] {message}"),
        dense_encoder=dense_encoder,
        reranker=reranker,
    )
    build_seconds = round(time.perf_counter() - started, 1)
    return {
        "manifest": result,
        "model_load_seconds": model_seconds,
        "build_seconds": build_seconds,
        "fixture": fixture,
    }, build_seconds


def _write_current(out_root: Path, generation_id: str) -> None:
    pointer_temp = out_root / f".CURRENT.{generation_id}"
    fd = os.open(pointer_temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, generation_id.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(pointer_temp, out_root / "CURRENT")
    descriptor = os.open(out_root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reseal(index_store: Any, out_root: Path, requested: str) -> dict[str, str]:
    requested = requested.strip()
    if not requested or Path(requested).name != requested:
        raise SystemExit(f"unsafe generation id: {requested!r}")
    current = (out_root / "CURRENT").read_text(encoding="utf-8").strip()
    if requested == current:
        return {"from": current, "to": requested, "resealed": False}
    generation = out_root / "generations" / current
    target = out_root / "generations" / requested
    if target.exists():
        raise SystemExit(f"generation id already exists: {target}")
    manifest_path = generation / "index_manifest.json"
    graph_path = generation / "graph.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    graph_text = graph_path.read_text(encoding="utf-8")
    with sqlite3.connect(generation / "evidence.sqlite3") as db:
        metadata_row = db.execute("SELECT value FROM index_metadata WHERE key='generation_id'").fetchone()
    if metadata_row is None:
        raise SystemExit("generation sqlite metadata has no generation_id row")

    def restore() -> None:
        manifest_path.write_text(manifest_text, encoding="utf-8")
        graph_path.write_text(graph_text, encoding="utf-8")
        with sqlite3.connect(generation / "evidence.sqlite3") as db:
            db.execute("UPDATE index_metadata SET value = ? WHERE key='generation_id'", (current,))

    os.rename(generation, target)
    try:
        manifest = json.loads(manifest_text)
        manifest["generation_id"] = requested
        if isinstance(manifest.get("identity"), dict):
            manifest["identity"]["generation_id"] = requested
        _write_json(target / "index_manifest.json", manifest, indent=2)
        graph = json.loads(graph_text)
        graph["generation_id"] = requested
        _write_json(target / "graph.json", graph)
        with sqlite3.connect(target / "evidence.sqlite3") as db:
            db.execute("UPDATE index_metadata SET value = ? WHERE key='generation_id'", (requested,))
        if not index_store._validate_index_files(target, manifest["identity"]):
            raise RuntimeError("resealed generation failed structural validation")
    except Exception as exc:
        os.rename(target, generation)
        restore()
        raise SystemExit(f"reseal failed and was rolled back: {exc}") from exc
    _write_current(out_root, requested)
    return {"from": current, "to": requested, "resealed": True}


def _verify_generation(index_store: Any, generation: Path, *, corpus: Path | None = None, deep: bool = True) -> dict[str, Any]:
    import numpy as np

    missing = [name for name in GENERATION_FILES if not (generation / name).is_file()]
    if missing:
        raise RuntimeError(f"generation {generation.name} is incomplete; missing {', '.join(missing)}")
    manifest = json.loads((generation / "index_manifest.json").read_text(encoding="utf-8"))
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or not index_store._validate_index_files(generation, identity):
        raise RuntimeError(f"generation {generation.name} failed structural integrity validation")
    checks: dict[str, Any] = {"structural": True, "row_counts": manifest.get("row_counts")}
    vectors = np.load(generation / "embeddings.npy", mmap_mode="r")
    checks["embeddings"] = {"shape": list(vectors.shape), "dtype": str(vectors.dtype)}
    count = int(vectors.shape[0])
    if count:
        sample_indexes = sorted({0, count // 2, count - 1, *(index for index in (1, 2, 3) if index < count)})
        norms = [float(np.linalg.norm(np.asarray(vectors[index], dtype=np.float32))) for index in sample_indexes]
        if min(norms) <= 0.0 or any(not math.isfinite(norm) for norm in norms):
            raise RuntimeError("embeddings contain a zero or non-finite vector")
        checks["sample_norms"] = {str(index): round(norm, 6) for index, norm in zip(sample_indexes, norms)}
    graph = json.loads((generation / "graph.json").read_text(encoding="utf-8"))
    ids = json.loads((generation / "id_map.json").read_text(encoding="utf-8"))
    checks["evidence_ids"] = len(ids)
    checks["nodes"] = len(graph.get("node_ids", []))
    checks["edges"] = len(graph.get("edges", []))
    checks["tables"] = {name: len(rows) for name, rows in (graph.get("tables") or {}).items() if rows}
    if deep and corpus is not None:
        checks["deep_reference"] = _deep_reference(index_store, graph, corpus)
    return checks


def _deep_reference(index_store: Any, graph: dict[str, Any], corpus: Path) -> dict[str, Any]:
    chunks = index_store._read_rows(corpus / "chunks.json")
    chunks_by_key = {(str(row.get("document_uid")), str(row.get("chunk_id"))): row for row in chunks}
    records = []
    for chunk in chunks:
        if chunk.get("status") != "accepted":
            continue
        row = dict(chunk)
        row["evidence_id"] = str(chunk["chunk_id"])
        row["text"] = str(chunk.get("text") or "")
        row["displayed_text"] = row["text"]
        row["chunk_sha256"] = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        records.append(row)
    extraction_path = corpus / "semantic_extractions.json"
    extraction_rows = index_store._read_rows(extraction_path) if extraction_path.is_file() else []
    candidates_path = corpus / "candidates.json"
    candidate_rows = index_store._read_rows(candidates_path) if candidates_path.is_file() else []
    expected, stats = index_store._simple_claim_graph(records, chunks_by_key, extraction_rows, candidate_rows)
    persisted = graph.get("tables") or {}

    def canonical(rows: list[dict[str, Any]]) -> list[str]:
        return sorted(json.dumps(row, sort_keys=True, default=str) for row in rows)

    for table in ("Entity", "Mention"):
        if canonical(expected.get(table, [])) != canonical(persisted.get(table, [])):
            raise RuntimeError(f"deep verification failed: {table} rows differ from re-derivation")
    if canonical(expected.get("edges", [])) != canonical(persisted.get("edges", [])):
        raise RuntimeError("deep verification failed: graph edges differ from re-derivation")
    return {
        "simple_claims": stats.get("simple_claims"),
        "valid_mappings": stats.get("valid_mappings"),
        "mentions": len(expected.get("Mention", [])),
        "entities": len(expected.get("Entity", [])),
        "edges": len(expected.get("edges", [])),
    }


def _load_check(index_store: Any, DemoConfig: Any, generation: Path, corpus: Path | None) -> dict[str, Any]:
    config = _build_config(DemoConfig, corpus or (PROJECT_ROOT / "semantic_graph_source"), generation.parent.parent, allow_download=False)
    config = replace(config, index_path=generation)

    class _NoBackend:
        pass

    original = index_store.build_or_validate_index
    index_store.build_or_validate_index = lambda *args, **kwargs: None
    try:
        retriever = index_store.LocalRetriever(config, dense_encoder=_NoBackend(), reranker=_NoBackend())
    finally:
        index_store.build_or_validate_index = original
    records = getattr(retriever, "_records", {}) or {}
    vectors = getattr(retriever, "vectors", None)
    return {
        "records": len(records),
        "evidence_ids": len(getattr(retriever, "evidence_ids", []) or []),
        "vectors_shape": list(vectors.shape) if vectors is not None else None,
        "graph_tables": {name: len(rows) for name, rows in (getattr(retriever, "_graph_tables", {}) or {}).items() if rows},
    }


def _resolve_generation(out_root: Path, generation_id: str | None) -> Path:
    current_file = out_root / "CURRENT"
    if not current_file.is_file():
        raise SystemExit(f"out-root has no CURRENT pointer: {out_root}")
    current = current_file.read_text(encoding="utf-8").strip()
    requested = (generation_id or current).strip()
    generation = out_root / "generations" / requested
    if not generation.is_dir():
        raise SystemExit(f"generation not found: {generation}")
    return generation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True, help="semantic_graph_source-style corpus directory (read-only)")
    parser.add_argument("--out-root", type=Path, required=True, help="scratch index root to build into")
    parser.add_argument("--docs", action="append", default=None, help="comma-separated document_uid allowlist; repeatable")
    parser.add_argument("--limit", type=int, default=None, help="max accepted chunks to stage (deterministic by chunk_id)")
    parser.add_argument("--generation-id", default=None, help="reseal the built generation to this id")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and print the plan; writes nothing")
    parser.add_argument("--verify-only", action="store_true", help="verify an existing generation; writes nothing")
    parser.add_argument("--staged-corpus", type=Path, default=None, help="filtered corpus dir used for --verify-only deep checks")
    parser.add_argument("--allow-download", action="store_true", help="allow Hugging Face network access if the cache is incomplete")
    parser.add_argument("--load-check", action="store_true", help="also construct LocalRetriever over the generation (read-only)")
    parser.add_argument("--report", type=Path, default=None, help="write a JSON summary to this path")
    args = parser.parse_args()

    corpus = args.corpus.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    if not corpus.is_dir():
        raise SystemExit(f"corpus directory not found: {corpus}")
    if not args.verify_only:
        _assert_scratch_root(out_root)
    index_store, DemoConfig = _ensure_package()

    report: dict[str, Any] = {
        "tool": "rebuild_v2/build_index_v2.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": str(corpus),
        "out_root": str(out_root),
    }

    if args.verify_only:
        generation = _resolve_generation(out_root, args.generation_id)
        staged = args.staged_corpus.expanduser().resolve() if args.staged_corpus else None
        verification = _verify_generation(index_store, generation, corpus=staged, deep=staged is not None)
        report.update({"mode": "verify_only", "generation": generation.name, "verification": verification})
        if args.load_check:
            report["load_check"] = _load_check(index_store, DemoConfig, generation, staged)
        print(json.dumps(report, indent=2, default=str))
        return 0

    docs = _parse_docs(args.docs)
    manifest = _corpus_manifest(corpus)
    needs_staging = bool(docs) or args.limit is not None
    if args.dry_run:
        selection = _select_rows(index_store, corpus, docs, args.limit)
        plan = {
            "mode": "dry_run",
            "corpus": str(corpus),
            "out_root": str(out_root),
            "staging_needed": needs_staging,
            "docs_filter": sorted(docs) if docs else None,
            "limit": args.limit,
            "selected": {
                "chunks": len(selection["chunks"]),
                "extractions": len(selection["extractions"]),
                "candidates": len(selection["candidates"]),
            },
            "source_counts": selection["source_counts"],
            "documents": sorted({str(row.get("document_uid")) for row in selection["chunks"]}),
            "fixture": (manifest.get("identity") or {}).get("stage_hash") == "tiny",
            "generation_id": args.generation_id,
            "allow_download": args.allow_download,
        }
        report.update(plan)
        print(json.dumps(report, indent=2, default=str))
        return 0

    staged_dir = None
    build_corpus = corpus
    selection: dict[str, Any] | None = None
    if needs_staging:
        selection = _select_rows(index_store, corpus, docs, args.limit)
        if not selection["chunks"]:
            raise SystemExit("staging selection produced zero accepted chunks")
        staged_dir = _stage_corpus(index_store, corpus, out_root, selection, manifest)
        build_corpus = staged_dir
    report["staged_corpus"] = str(staged_dir) if staged_dir else None
    if selection is not None:
        report["staged_counts"] = {
            "chunks": len(selection["chunks"]),
            "extractions": len(selection["extractions"]),
            "candidates": len(selection["candidates"]),
        }

    build_info, _ = _build(index_store, DemoConfig, build_corpus, out_root, allow_download=args.allow_download)
    result = build_info.pop("manifest")
    report["build"] = build_info
    report["row_counts"] = dict(result.row_counts)
    report["readiness_warnings"] = [dict(warning) for warning in result.readiness_warnings]

    if args.generation_id:
        report["reseal"] = _reseal(index_store, out_root, args.generation_id)
    generation = _resolve_generation(out_root, None)
    report["generation"] = generation.name

    report["verification"] = _verify_generation(index_store, generation, corpus=build_corpus, deep=True)
    if args.load_check:
        report["load_check"] = _load_check(index_store, DemoConfig, generation, build_corpus)

    print(json.dumps({key: report[key] for key in ("generation", "row_counts", "readiness_warnings", "verification")}, indent=2, default=str))
    if args.report:
        report_path = args.report.expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(report_path, report, indent=2)
        print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
