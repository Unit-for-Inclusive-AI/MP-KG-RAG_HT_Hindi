#!/usr/bin/env python3
"""Read-only index smoke test for a built local_mpkg_demo index generation.

Loads a generation through the shipped demo ``local_mpkg_demo.config`` /
``index_store`` classes without ever building, swapping, or writing to the
index.  ``index_store.build_or_validate_index`` is replaced in-memory with a
no-op so ``LocalRetriever`` cannot rebuild a stale generation; the index
directory and ``CURRENT`` pointer are only read.

Runs 8 queries on CPU (BGE-M3 embeddings + BGE reranker; no MLX/Qwen):

* 3 Hinglish homophobic / homophobic-adjacent posts from test_homophobic.jsonl
* 3 English topic queries
* 2 authority / safety queries

For each query it prints the top-5 evidence chunk_ids with source_id, a
120-character snippet, dense and rerank scores, and timing.  Any exception
while loading or querying is recorded and forces exit code 1.

Usage:
  python3 smoke_index_v2.py [--index-root PATH] [--generation-id ID]
                            [--queries N] [--out DIR] [--allow-download]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

EXPORT_ROOT = Path("/Users/pika/Downloads/MP_KG_RAG_v2/exports/MP_KG_RAG_Gradio_UI_Windows_Claude_Port")
PROJECT_ROOT = Path("/Users/pika/Downloads/MP_KG_RAG_v2")
DEFAULT_INDEX_ROOT = PROJECT_ROOT / "local_mpkg_demo_index"
DEFAULT_OUT_DIR = Path(
    "/Users/pika/Downloads/Hate_Speech_Counter_Narrative_RAG_Project/.audit/rebuild_v2_acceptance_20261019"
)
TEST_HOMOPHOBIC = PROJECT_ROOT / "outputs/2026-10-16-local-mlx-test155/test_homophobic.jsonl"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

FALLBACK_HINGLISH = [
    {"id": "fallback-1", "text": "मेरे हिसाब से classroom poster में queer log wali discussion avoid karni chahiye, children ke saamne unnecessary confusion create hota hai."},
    {"id": "fallback-2", "text": "इतनी openness ke naam par metro station poster mein rainbow wale ko push karna mujhe sahi nahi lagta."},
    {"id": "fallback-3", "text": "bi logon ke liye labels badalte rehte hain, serious discussion kaise ho"},
]

ENGLISH_TOPICS = [
    {"id": "topic-1", "text": "conversion therapy ban India", "target": "", "category": ""},
    {"id": "topic-2", "text": "minority stress LGBTQ health", "target": "", "category": ""},
    {"id": "topic-3", "text": "UN hate speech strategy", "target": "", "category": ""},
]

AUTHORITY_QUERIES = [
    {"id": "auth-1", "text": "OHCHR report conversion therapy human rights", "target": "", "category": "authority"},
    {"id": "auth-2", "text": "WHO guidelines health services key populations transgender", "target": "", "category": "authority"},
]


def snippet(text: str, limit: int = 120) -> str:
    collapsed = " ".join(str(text or "").split())
    return collapsed[:limit]


def load_hinglish_queries(limit: int = 3):
    rows = []
    fallback_used = False
    if TEST_HOMOPHOBIC.exists():
        for line in TEST_HOMOPHOBIC.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    picked, seen = [], set()
    for row in rows:
        text = str(row.get("Text") or row.get("text") or "").strip()
        key = " ".join(text.split()).casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        picked.append({
            "id": f"hinglish-{row.get('ID', len(picked) + 1)}",
            "text": text,
            "target": str(row.get("Target") or ""),
            "category": str(row.get("Category") or ""),
        })
        if len(picked) >= limit:
            break
    if not picked:
        picked = [dict(item) for item in FALLBACK_HINGLISH]
        fallback_used = True
    return picked, fallback_used, len(rows)


def build_queries(limit: int):
    hinglish, fallback_used, test_rows = load_hinglish_queries(3)
    queries = (
        [{**item, "kind": "hinglish_homophobic"} for item in hinglish]
        + [{**item, "kind": "english_topic"} for item in ENGLISH_TOPICS]
        + [{**item, "kind": "authority_safety"} for item in AUTHORITY_QUERIES]
    )
    return queries[: max(1, min(limit, len(queries)))], fallback_used, test_rows


def manifest_summary(generation: Path):
    manifest = json.loads((generation / "index_manifest.json").read_text(encoding="utf-8"))
    return {
        "generation_id": manifest.get("generation_id"),
        "created_at": manifest.get("created_at"),
        "row_counts": manifest.get("row_counts"),
        "embedding_model": (manifest.get("identity") or {}).get("embedding_model"),
        "embedding_dimension": (manifest.get("identity") or {}).get("embedding_dimension"),
        "reranker_model": (manifest.get("identity") or {}).get("reranker_model"),
        "extraction_adapter": (manifest.get("identity") or {}).get("extraction_adapter"),
        "readiness_warnings": manifest.get("readiness_warnings"),
    }


def run_query(retriever, config, query):
    from local_mpkg_demo.config import QueryInput
    from work.mpkg_rag_core import reciprocal_rank_fusion

    query_text = " ".join(part for part in (query.get("text"), query.get("target"), query.get("category")) if part)
    timings = {}
    t_start = time.perf_counter()
    t0 = time.perf_counter()
    dense = retriever._dense_search(query_text, config.dense_top_k)
    t1 = time.perf_counter()
    bm25 = retriever._bm25_search(query_text, config.bm25_top_k)
    t2 = time.perf_counter()
    fused = reciprocal_rank_fusion(
        {"dense": dense, "bm25": bm25, "graph": []},
        {key: value for key, value in config.rrf_weights},
        config.rrf_constant,
    )
    # LocalRetriever.retrieve merges the stored record into each fused row
    # before reranking (fused rows carry no text on their own).  Mirror it.
    for row in fused:
        row.update(retriever._records.get(row["evidence_chunk_id"], {}))
    top = fused[: config.rerank_top_k]
    t3 = time.perf_counter()
    rerank = retriever._rerank(query_text, top)
    t4 = time.perf_counter()
    timings.update({
        "dense_ms": round((t1 - t0) * 1000.0, 1),
        "bm25_ms": round((t2 - t1) * 1000.0, 1),
        "fusion_ms": round((t3 - t2) * 1000.0, 1),
        "rerank_ms": round((t4 - t3) * 1000.0, 1),
    })

    hits = []
    ordered = sorted(
        top,
        key=lambda row: (
            -rerank.get(row["evidence_chunk_id"], float("-inf")),
            -float(row.get("rrf_score", 0.0)),
            row["evidence_chunk_id"],
        ),
    )
    for row in ordered:
        record = retriever._records.get(row["evidence_chunk_id"], {})
        traces = row.get("branch_traces", [])
        dense_score = max(
            (float(trace.get("raw_score")) for trace in traces if trace.get("branch") == "dense" and trace.get("raw_score") is not None),
            default=None,
        )
        bm25_score = max(
            (float(trace.get("raw_score")) for trace in traces if trace.get("branch") == "bm25" and trace.get("raw_score") is not None),
            default=None,
        )
        hits.append({
            "evidence_chunk_id": row["evidence_chunk_id"],
            "chunk_id": record.get("chunk_id") or row.get("chunk_id"),
            "document_uid": record.get("document_uid") or row.get("document_uid"),
            "source_id": record.get("source_id"),
            "source_type": record.get("source_type"),
            "page": record.get("page"),
            "snippet": snippet(record.get("text")),
            "dense_score": dense_score,
            "bm25_score": bm25_score,
            "rerank_score": rerank.get(row["evidence_chunk_id"]),
            "rrf_score": round(float(row.get("rrf_score", 0.0)), 6),
        })
    timings["total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 1)

    public = {"ok": True, "abstained": None, "reason": None, "selected_count": 0, "selected": []}
    t_public = time.perf_counter()
    try:
        result = retriever.retrieve(QueryInput(text=query_text), limit=5)
        public.update({
            "abstained": bool(result.abstained),
            "reason": result.reason,
            "selected_count": len(result.evidence),
            "selected": [
                {
                    "evidence_chunk_id": item.internal_evidence_id,
                    "source_id": item.source_id,
                    "rerank_score": item.scores.get("rerank"),
                    "rrf_score": item.scores.get("rrf"),
                    "dense_score": item.scores.get("dense"),
                    "bm25_score": item.scores.get("bm25"),
                }
                for item in result.evidence
            ],
            "trace": dict(result.trace),
        })
    except Exception as exc:  # surfaced as a hard smoke failure
        public = {"ok": False, "error": repr(exc)}
    public["ms"] = round((time.perf_counter() - t_public) * 1000.0, 1)

    return {
        "query": query,
        "query_text": query_text,
        "timings_ms": timings,
        "hits": hits,
        "public_retrieve": public,
        "error": None if public.get("ok") else public.get("error", "unknown"),
    }


def render_markdown(report):
    lines = [
        "# rebuild_v2 index smoke report",
        "",
        f"- generated_at: {report['meta']['generated_at']}",
        f"- index_root: `{report['meta']['index_root']}`",
        f"- generation_id: `{report['meta']['generation_id']}` (CURRENT: `{report['meta']['current_generation_id']}`)",
        f"- device: {report['meta']['device']}; offline: {report['meta']['offline']}",
        f"- models: {report['meta']['embedding_model']} + {report['meta']['reranker_model']} (CPU)",
        f"- queries: {report['summary']['queries']}; errors: {report['summary']['errors']}",
        f"- model load: {report['summary']['model_load_s']}s; retriever init: {report['summary']['retriever_init_s']}s; total: {report['summary']['total_s']}s",
        "",
        "## Per-query results",
        "",
    ]
    for item in report["queries"]:
        query = item["query"]
        timings = item["timings_ms"]
        lines.append(f"### {query['id']} ({query['kind']})")
        lines.append("")
        lines.append(f"- query: `{item['query_text'][:240]}`")
        lines.append(f"- timing_ms: dense={timings['dense_ms']} bm25={timings['bm25_ms']} rerank={timings['rerank_ms']} total={timings['total_ms']}")
        public = item["public_retrieve"]
        if public.get("ok"):
            lines.append(f"- public retrieve: abstained={public['abstained']} ({public['reason']}), selected={public['selected_count']}, {public['ms']}ms")
        else:
            lines.append(f"- public retrieve: FAILED {public.get('error')}")
        lines.append("")
        lines.append("| rank | evidence_chunk_id | source_id | dense | rerank | rrf | snippet (120 chars) |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for rank, hit in enumerate(item["hits"], 1):
            dense = "-" if hit["dense_score"] is None else f"{hit['dense_score']:.4f}"
            rerank = "-" if hit["rerank_score"] is None else f"{hit['rerank_score']:.4f}"
            snippet_text = hit["snippet"].replace("|", "/")
            lines.append(
                f"| {rank} | `{hit['evidence_chunk_id']}` | {hit['source_id']} | {dense} | {rerank} | {hit['rrf_score']:.5f} | {snippet_text} |"
            )
        lines.append("")
    if report["errors"]:
        lines.append("## Errors")
        lines.append("")
        for error in report["errors"]:
            lines.append(f"- {error}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--generation-id", default=None)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--allow-download", action="store_true", help="allow Hugging Face network access if the cache is incomplete")
    args = parser.parse_args()

    index_root = args.index_root.expanduser().resolve()
    current_file = index_root / "CURRENT"
    if not current_file.is_file():
        print(f"index root has no CURRENT pointer: {index_root}", file=sys.stderr)
        return 2
    current_generation_id = current_file.read_text(encoding="utf-8").strip()
    generation_id = (args.generation_id or current_generation_id).strip()
    generation = index_root / "generations" / generation_id
    required = ("index_manifest.json", "embeddings.npy", "id_map.json", "graph.json", "evidence.sqlite3")
    missing = [name for name in required if not (generation / name).is_file()]
    if missing:
        print(f"generation {generation_id} is incomplete; missing {', '.join(missing)}", file=sys.stderr)
        return 2

    if str(EXPORT_ROOT) not in sys.path:
        sys.path.insert(0, str(EXPORT_ROOT))
    import local_mpkg_demo.index_store as index_store
    from local_mpkg_demo.config import DemoConfig
    from local_mpkg_demo.index_store import LocalRetriever

    # Read-only seam: the demo would otherwise rebuild this generation because
    # the shipped index predates the current extraction_adapter identity.
    index_store.build_or_validate_index = lambda *args, **kwargs: None

    offline = not args.allow_download
    started = time.perf_counter()
    model_load_s = None
    retriever_init_s = None
    errors = []
    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer

        t0 = time.perf_counter()
        embedding = SentenceTransformer(
            EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device="cpu", local_files_only=offline
        )
        embedding.max_seq_length = 512
        reranker = CrossEncoder(
            RERANKER_MODEL, revision=RERANKER_REVISION, max_length=512, device="cpu", local_files_only=offline
        )
        model_load_s = round(time.perf_counter() - t0, 1)
    except Exception as exc:
        errors.append(f"model_load: {exc!r}")
        print(f"model load failed: {exc!r}", file=sys.stderr)
        return 1

    config = replace(DemoConfig.load(PROJECT_ROOT), index_path=generation, offline=offline)
    t0 = time.perf_counter()
    try:
        retriever = LocalRetriever(config, dense_encoder=embedding, reranker=reranker)
    except Exception as exc:
        errors.append(f"retriever_init: {exc!r}")
        print(f"retriever init failed: {exc!r}", file=sys.stderr)
        return 1
    retriever_init_s = round(time.perf_counter() - t0, 1)

    queries, fallback_used, test_rows = build_queries(args.queries)
    results = []
    for query in queries:
        try:
            results.append(run_query(retriever, config, query))
        except Exception as exc:
            errors.append(f"query {query['id']}: {exc!r}")
            results.append({
                "query": query,
                "query_text": query.get("text"),
                "timings_ms": {},
                "hits": [],
                "public_retrieve": {"ok": False, "error": repr(exc)},
                "error": repr(exc),
            })

    total_s = round(time.perf_counter() - started, 1)
    summary_manifest = manifest_summary(generation)
    report = {
        "meta": {
            "tool": "rebuild_v2/smoke_index_v2.py",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "index_root": str(index_root),
            "generation_id": generation_id,
            "current_generation_id": current_generation_id,
            "is_current_generation": generation_id == current_generation_id,
            "project_root": str(PROJECT_ROOT),
            "export_root": str(EXPORT_ROOT),
            "device": "cpu",
            "offline": offline,
            "read_only": True,
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": RERANKER_MODEL,
            "index_manifest": summary_manifest,
            "test_jsonl": str(TEST_HOMOPHOBIC),
            "test_rows_read": test_rows,
            "hinglish_fallback_used": fallback_used,
        },
        "summary": {
            "queries": len(queries),
            "errors": len(errors),
            "model_load_s": model_load_s,
            "retriever_init_s": retriever_init_s,
            "total_s": total_s,
            "dense_ms_total": round(sum(item["timings_ms"].get("dense_ms", 0.0) for item in results), 1),
            "rerank_ms_total": round(sum(item["timings_ms"].get("rerank_ms", 0.0) for item in results), 1),
            "public_retrieve_abstained": sum(1 for item in results if item["public_retrieve"].get("abstained")),
        },
        "errors": errors,
        "queries": results,
    }
    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"smoke_index_v2_{generation_id}_{stamp}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    for item in results:
        top = item["hits"][0] if item["hits"] else None
        status = "OK" if not item["error"] else "ERROR"
        timing = item["timings_ms"].get("total_ms", "-")
        if top:
            print(f"[{status}] {item['query']['id']} ({item['query']['kind']}) total={timing}ms "
                  f"top={top['evidence_chunk_id'][:16]} {top['source_id']} dense={top['dense_score']} rerank={top['rerank_score']}")
        else:
            print(f"[{status}] {item['query']['id']} ({item['query']['kind']}) total={timing}ms no hits")
    print(f"queries={len(queries)} errors={len(errors)} model_load={model_load_s}s retriever_init={retriever_init_s}s total={total_s}s")
    print(f"reports: {json_path.name}, {md_path.name}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
