"""Read-only loader and retrieval primitives for an existing index generation.

The driver intentionally does not call the demo's ``build_or_validate_index``:
that path may rebuild a generation when its identity check fails, and the
existing root generation carries an older extraction-adapter identity.  This
module loads the five generation files directly and reproduces the production
retrieval primitives (dense cosine, BM25Okapi, graph expansion, reranking)
with the generation's own retrieval contract.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import mpkg_rag_core as core

from .bm25_vendored import BM25Okapi, tokenize
from .config_v2 import load_generation


class ProductionIndex:
    def __init__(
        self,
        generation: Mapping[str, Any],
        *,
        embedding: Any | None = None,
        reranker: Any | None = None,
    ) -> None:
        self.generation = dict(generation)
        self.generation_id = generation["generation_id"]
        self.generation_dir = Path(generation["generation_dir"])
        self.identity = dict(generation["identity"])
        self.retrieval = dict(generation["retrieval"])
        self.minimum_authority = float(self.retrieval["minimum_authority"])
        self.minimum_dense_score = float(self.retrieval["minimum_dense_score"])
        self.embedding = embedding
        self.reranker = reranker
        self._load_files()

    @classmethod
    def load(
        cls,
        index_root: Path,
        generation_id: str | None = None,
        *,
        embedding: Any | None = None,
        reranker: Any | None = None,
    ) -> "ProductionIndex":
        generation = load_generation(Path(index_root), generation_id)
        return cls(generation, embedding=embedding, reranker=reranker)

    def _load_files(self) -> None:
        self.evidence_ids: list[str] = json.loads(
            (self.generation_dir / "id_map.json").read_text(encoding="utf-8")
        )
        if not isinstance(self.evidence_ids, list) or not self.evidence_ids:
            raise RuntimeError("id_map.json must be a non-empty list")
        self.vectors = np.load(self.generation_dir / "embeddings.npy", mmap_mode="r")
        if self.vectors.shape[0] != len(self.evidence_ids):
            raise RuntimeError("embeddings.npy row count does not match id_map.json")
        self.records: dict[str, dict[str, Any]] = {}
        with sqlite3.connect(self.generation_dir / "evidence.sqlite3") as db:
            for evidence_id, row_json in db.execute("SELECT evidence_id, row_json FROM evidence"):
                self.records[str(evidence_id)] = json.loads(row_json)
        missing = [evidence_id for evidence_id in self.evidence_ids if evidence_id not in self.records]
        if missing:
            raise RuntimeError(f"evidence.sqlite3 is missing {len(missing)} id_map rows")
        graph = json.loads((self.generation_dir / "graph.json").read_text(encoding="utf-8"))
        tables = graph.get("tables")
        if not isinstance(tables, Mapping):
            raise RuntimeError("graph.json is missing tables")
        self.graph_tables: dict[str, Any] = dict(tables)
        graph_evidence_rows = self.graph_tables.get("EvidenceChunk")
        if not isinstance(graph_evidence_rows, list):
            raise RuntimeError("graph tables are missing EvidenceChunk rows")
        self.evidence_by_id: dict[str, dict[str, Any]] = {}
        for row in graph_evidence_rows:
            key = row.get("evidence_chunk_id")
            if isinstance(key, str):
                self.evidence_by_id[key] = row
        uncovered = [evidence_id for evidence_id in self.evidence_ids if evidence_id not in self.evidence_by_id]
        if uncovered:
            raise RuntimeError(
                f"graph EvidenceChunk does not cover {len(uncovered)} id_map rows; "
                "the generation's id spaces disagree"
            )
        self.allowed_ids: set[str] = {
            evidence_id
            for evidence_id in self.evidence_ids
            if self._allowed(self.records[evidence_id])
        }
        self.bm25_ids: list[str] = [
            evidence_id for evidence_id in self.evidence_ids if evidence_id in self.allowed_ids
        ]
        corpus_tokens = [tokenize(self.records[evidence_id].get("text", "")) for evidence_id in self.bm25_ids]
        self.bm25 = BM25Okapi(corpus_tokens)
        self.graph_config = dict(self.retrieval)

    def _allowed(self, row: Mapping[str, Any]) -> bool:
        if row.get("factual_index_allowed") is not True:
            return False
        authority = row.get("authority_score")
        return (
            not isinstance(authority, bool)
            and isinstance(authority, (int, float))
            and math.isfinite(float(authority))
            and float(authority) >= self.minimum_authority
        )

    def attach_models(self, embedding: Any | None, reranker: Any | None) -> None:
        self.embedding = embedding
        self.reranker = reranker

    def _encode(self, text: str) -> np.ndarray:
        if self.embedding is None:
            raise RuntimeError("embedding backend is not attached")
        vector = np.asarray(
            self.embedding.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0],
            dtype=np.float32,
        )
        if vector.shape[0] != self.vectors.shape[1]:
            raise RuntimeError("query embedding dimension does not match the index")
        return vector

    def dense_search(self, query: str, k: int) -> list[dict[str, Any]]:
        if not self.evidence_ids:
            return []
        vector = self._encode(query)
        scores = np.asarray(self.vectors @ vector, dtype=np.float32)
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.evidence_ids[i]))[
            :k
        ]
        rows: list[dict[str, Any]] = []
        for rank, index in enumerate(order, 1):
            score = float(scores[index])
            if score < self.minimum_dense_score:
                continue
            evidence_id = self.evidence_ids[index]
            if evidence_id not in self.allowed_ids:
                continue
            row = dict(self.records[evidence_id])
            row.update(
                {
                    "branch": "dense",
                    "rank": rank,
                    "raw_score": score,
                    "evidence_chunk_id": evidence_id,
                }
            )
            rows.append(row)
        return rows

    def bm25_search(self, query: str, k: int) -> list[dict[str, Any]]:
        if not self.bm25_ids:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        rows: list[dict[str, Any]] = []
        for rank, index in enumerate(np.argsort(scores)[::-1][:k], 1):
            score = float(scores[index])
            if score <= 0:
                continue
            evidence_id = self.bm25_ids[int(index)]
            row = dict(self.records[evidence_id])
            row.update(
                {
                    "branch": "bm25",
                    "rank": rank,
                    "raw_score": score,
                    "evidence_chunk_id": evidence_id,
                }
            )
            rows.append(row)
        return rows

    def graph_search(
        self,
        target: str,
        k: int,
        post: str = "",
        category: str = "",
        query_signature: Mapping[str, Any] | None = None,
        graph_enabled: bool = True,
    ) -> list[dict[str, Any]]:
        if not graph_enabled:
            return []
        signature = dict(query_signature or {})
        seeds = self.dense_search(
            f"evidence addressing {target} {post}", int(self.retrieval["dense_top_k"])
        ) + self.bm25_search(f"{target} {post}", int(self.retrieval["bm25_top_k"]))
        return core.expand_graph_from_seeds(
            seeds,
            query_signature=signature,
            graph_tables=self.graph_tables,
            config=self.graph_config,
        )[:k]

    def rerank(self, post: str, texts: Sequence[str]) -> list[float]:
        if self.reranker is None:
            raise RuntimeError("reranker backend is not attached")
        if not texts:
            return []
        logits = self.reranker.predict([(post, str(text)) for text in texts], batch_size=16)
        return [float(value) for value in np.asarray(logits).reshape(-1).tolist()]

    def resolve_target_anchor_entities(self, target: str) -> dict[str, Any]:
        normalized = str(target or "").strip().casefold()
        entities = self.graph_tables.get("Entity") or []
        names: list[tuple[str, str]] = []
        for row in entities:
            if not isinstance(row, Mapping):
                continue
            entity_id = row.get("entity_id")
            canonical = str(row.get("canonical_name", "")).casefold().strip()
            if isinstance(entity_id, str) and canonical:
                names.append((entity_id, canonical))
        if not normalized:
            return {"entity_ids": [], "match_mode": "empty_target", "canonical_names": []}
        exact = [(entity_id, canonical) for entity_id, canonical in names if canonical == normalized]
        if exact:
            return _anchor_result(exact, "exact_canonical_name")
        containing = [
            (entity_id, canonical) for entity_id, canonical in names if normalized in canonical
        ]
        if containing:
            return _anchor_result(containing, "target_contained_in_canonical_name")
        contained = [
            (entity_id, canonical) for entity_id, canonical in names if canonical in normalized
        ]
        if contained:
            return _anchor_result(contained, "canonical_name_contained_in_target")
        return _anchor_token_subset(names, normalized)

    def stats(self) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        for row in self.records.values():
            status = str(row.get("status", "unknown"))
            statuses[status] = statuses.get(status, 0) + 1
        return {
            "generation_id": self.generation_id,
            "evidence_rows": len(self.evidence_ids),
            "allowed_rows": len(self.allowed_ids),
            "status_counts": statuses,
            "graph_evidence_chunks": len(self.evidence_by_id),
        }


def _anchor_result(matches: Iterable[tuple[str, str]], mode: str) -> dict[str, Any]:
    ordered = sorted(matches)
    return {
        "entity_ids": [entity_id for entity_id, _ in ordered],
        "match_mode": mode,
        "canonical_names": [canonical for _, canonical in ordered],
    }


_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)\s*$")


def _anchor_token_subset(
    names: Sequence[tuple[str, str]], normalized_target: str
) -> dict[str, Any]:
    target_tokens = set(tokenize(normalized_target))
    matches: list[tuple[str, str]] = []
    for entity_id, canonical in names:
        stripped = _PARENTHETICAL_RE.sub("", canonical).strip()
        name_tokens = tokenize(stripped)
        if not name_tokens or any(len(token) < 3 for token in name_tokens):
            continue
        if all(token in target_tokens for token in name_tokens):
            matches.append((entity_id, canonical))
    if matches:
        return _anchor_result(matches, "canonical_token_subset")
    return {"entity_ids": [], "match_mode": "no_canonical_match", "canonical_names": []}
