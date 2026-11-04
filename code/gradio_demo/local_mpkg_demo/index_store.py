"""Reusable disk-backed lexical, dense, and bounded graph retrieval index."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from work.mpkg_rag_core import (
    FACTUAL_SOURCE_TYPES,
    build_semantic_graph,
    expand_graph_from_seeds,
    reciprocal_rank_fusion,
    select_evidence,
    stable_id,
    validate_extraction,
)

from .config import DemoConfig, EvidenceRecord, IndexManifest, QueryInput, RetrievalResult


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    # Windows cannot open a directory as a file descriptor; NTFS journals the
    # rename metadata itself, so directory fsync is a POSIX-only step.
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("records", data.get("rows", data.get("chunks", data)))
    return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.casefold())


def _hash_vector(text: str, dimension: int) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dimension
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _manifest_from_json(data: Mapping[str, Any], path: Path) -> IndexManifest:
    return IndexManifest(identity=data.get("identity", {}), source_hashes=data.get("source_hashes", {}), model_revisions=data.get("model_revisions", {}), dimensions=data.get("dimensions", {}), row_counts=data.get("row_counts", {}), index_path=path, created_at=str(data.get("created_at", "")), readiness_warnings=tuple(data.get("readiness_warnings", ())))


def _extract_entity_keys(row: Mapping[str, Any]) -> set[str]:
    keys: set[str] = set()
    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for name, item in value.items():
                if name in {"entity_id", "entity_ids", "candidate_ids", "subject_entity_id", "object_entity_id"}:
                    walk(item, name)
                elif name in {"subject", "object", "entities", "mentions", "claims"}:
                    walk(item, name)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                walk(item, key)
        elif key and isinstance(value, str) and value.strip():
            keys.add(value.strip())
    walk(row)
    return keys


def _backend_name(backend: Any | None) -> str:
    if backend is None:
        return "unconfigured"
    return str(getattr(backend, "name", None) or getattr(backend, "model_name", None) or f"{type(backend).__module__}.{type(backend).__qualname__}")


def _encode_backend(backend: Any, text: str) -> np.ndarray:
    output = backend.encode([text], normalize_embeddings=True, convert_to_numpy=True) if hasattr(backend, "encode") else backend(text)
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], (list, tuple, np.ndarray)):
        output = output[0]
    vector = np.asarray(output, dtype=np.float32).reshape(-1)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise RuntimeError("embedding backend returned an invalid vector")
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm


def _validate_source_record(row: Mapping[str, Any], *, strict_spans: bool = True) -> None:
    """Reject malformed ledger rows before they can enter a persisted index."""
    text = str(row.get("text") or row.get("displayed_text") or "")
    if not row.get("evidence_id") or not row.get("document_uid") or not row.get("chunk_id"):
        raise ValueError("evidence ledger row is missing evidence/document/chunk identity")
    if "displayed_text" in row and str(row["displayed_text"]) != text:
        raise ValueError(f"evidence {row['evidence_id']} displayed text does not match chunk text")
    for key in ("span_start", "span_end", "start_char", "end_char"):
        if key in row and row[key] is not None and (isinstance(row[key], bool) or not isinstance(row[key], int)):
            raise ValueError(f"evidence {row['evidence_id']} has an invalid {key}")
    start, end = row.get("span_start", row.get("start_char")), row.get("span_end", row.get("end_char"))
    if start is not None and end is not None and (start < 0 or end < start or (strict_spans and end > len(text))):
        raise ValueError(f"evidence {row['evidence_id']} has an invalid span")
    if "status" in row and row["status"] not in {"accepted", "validated", "reviewed"}:
        raise ValueError(f"evidence {row['evidence_id']} has unsupported status")
    for key in ("displayed_text_sha256", "text_sha256"):
        if row.get(key) is not None and str(row[key]).casefold() != hashlib.sha256(text.encode("utf-8")).hexdigest():
            raise ValueError(f"evidence {row['evidence_id']} has a mismatched {key}")
    source_text = row.get("source_text")
    if row.get("source_text_sha256") is not None and source_text is not None:
        source_value = str(source_text)
        if str(row["source_text_sha256"]).casefold() != hashlib.sha256(source_value.encode("utf-8")).hexdigest():
            raise ValueError(f"evidence {row['evidence_id']} has a mismatched source_text_sha256")


def _adapt_normalized_extractions(
    extraction_rows: Iterable[Mapping[str, Any]],
    chunks_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    candidate_rows: Iterable[Mapping[str, Any]] = (),
    *, fixture: bool = False,
) -> list[dict[str, Any]]:
    """Adapt only normalized claims with explicit structured semantics."""
    candidate_rows = [candidate for candidate in candidate_rows if isinstance(candidate, Mapping) and candidate.get("candidate_id")]

    def candidate_belongs_to_chunk(candidate: Mapping[str, Any], chunk: Mapping[str, Any]) -> bool:
        """Bind a candidate to the exact canonical chunk that owns its span."""
        if str(candidate.get("chunk_id") or "") != str(chunk.get("chunk_id") or ""):
            return False
        for key in ("document_uid", "source_id"):
            supplied = candidate.get(key)
            if supplied is not None and str(supplied) != str(chunk.get(key)):
                return False
        return True

    def candidate_span_matches(candidate: Mapping[str, Any], chunk: Mapping[str, Any]) -> bool:
        surface = candidate.get("text") or candidate.get("label") or candidate.get("normalized_text")
        start, end = candidate.get("start"), candidate.get("end")
        return (
            isinstance(surface, str)
            and bool(surface)
            and isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and start >= 0
            and end > start
            and str(chunk.get("text", ""))[start:end] == surface
        )

    # Candidate IDs are local to an owning chunk/source/document/span.  Keep
    # the owner in the key so a reused ID in another chunk is not a conflict.
    candidate_by_context: dict[tuple[str, str, str, str, int, int], Mapping[str, Any]] = {}
    conflicting_contexts: set[tuple[str, str, str, str, int, int]] = set()
    chunks_by_chunk_id: dict[str, list[tuple[tuple[str, str], Mapping[str, Any]]]] = {}
    for chunk_key, chunk in chunks_by_key.items():
        chunks_by_chunk_id.setdefault(str(chunk.get("chunk_id") or ""), []).append((chunk_key, chunk))
    for candidate in candidate_rows:
        for (document_uid, chunk_id), chunk in chunks_by_chunk_id.get(str(candidate.get("chunk_id") or ""), ()):
            if not candidate_belongs_to_chunk(candidate, chunk) or not candidate_span_matches(candidate, chunk):
                continue
            key = (str(candidate["candidate_id"]), str(document_uid), str(chunk_id), str(chunk.get("source_id") or ""), int(candidate["start"]), int(candidate["end"]))
            previous = candidate_by_context.get(key)
            if previous is not None and dict(previous) != dict(candidate):
                conflicting_contexts.add(key)
            else:
                candidate_by_context[key] = candidate
    candidate_by_owner: dict[tuple[str, str, str], list[tuple[tuple[str, str, str, str, int, int], Mapping[str, Any]]]] = {}
    for key, candidate in candidate_by_context.items():
        candidate_by_owner.setdefault(key[1:4], []).append((key, candidate))

    def candidates_for_chunk(candidate_id: Any, chunk: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        owner = (str(chunk.get("document_uid") or ""), str(chunk.get("chunk_id") or ""), str(chunk.get("source_id") or ""))
        return [candidate for key, candidate in candidate_by_owner.get(owner, ()) if (candidate_id is None or key[0] == str(candidate_id)) and key not in conflicting_contexts]

    def candidate_entity_id(candidate: Mapping[str, Any]) -> str | None:
        label = candidate.get("label") or candidate.get("normalized_text") or candidate.get("text")
        if candidate and isinstance(label, str) and label.strip():
            return "ENT:" + stable_id("candidate-catalog", candidate.get("entity_type", "unknown"), " ".join(label.split()).casefold())
        return None

    def candidate_mention(candidate_id: Any, chunk: Mapping[str, Any], mention_id: str) -> dict[str, Any] | None:
        if not isinstance(candidate_id, str) or not candidate_id:
            return None
        owned = candidates_for_chunk(candidate_id, chunk)
        if len(owned) != 1:
            return None
        candidate = owned[0]
        entity_id = candidate_entity_id(candidate)
        surface = candidate.get("text") or candidate.get("label") or candidate.get("normalized_text")
        start, end = candidate.get("start"), candidate.get("end")
        if not candidate or not entity_id or not isinstance(surface, str) or not surface:
            return None
        if not candidate_belongs_to_chunk(candidate, chunk) or not candidate_span_matches(candidate, chunk):
            return None
        return {"mention_id": mention_id, "text": surface, "start": start, "end": end, "candidate_index": None, "canonical_name": str(candidate.get("label") or surface), "entity_id": entity_id, "entity_status": "canonical"}

    required = {"predicate", "polarity", "modality", "attribution", "evidence_stance", "model_confidence"}
    adapted: list[dict[str, Any]] = []
    for row in extraction_rows:
        if not isinstance(row, Mapping):
            continue
        source_id, chunk_id, document_uid = row.get("source_id"), row.get("chunk_id"), row.get("document_uid")
        if not isinstance(chunk_id, str) or not chunk_id:
            continue
        if document_uid is not None:
            chunk = chunks_by_key.get((str(document_uid), chunk_id))
        else:
            matches = [value for (source, key), value in chunks_by_key.items() if key == chunk_id and (not source_id or source == str(source_id))]
            chunk = matches[0] if len(matches) == 1 else None
        if chunk is None:
            continue
        if source_id is not None and str(source_id) != str(chunk.get("source_id")):
            continue
        if any(key in row and row.get(key) != chunk.get(key) for key in ("chunk_sha256", "page", "span_start", "span_end")):
            continue
        text = str(chunk.get("text") or "")
        claims = row.get("claims", [])
        if isinstance(claims, str):
            try:
                claims = json.loads(claims)
            except (TypeError, ValueError):
                continue
        if not isinstance(claims, list):
            continue
        promoted: list[dict[str, Any]] = []
        for index, normalized in enumerate(claims):
            if not isinstance(normalized, Mapping):
                continue
            if {"mentions", "subject", "predicate", "object"}.issubset(normalized):
                mentions = normalized.get("mentions")
                if not isinstance(mentions, list):
                    continue
                # Structured claims may only carry IDs derived from the
                # corpus-local candidate catalog.  A raw model-supplied ID
                # (for example ENT:forged) is never a graph endpoint.
                structured_is_owned = True
                for mention in mentions:
                    if not isinstance(mention, Mapping):
                        structured_is_owned = False
                        break
                    entity_id = mention.get("entity_id")
                    owners = [
                        candidate
                        for candidate in candidates_for_chunk(mention.get("candidate_id"), chunk)
                        if candidate_entity_id(candidate) == entity_id
                        and str(candidate.get("text") or candidate.get("label") or candidate.get("normalized_text") or "") == str(mention.get("text") or "")
                        and candidate.get("start") == mention.get("start")
                        and candidate.get("end") == mention.get("end")
                    ]
                    if not owners:
                        structured_is_owned = False
                        break
                if not structured_is_owned:
                    continue
                promoted.append(dict(normalized))
                continue
            if not required.issubset(normalized):
                continue
            subject_id = normalized.get("subject_candidate_id")
            if isinstance(normalized.get("subject"), Mapping):
                subject_id = normalized["subject"].get("candidate_id") or subject_id
            object_value = normalized.get("object")
            object_id = normalized.get("object_candidate_id")
            if isinstance(object_value, Mapping):
                object_id = object_value.get("candidate_id") or object_id
            subject_mention_id = f"candidate-{index}-subject"
            subject_mention = candidate_mention(subject_id, chunk, subject_mention_id)
            if subject_mention is None:
                continue
            mentions = [subject_mention]
            if object_id is not None:
                object_mention_id = f"candidate-{index}-object"
                object_mention = candidate_mention(object_id, chunk, object_mention_id)
                if object_mention is None:
                    continue
                mentions.append(object_mention)
                object_ref = {"mention_id": object_mention_id}
            elif isinstance(object_value, Mapping) and set(object_value) == {"value", "value_type"}:
                object_ref = dict(object_value)
            else:
                continue
            claim = {"claim_id": normalized.get("claim_id") or f"normalized-{index}", "mentions": mentions, "subject": {"mention_id": subject_mention_id}, "predicate": normalized["predicate"], "object": object_ref, "polarity": normalized["polarity"], "modality": normalized["modality"], "attribution": normalized["attribution"], "evidence_stance": normalized["evidence_stance"], "model_confidence": normalized["model_confidence"]}
            if "polarity_scope" in normalized:
                claim["polarity_scope"] = normalized["polarity_scope"]
            promoted.append(claim)
        if not promoted:
            continue
        payload = {"schema_version": "semantic-claims.v1", "claims": promoted}
        context = dict(chunk)
        context["source_id"] = str(chunk.get("source_id"))
        result = validate_extraction(payload, text, context)
        if result.get("accepted") and not result.get("quarantined") and not result.get("reviewed"):
            adapted.append(result)
    return adapted


def _simple_claim_graph(
    records: Iterable[Mapping[str, Any]],
    chunks_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    extraction_rows: Iterable[Mapping[str, Any]],
    candidate_rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Build the graph supported by the shipped simple-claim artifact.

    The semantic extraction artifact deliberately contains no claim semantics:
    a claim is text, a source span, and candidate IDs.  Candidate IDs are only
    usable after the candidate record owns the exact chunk and span.  Entity
    IDs are then deterministic catalog IDs derived from the candidate label;
    no model-provided subject/predicate/object values are promoted.
    """
    record_rows = [dict(row) for row in records]
    evidence_by_key = {
        (str(row.get("document_uid")), str(row.get("chunk_id"))): str(row["evidence_id"])
        for row in record_rows
    }
    graph: dict[str, Any] = {
        "schema_version": "simple-claims-candidate-graph.v1",
        "Document": [],
        "EvidenceChunk": [],
        "Mention": [],
        "Entity": [],
        "Claim": [],
        "Value": [],
        "edges": [],
        "quarantined": [],
        "reviewed": [],
    }
    documents: dict[str, dict[str, Any]] = {}
    entities: dict[str, dict[str, Any]] = {}
    mentions: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    edge_keys: dict[tuple[str, str, str, str], int] = {}

    def candidate_belongs_to_chunk(candidate: Mapping[str, Any], chunk: Mapping[str, Any]) -> bool:
        if str(candidate.get("chunk_id") or "") != str(chunk.get("chunk_id") or ""):
            return False
        for key in ("document_uid", "source_id"):
            supplied = candidate.get(key)
            if supplied is not None and str(supplied) != str(chunk.get(key)):
                return False
        return True

    def candidate_surface_and_span(candidate: Mapping[str, Any], chunk: Mapping[str, Any]) -> tuple[str, int, int] | None:
        surface = candidate.get("text") or candidate.get("label") or candidate.get("normalized_text")
        start, end = candidate.get("start"), candidate.get("end")
        if not isinstance(surface, str) or not surface or isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int):
            return None
        text = str(chunk.get("text") or "")
        if start < 0 or end <= start or end > len(text) or text[start:end] != surface:
            return None
        return surface, start, end

    # Candidate IDs are contextual.  Canonicalize each row to every exact
    # owner it can bind to, including the candidate span, and only quarantine
    # a conflicting duplicate within that same context.
    candidate_by_context: dict[tuple[str, str, str, str, int, int], Mapping[str, Any]] = {}
    conflicting_contexts: set[tuple[str, str, str, str, int, int]] = set()
    chunks_by_chunk_id: dict[str, list[tuple[tuple[str, str], Mapping[str, Any]]]] = {}
    for chunk_key, chunk in chunks_by_key.items():
        chunks_by_chunk_id.setdefault(str(chunk.get("chunk_id") or ""), []).append((chunk_key, chunk))
    for candidate in candidate_rows:
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get("candidate_id"), str):
            continue
        for (document_uid, chunk_id), chunk in chunks_by_chunk_id.get(str(candidate.get("chunk_id") or ""), ()):
            span = candidate_surface_and_span(candidate, chunk)
            if not candidate_belongs_to_chunk(candidate, chunk) or span is None:
                continue
            _, start, end = span
            key = (candidate["candidate_id"], str(document_uid), str(chunk_id), str(chunk.get("source_id") or ""), start, end)
            previous = candidate_by_context.get(key)
            if previous is not None and dict(previous) != dict(candidate):
                conflicting_contexts.add(key)
            else:
                candidate_by_context[key] = candidate

    candidate_by_owner: dict[tuple[str, str, str, str], list[tuple[tuple[str, str, str, str, int, int], Mapping[str, Any]]]] = {}
    for key, candidate in candidate_by_context.items():
        candidate_by_owner.setdefault(key[:4], []).append((key, candidate))

    def canonical_entity_id(candidate: Mapping[str, Any]) -> tuple[str, str, str]:
        label = str(candidate.get("label") or candidate.get("normalized_text") or candidate.get("text") or "")
        canonical_name = " ".join(label.split())
        normalized_name = canonical_name.casefold()
        raw_type = str(candidate.get("entity_type") or candidate.get("candidate_type") or "unknown").casefold()
        entity_type = raw_type if raw_type in {"organization", "person", "group", "place", "policy", "event", "concept", "dataset_target"} else "unknown"
        return "ENT:" + stable_id("catalog-entity", "corpus", entity_type, normalized_name), canonical_name, entity_type

    def owned_candidate(candidate_id: Any, chunk: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if not isinstance(candidate_id, str):
            return None
        owner = (candidate_id, str(chunk.get("document_uid") or ""), str(chunk.get("chunk_id") or ""), str(chunk.get("source_id") or ""))
        matches = [candidate for key, candidate in candidate_by_owner.get(owner, ()) if key not in conflicting_contexts]
        if len(matches) != 1:
            return None
        return matches[0]

    def verified_claim_span(claim: Mapping[str, Any], text: str) -> tuple[int, int] | None:
        start, end = claim.get("span_start"), claim.get("span_end")
        if isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int):
            return None
        if start < 0 or end <= start or end > len(text):
            return None
        return start, end

    def candidate_is_in_claim(candidate: Mapping[str, Any], claim_span: tuple[int, int] | None) -> bool:
        if claim_span is None:
            return False
        start, end = int(candidate["start"]), int(candidate["end"])
        return claim_span[0] <= start and end <= claim_span[1]

    def add_edge(source_id: str, edge_type: str, target_id: str, **extra: Any) -> None:
        # Repeated mentions of one entity in different claims remain distinct
        # evidence links; co-occurrence/document links still deduplicate.
        key = (source_id, edge_type, target_id, str(extra.get("mention_id") or ""))
        existing_index = edge_keys.get(key)
        if existing_index is not None:
            # A verified claim relation is more specific than a chunk-only
            # relation emitted by the bounded fallback pass.
            if extra.get("scope") == "claim" and edges[existing_index].get("scope") == "chunk":
                edges[existing_index] = {"source_id": source_id, "type": edge_type, "target_id": target_id, **extra}
            return
        edge_keys[key] = len(edges)
        edges.append({"source_id": source_id, "type": edge_type, "target_id": target_id, **extra})

    for row in record_rows:
        document_uid = str(row.get("document_uid") or "")
        evidence_id = str(row["evidence_id"])
        if document_uid not in documents:
            documents[document_uid] = {"document_uid": document_uid, "source_id": row.get("source_id"), "source_type": row.get("source_type"), "authority_score": row.get("authority_score"), "status": row.get("status", "accepted")}
        graph["EvidenceChunk"].append({**row, "evidence_chunk_id": evidence_id})
        add_edge(evidence_id, "from_document", document_uid, target_type="document")

    simple_claims = 0
    valid_mappings = 0
    chunk_entity_ids: dict[str, set[str]] = {}
    for extraction in extraction_rows:
        if not isinstance(extraction, Mapping):
            continue
        chunk_id = extraction.get("chunk_id")
        if not isinstance(chunk_id, str):
            continue
        document_uid = extraction.get("document_uid")
        matches = [
            ((doc, cid), chunk)
            for (doc, cid), chunk in chunks_by_key.items()
            if cid == chunk_id and (document_uid is None or str(doc) == str(document_uid))
        ]
        if len(matches) != 1:
            continue
        (doc_key, _), chunk = matches[0]
        evidence_id = evidence_by_key.get((str(doc_key), chunk_id))
        if evidence_id is None:
            continue
        if extraction.get("source_id") is not None and str(extraction["source_id"]) != str(chunk.get("source_id")):
            continue
        for provenance_key in ("chunk_sha256", "page"):
            if extraction.get(provenance_key) is not None and str(extraction[provenance_key]) != str(chunk.get(provenance_key)):
                evidence_id = None
                break
        if chunk.get("status") != "accepted":
            continue
        claims = extraction.get("claims")
        if not isinstance(claims, list):
            continue
        for claim_index, claim in enumerate(claims):
            if not isinstance(claim, Mapping) or not isinstance(claim.get("claim"), str) or not isinstance(claim.get("candidate_ids"), list):
                continue
            simple_claims += 1
            claim_span = verified_claim_span(claim, str(chunk.get("text") or ""))
            valid_for_claim: list[tuple[Mapping[str, Any], str, str]] = []
            for candidate_id in dict.fromkeys(value for value in claim["candidate_ids"] if isinstance(value, str)):
                candidate = owned_candidate(candidate_id, chunk)
                if candidate is None:
                    continue
                entity_id, canonical_name, entity_type = canonical_entity_id(candidate)
                valid_for_claim.append((candidate, entity_id, canonical_name))
                valid_mappings += 1
                chunk_entity_ids.setdefault(evidence_id, set()).add(entity_id)
                entities.setdefault(entity_id, {"entity_id": entity_id, "canonical_name": canonical_name, "entity_type": entity_type, "catalog_status": "active", "link_status": "linked", "status": "accepted"})
                mention_id = "MNT:" + stable_id("mention", evidence_id, claim_index, candidate.get("candidate_id"), candidate.get("start"), candidate.get("end"))
                mentions.setdefault(mention_id, {"mention_id": mention_id, "document_uid": str(doc_key), "evidence_chunk_id": evidence_id, "claim_index": claim_index, "candidate_id": candidate.get("candidate_id"), "text": candidate.get("text") or candidate.get("label"), "start": candidate.get("start"), "end": candidate.get("end"), "entity_id": entity_id, "entity_status": "canonical"})
                claim_scoped = candidate_is_in_claim(candidate, claim_span)
                edge_extra = {"target_type": "entity", "mention_id": mention_id, "candidate_id": candidate.get("candidate_id"), "scope": "claim" if claim_scoped else "chunk"}
                if claim_scoped:
                    edge_extra.update({"claim_text": claim["claim"], "span_start": claim_span[0], "span_end": claim_span[1]})
                add_edge(evidence_id, "mentions", entity_id, **edge_extra)
            entity_ids = sorted({item[1] for item in valid_for_claim})
            claim_scoped_entities = {entity_id for candidate, entity_id, _ in valid_for_claim if candidate_is_in_claim(candidate, claim_span)}
            for left_index, left in enumerate(entity_ids[:16]):
                for right in entity_ids[left_index + 1:16]:
                    claim_scoped = left in claim_scoped_entities and right in claim_scoped_entities
                    edge_extra = {"target_type": "entity", "scope": "claim" if claim_scoped else "chunk", "evidence_chunk_id": evidence_id, "claim_index": claim_index, "bounded": True}
                    if claim_scoped:
                        edge_extra.update({"claim_text": claim["claim"], "span_start": claim_span[0], "span_end": claim_span[1]})
                    add_edge(left, "co_occurs", right, **edge_extra)

    # Claims are not guaranteed to mention every candidate in a chunk. Preserve
    # only a bounded, undirected co-occurrence relation for the first 32
    # canonical entities in each accepted chunk (at most 128 pairs per chunk).
    for evidence_id, entity_ids in sorted(chunk_entity_ids.items()):
        bounded_ids = sorted(entity_ids)[:32]
        pair_count = 0
        for left_index, left in enumerate(bounded_ids):
            for right in bounded_ids[left_index + 1:]:
                if pair_count >= 128:
                    break
                add_edge(left, "co_occurs", right, target_type="entity", scope="chunk", evidence_chunk_id=evidence_id, bounded=True)
                pair_count += 1
            if pair_count >= 128:
                break

    graph["Document"] = list(documents.values())
    graph["Mention"] = list(mentions.values())
    graph["Entity"] = sorted(entities.values(), key=lambda row: row["entity_id"])
    graph["edges"] = edges
    graph["Edges"] = edges
    graph["_simple_claim_stats"] = {"simple_claims": simple_claims, "valid_mappings": valid_mappings}
    return graph, graph["_simple_claim_stats"]


def _active_generation(index_path: Path) -> Path | None:
    pointer = index_path / "CURRENT"
    try:
        generation_id = pointer.read_text(encoding="utf-8").strip()
        if not generation_id or Path(generation_id).name != generation_id:
            return None
        generation = index_path / "generations" / generation_id
        return generation if generation.is_dir() else None
    except OSError:
        return None


def _validate_index_files(index_path: Path, identity: Mapping[str, Any]) -> bool:
    try:
        manifest = json.loads((index_path / "index_manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            return False
        expected_generation = str(manifest.get("generation_id") or "")
        manifest_identity_value = manifest.get("identity")
        if not isinstance(manifest_identity_value, Mapping):
            return False
        source_hashes = manifest.get("source_hashes")
        identity_source_hashes = manifest_identity_value.get("source_hashes")
        if not isinstance(source_hashes, Mapping) or not isinstance(identity_source_hashes, Mapping) or dict(source_hashes) != dict(identity_source_hashes):
            return False
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value) for value in source_hashes.values()):
            return False
        if not all(isinstance(manifest.get(name), Mapping) for name in ("model_revisions", "dimensions", "row_counts")):
            return False
        manifest_identity = dict(manifest_identity_value)
        expected_identity = dict(identity)
        manifest_identity.pop("generation_id", None)
        expected_identity.pop("generation_id", None)
        if manifest_identity != expected_identity or (not index_path.name.startswith(".") and expected_generation != index_path.name):
            return False
        vectors = np.load(index_path / "embeddings.npy", mmap_mode="r")
        dimension = int(identity["embedding_dimension"])
        expected_count = int(manifest.get("row_counts", {}).get("evidence", -1))
        if vectors.dtype != np.dtype("float32") or vectors.ndim != 2 or vectors.shape != (expected_count, dimension):
            return False
        ids = json.loads((index_path / "id_map.json").read_text(encoding="utf-8"))
        if not isinstance(ids, list) or len(ids) != expected_count or len(set(ids)) != len(ids) or not all(isinstance(item, str) and item for item in ids):
            return False
        if ids is None:
            return False
        graph = json.loads((index_path / "graph.json").read_text(encoding="utf-8"))
        if not isinstance(graph, Mapping):
            return False
        required_graph_fields = {"generation_id", "evidence_ids", "node_ids", "adjacency", "edges", "tables"}
        if not required_graph_fields <= set(graph):
            return False
        adjacency = graph.get("adjacency", graph)
        tables = graph.get("tables", {})
        evidence_ids = graph.get("evidence_ids")
        node_ids = graph.get("node_ids")
        if not isinstance(evidence_ids, list) or not isinstance(node_ids, list) or evidence_ids != ids or len(node_ids) != len(set(node_ids)):
            return False
        graph_nodes = set(node_ids)
        if not set(ids) <= graph_nodes:
            return False
        table_nodes = set()
        for table, key in (("Document", "document_uid"), ("EvidenceChunk", "evidence_chunk_id"), ("Claim", "claim_id"), ("Entity", "entity_id"), ("Mention", "mention_id"), ("Value", "value_id")):
            rows = tables.get(table, []) if isinstance(tables, dict) else []
            if not isinstance(rows, list):
                return False
            table_nodes.update(str(row[key]) for row in rows if isinstance(row, dict) and isinstance(row.get(key), str))
        if not isinstance(adjacency, dict) or set(adjacency) != graph_nodes or any(not isinstance(neighbors, list) or any(item not in graph_nodes for item in neighbors) for neighbors in adjacency.values()):
            return False
        if set(graph.get("evidence_ids", ids)) != set(ids) or graph_nodes != table_nodes:
            return False
        edges = graph.get("edges", [])
        if not isinstance(edges, list) or any(not isinstance(edge, dict) or edge.get("source_id") not in graph_nodes or edge.get("target_id") not in graph_nodes for edge in edges):
            return False
        if int(manifest.get("row_counts", {}).get("graph_edges", -1)) != len(edges):
            return False
        if graph.get("generation_id") != expected_generation:
            return False
        with closing(sqlite3.connect(index_path / "evidence.sqlite3")) as db, db:
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "evidence" not in names or "evidence_fts" not in names:
                return False
            db_ids = [row[0] for row in db.execute("SELECT evidence_id FROM evidence ORDER BY rowid")]
            if db_ids != ids or db.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] != expected_count:
                return False
            fts_ids = [row[0] for row in db.execute("SELECT evidence_id FROM evidence_fts ORDER BY rowid")]
            if fts_ids != ids or db.execute("SELECT COUNT(*) FROM evidence_fts").fetchone()[0] != expected_count:
                return False
            metadata = db.execute("SELECT value FROM index_metadata WHERE key='generation_id'").fetchone()
            if metadata is None or metadata[0] != expected_generation:
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error):
        return False


def resolve_current_generation(config: Any) -> Path:
    """Resolve and structurally validate the generation selected by ``CURRENT``.

    The pointer is only a selector; readiness must never treat a non-empty
    pointer as proof that its immutable generation is usable.  Validate the
    manifest and every retrieval artifact, then compare the generation's
    source hashes with the currently configured semantic graph inputs so a
    stale index is reported before the first Run.
    """

    index_path = Path(config.index_path)
    pointer = index_path / "CURRENT"
    try:
        generation_id = pointer.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError("index CURRENT pointer is missing or unreadable") from exc
    if not generation_id or Path(generation_id).name != generation_id:
        raise RuntimeError("index CURRENT pointer is empty or unsafe")
    generation = index_path / "generations" / generation_id
    if not generation.is_dir():
        raise RuntimeError(f"index CURRENT points to missing generation {generation_id}")
    required = ("index_manifest.json", "embeddings.npy", "id_map.json", "graph.json", "evidence.sqlite3")
    missing = [name for name in required if not (generation / name).is_file()]
    if missing:
        raise RuntimeError("index generation is incomplete; missing " + ", ".join(missing))
    try:
        manifest = json.loads((generation / "index_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError("index generation manifest is missing or corrupt") from exc
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("identity"), Mapping):
        raise RuntimeError("index generation manifest has no valid identity")
    if str(manifest.get("generation_id") or "") != generation_id:
        raise RuntimeError("index generation manifest does not match CURRENT")
    identity = manifest["identity"]
    if not _validate_index_files(generation, identity):
        raise RuntimeError("index generation failed structural integrity validation")

    configured_sources = {
        "chunks": Path(config.chunks_path),
        "evidence_ledger": Path(config.evidence_ledger_path),
        "semantic_extractions": Path(config.semantic_extractions_path),
        "candidates": Path(config.candidates_path),
        "manifest": Path(config.source_manifest_path),
    }
    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, Mapping):
        raise RuntimeError("index generation manifest has no valid source hashes")
    required_sources = {"chunks", "evidence_ledger", "semantic_extractions", "manifest"}
    if not required_sources <= {str(name) for name in source_hashes}:
        raise RuntimeError("index generation manifest is missing required source hashes")
    stale = []
    for name, source in configured_sources.items():
        expected = source_hashes.get(name)
        if source.is_file():
            if expected is None or _sha256(source) != str(expected):
                stale.append(name)
        elif expected is not None:
            stale.append(name)
    if stale:
        raise RuntimeError("index generation is stale for source artifact(s): " + ", ".join(sorted(stale)))
    return generation


def build_or_validate_index(config: DemoConfig, progress: Callable[[str, float], None] | None = None, *, dense_encoder: Any | None = None, reranker: Any | None = None) -> IndexManifest:
    """Build the local index once, or return it when its source identity is unchanged."""
    index_path = Path(config.index_path)
    index_path.mkdir(parents=True, exist_ok=True)
    # A single CURRENT pointer selects a complete immutable generation.
    source_manifest_fallback = Path(config.source_manifest_path)
    fixture_marker = False
    if source_manifest_fallback.exists():
        try:
            fixture_marker = json.loads(source_manifest_fallback.read_text(encoding="utf-8")).get("identity", {}).get("stage_hash") == "tiny"
        except (OSError, ValueError, TypeError):
            fixture_marker = False
    source_paths = {"chunks": Path(config.chunks_path), "evidence_ledger": Path(config.evidence_ledger_path), "semantic_extractions": Path(config.semantic_extractions_path), "candidates": Path(config.candidates_path), "manifest": Path(config.source_manifest_path)}
    backend = dense_encoder or getattr(config, "embedding_backend", None)
    reranker_backend = reranker or getattr(config, "reranker_backend", None)
    if not fixture_marker and backend is None:
        raise RuntimeError("embedding backend is required; synthetic/hash vectors are test-only")
    if not fixture_marker and reranker_backend is None:
        raise RuntimeError("reranker backend is required; retrieval fails closed without one")
    missing = [str(path) for name, path in source_paths.items() if name != "candidates" and not path.exists()]
    if missing:
        raise FileNotFoundError("missing configured semantic graph artifacts: " + ", ".join(missing))
    source_hashes = {name: _sha256(path) for name, path in source_paths.items() if path.exists()}
    source_manifest = json.loads(source_paths["manifest"].read_text(encoding="utf-8"))
    if source_manifest.get("complete") is False:
        raise RuntimeError("source manifest is incomplete")
    expected_source_hashes = source_manifest.get("source_hashes")
    if isinstance(expected_source_hashes, Mapping):
        mismatches = [name for name, expected in expected_source_hashes.items() if name in source_hashes and str(expected) != source_hashes[name]]
        if mismatches:
            raise RuntimeError("source manifest binding mismatch: " + ", ".join(sorted(mismatches)))
    test_fixture = source_manifest.get("identity", {}).get("stage_hash") == "tiny"
    ledger_document = json.loads(source_paths["evidence_ledger"].read_text(encoding="utf-8"))
    manifest_stage_hash = source_manifest.get("stage_hash") or source_manifest.get("identity", {}).get("stage_hash") or source_manifest.get("identity", {}).get("payload", {}).get("stage_hash")
    ledger_stage_hash = ledger_document.get("stage_hash") if isinstance(ledger_document, Mapping) else None
    if backend is None and not test_fixture:
        raise RuntimeError("embedding backend is required; synthetic/hash vectors are test-only")
    if reranker_backend is None and not test_fixture:
        raise RuntimeError("reranker backend is required; retrieval fails closed without one")
    identity = {
        "evidence_source": "canonical_chunks.v2",
        "extraction_adapter": "simple-claims-candidate-graph.v1",
        "source_hashes": source_hashes,
        "embedding_backend": _backend_name(backend),
        "embedding_model": config.embedding_model,
        "embedding_revision": config.embedding_revision,
        "embedding_dimension": config.embedding_dimension,
        "reranker_backend": _backend_name(reranker_backend),
        "reranker_model": config.reranker_model,
        "reranker_revision": config.reranker_revision,
        "retrieval": config.retrieval_config,
    }
    active = _active_generation(index_path)
    if active is not None:
        try:
            active_manifest = json.loads((active / "index_manifest.json").read_text(encoding="utf-8"))
            if _validate_index_files(active, identity):
                return _manifest_from_json(active_manifest, index_path)
        except (OSError, ValueError, TypeError):
            pass
    if progress:
        progress("Loading canonical chunks", 0.1)
    # Every artifact is assembled in a sibling generation and only validated
    # files are swapped into the configured directory.
    generations = index_path / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    generation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + os.urandom(4).hex()
    temp = Path(tempfile.mkdtemp(prefix=f".{generation_id}.", dir=str(generations)))
    chunks = _read_rows(source_paths["chunks"])
    chunks_by_key = {(str(row.get("document_uid")), str(row.get("chunk_id"))): row for row in chunks}
    if len(chunks_by_key) != len(chunks):
        raise ValueError("canonical chunks contain duplicate document_uid/chunk_id identities")
    ledger = _read_rows(source_paths["evidence_ledger"])
    ledger_by_chunk = {str(row.get("chunk_id")): row for row in ledger if row.get("chunk_id") is not None}
    mismatch_fields = {key: 0 for key in ("document_uid", "source_id", "page", "chunk_sha256", "span_start", "span_end", "text")}
    for chunk in chunks:
        ledger_row = ledger_by_chunk.get(str(chunk.get("chunk_id")))
        for key in mismatch_fields:
            chunk_value = chunk.get(key, chunk.get("text") if key == "text" else None)
            if ledger_row is None:
                ledger_value = None
            elif key == "text":
                ledger_value = ledger_row.get("text") or ledger_row.get("displayed_text")
            elif key == "chunk_sha256":
                ledger_value = ledger_row.get("chunk_sha256") or ledger_row.get("text_sha256")
            else:
                ledger_value = ledger_row.get(key)
            if ledger_value != chunk_value:
                mismatch_fields[key] += 1
    readiness_warnings: list[dict[str, Any]] = []
    if len(ledger) != len(chunks) or any(mismatch_fields.values()) or (ledger_stage_hash and manifest_stage_hash and str(ledger_stage_hash) != str(manifest_stage_hash)):
        readiness_warnings.append({"code": "stale_evidence_ledger", "message": "evidence_ledger.json is stale or misaligned; canonical chunks.json is the retrieval/evidence source and ledger rows were not ingested", "mismatches": mismatch_fields, "ledger_rows": len(ledger), "chunk_rows": len(chunks)})
    records: list[dict[str, Any]] = []
    for chunk in chunks:
        if chunk.get("status") != "accepted":
            continue
        row = dict(chunk)
        row["evidence_id"] = str(chunk["chunk_id"])
        row["text"] = str(chunk.get("text") or "")
        row["displayed_text"] = row["text"]
        canonical_chunk_sha256 = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        supplied_chunk_sha256 = row.get("chunk_sha256")
        if supplied_chunk_sha256 is not None and str(supplied_chunk_sha256).casefold() != canonical_chunk_sha256:
            raise ValueError(f"evidence {row['evidence_id']} has a mismatched chunk_sha256")
        row["chunk_sha256"] = canonical_chunk_sha256
        if test_fixture:
            row.setdefault("factual_index_allowed", True)
            row.setdefault("authority_score", 1.0)
        elif "factual_index_allowed" not in row:
            # Older chunk exports omit this derived policy bit. Recompute it
            # from the versioned source type/status; an explicit false is
            # never overridden.
            row["factual_index_allowed"] = row.get("source_type") in FACTUAL_SOURCE_TYPES and row.get("status") in {"accepted", "validated"}
        if row.get("factual_index_allowed") is not True:
            raise ValueError(f"evidence {row['evidence_id']} factual_index_allowed is not true")
        authority = row.get("authority_score")
        if isinstance(authority, bool) or not isinstance(authority, (int, float)) or not math.isfinite(float(authority)) or float(authority) < config.minimum_authority:
            raise ValueError(f"evidence {row['evidence_id']} authority_score is below minimum_authority")
        # Chunk spans are source/page offsets and may include the page's
        # leading/trailing offsets beyond the extracted chunk text. Identity
        # equality with the chunk is authoritative; only ordering/type is
        # validated here.
        _validate_source_record(row, strict_spans=False)
        records.append(row)
    if len({str(row["evidence_id"]) for row in records}) != len(records):
        raise ValueError("canonical chunks contain duplicate evidence IDs")
    extraction_rows = _read_rows(source_paths["semantic_extractions"])
    candidate_rows = _read_rows(source_paths["candidates"]) if source_paths["candidates"].exists() else []
    simple_graph, simple_stats = _simple_claim_graph(records, chunks_by_key, extraction_rows, candidate_rows)
    if simple_stats["simple_claims"]:
        graph_tables = simple_graph
        if not test_fixture and simple_stats["valid_mappings"] == 0:
            raise RuntimeError("graph readiness failed: no valid canonical candidate mappings")
    else:
        validated_extractions = _adapt_normalized_extractions(extraction_rows, chunks_by_key, candidate_rows, fixture=test_fixture)
        graph_tables = build_semantic_graph(
            [{"evidence_id": str(row["evidence_id"]), **row} for row in records],
            validated_extractions,
        )
    core_evidence_to_ledger = {
        str(item.get("evidence_chunk_id")): str(row["evidence_id"])
        for item in graph_tables.get("EvidenceChunk", [])
        for row in records
        if item.get("document_uid") == row.get("document_uid") and item.get("chunk_id") == row.get("chunk_id")
    }
    for edge in graph_tables.get("edges", []):
        if edge.get("source_id") in core_evidence_to_ledger:
            edge["source_id"] = core_evidence_to_ledger[edge["source_id"]]
        if edge.get("target_id") in core_evidence_to_ledger:
            edge["target_id"] = core_evidence_to_ledger[edge["target_id"]]
    for mention in graph_tables.get("Mention", []):
        if mention.get("evidence_chunk_id") in core_evidence_to_ledger:
            mention["evidence_chunk_id"] = core_evidence_to_ledger[mention["evidence_chunk_id"]]
    graph_tables["EvidenceChunk"] = [{**row, "evidence_chunk_id": str(row["evidence_id"])} for row in records]
    existing_documents = {str(row.get("document_uid")) for row in graph_tables.get("Document", [])}
    for row in records:
        if str(row.get("document_uid")) not in existing_documents:
            graph_tables.setdefault("Document", []).append({"document_uid": str(row.get("document_uid")), **row})
    evidence_ids = [str(row["evidence_id"]) for row in records]
    # Every legitimate endpoint must come from a persisted node table. Literal
    # object endpoints are represented explicitly in a Value table.
    values = {}
    for edge in graph_tables.get("edges", []):
        if edge.get("target_type") == "value" and edge.get("target_id"):
            values[str(edge["target_id"])] = {"value_id": str(edge["target_id"]), "value": edge.get("value")}
    graph_tables["Value"] = list(values.values())
    graph_tables.pop("_simple_claim_stats", None)
    node_ids = set(evidence_ids)
    for table, key in (("Document", "document_uid"), ("EvidenceChunk", "evidence_chunk_id"), ("Claim", "claim_id"), ("Entity", "entity_id"), ("Mention", "mention_id"), ("Value", "value_id")):
        node_ids.update(str(row[key]) for row in graph_tables.get(table, []) if isinstance(row, dict) and isinstance(row.get(key), str))
    # Keep typed graph endpoints in the adjacency index. Earlier builds
    # retained only evidence-to-evidence edges, making canonical entity seeds
    # unreachable despite a non-empty graph edge count.
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for edge in graph_tables.get("edges", []):
        source_id, target_id = edge.get("source_id"), edge.get("target_id")
        if source_id in adjacency and target_id in adjacency:
            adjacency[source_id].add(target_id)
            adjacency[target_id].add(source_id)
    if test_fixture and not graph_tables.get("Claim"):
        owners: dict[str, set[str]] = {}
        for extraction in extraction_rows:
            key = (str(extraction.get("source_id", "")), str(extraction.get("chunk_id", "")))
            evidence_id = next((str(row["evidence_id"]) for row in records if str(row.get("source_id")) == key[0] and str(row.get("chunk_id")) == key[1]), None)
            if evidence_id is None:
                evidence_id = next((str(row["evidence_id"]) for row in records if str(extraction.get("document_uid")) == str(row.get("document_uid")) and str(extraction.get("chunk_id")) == str(row.get("chunk_id"))), None)
            if evidence_id:
                for entity in _extract_entity_keys(extraction):
                    owners.setdefault(entity, set()).add(evidence_id)
        for evidence_ids_for_entity in owners.values():
            for left in evidence_ids_for_entity:
                adjacency.setdefault(left, set()).update(evidence_ids_for_entity - {left})
    with closing(sqlite3.connect(temp / "evidence.sqlite3")) as db, db:
        db.execute("PRAGMA journal_mode = DELETE")
        db.execute("DROP TABLE IF EXISTS evidence")
        db.execute("DROP TABLE IF EXISTS evidence_fts")
        db.execute("CREATE TABLE evidence (evidence_id TEXT PRIMARY KEY, row_json TEXT NOT NULL)")
        db.execute("CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("INSERT INTO index_metadata VALUES ('generation_id', ?)", (generation_id,))
        try:
            db.execute("CREATE VIRTUAL TABLE evidence_fts USING fts5(evidence_id UNINDEXED, text)")
        except sqlite3.OperationalError:
            db.execute("CREATE TABLE evidence_fts (evidence_id TEXT PRIMARY KEY, text TEXT NOT NULL)")
        for row in records:
            evidence_id = str(row["evidence_id"])
            db.execute("INSERT INTO evidence VALUES (?, ?)", (evidence_id, json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)))
            db.execute("INSERT INTO evidence_fts VALUES (?, ?)", (evidence_id, row["text"]))
    if progress:
        progress("Writing memory-mapped embeddings", 0.55)
    dimension = config.embedding_dimension
    if backend is not None and test_fixture:
        vectors_for_dimension = []
        for row in records:
            value = _encode_backend(backend, row["text"])
            if not np.linalg.norm(value):
                value = next((_encode_backend(backend, token) for token in _tokens(row["text"]) if np.linalg.norm(_encode_backend(backend, token))), value)
            vectors_for_dimension.append(value)
        dimension = next((int(vector.size) for vector in vectors_for_dimension if np.linalg.norm(vector) > 0), dimension)
    if backend is not None and not test_fixture and records:
        actual_dimension = int(_encode_backend(backend, records[0]["text"]).size)
        if actual_dimension != dimension:
            raise RuntimeError(f"embedding backend dimension {actual_dimension} does not match configured index dimension {dimension}")
    vectors = np.lib.format.open_memmap(temp / "embeddings.npy", mode="w+", dtype="float32", shape=(len(records), dimension))
    if backend is not None and not test_fixture and records and hasattr(backend, "encode"):
        # SentenceTransformers is dramatically faster when the corpus is encoded
        # in batches. Encoding each chunk separately under-utilises Apple Silicon
        # and turns a demo index build into an hours-long operation.
        batch_size = 32
        for start in range(0, len(records), batch_size):
            texts = [str(row["text"]) for row in records[start : start + batch_size]]
            batch = np.asarray(
                backend.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=batch_size),
                dtype="float32",
            )
            if batch.shape != (len(texts), dimension):
                raise RuntimeError(f"embedding backend returned batch shape {batch.shape}, expected {(len(texts), dimension)}")
            if np.any(np.linalg.norm(batch, axis=1) == 0):
                raise RuntimeError("embedding backend returned a zero corpus vector")
            vectors[start : start + len(texts)] = batch
    else:
        for index, row in enumerate(records):
            value = _encode_backend(backend, row["text"]) if backend is not None else _hash_vector(row["text"], dimension)
            if backend is not None and test_fixture and not np.linalg.norm(value):
                value = next((_encode_backend(backend, token) for token in _tokens(row["text"]) if np.linalg.norm(_encode_backend(backend, token))), value)
            if backend is not None and not test_fixture and not np.linalg.norm(value):
                raise RuntimeError("embedding backend returned a zero corpus vector")
            vectors[index] = value
    vectors.flush()
    # Release the memmap before the generation directory is renamed; Windows
    # refuses to move a directory that contains an open mapped file.
    del vectors
    manifest = IndexManifest(
        identity={**identity, "embedding_dimension": dimension, "generation_id": generation_id},
        source_hashes=source_hashes,
        model_revisions={"embedding": config.embedding_revision, "reranker": config.reranker_revision, "generator": getattr(config, "generator_label", config.qwen_revision)},
        dimensions={"embedding": dimension},
        row_counts={
            "evidence": len(records),
            "documents": len({row.get("document_uid") for row in records}),
            "entities": len(graph_tables.get("Entity", [])),
            "mentions": len(graph_tables.get("Mention", [])),
            "graph_edges": len(graph_tables.get("edges", [])),
        },
        index_path=index_path,
        created_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        readiness_warnings=tuple(readiness_warnings),
    )
    identity = dict(manifest.identity)
    graph_payload = {"generation_id": generation_id, "evidence_ids": evidence_ids, "node_ids": sorted(node_ids), "adjacency": {key: sorted(value) for key, value in adjacency.items()}, "edges": graph_tables.get("edges", []), "tables": graph_tables}
    try:
        (temp / "id_map.json").write_text(json.dumps([str(row["evidence_id"]) for row in records], ensure_ascii=False), encoding="utf-8")
        (temp / "graph.json").write_text(json.dumps(graph_payload, sort_keys=True), encoding="utf-8")
        manifest_payload = {"generation_id": generation_id, "identity": identity, "source_hashes": source_hashes, "model_revisions": dict(manifest.model_revisions), "dimensions": dict(manifest.dimensions), "row_counts": dict(manifest.row_counts), "index_path": str(index_path), "created_at": manifest.created_at, "readiness_warnings": list(manifest.readiness_warnings)}
        (temp / "index_manifest.json").write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
        if not _validate_index_files(temp, identity):
            raise RuntimeError("built index failed integrity validation")
        final_generation = generations / generation_id
        os.replace(temp, final_generation)
        _fsync_directory(generations)
        pointer_temp = generations / f".CURRENT.{generation_id}"
        fd = os.open(pointer_temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            os.write(fd, generation_id.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(pointer_temp, index_path / "CURRENT")
        _fsync_directory(index_path)
    finally:
        if temp.exists():
            for path in temp.iterdir():
                path.unlink(missing_ok=True)
            temp.rmdir()
    if progress:
        progress("Index ready", 1.0)
    return manifest


class RetrievalCandidate:
    def __init__(self, evidence_id: str, text: str, routes: Iterable[str], scores: Mapping[str, float], metadata: Mapping[str, Any]):
        self.evidence_id, self.text, self.routes, self.scores, self.metadata = evidence_id, text, tuple(sorted(set(routes))), dict(scores), dict(metadata)
    def __getitem__(self, key: str) -> Any:
        if key == "evidence_chunk_id":
            return self.evidence_id
        if key == "routes":
            return self.routes
        if key in self.metadata:
            return self.metadata[key]
        if key in self.scores:
            return self.scores[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class LocalRetriever:
    def __init__(self, config: DemoConfig, *, dense_encoder: Any | None = None, reranker: Any | None = None) -> None:
        self.config = config
        self.dense_encoder, self.reranker = dense_encoder or getattr(config, "embedding_backend", None), reranker or getattr(config, "reranker_backend", None)
        build_or_validate_index(config, dense_encoder=self.dense_encoder, reranker=self.reranker)
        self.index_path = _active_generation(Path(config.index_path)) or Path(config.index_path)
        self.vectors = np.load(self.index_path / "embeddings.npy", mmap_mode="r")
        self.evidence_ids = json.loads((self.index_path / "id_map.json").read_text(encoding="utf-8"))
        self._records: dict[str, dict[str, Any]] = {}
        self._graph = json.loads((self.index_path / "graph.json").read_text(encoding="utf-8"))
        self._adjacency = self._graph.get("adjacency", self._graph)
        self._graph_tables = self._graph.get("tables", {})
        try:
            self._test_fixture = json.loads(Path(config.source_manifest_path).read_text(encoding="utf-8")).get("identity", {}).get("stage_hash") == "tiny"
        except (OSError, ValueError, TypeError):
            self._test_fixture = False
        with closing(sqlite3.connect(self.index_path / "evidence.sqlite3")) as db, db:
            self._records = {row[0]: json.loads(row[1]) for row in db.execute("SELECT evidence_id, row_json FROM evidence")}

    def _encode(self, text: str) -> np.ndarray:
        encoder = self.dense_encoder
        if encoder is None:
            raise RuntimeError("embedding backend is not configured; retrieval abstains")
        vector = _encode_backend(encoder, text)
        marker = None
        if not np.linalg.norm(vector) and Path(self.config.source_manifest_path).exists():
            try:
                marker = json.loads(Path(self.config.source_manifest_path).read_text()).get("identity", {}).get("stage_hash")
            except (OSError, ValueError, TypeError):
                marker = None
            if marker == "tiny":
                vectors = [_encode_backend(encoder, token) for token in _tokens(text)]
                nonzero = [item for item in vectors if np.linalg.norm(item)]
                if nonzero:
                    vector = np.sum(nonzero, axis=0)
                    vector /= np.linalg.norm(vector)
        if vector.size != self.vectors.shape[1]:
            raise RuntimeError(f"query embedding dimension {vector.size} does not match index dimension {self.vectors.shape[1]}")
        if not np.linalg.norm(vector) and marker != "tiny":
            raise RuntimeError("embedding backend returned a zero query vector")
        return vector

    def _dense_search(self, text: str, top_k: int) -> list[dict[str, Any]]:
        if not self.evidence_ids:
            return []
        query = self._encode(text)
        matrix = self.vectors
        # Corpus vectors are built once and queried directly from the memmap;
        # re-encoding documents at query time would defeat the disk-backed index.
        rows = matrix
        scores = np.asarray(rows @ query, dtype=np.float32)
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.evidence_ids[i]))[:top_k]
        return [{**self._records[self.evidence_ids[i]], "evidence_chunk_id": self.evidence_ids[i], "branch": "dense", "rank": rank, "raw_score": float(scores[i]), "raw_branch_score": float(scores[i])} for rank, i in enumerate(order, 1) if float(scores[i]) >= self.config.minimum_dense_score and self._allowed(self._records[self.evidence_ids[i]])]

    def _allowed(self, row: Mapping[str, Any]) -> bool:
        if row.get("factual_index_allowed") is not True:
            return False
        authority = row.get("authority_score")
        return (not isinstance(authority, bool) and isinstance(authority, (int, float)) and math.isfinite(float(authority)) and float(authority) >= self.config.minimum_authority)

    def _simple_graph_search(self, query: QueryInput, top_k: int) -> list[dict[str, Any]]:
        """Seed canonical entities and expand to accepted evidence in <=2 hops."""
        entity_ids = query.query_signature.get("entity_ids", query.query_signature.get("canonical_entity_ids", ()))
        if isinstance(entity_ids, str):
            entity_ids = (entity_ids,)
        if not isinstance(entity_ids, (list, tuple, set, frozenset)):
            return []
        known_entities = {
            str(row.get("entity_id"))
            for row in self._graph_tables.get("Entity", [])
            if isinstance(row, Mapping) and isinstance(row.get("entity_id"), str)
        }
        seeds = sorted({str(value) for value in entity_ids if isinstance(value, str) and value in known_entities})
        if not seeds:
            return []
        evidence_ids = set(self.evidence_ids)
        paths_by_evidence: dict[str, list[dict[str, Any]]] = {}
        for seed in seeds:
            queue: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [(seed, (seed,), ())]
            while queue:
                node, path, relations = queue.pop(0)
                hop = len(relations)
                if node in evidence_ids:
                    paths_by_evidence.setdefault(node, []).append({"nodes": list(path), "relations": list(relations), "hop": hop})
                if hop >= min(2, max(0, int(self.config.retrieval_max_hops))):
                    continue
                for neighbor in self._adjacency.get(node, ()):
                    if neighbor in path:
                        continue
                    relation = next((str(edge.get("type")) for edge in self._graph.get("edges", []) if {edge.get("source_id"), edge.get("target_id")} == {node, neighbor}), "related")
                    queue.append((neighbor, path + (neighbor,), relations + (relation,)))
        ranked = sorted(paths_by_evidence, key=lambda evidence_id: (min(path["hop"] for path in paths_by_evidence[evidence_id]), evidence_id))[:top_k]
        result: list[dict[str, Any]] = []
        for rank, evidence_id in enumerate(ranked, 1):
            row = self._records[evidence_id]
            if not self._allowed(row):
                continue
            paths = sorted(paths_by_evidence[evidence_id], key=lambda path: (path["hop"], path["nodes"], path["relations"]))
            hop = paths[0]["hop"]
            score = 1.0 / (hop + 1)
            if score <= self.config.minimum_graph_score:
                continue
            result.append({**row, "evidence_chunk_id": evidence_id, "branch": "graph", "rank": rank, "raw_score": score, "graph_score": score, "graph_paths": paths, "source_trace": {"query_signature": dict(query.query_signature), "seed_entity_ids": seeds, "max_hops": 2, "graph_paths": paths}})
        return result

    def _bm25_search(self, text: str, top_k: int) -> list[dict[str, Any]]:
        terms = _tokens(text)
        if not terms:
            return []
        query = " OR ".join('"' + term.replace('"', '') + '"' for term in terms)
        with closing(sqlite3.connect(self.index_path / "evidence.sqlite3")) as db, db:
            try:
                rows = db.execute("SELECT evidence_id, bm25(evidence_fts) FROM evidence_fts WHERE evidence_fts MATCH ? ORDER BY bm25(evidence_fts), evidence_id LIMIT ?", (query, top_k)).fetchall()
            except sqlite3.OperationalError:
                rows = []
                for evidence_id, record in self._records.items():
                    overlap = len(set(terms) & set(_tokens(record["text"])))
                    if overlap:
                        rows.append((evidence_id, -float(overlap)))
                rows.sort(key=lambda item: (item[1], item[0]))
                rows = rows[:top_k]
        scored = [(evidence_id, float(score)) for evidence_id, score in rows if self._allowed(self._records[evidence_id])]
        raw_values = [-score for _, score in scored]
        low, high = (min(raw_values), max(raw_values)) if raw_values else (0.0, 0.0)
        result = []
        for rank, ((evidence_id, score), raw) in enumerate(zip(scored, raw_values), 1):
            normalized = 1.0 if high == low else (raw - low) / (high - low)
            result.append({**self._records[evidence_id], "evidence_chunk_id": evidence_id, "branch": "bm25", "rank": rank, "raw_score": normalized, "raw_branch_score": score, "normalized_fts_score": normalized})
        return result

    def _graph_search(self, seeds: Iterable[dict[str, Any]], query: QueryInput, top_k: int) -> list[dict[str, Any]]:
        if self._graph_tables.get("EvidenceChunk") and self._graph_tables.get("Claim") and query.query_signature:
            graph_config = dict(self.config.retrieval_config)
            graph_config["max_hops"] = min(2, int(graph_config.get("max_hops", 2)))
            expanded = expand_graph_from_seeds(seeds, dict(query.query_signature), self._graph_tables, graph_config)
            return [row for row in expanded if self._allowed(row)][:top_k]
        if query.query_signature and self._graph_tables.get("Entity"):
            return self._simple_graph_search(query, top_k)
        if not self._test_fixture:
            return []
        distances: dict[str, int] = {}
        frontier = [str(hit["evidence_chunk_id"]) for hit in seeds]
        parents: dict[str, str] = {}
        seen = set(frontier)
        for hop in range(1, self.config.retrieval_max_hops + 1):
            next_frontier: list[str] = []
            for evidence_id in sorted(frontier):
                for neighbor in sorted(self._adjacency.get(evidence_id, [])):
                    if neighbor not in seen:
                        seen.add(neighbor); distances[neighbor] = hop; parents[neighbor] = evidence_id; next_frontier.append(neighbor)
            frontier = next_frontier
        ranked = sorted(distances.items(), key=lambda item: (item[1], item[0]))[:top_k]
        return [{**self._records[evidence_id], "evidence_chunk_id": evidence_id, "branch": "graph", "rank": rank, "raw_score": 1.0 / (hop + 1), "graph_score": 1.0 / (hop + 1), "graph_paths": [{"nodes": [parents.get(evidence_id, evidence_id), evidence_id], "hop": hop}], "source_trace": {"query_signature": dict(query.query_signature), "hop": hop}} for rank, (evidence_id, hop) in enumerate(ranked, 1) if 1.0 / (hop + 1) > self.config.minimum_graph_score and self._allowed(self._records[evidence_id])]

    def _rerank(self, query: str, candidates: list[dict[str, Any]]) -> dict[str, float]:
        if not candidates:
            return {}
        if self.reranker is not None:
            try:
                values = self.reranker.predict([(query, row.get("text", "")) for row in candidates]) if hasattr(self.reranker, "predict") else self.reranker(query, [row.get("text", "") for row in candidates])
                if isinstance(values, Mapping):
                    scores = {str(key): float(value) for key, value in values.items()}
                    return {key: (score if 0.0 <= score <= 1.0 else 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, score))))) for key, score in scores.items()}
                scores = {row["evidence_chunk_id"]: float(value) for row, value in zip(candidates, values)}
                # BGE reranker checkpoints commonly expose logits through
                # CrossEncoder.predict.  The selection contract is a
                # probability threshold, so normalize logits at this seam;
                # injected fixture probabilities in [0, 1] remain unchanged.
                return {key: (score if 0.0 <= score <= 1.0 else 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, score))))) for key, score in scores.items()}
            except Exception as exc:
                raise RuntimeError("reranker backend failed; retrieval refused") from exc
        return {}

    def retrieve(self, query: QueryInput, limit: int = 5) -> RetrievalResult:
        limit = max(1, min(int(limit), self.config.max_evidence))
        query_text = " ".join(part for part in (query.text, query.target, query.category) if part)
        bm25 = self._bm25_search(query_text, self.config.bm25_top_k)
        dense = self._dense_search(query_text, self.config.dense_top_k)
        graph = self._graph_search([*bm25, *dense], query, self.config.graph_top_k)
        fused = reciprocal_rank_fusion({"dense": dense, "bm25": bm25, "graph": graph}, {key: value for key, value in self.config.rrf_weights}, self.config.rrf_constant)
        for row in fused:
            source = self._records.get(row["evidence_chunk_id"], {})
            row.update(source)
            row["routes"] = tuple(trace["branch"] for trace in row.get("branch_traces", []))
        if self.reranker is None:
            candidates = tuple(RetrievalCandidate(row["evidence_chunk_id"], str(row.get("text", "")), row.get("routes", ()), {"rrf": float(row.get("rrf_score", 0.0))}, row) for row in fused)
            return RetrievalResult(query=query, evidence=(), candidates=candidates, trace={"bm25_hits": len(bm25), "dense_hits": len(dense), "graph_hits": len(graph), "candidate_count": len(fused)}, abstained=True, reason="reranker backend is not configured")
        rerank_candidates = fused[: max(1, min(self.config.rerank_top_k, 5))]
        selection = select_evidence(rerank_candidates, self._rerank(query_text, rerank_candidates), {"minimum_rerank_probability": self.config.minimum_rerank_probability, "max_evidence": min(limit, 5)})
        def scores(row: Mapping[str, Any]) -> dict[str, float]:
            values = {"rrf": float(row.get("rrf_score", 0.0)), "rerank": float(row.get("rerank_probability", 0.0))}
            for trace in row.get("branch_traces", []):
                branch = str(trace.get("branch"))
                values[branch] = max(values.get(branch, 0.0), float(trace.get("raw_score", trace.get("score", 0.0))))
            if row.get("graph_score") is not None:
                values["graph"] = float(row["graph_score"])
            return values
        evidence = tuple(
            EvidenceRecord.from_mapping(
                row,
                evidence_id=f"E{rank}",
                internal_evidence_id=str(row["evidence_chunk_id"]),
                routes=row.get("routes", ()),
                scores=scores(row),
            )
            for rank, row in enumerate(selection["selected"], 1)
        )
        candidates = tuple(RetrievalCandidate(row["evidence_chunk_id"], str(row.get("text", "")), row.get("routes", ()), scores(row), row) for row in fused)
        return RetrievalResult(query=query, evidence=evidence, candidates=candidates, trace={"bm25_hits": len(bm25), "dense_hits": len(dense), "graph_hits": len(graph), "candidate_count": len(fused), "weights": dict(self.config.rrf_weights), "rrf_constant": self.config.rrf_constant}, abstained=bool(selection["abstained"]), reason=selection["reason"])
