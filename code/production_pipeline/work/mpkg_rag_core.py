"""Manifest-driven corpus identity and provenance registry."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_SUFFIXES = {".pdf", ".html", ".htm"}
HIDDEN_NAMES = {".ds_store", "thumbs.db"}
FACTUAL_SOURCE_TYPES = {
    "official",
    "peer_reviewed",
    "court",
    "government",
    "un_agency",
    "ngo_report",
    "survey",
    "education",
}
AUTHORITY_SCORES = {
    "official": 1.00,
    "peer_reviewed": 0.95,
    "court": 0.95,
    "government": 0.92,
    "un_agency": 0.92,
    "ngo_report": 0.75,
    "survey": 0.75,
    "education": 0.72,
    "oral_history": 0.55,
    "news_archive": 0.50,
    "public_debate": 0.35,
    "harmful_examples": 0.20,
}
SOURCE_ID_PATTERN = re.compile(r"\bSRC\d+\b", re.IGNORECASE)
TYPE_SOURCE_LABELS = {
    "peer reviewed full text article": "peer_reviewed",
    "peer reviewed research": "peer_reviewed",
    "peer reviewed review": "peer_reviewed",
    "systematic review source page": "peer_reviewed",
    "narrative review source page": "peer_reviewed",
    "european journal of public health pmc": "peer_reviewed",
    "official guidance": "official",
    "policy guidance": "official",
    "global legal report page": "court",
    "un fact sheet": "un_agency",
    "un hate speech framework": "un_agency",
    "un human rights booklet": "un_agency",
    "un report": "un_agency",
    "education guidance": "education",
    "education survey report": "survey",
    "human rights report": "ngo_report",
    "population survey report": "survey",
}
ORGANISATION_SOURCE_LABELS = {
    "ohchr": "un_agency",
    "unesco": "un_agency",
    "who": "un_agency",
    "unicef": "un_agency",
    "ilo": "un_agency",
    "united nations": "un_agency",
    "united nations free equal": "un_agency",
    "human rights watch": "ngo_report",
    "amnesty international": "ngo_report",
    "ilga world": "ngo_report",
    "pubmed": "peer_reviewed",
    "pubmed central": "peer_reviewed",
}
SOURCE_TYPE_PRIORITY = (
    "court",
    "official",
    "government",
    "un_agency",
    "peer_reviewed",
    "ngo_report",
    "survey",
    "education",
    "oral_history",
    "news_archive",
    "public_debate",
    "harmful_examples",
)
RESERVED_DERIVED_FIELDS = frozenset(
    {
        "document_uid",
        "legacy_source_id",
        "legacy_source_ids",
        "relative_path",
        "relative_paths",
        "path",
        "content_sha256",
        "document_type",
        "source_type",
        "authority_score",
        "factual_index_allowed",
        "status",
        "status_reason",
        "quarantine_reasons",
        "provenance",
        "manifest_metadata",
        "manifest_sources",
        "validation_errors",
        "source_id",
        "sha256",
        "local_file",
        "filename",
    }
)


class CorpusValidationError(ValueError):
    """Raised when strict registry loading rejects validation errors."""


class SourceRegistry(list):
    """List of deduplicated source rows with non-row audit information."""

    file_records_before_deduplication: int
    audit_events: list[dict[str, Any]]
    validation_errors: list[dict[str, Any]]

    def __init__(
        self,
        rows: Iterable[dict[str, Any]] = (),
        *,
        file_records_before_deduplication: int = 0,
        audit_events: list[dict[str, Any]] | None = None,
        validation_errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(rows)
        self.file_records_before_deduplication = file_records_before_deduplication
        self.audit_events = audit_events or []
        self.validation_errors = validation_errors or []


def stable_id(*parts: object) -> str:
    """Return a deterministic content-safe identifier for ordered parts."""

    payload = json.dumps(
        [str(part) for part in parts],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _entity_normalize(value: object) -> str:
    """Normalize a surface form for deterministic catalog matching only."""

    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def _catalog_entity_id(namespace: str, canonical_name: str, entity_type: str) -> str:
    prefix = "TGT" if namespace == "target" else "ENT"
    return f"{prefix}:{stable_id('catalog-entity', namespace, entity_type, _entity_normalize(canonical_name))}"


ENTITY_TYPE_ALIASES = {
    "org": "organization",
    "organisation": "organization",
    "institution": "organization",
    "company": "organization",
    "agency": "organization",
    "person": "person",
    "human": "person",
    "individual": "person",
    "people": "group",
    "population": "group",
    "community": "group",
    "collective": "group",
    "location": "place",
    "country": "place",
    "city": "place",
    "law": "policy",
    "legislation": "policy",
    "regulation": "policy",
    "event": "event",
    "concept": "concept",
    "issue": "concept",
    "dataset_target": "dataset_target",
}


def _normalize_entity_type(value: object) -> str:
    normalized = _entity_normalize(value or "unknown") or "unknown"
    return ENTITY_TYPE_ALIASES.get(normalized, normalized if normalized in {"organization", "person", "group", "place", "policy", "event", "concept", "dataset_target", "unknown"} else "unknown")


def _entity_tokens(value: object) -> list[str]:
    return re.findall(r"[\w]+", _entity_normalize(value))


def _acronym_key(value: object) -> str | None:
    surface = unicodedata.normalize("NFKC", str(value or "")).strip()
    tokens = _entity_tokens(surface)
    if not tokens:
        return None
    if len(tokens) == 1 and 2 <= len(tokens[0]) <= 8 and (surface.isupper() or len(tokens[0]) <= 4):
        return tokens[0].casefold()
    if 2 <= len(tokens) <= 8 and all(len(token) == 1 for token in tokens) and any(not char.isalnum() and not char.isspace() for char in surface):
        return "".join(tokens).casefold()
    return None


def _is_conservative_acronym(left: str, right: str) -> bool:
    acronym, right_tokens = _acronym_key(left), _entity_tokens(right)
    if acronym is None or len(right_tokens) < 2:
        return False
    stopwords = {"a", "an", "and", "for", "of", "the", "to"}
    initials = "".join(token[0] for token in right_tokens if token not in stopwords)
    return acronym.casefold() == initials.casefold()


def build_entity_catalog(
    corpus_rows: Iterable[dict[str, Any]] = (),
    dataset_targets: Iterable[str] = (),
    reviewed_manifest_organizations: Iterable[str] = (),
    mention_spans: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Build a closed, deterministic corpus-local entity catalog.

    Model output never supplies IDs to this function.  IDs are derived from a
    normalized label, entity type, and namespace; aliases are retained for
    candidate generation and are never used as graph identity.
    """

    candidates: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add(label: object, entity_type: object, namespace: str, provenance: dict[str, Any] | None = None) -> None:
        canonical = " ".join(str(label or "").split())
        normalized = _entity_normalize(canonical)
        if not normalized:
            return
        kind = _normalize_entity_type(entity_type)
        key = (namespace, kind, normalized)
        row = candidates.setdefault(key, {"namespace": namespace, "entity_type": kind, "labels": set(), "provenance": []})
        row["labels"].add(canonical)
        if provenance and provenance not in row["provenance"]:
            row["provenance"].append(provenance)

    for target in dataset_targets or ():
        add(target, "dataset_target", "target", {"kind": "dataset_target"})
    for organization in reviewed_manifest_organizations or ():
        add(organization, "organization", "corpus", {"kind": "reviewed_manifest_organization"})
    for row in corpus_rows or ():
        if not isinstance(row, dict):
            continue
        metadata = row.get("manifest_metadata") if isinstance(row.get("manifest_metadata"), dict) else row
        organization = metadata.get("organisation") or metadata.get("organization")
        if organization:
            add(organization, "organization", "corpus", {"kind": "manifest", "document_uid": row.get("document_uid")})
    for mention in mention_spans or ():
        if not isinstance(mention, dict):
            continue
        text = mention.get("text")
        if isinstance(text, str) and text.strip():
            add(text, mention.get("entity_type", mention.get("mention_type", "unknown")), "corpus", {"kind": "validated_mention", "mention_id": mention.get("mention_id"), "document_uid": mention.get("document_uid"), "chunk_id": mention.get("chunk_id"), "start": mention.get("start"), "end": mention.get("end"), "text": text})

    nodes = sorted(candidates)
    parent = {node: node for node in nodes}
    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)
    # Exact aliases are safe within a namespace/type-compatible catalog. An
    # acronym expansion is deliberately stricter: only a one-to-one pairing
    # may be merged, so ABC cannot absorb two organizations sharing initials
    # and cannot create a transitive alias chain.
    for index, left in enumerate(nodes):
        for right in nodes[index + 1:]:
            if left[0] == right[0] and (left[1] == right[1] or "unknown" in {left[1], right[1]}):
                left_labels, right_labels = candidates[left]["labels"], candidates[right]["labels"]
                if any(_entity_normalize(a) == _entity_normalize(b) for a in left_labels for b in right_labels):
                    union(left, right)
    ambiguous_acronym_nodes: set[tuple[str, str, str]] = set()
    for acronym in nodes:
        acronym_labels = candidates[acronym]["labels"]
        if not any(_acronym_key(label) for label in acronym_labels):
            continue
        compatible = []
        compatible_roots = set()
        for expansion in nodes:
            if expansion == acronym or expansion[0] != acronym[0] or not (expansion[1] == acronym[1] or "unknown" in {expansion[1], acronym[1]}) or not any(len(_entity_tokens(label)) >= 2 for label in candidates[expansion]["labels"]):
                continue
            if any(_is_conservative_acronym(a, b) for a in acronym_labels for b in candidates[expansion]["labels"]):
                expansion_root = find(expansion)
                if expansion_root not in compatible_roots:
                    compatible.append(expansion)
                    compatible_roots.add(expansion_root)
        if len(compatible) != 1:
            if compatible:
                ambiguous_acronym_nodes.add(acronym)
            continue
        expansion = compatible[0]
        reverse_roots = set()
        for other in nodes:
            if other == expansion or other[0] != expansion[0] or not (other[1] == expansion[1] or "unknown" in {other[1], expansion[1]}) or not any(_acronym_key(label) for label in candidates[other]["labels"]):
                continue
            if any(_is_conservative_acronym(a, b) for a in candidates[other]["labels"] for b in candidates[expansion]["labels"]):
                reverse_roots.add(find(other))
        if len(reverse_roots) == 1:
            union(acronym, expansion)
        else:
            ambiguous_acronym_nodes.add(acronym)

    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        grouped[find(node)].append(candidates[node])
    rows = []
    ambiguous_roots = {find(node) for node in ambiguous_acronym_nodes}
    for root, group in grouped.items():
        labels = sorted({label for item in group for label in item["labels"]}, key=lambda value: (-len(_entity_tokens(value)), -len(value), _entity_normalize(value), value))
        canonical = labels[0]
        namespace = group[0]["namespace"]
        known_types = sorted({item["entity_type"] for item in group if item["entity_type"] != "unknown"})
        kind = known_types[0] if known_types else "unknown"
        ambiguous = root in ambiguous_roots
        rows.append({"entity_id": _catalog_entity_id(namespace, canonical, kind), "canonical_name": canonical, "aliases": sorted(labels, key=lambda value: (_entity_normalize(value), value)), "entity_type": kind, "namespace": namespace, "external_refs": [], "catalog_status": "ambiguous" if ambiguous else "active", "link_status": "ambiguous" if ambiguous else "linked", "retrieval_allowed": not ambiguous, "factual_identity_allowed": namespace != "target", "provenance": sorted({json.dumps(value, ensure_ascii=False, sort_keys=True, default=str): value for item in group for value in item["provenance"]}.values(), key=lambda value: json.dumps(value, sort_keys=True, default=str))})
    alias_to_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        for alias in row["aliases"]:
            alias_to_ids[(row["namespace"], _entity_normalize(alias))].add(row["entity_id"])
    for row in rows:
        collisions = [alias for alias in row["aliases"] if len(alias_to_ids[(row["namespace"], _entity_normalize(alias))]) > 1]
        if collisions:
            row["catalog_status"] = "ambiguous"; row["link_status"] = "ambiguous"; row["retrieval_allowed"] = False; row["ambiguity_aliases"] = sorted(collisions)
        elif row.get("link_status") != "ambiguous":
            row["link_status"] = "linked"
    rows = sorted(rows, key=lambda row: row["entity_id"])
    revision = "entity-catalog.v4.namespace-aware-acronym-normalization"
    catalog_hash = stable_id(revision, json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str))
    return {"catalog_revision": revision, "catalog_hash": catalog_hash, "entities": rows}


def build_entity_candidates(
    mention_text: str,
    catalog: dict[str, Any],
    max_candidates: int = 8,
    *,
    namespace_filter: str | Iterable[str] | None = None,
    namespace_preference: str | None = None,
    allow_target_fallback: bool = False,
    factual_only: bool = False,
) -> dict[str, Any]:
    """Return a deterministic, hashed candidate list for one validated mention."""

    if not isinstance(catalog, dict) or not isinstance(catalog.get("entities"), list):
        raise TypeError("entity catalog must contain an entities list")
    needle = _entity_normalize(mention_text)
    if namespace_filter is None:
        allowed_namespaces = None
        namespace_filter_key: list[str] | None = None
    elif isinstance(namespace_filter, str):
        allowed_namespaces = {namespace_filter}
        namespace_filter_key = [namespace_filter]
    else:
        namespace_filter_key = sorted({str(value) for value in namespace_filter})
        allowed_namespaces = set(namespace_filter_key)
    rows = []
    for entity in catalog["entities"]:
        if not isinstance(entity, dict) or entity.get("retrieval_allowed") is not True:
            continue
        if allowed_namespaces is not None and entity.get("namespace") not in allowed_namespaces:
            continue
        if factual_only and not (entity.get("namespace") == "corpus" and entity.get("factual_identity_allowed") is True):
            continue
        aliases = entity.get("aliases", [])
        if needle and any(_entity_normalize(alias) == needle for alias in aliases if isinstance(alias, str)):
            rows.append({
                "entity_id": entity["entity_id"],
                "canonical_name": entity["canonical_name"],
                "entity_type": entity["entity_type"],
                "namespace": entity["namespace"],
                "catalog_revision": catalog.get("catalog_revision"),
            })
    if namespace_preference is not None:
        preferred = [row for row in rows if row["namespace"] == namespace_preference]
        if preferred:
            rows = preferred
        elif allow_target_fallback:
            rows = [row for row in rows if row["namespace"] == "target"]
        else:
            rows = []
    rows.sort(key=lambda row: (row["namespace"] != "corpus", row["namespace"] != "target", row["canonical_name"].casefold(), row["entity_id"]))
    rows = rows[:max_candidates]
    policy = {"max_candidates": int(max_candidates), "namespace_filter": namespace_filter_key, "namespace_preference": namespace_preference, "allow_target_fallback": bool(allow_target_fallback), "factual_only": bool(factual_only)}
    candidate_hash = stable_id("entity-candidates.v3", catalog.get("catalog_hash"), needle, json.dumps(policy, sort_keys=True, ensure_ascii=False), json.dumps(rows, sort_keys=True, ensure_ascii=False))
    return {
        "mention_text": mention_text,
        "normalized_mention": needle,
        "catalog_revision": catalog.get("catalog_revision"),
        "catalog_hash": catalog.get("catalog_hash"),
        "max_candidates": int(max_candidates),
        "namespace_filter": namespace_filter_key,
        "namespace_preference": namespace_preference,
        "allow_target_fallback": bool(allow_target_fallback),
        "factual_only": bool(factual_only),
        "candidates": rows,
        "candidate_set_hash": candidate_hash,
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Expected a JSON array of objects: {path}")
    return value


def _normalized(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).replace("\\", "/")
    return re.sub(r"/+", "/", text).strip().casefold()


def _normalized_path(value: object) -> str:
    text = _normalized(value)
    while text.startswith("./"):
        text = text[2:]
    return text.rstrip("/")


def _basename(value: object) -> str:
    return _normalized_path(value).rsplit("/", 1)[-1]


def _normalized_label(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _source_id(value: object) -> str | None:
    match = SOURCE_ID_PATTERN.search(str(value or ""))
    return match.group(0).upper() if match else None


def _record_paths(record: dict[str, Any]) -> tuple[set[str], str | None]:
    paths: set[str] = set()
    for key in ("local_file", "path", "filename", "relative_path"):
        value = record.get(key)
        if value:
            normalized = _normalized_path(value)
            paths.add(normalized)
    basename = next(iter({_basename(path) for path in paths}), None)
    return paths, basename


def _path_matches_file(record: dict[str, Any], relative_path: str) -> bool:
    paths, _ = _record_paths(record)
    relative = _normalized_path(relative_path)
    return relative in paths or any(path.endswith("/" + relative) for path in paths)


def _hash_matches_file(record: dict[str, Any], content_sha256: str) -> bool:
    expected_hash = str(record.get("sha256") or record.get("content_sha256") or "").casefold()
    return bool(expected_hash) and expected_hash == content_sha256


def _unique_values(records: Iterable[dict[str, Any]], key: str) -> list[Any]:
    values: list[Any] = []
    seen: set[str] = set()
    for record in records:
        value = record.get(key)
        if value in (None, "", []):
            continue
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if marker not in seen:
            values.append(value)
            seen.add(marker)
    return values


def _field_values(records: Iterable[dict[str, Any]], key: str) -> list[Any]:
    values: list[Any] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, list):
            values.extend(value)
        elif value not in (None, ""):
            values.append(value)
    return values


def _merged_metadata(records: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in sorted({key for _, record in records for key in record}):
        values = _unique_values((record for _, record in records), key)
        if not values:
            continue
        result[key] = values[0] if len(values) == 1 else values
    return result


def _classify_source_type(records: list[tuple[str, dict[str, Any]]]) -> str:
    record_values = [record for _, record in records]
    type_labels = {
        _normalized_label(value)
        for value in _field_values(record_values, "type")
        if _normalized_label(value)
    }
    organisation_labels = {
        _normalized_label(value)
        for value in _field_values(record_values, "organisation")
        if _normalized_label(value)
    }
    has_doi = any(record.get("doi") or record.get("openalex") for _, record in records)

    candidates = {
        TYPE_SOURCE_LABELS[label]
        for label in type_labels
        if label in TYPE_SOURCE_LABELS
    }
    candidates.update(
        ORGANISATION_SOURCE_LABELS[label]
        for label in organisation_labels
        if label in ORGANISATION_SOURCE_LABELS
    )
    if has_doi:
        candidates.add("peer_reviewed")
    for source_type in SOURCE_TYPE_PRIORITY:
        if source_type in candidates:
            return source_type
    return "unknown"


def _source_ids(
    relative_paths: Iterable[str], records: list[tuple[str, dict[str, Any]]]
) -> list[str]:
    values: set[str] = set()
    for _, record in records:
        value = _source_id(record.get("source_id"))
        if value:
            values.add(value)
    for relative_path in relative_paths:
        value = _source_id(relative_path)
        if value:
            values.add(value)
    return sorted(values)


def _record_validation_errors(
    relative_path: str,
    actual_sha256: str,
    records: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for origin, record in records:
        expected = str(record.get("sha256") or record.get("content_sha256") or "").casefold()
        if expected and expected != actual_sha256:
            errors.append(
                {
                    "event": "hash_mismatch",
                    "relative_path": relative_path,
                    "manifest_source": origin,
                    "source_id": _source_id(record.get("source_id")),
                    "expected_sha256": expected,
                    "actual_sha256": actual_sha256,
                }
            )
    return errors


def load_source_registry(corpus_root: Path, *, strict: bool = False) -> SourceRegistry:
    """Load, validate, merge, and content-deduplicate a corpus registry.

    Supported files on disk define the accounting baseline. The three JSON
    inventories are metadata overlays; no source ID is used as a document
    identity. By default validation errors are retained on the result so a
    corpus audit can complete. ``strict=True`` raises after the full audit.
    """

    corpus_root = Path(corpus_root)
    documents_root = corpus_root / "documents"
    if not documents_root.is_dir():
        raise FileNotFoundError(f"Expected corpus documents directory: {documents_root}")

    sources = [
        ("source_manifest", _load_records(corpus_root / "source_manifest.json")),
        ("added_openalex_sources", _load_records(corpus_root / "added_openalex_sources.json")),
        ("local_file_inventory", _load_records(corpus_root / "local_file_inventory.json")),
    ]
    overlays = [(origin, record) for origin, records in sources for record in records]

    audit_events: list[dict[str, Any]] = []
    file_records: list[dict[str, Any]] = []
    for path in sorted(documents_root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(corpus_root).as_posix()
        if path.name.casefold() in HIDDEN_NAMES or path.name.startswith("."):
            audit_events.append({"event": "hidden_metadata_file", "relative_path": relative_path})
            continue
        if path.suffix.casefold() not in SUPPORTED_SUFFIXES:
            audit_events.append({"event": "unsupported_file", "relative_path": relative_path})
            continue

        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        file_records.append(
            {
                "relative_path": relative_path,
                "path": str(path),
                "content_sha256": actual_sha256,
                "document_type": "html" if path.suffix.casefold() in {".html", ".htm"} else "pdf",
                "matched_manifest_records": [],
                "validation_errors": [],
            }
        )

    supported_by_basename: dict[str, list[str]] = defaultdict(list)
    for file_record in file_records:
        supported_by_basename[_basename(file_record["relative_path"])].append(
            file_record["relative_path"]
        )

    for file_record in file_records:
        relative_path = file_record["relative_path"]
        content_sha256 = file_record["content_sha256"]
        basename = _basename(relative_path)
        matched: list[tuple[str, dict[str, Any]]] = []
        basename_candidates: list[tuple[str, dict[str, Any]]] = []
        for origin, record in overlays:
            if _path_matches_file(record, relative_path) or _hash_matches_file(record, content_sha256):
                matched.append((origin, record))
            elif _record_paths(record)[1] == basename:
                basename_candidates.append((origin, record))

        if basename_candidates:
            basename_paths = supported_by_basename[basename]
            if len(basename_paths) == 1 and len(basename_candidates) == 1:
                matched.extend(basename_candidates)
            else:
                ambiguity = {
                    "event": "ambiguous_basename",
                    "relative_path": relative_path,
                    "basename": basename,
                    "affected_paths": sorted(basename_paths),
                    "candidate_manifest_sources": sorted(
                        {origin for origin, _ in basename_candidates}
                    ),
                    "reason": "basename-only overlay is not unique",
                }
                file_record["validation_errors"].append(ambiguity)
                audit_events.append(ambiguity.copy())

        file_record["matched_manifest_records"] = matched
        file_record["validation_errors"].extend(
            _record_validation_errors(relative_path, content_sha256, matched)
        )

    validation_errors = [
        error for file_record in file_records for error in file_record["validation_errors"]
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for file_record in file_records:
        groups[file_record["content_sha256"]].append(file_record)

    rows: list[dict[str, Any]] = []
    for content_sha256, group in sorted(groups.items()):
        relative_paths = sorted(file_record["relative_path"] for file_record in group)
        matched = [
            item
            for file_record in group
            for item in file_record["matched_manifest_records"]
        ]
        metadata = _merged_metadata(matched)
        source_type = _classify_source_type(matched)
        source_ids = _source_ids(relative_paths, matched)
        row_errors = [
            error
            for file_record in group
            for error in file_record["validation_errors"]
        ]
        quarantine_reasons = sorted(
            {
                "manifest_hash_mismatch"
                if error["event"] == "hash_mismatch"
                else "ambiguous_basename"
                if error["event"] == "ambiguous_basename"
                else str(error["event"])
                for error in row_errors
            }
        )
        quarantined = bool(quarantine_reasons)
        provenance = [
            {
                "relative_path": file_record["relative_path"],
                "legacy_source_ids": _source_ids(
                    [file_record["relative_path"]], file_record["matched_manifest_records"]
                ),
                "manifest_sources": sorted({origin for origin, _ in file_record["matched_manifest_records"]}),
            }
            for file_record in group
        ]
        row: dict[str, Any] = {
            "document_uid": stable_id("document", content_sha256),
            "legacy_source_id": source_ids[0] if source_ids else None,
            "legacy_source_ids": source_ids,
            "relative_path": relative_paths[0],
            "relative_paths": relative_paths,
            "path": str(corpus_root / relative_paths[0]),
            "content_sha256": content_sha256,
            "document_type": group[0]["document_type"],
            "source_type": source_type,
            "authority_score": AUTHORITY_SCORES.get(source_type),
            "factual_index_allowed": source_type in FACTUAL_SOURCE_TYPES and not quarantined,
            "status": "quarantined" if quarantined else "accepted",
            "status_reason": quarantine_reasons[0] if len(quarantine_reasons) == 1 else (
                "multiple_validation_errors" if quarantine_reasons else "validated"
            ),
            "quarantine_reasons": quarantine_reasons,
            "provenance": provenance,
            "manifest_metadata": metadata,
            "manifest_sources": sorted({origin for origin, _ in matched}),
            "validation_errors": row_errors,
        }
        for key, value in metadata.items():
            if key not in RESERVED_DERIVED_FIELDS:
                row[key] = value
        rows.append(row)

    result = SourceRegistry(
        rows,
        file_records_before_deduplication=len(file_records),
        audit_events=audit_events,
        validation_errors=validation_errors,
    )
    if strict and validation_errors:
        raise CorpusValidationError(
            f"{len(validation_errors)} corpus validation error(s); first: {validation_errors[0]}"
        )
    return result


SEMANTIC_PREDICATES = frozenset(
    {
        "associated_with",
        "calls_for",
        "causes",
        "contains",
        "describes",
        "denies",
        "excludes",
        "has_property",
        "includes",
        "is",
        "opposes",
        "supports",
        "targets",
    }
)
SEMANTIC_POLARITIES = frozenset({"affirmed", "negated", "uncertain", "mixed"})
SEMANTIC_MODALITIES = frozenset(
    {"asserted", "conditional", "necessary", "possible", "prohibited", "recommended", "reported"}
)
EVIDENCE_STANCES = frozenset({"supports", "refutes", "quotes", "contextualizes"})
ENTITY_STATUSES = frozenset({"canonical", "ambiguous", "nil"})
VALIDATED_EXTRACTION_MARKER = "mpkg-rag.validated-extraction.v1"
_ENTITY_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+._:/?#%=-]{1,255}$")
_TOP_LEVEL_KEYS = frozenset({"schema_version", "claims"})
_CLAIM_KEYS = frozenset(
    {
        "claim_id",
        "mentions",
        "subject",
        "predicate",
        "object",
        "polarity",
        "polarity_scope",
        "modality",
        "attribution",
        "evidence_stance",
        "model_confidence",
    }
)
_MENTION_KEYS = frozenset(
    {"mention_id", "text", "start", "end", "entity_id", "candidate_index", "entity_status", "canonical_name"}
)


class ValidatedExtractionResult(dict):
    """JSON-persistable result marker for the validation-to-graph boundary."""

    validation_marker = VALIDATED_EXTRACTION_MARKER


SEMANTIC_EXTRACTION_SCHEMA = {
    "version": "semantic-claims.v1",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "claims"],
    "properties": {
        "schema_version": {"const": "semantic-claims.v1"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "mentions",
                    "subject",
                    "predicate",
                    "object",
                    "polarity",
                    "modality",
                    "attribution",
                    "evidence_stance",
                    "model_confidence",
                ],
                "properties": {
                    "claim_id": {"type": "string"},
                    "mentions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "mention_id",
                                "text",
                                "start",
                                "end",
                                "candidate_index",
                                "canonical_name",
                            ],
                            "properties": {
                                "mention_id": {"type": "string"},
                                "text": {"type": "string"},
                                "start": {"type": "integer", "minimum": 0},
                                "end": {"type": "integer", "minimum": 1},
                                "candidate_index": {"type": ["integer", "null"], "minimum": 0},
                                "canonical_name": {"type": "string", "minLength": 1},
                            },
                        },
                    },
                    "subject": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["mention_id"],
                        "properties": {"mention_id": {"type": "string", "minLength": 1}},
                    },
                    "predicate": {"type": "string", "enum": sorted(SEMANTIC_PREDICATES)},
                    "object": {"type": "object"},
                    "polarity": {"type": "string", "enum": sorted(SEMANTIC_POLARITIES)},
                    "polarity_scope": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["text", "start", "end"],
                        "properties": {
                            "text": {"type": "string"},
                            "start": {"type": "integer", "minimum": 0},
                            "end": {"type": "integer", "minimum": 1},
                        },
                    },
                    "modality": {"type": "string", "enum": sorted(SEMANTIC_MODALITIES)},
                    "attribution": {"type": ["object", "string", "null"]},
                    "evidence_stance": {"type": "string", "enum": sorted(EVIDENCE_STANCES)},
                    "model_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
    },
}


def build_extraction_prompt(
    text: str, source_context: dict[str, Any], mode: str
) -> list[dict[str, str]]:
    """Build a model-agnostic extraction prompt for the versioned payload."""

    schema = json.dumps(SEMANTIC_EXTRACTION_SCHEMA, ensure_ascii=False, sort_keys=True)
    context = json.dumps(source_context, ensure_ascii=False, sort_keys=True)
    system = (
        "Extract only claims explicitly supported by the supplied text. Return one JSON object "
        "with schema_version=semantic-claims.v1 and a claims list. Preserve subject/object roles, "
        "use exact Python string offsets and mention text, and never infer a relation from keywords. "
        "Never emit entity IDs. For every mention, emit candidate_index only; the validator assigns the "
        "closed-catalog entity ID. Use null when no supplied candidate is supported by the text. "
        f"Mode: {mode}. Schema: {schema}"
    )
    user = f"Source context: {context}\nText (do not normalize):\n{text}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


MENTION_DISCOVERY_SCHEMA = {
    "version": "mention-discovery.v1",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "mentions"],
    "properties": {
        "schema_version": {"const": "mention-discovery.v1"},
        "mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["mention_id", "text", "start", "end", "mention_type"],
                "properties": {
                    "mention_id": {"type": "string", "minLength": 1},
                    "text": {"type": "string", "minLength": 1},
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 1},
                    "mention_type": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}


def build_mention_prompt(text: str, source_context: dict[str, Any]) -> list[dict[str, str]]:
    schema = json.dumps(MENTION_DISCOVERY_SCHEMA, ensure_ascii=False, sort_keys=True)
    return [
        {"role": "system", "content": "Discover only exact entity mention spans. Never emit entity IDs. Return JSON matching this schema: " + schema},
        {"role": "user", "content": f"Source context: {json.dumps(source_context, ensure_ascii=False, sort_keys=True)}\nText (do not normalize):\n{text}"},
    ]


def validate_mentions(payload: Any, text: str, source_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate mention spans before they can seed catalog construction."""

    result = {"schema_version": MENTION_DISCOVERY_SCHEMA["version"], "accepted": [], "quarantined": [], "status": "quarantined"}
    if not isinstance(text, str) or not isinstance(payload, dict) or set(payload) != {"schema_version", "mentions"} or payload.get("schema_version") != MENTION_DISCOVERY_SCHEMA["version"] or not isinstance(payload.get("mentions"), list):
        result["quarantined"].append({"record": payload, "reasons": ["mention_payload_schema_invalid"], "source_context": source_context, "status": "quarantined"})
        return result
    seen: set[str] = set()
    for mention in payload["mentions"]:
        reasons: list[str] = []
        if not isinstance(mention, dict):
            reasons.append("mention_not_object")
        else:
            if set(mention) != {"mention_id", "text", "start", "end", "mention_type"}:
                reasons.append("mention_schema_invalid")
            mention_id, surface, start, end = (mention.get(key) for key in ("mention_id", "text", "start", "end"))
            if not isinstance(mention_id, str) or not mention_id or mention_id in seen:
                reasons.append("mention_id_invalid_or_duplicate")
            if not isinstance(surface, str) or not surface:
                reasons.append("mention_text_invalid")
            if isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int) or start < 0 or end <= start or end > len(text) or text[start:end] != surface:
                reasons.append("mention_span_mismatch")
            if not isinstance(mention.get("mention_type"), str) or not mention["mention_type"]:
                reasons.append("mention_type_invalid")
        if reasons:
            result["quarantined"].append({"record": mention, "reasons": sorted(set(reasons)), "source_context": source_context, "status": "quarantined"})
        else:
            seen.add(mention["mention_id"])
            result["accepted"].append(dict(mention))
    result["status"] = "accepted" if not result["quarantined"] else ("partial" if result["accepted"] else "quarantined")
    return result


def resolve_query_signature_entities(payload: Any, candidate_sets: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    """Map query candidate indices to catalog IDs; raw model IDs are forbidden."""

    if not isinstance(payload, dict):
        return {"valid": False, "reasons": ["query_signature_not_object"], "entity_ids": []}
    if payload.get("entity_ids"):
        return {"valid": False, "reasons": ["raw_query_entity_ids_forbidden"], "entity_ids": []}
    indices = payload.get("entity_candidate_indices", [])
    if not isinstance(indices, list):
        return {"valid": False, "reasons": ["query_candidate_indices_not_list"], "entity_ids": []}
    entity_ids: list[str] = []
    reasons: list[str] = []
    for item in indices:
        if not isinstance(item, dict) or not isinstance(item.get("mention_id"), str):
            reasons.append("query_candidate_schema_invalid")
            continue
        candidate_set = candidate_sets.get(item["mention_id"])
        if not isinstance(candidate_set, dict):
            reasons.append("query_candidate_set_missing")
            continue
        try:
            expected = build_entity_candidates(
                candidate_set.get("mention_text", ""),
                catalog,
                max_candidates=int(candidate_set.get("max_candidates", 8)),
                namespace_filter=candidate_set.get("namespace_filter"),
                namespace_preference=candidate_set.get("namespace_preference"),
                allow_target_fallback=bool(candidate_set.get("allow_target_fallback", False)),
                factual_only=bool(candidate_set.get("factual_only", False)),
            )
        except (TypeError, ValueError):
            reasons.append("query_candidate_set_authentication_failed")
            continue
        if candidate_set.get("candidate_set_hash") != expected.get("candidate_set_hash"):
            reasons.append("query_candidate_set_hash_mismatch")
        if candidate_set != expected:
            reasons.append("query_candidate_set_authentication_failed")
            continue
        index = item.get("candidate_index")
        if index is None:
            continue
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(candidate_set.get("candidates", [])):
            reasons.append("query_candidate_index_unknown")
            continue
        entity_ids.append(expected["candidates"][index]["entity_id"])
    return {"valid": not reasons and len(entity_ids) == len(set(entity_ids)), "reasons": sorted(set(reasons)), "entity_ids": sorted(set(entity_ids))}


def _semantic_context(source_context: Any) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(source_context, dict):
        return None, ["source_context_not_object"]
    context = dict(source_context)
    reasons = []
    for key in ("document_uid", "chunk_id"):
        if not isinstance(context.get(key), str) or not context[key]:
            reasons.append(f"missing_{key}")
    return (context if not reasons else None), reasons


def _claim_ref(value: Any) -> str | None:
    if isinstance(value, dict) and set(value) == {"mention_id"} and isinstance(value["mention_id"], str) and value["mention_id"]:
        return value["mention_id"]
    return None


def _nil_entity_id(context: dict[str, Any], mention: dict[str, Any], surface: str) -> str:
    return "NIL:" + stable_id(
        "entity",
        context["document_uid"],
        context["chunk_id"],
        surface,
        mention["start"],
        mention["end"],
    )


def _strict_catalog_context(context: dict[str, Any]) -> bool:
    return isinstance(context.get("entity_catalog"), dict)


def _validated_candidate_for_mention(
    mention: dict[str, Any], context: dict[str, Any], mention_id: str
) -> tuple[str | None, str | None, list[str]]:
    """Resolve a model candidate index against the pinned catalog, never a raw ID."""

    if not _strict_catalog_context(context):
        return mention.get("entity_id"), mention.get("entity_status"), []
    reasons: list[str] = []
    if mention.get("entity_id") is not None:
        reasons.append("raw_entity_id_forbidden")
    candidate_sets = context.get("candidate_sets")
    candidate_set = candidate_sets.get(mention_id) if isinstance(candidate_sets, dict) else None
    if not isinstance(candidate_set, dict):
        reasons.append("missing_candidate_set")
        return None, "nil", reasons
    catalog = context["entity_catalog"]
    catalog_rows = {
        row.get("entity_id"): row
        for row in catalog.get("entities", [])
        if isinstance(row, dict) and isinstance(row.get("entity_id"), str)
    }
    try:
        expected_candidate_set = build_entity_candidates(
            candidate_set.get("mention_text", ""),
            catalog,
            max_candidates=int(candidate_set.get("max_candidates", 8)),
            namespace_filter=candidate_set.get("namespace_filter"),
            namespace_preference=candidate_set.get("namespace_preference"),
            allow_target_fallback=bool(candidate_set.get("allow_target_fallback", False)),
            factual_only=bool(candidate_set.get("factual_only", False)),
        )
    except (TypeError, ValueError):
        reasons.append("candidate_set_authentication_failed")
        return None, "nil", reasons
    candidates = expected_candidate_set.get("candidates")
    if candidate_set.get("candidate_set_hash") != expected_candidate_set.get("candidate_set_hash"):
        reasons.append("candidate_set_hash_mismatch")
    if candidate_set != expected_candidate_set:
        reasons.append("candidate_set_authentication_failed")
    if candidate_set.get("catalog_revision") != catalog.get("catalog_revision"):
        reasons.append("catalog_revision_mismatch")
    if not isinstance(candidates, list):
        reasons.append("candidate_list_invalid")
        return None, "nil", reasons
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("entity_id") not in catalog_rows:
            reasons.append("candidate_not_in_catalog")
            break
        if candidate_set.get("factual_only") and not (candidate.get("namespace") == "corpus" and catalog_rows[candidate["entity_id"]].get("factual_identity_allowed") is True):
            reasons.append("candidate_namespace_policy_violation")
            break
    candidate_index = mention.get("candidate_index")
    if candidate_index is None:
        return None, "nil", reasons
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int) or not 0 <= candidate_index < len(candidates):
        reasons.append("candidate_index_unknown")
        return None, "nil", reasons
    candidate = expected_candidate_set["candidates"][candidate_index]
    entity_id = candidate.get("entity_id")
    entity = catalog_rows.get(entity_id)
    if not isinstance(entity_id, str) or entity is None or entity.get("retrieval_allowed") is not True:
        reasons.append("candidate_not_retrieval_allowed")
        return None, "nil", reasons
    return entity_id, "canonical", reasons


def _quarantined_record(record: Any, reasons: list[str], context: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "record": record,
        "reasons": sorted(set(reasons)),
        "source_context": context,
        "status": "quarantined",
    }


def _reviewed_record(record: Any, reasons: list[str], context: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "record": record,
        "reasons": sorted(set(reasons)),
        "source_context": context,
        "status": "reviewed",
    }


def _normalized_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _validation_fingerprint(result: dict[str, Any]) -> str:
    digest_material = {
        key: value for key, value in result.items() if key != "validation_fingerprint"
    }
    serialized = json.dumps(digest_material, ensure_ascii=False, sort_keys=True, default=str)
    return stable_id("validated-extraction", serialized)


def _new_validation_result(payload: Any, text: Any, source_context: Any) -> ValidatedExtractionResult:
    return ValidatedExtractionResult(
        {
            "validation_marker": VALIDATED_EXTRACTION_MARKER,
            "validation_fingerprint": "",
            "schema_version": SEMANTIC_EXTRACTION_SCHEMA["version"],
            "payload": payload,
            "normalized_payload": None,
            "source_context": dict(source_context) if isinstance(source_context, dict) else source_context,
            "text": text,
            "accepted": [],
            "quarantined": [],
            "reviewed": [],
        }
    )


def _finish_validation_result(result: ValidatedExtractionResult) -> ValidatedExtractionResult:
    normalized_claims = list(result.get("accepted", []))
    normalized_claims.extend(
        item["record"]
        for item in result.get("reviewed", [])
        if isinstance(item, dict) and isinstance(item.get("record"), dict)
    )
    result["normalized_payload"] = {
        "schema_version": SEMANTIC_EXTRACTION_SCHEMA["version"],
        "claims": _normalized_json(normalized_claims),
    }
    result["validation_fingerprint"] = _validation_fingerprint(result)
    return result


def validate_extraction(
    payload: Any, text: str, source_context: dict[str, Any]
) -> dict[str, Any]:
    """Validate a Qwen extraction payload without loading or invoking a model."""

    context, context_reasons = _semantic_context(source_context)
    result = _new_validation_result(payload, text, source_context)
    if not isinstance(text, str):
        result["quarantined"].append(_quarantined_record(payload, ["source_text_not_string"], context))
        result["status"] = "quarantined"
        return _finish_validation_result(result)
    if not isinstance(payload, dict):
        result["quarantined"].append(_quarantined_record(payload, ["payload_not_object"], context))
        result["status"] = "quarantined"
        return _finish_validation_result(result)
    top_level_reasons = []
    if set(payload) - _TOP_LEVEL_KEYS:
        top_level_reasons.append("extra_payload_fields")
    if "schema_version" not in payload:
        top_level_reasons.append("missing_schema_version")
    if "claims" not in payload:
        top_level_reasons.append("missing_claims")
    if top_level_reasons:
        result["quarantined"].append(_quarantined_record(payload, top_level_reasons, context))
        result["status"] = "quarantined"
        return _finish_validation_result(result)
    if payload.get("schema_version") != SEMANTIC_EXTRACTION_SCHEMA["version"]:
        result["quarantined"].append(
            _quarantined_record(payload, ["unsupported_schema_version"], context)
        )
        result["status"] = "quarantined"
        return _finish_validation_result(result)
    if not isinstance(payload.get("claims"), list):
        result["quarantined"].append(_quarantined_record(payload, ["claims_not_list"], context))
        result["status"] = "quarantined"
        return _finish_validation_result(result)
    if context_reasons:
        result["quarantined"].extend(
            _quarantined_record(claim, context_reasons, context) for claim in payload["claims"]
        )
        result["status"] = "quarantined"
        return _finish_validation_result(result)

    for claim_index, claim in enumerate(payload["claims"]):
        reasons: list[str] = []
        if not isinstance(claim, dict):
            result["quarantined"].append(
                _quarantined_record(claim, ["claim_not_object"], context)
            )
            continue
        if set(claim) - _CLAIM_KEYS:
            reasons.append("extra_claim_fields")
        if "claim_id" in claim and (not isinstance(claim["claim_id"], str) or not claim["claim_id"]):
            reasons.append("invalid_claim_id")
        required = {
            "mentions",
            "subject",
            "predicate",
            "object",
            "polarity",
            "modality",
            "attribution",
            "evidence_stance",
            "model_confidence",
        }
        reasons.extend(f"missing_{key}" for key in sorted(required - set(claim)))
        mentions = claim.get("mentions")
        normalized_mentions: list[dict[str, Any]] = []
        mention_by_id: dict[str, dict[str, Any]] = {}
        if not isinstance(mentions, list) or not mentions:
            reasons.append("mentions_not_nonempty_list")
        else:
            for mention_index, mention in enumerate(mentions):
                if not isinstance(mention, dict):
                    reasons.append("mention_not_object")
                    continue
                mention_reasons: list[str] = []
                if set(mention) - _MENTION_KEYS:
                    mention_reasons.append("extra_mention_fields")
                required_mention_keys = _MENTION_KEYS - ({"entity_id", "candidate_index", "entity_status"} if _strict_catalog_context(context or {}) else {"candidate_index"})
                missing_mention_keys = required_mention_keys - set(mention)
                mention_reasons.extend(f"missing_{key}" for key in sorted(missing_mention_keys))
                mention_id = mention.get("mention_id")
                if not isinstance(mention_id, str) or not mention_id:
                    mention_reasons.append("invalid_mention_id")
                    reasons.extend(mention_reasons)
                    continue
                if mention_id in mention_by_id:
                    mention_reasons.append("duplicate_mention_id")
                surface = mention.get("text")
                start = mention.get("start")
                end = mention.get("end")
                if not isinstance(surface, str) or not surface:
                    mention_reasons.append("mention_text_not_string")
                    reasons.extend(mention_reasons)
                    continue
                if isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int):
                    mention_reasons.append("mention_offsets_not_integer")
                    reasons.extend(mention_reasons)
                    continue
                if start < 0 or end <= start or end > len(text) or text[start:end] != surface:
                    mention_reasons.append("mention_span_mismatch")
                    reasons.extend(mention_reasons)
                    continue
                supplied_entity_id = mention.get("entity_id")
                entity_status = mention.get("entity_status")
                entity_id, resolved_status, candidate_reasons = _validated_candidate_for_mention(
                    mention, context or {}, mention_id
                )
                if _strict_catalog_context(context or {}):
                    entity_status = resolved_status
                    mention_reasons.extend(candidate_reasons)
                else:
                    entity_id = supplied_entity_id
                if not isinstance(entity_status, str) or entity_status not in ENTITY_STATUSES:
                    mention_reasons.append("unknown_entity_status")
                if entity_id is not None and not isinstance(entity_id, str):
                    mention_reasons.append("invalid_entity_id_type")
                    reasons.extend(mention_reasons)
                    continue
                if isinstance(entity_id, str) and (
                    not _ENTITY_ID_PATTERN.fullmatch(entity_id)
                    or (entity_id.startswith(("NIL-", "NIL:")) and entity_status == "canonical")
                ):
                    mention_reasons.append("invalid_entity_id")
                if entity_status == "canonical" and not entity_id:
                    mention_reasons.append("missing_canonical_entity_id")
                if entity_status in {"ambiguous", "nil"}:
                    if entity_id is not None and not entity_id.startswith(("NIL-", "NIL:")):
                        mention_reasons.append("unresolved_entity_id_must_be_nil")
                    if entity_id is None and context is not None:
                        entity_id = _nil_entity_id(context, mention, surface)
                canonical_name = mention.get("canonical_name")
                if not isinstance(canonical_name, str) or not canonical_name:
                    mention_reasons.append("invalid_canonical_name")
                normalized_mention = {
                    "mention_id": mention_id,
                    "text": surface,
                    "start": start,
                    "end": end,
                    "entity_id": entity_id,
                    "candidate_index": mention.get("candidate_index"),
                    "entity_status": entity_status,
                    "canonical_name": canonical_name,
                }
                if mention_reasons:
                    reasons.extend(mention_reasons)
                else:
                    mention_by_id[mention_id] = normalized_mention
                    normalized_mentions.append(normalized_mention)

        predicate = claim.get("predicate")
        if predicate not in SEMANTIC_PREDICATES:
            reasons.append("unknown_predicate")
        if claim.get("polarity") not in SEMANTIC_POLARITIES:
            reasons.append("unknown_polarity")
        if claim.get("modality") not in SEMANTIC_MODALITIES:
            reasons.append("unknown_modality")
        if claim.get("evidence_stance") not in EVIDENCE_STANCES:
            reasons.append("unknown_evidence_stance")
        if not isinstance(claim.get("attribution"), (dict, str, type(None))):
            reasons.append("invalid_attribution")
        confidence = claim.get("model_confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            reasons.append("invalid_model_confidence")

        if not isinstance(claim.get("subject"), dict):
            reasons.append("subject_not_object")
        elif set(claim["subject"]) != {"mention_id"}:
            reasons.append("subject_schema_invalid")
        subject_ref = _claim_ref(claim.get("subject"))
        if subject_ref is None or subject_ref not in mention_by_id:
            reasons.append("subject_endpoint_missing")
        object_value: dict[str, Any] | None = None
        object_ref = _claim_ref(claim.get("object"))
        if not isinstance(claim.get("object"), dict):
            reasons.append("object_not_object")
        elif object_ref is not None:
            if object_ref not in mention_by_id:
                reasons.append("object_endpoint_missing")
        else:
            raw_object = claim.get("object")
            if isinstance(raw_object, dict) and set(raw_object) == {"value", "value_type"}:
                value = raw_object["value"]
                if not isinstance(raw_object["value_type"], str) or not isinstance(value, (str, int, float, bool)) or value == "":
                    reasons.append("object_value_invalid")
                else:
                    object_value = {"value": value, "value_type": raw_object["value_type"]}
            else:
                reasons.append("object_schema_invalid")

        polarity_scope = claim.get("polarity_scope")
        if claim.get("polarity") in {"negated", "uncertain", "mixed"} and polarity_scope is None:
            reasons.append("missing_polarity_scope")
        if polarity_scope is not None:
            if not isinstance(polarity_scope, dict) or set(polarity_scope) != {"text", "start", "end"}:
                reasons.append("polarity_scope_schema_invalid")
            else:
                scope_text = polarity_scope["text"]
                scope_start = polarity_scope["start"]
                scope_end = polarity_scope["end"]
                if (
                    not isinstance(scope_text, str)
                    or isinstance(scope_start, bool)
                    or not isinstance(scope_start, int)
                    or isinstance(scope_end, bool)
                    or not isinstance(scope_end, int)
                    or scope_start < 0
                    or scope_end <= scope_start
                    or scope_end > len(text)
                    or text[scope_start:scope_end] != scope_text
                ):
                    reasons.append("polarity_scope_mismatch")

        if reasons:
            result["quarantined"].append(_quarantined_record(claim, reasons, context))
            continue
        normalized_claim = {
            "claim_id": claim.get("claim_id") or "claim-" + stable_id("claim", claim_index, context["chunk_id"]),
            "mentions": normalized_mentions,
            "subject": {"mention_id": subject_ref},
            "predicate": predicate,
            "object": {"mention_id": object_ref} if object_ref is not None else object_value,
            "polarity": claim["polarity"],
            "modality": claim["modality"],
            "attribution": claim["attribution"],
            "evidence_stance": claim["evidence_stance"],
            "model_confidence": confidence,
            "source_context": context,
        }
        if polarity_scope is not None:
            normalized_claim["polarity_scope"] = dict(polarity_scope)
        unresolved = any(
            mention["entity_status"] in {"ambiguous", "nil"}
            for mention in normalized_mentions
        )
        if unresolved or claim["evidence_stance"] == "quotes":
            review_reasons = ["unresolved_entity"] if unresolved else []
            if claim["evidence_stance"] == "quotes":
                review_reasons.append("quoted_only")
            result["reviewed"].append(_reviewed_record(normalized_claim, review_reasons, context))
        else:
            result["accepted"].append(normalized_claim)
    if result["accepted"] and not result["quarantined"] and not result["reviewed"]:
        result["status"] = "accepted"
    elif result["reviewed"] and not result["accepted"] and not result["quarantined"]:
        result["status"] = "reviewed"
    elif result["quarantined"] and not result["accepted"] and not result["reviewed"]:
        result["status"] = "quarantined"
    else:
        result["status"] = "partial"
    return _finish_validation_result(result)


def _semantic_key(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


_VALIDATED_RESULT_KEYS = frozenset(
    {
        "validation_marker",
        "validation_fingerprint",
        "schema_version",
        "payload",
        "normalized_payload",
        "source_context",
        "text",
        "accepted",
        "quarantined",
        "reviewed",
        "status",
    }
)


def _iter_accepted_extractions(
    accepted_extractions: Any,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], str]], list[Any], list[Any]]:
    accepted: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    quarantined: list[Any] = []
    reviewed: list[Any] = []
    records = (
        [(None, accepted_extractions)]
        if isinstance(accepted_extractions, dict)
        else list(enumerate(accepted_extractions or []))
    )
    for _, record in records:
        if not isinstance(record, dict) or record.get("validation_marker") != VALIDATED_EXTRACTION_MARKER:
            quarantined.append(_quarantined_record(record, ["unvalidated_extraction_result"], None))
            continue
        if not _VALIDATED_RESULT_KEYS.issubset(record) or set(record) - _VALIDATED_RESULT_KEYS:
            quarantined.append(_quarantined_record(record, ["validated_result_schema_invalid"], None))
            continue
        if record.get("status") not in {"accepted", "partial", "reviewed", "quarantined"}:
            quarantined.append(_quarantined_record(record, ["validation_result_status_invalid"], record.get("source_context")))
            continue
        source_context = record.get("source_context")
        text = record.get("text")
        if not isinstance(source_context, dict) or not isinstance(text, str):
            quarantined.append(_quarantined_record(record, ["validated_result_provenance_invalid"], source_context))
            continue
        if record.get("schema_version") != SEMANTIC_EXTRACTION_SCHEMA["version"]:
            quarantined.append(_quarantined_record(record, ["validated_result_schema_invalid"], source_context))
            continue
        revalidated = validate_extraction(record.get("payload"), text, source_context)
        comparable_fields = (
            "normalized_payload",
            "text",
            "source_context",
            "status",
            "accepted",
            "reviewed",
            "quarantined",
        )
        if any(record.get(field) != revalidated.get(field) for field in comparable_fields):
            quarantined.append(_quarantined_record(record, ["validation_result_stale"], source_context))
            continue
        if record.get("validation_fingerprint") != revalidated.get("validation_fingerprint"):
            quarantined.append(_quarantined_record(record, ["validation_result_tampered"], source_context))
            continue
        quarantined.extend(revalidated["quarantined"])
        reviewed.extend(revalidated["reviewed"])
        if record.get("status") == "quarantined":
            quarantined.extend(
                _quarantined_record(claim, ["quarantined_extraction_result"], source_context)
                for claim in revalidated["accepted"]
            )
            continue
        if record.get("status") == "reviewed":
            reviewed.extend(
                _reviewed_record(claim, ["reviewed_only"], source_context)
                for claim in revalidated["accepted"]
            )
            continue
        for claim in revalidated["accepted"]:
            accepted.append((claim, source_context, text))
    return accepted, quarantined, reviewed


def _chunk_provenance_reasons(
    context: dict[str, Any], extraction_text: str, chunk: dict[str, Any] | None
) -> list[str]:
    if chunk is None:
        return ["missing_chunk_provenance"]
    reasons: list[str] = []
    chunk_text = chunk.get("text")
    if not isinstance(chunk_text, str):
        reasons.append("missing_chunk_text")
    elif chunk_text != extraction_text:
        reasons.append("chunk_text_mismatch")
    if chunk.get("factual_index_allowed") is not True:
        reasons.append("factual_index_policy_missing_or_denied")
    if chunk.get("status") not in {"accepted", "validated"}:
        reasons.append("chunk_status_not_accepted")
    if context.get("factual_index_allowed") is not None and context.get("factual_index_allowed") is not True:
        reasons.append("factual_index_policy_mismatch")
    if context.get("status") is not None and context.get("status") not in {"accepted", "validated"}:
        reasons.append("chunk_status_not_accepted")
    if "text" in context and context["text"] != extraction_text:
        reasons.append("source_context_mismatch")
    for key in ("source_id", "source_type", "authority_score", "factual_index_allowed", "status"):
        if key in context and key in chunk and context[key] != chunk[key]:
            reasons.append("source_context_mismatch")
    if chunk.get("source_type") == "harmful_examples" or context.get("source_type") == "harmful_examples":
        reasons.append("harmful_source_type")
    if isinstance(chunk_text, str):
        actual_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        text_hashes = [
            owner["text_sha256"]
            for owner in (context, chunk)
            if owner.get("text_sha256") is not None
        ]
        if any(expected != actual_hash for expected in text_hashes):
            reasons.append("chunk_hash_mismatch")
        if len(set(text_hashes)) > 1:
            reasons.append("source_context_hash_mismatch")
    document_hashes = [
        owner[key]
        for owner in (context, chunk)
        for key in ("content_sha256", "document_sha256")
        if owner.get(key) is not None
    ]
    if len(set(document_hashes)) > 1:
        reasons.append("document_hash_mismatch")
    return sorted(set(reasons))


def _audit_field(chunk: dict[str, Any], context: dict[str, Any], key: str) -> Any:
    return chunk[key] if key in chunk else context.get(key)


def build_semantic_graph(chunks: Iterable[dict[str, Any]], accepted_extractions: Any) -> dict[str, Any]:
    """Build accepted semantic tables and typed edges from validated extractions."""

    chunk_rows = list(chunks or [])
    chunk_by_key = {
        (row.get("document_uid"), row.get("chunk_id")): row
        for row in chunk_rows
        if isinstance(row, dict)
    }
    accepted, quarantined, reviewed = _iter_accepted_extractions(accepted_extractions)
    graph: dict[str, Any] = {
        "schema_version": SEMANTIC_EXTRACTION_SCHEMA["version"],
        "Document": [],
        "EvidenceChunk": [],
        "Mention": [],
        "Entity": [],
        "Claim": [],
        "edges": [],
        "quarantined": quarantined,
        "reviewed": reviewed,
    }
    documents: dict[str, dict[str, Any]] = {}
    evidence_chunks: dict[str, dict[str, Any]] = {}
    mentions: dict[str, dict[str, Any]] = {}
    entities: dict[str, dict[str, Any]] = {}
    claims: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def add_edge(source_id: str, edge_type: str, target_id: str, **extra: Any) -> None:
        edges.append({"source_id": source_id, "type": edge_type, "target_id": target_id, **extra})

    for extraction_index, (claim, context, extraction_text) in enumerate(accepted):
        document_uid = context.get("document_uid")
        chunk_id = context.get("chunk_id")
        if not isinstance(document_uid, str) or not isinstance(chunk_id, str):
            graph["quarantined"].append(_quarantined_record(claim, ["missing_source_context"], context))
            continue
        chunk = chunk_by_key.get((document_uid, chunk_id))
        provenance_reasons = _chunk_provenance_reasons(context, extraction_text, chunk)
        if provenance_reasons:
            graph["quarantined"].append(_quarantined_record(claim, provenance_reasons, context))
            continue
        assert chunk is not None
        evidence_id = stable_id("evidence", document_uid, chunk_id)
        documents.setdefault(
            document_uid,
            {
                "document_uid": document_uid,
                "source_id": _audit_field(chunk, context, "source_id"),
                "source_type": _audit_field(chunk, context, "source_type"),
                "authority_score": _audit_field(chunk, context, "authority_score"),
                "factual_index_allowed": _audit_field(chunk, context, "factual_index_allowed"),
                "status": _audit_field(chunk, context, "status") or "accepted",
                "content_sha256": _audit_field(chunk, context, "content_sha256"),
                "document_sha256": _audit_field(chunk, context, "document_sha256"),
                "text_sha256": _audit_field(chunk, context, "text_sha256"),
                "sha256": _audit_field(chunk, context, "sha256"),
            },
        )
        evidence_chunks.setdefault(
            evidence_id,
            {
                "evidence_chunk_id": evidence_id,
                "document_uid": document_uid,
                "chunk_id": chunk_id,
                "text": chunk["text"],
                "source_id": _audit_field(chunk, context, "source_id"),
                "source_type": _audit_field(chunk, context, "source_type"),
                "authority_score": _audit_field(chunk, context, "authority_score"),
                "factual_index_allowed": _audit_field(chunk, context, "factual_index_allowed"),
                "status": _audit_field(chunk, context, "status") or "accepted",
                "content_sha256": _audit_field(chunk, context, "content_sha256"),
                "document_sha256": _audit_field(chunk, context, "document_sha256"),
                "text_sha256": _audit_field(chunk, context, "text_sha256"),
                "sha256": _audit_field(chunk, context, "sha256"),
            },
        )
        mention_by_id = {mention["mention_id"]: mention for mention in claim.get("mentions", [])}
        subject_mention = mention_by_id.get(claim.get("subject", {}).get("mention_id"))
        object_ref = claim.get("object", {}).get("mention_id") if isinstance(claim.get("object"), dict) else None
        object_mention = mention_by_id.get(object_ref) if object_ref else None
        if subject_mention is None or (object_ref and object_mention is None):
            graph["quarantined"].append(_quarantined_record(claim, ["graph_endpoint_missing"], context))
            continue
        occurrence_id = stable_id("occurrence", document_uid, chunk_id, extraction_index, claim.get("claim_id"))

        def entity_semantic_id(mention: dict[str, Any]) -> str:
            if not str(mention["entity_id"]).startswith("NIL-"):
                return "entity:" + str(mention["entity_id"])
            return "entity-name:" + _semantic_key(mention.get("canonical_name") or mention["text"])

        subject_key = entity_semantic_id(subject_mention)
        if object_mention is not None:
            object_key = entity_semantic_id(object_mention)
        else:
            object_value = claim.get("object", {}).get("value")
            object_key = "value:" + _semantic_key(object_value)
        semantic_signature = json.dumps(
            [
                subject_key,
                claim.get("predicate"),
                object_key,
                claim.get("polarity"),
                claim.get("modality"),
                claim.get("attribution"),
            ],
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        claim_id = "CLM-" + stable_id("claim", document_uid, semantic_signature)
        claim_row = claims.setdefault(
            claim_id,
            {
                "claim_id": claim_id,
                "document_uid": document_uid,
                "subject_entity_id": subject_mention["entity_id"],
                "predicate": claim["predicate"],
                "object_entity_id": object_mention["entity_id"] if object_mention else None,
                "object_value": claim.get("object", {}).get("value") if not object_mention else None,
                "model_confidence": claim["model_confidence"],
                "model_confidences": [claim["model_confidence"]],
                "polarity": claim["polarity"],
                "modality": claim["modality"],
                "attribution": claim["attribution"],
                "stance": claim["evidence_stance"],
                "evidence_stances": [claim["evidence_stance"]],
                "review_status": "accepted",
                "status": "accepted",
                "evidence_occurrence_ids": [],
            },
        )
        if claim_row["model_confidences"][-1] != claim["model_confidence"]:
            claim_row["model_confidences"].append(claim["model_confidence"])
        if claim["evidence_stance"] not in claim_row["evidence_stances"]:
            claim_row["evidence_stances"].append(claim["evidence_stance"])
        for mention in claim.get("mentions", []):
            mention_id = stable_id("mention", occurrence_id, mention["mention_id"])
            mentions[mention_id] = {
                "mention_id": mention_id,
                "document_uid": document_uid,
                "evidence_chunk_id": evidence_id,
                "claim_occurrence_id": occurrence_id,
                "text": mention["text"],
                "start": mention["start"],
                "end": mention["end"],
                "entity_id": mention["entity_id"],
                "entity_status": mention["entity_status"],
            }
            entities.setdefault(
                mention["entity_id"],
                {
                    "entity_id": mention["entity_id"],
                    "canonical_name": mention["canonical_name"],
                    "entity_status": mention["entity_status"],
                    "status": "accepted",
                },
            )
        if not any(edge["source_id"] == claim_id and edge["type"] == "has_subject" for edge in edges):
            add_edge(claim_id, "has_subject", subject_mention["entity_id"], target_type="entity")
            if object_mention is not None:
                add_edge(claim_id, "has_object", object_mention["entity_id"], target_type="entity")
            else:
                value_id = "VAL-" + stable_id("value", document_uid, object_key)
                add_edge(
                    claim_id,
                    "has_object",
                    value_id,
                    target_type="value",
                    value=claim.get("object", {}).get("value"),
                )
        add_edge(evidence_id, claim["evidence_stance"], claim_id, stance=claim["evidence_stance"], occurrence_id=occurrence_id)
        add_edge(evidence_id, "from_document", document_uid, occurrence_id=occurrence_id)
        add_edge(claim_id, "evidenced_by", evidence_id, occurrence_id=occurrence_id)
        claim_row["evidence_occurrence_ids"].append(occurrence_id)

    graph["Document"] = list(documents.values())
    graph["EvidenceChunk"] = list(evidence_chunks.values())
    graph["Mention"] = list(mentions.values())
    graph["Entity"] = list(entities.values())
    graph["Claim"] = list(claims.values())
    graph["edges"] = edges
    graph["Edges"] = edges
    return graph


_RETRIEVAL_WEIGHT_KEYS = (
    "query_entity",
    "predicate",
    "polarity",
    "modality",
    "stance",
    "review_state",
    "authority",
    "extraction_confidence",
    "seed_score",
    "hop_decay",
)
_RRF_BRANCH_NAMES = frozenset({"dense", "bm25", "graph"})


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _validate_retrieval_config(config: Any, *, expansion: bool = True) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError("retrieval config must be an object")
    required = {"minimum_rerank_probability", "max_evidence"}
    if expansion:
        required |= {
            "max_hops",
            "minimum_dense_score",
            "minimum_graph_score",
            "hop_decay",
            "weights",
            "review_state_scores",
        }
    missing = sorted(key for key in required if key not in config)
    if missing:
        raise ValueError(f"retrieval config missing required fields: {', '.join(missing)}")

    normalized = dict(config)
    if "max_hops" in config:
        max_hops = config["max_hops"]
        if isinstance(max_hops, bool) or not isinstance(max_hops, int) or not 0 <= max_hops <= 8:
            raise ValueError("max_hops must be an integer from 0 through 8")
        normalized["max_hops"] = max_hops
    for key in ("minimum_dense_score", "minimum_graph_score", "hop_decay"):
        if key in config:
            normalized[key] = _finite_number(config[key], key)
    for key in ("minimum_dense_score", "minimum_graph_score"):
        if key in normalized and normalized[key] < 0:
            raise ValueError(f"{key} must be non-negative")
    if "hop_decay" in config and not 0 < normalized["hop_decay"] <= 1:
        raise ValueError("hop_decay must be greater than 0 and at most 1")
    minimum_probability = _finite_number(
        config["minimum_rerank_probability"], "minimum_rerank_probability"
    )
    if not 0 <= minimum_probability <= 1:
        raise ValueError("minimum_rerank_probability must be between 0 and 1")
    normalized["minimum_rerank_probability"] = minimum_probability
    max_evidence = config["max_evidence"]
    if isinstance(max_evidence, bool) or not isinstance(max_evidence, int) or max_evidence < 1:
        raise ValueError("max_evidence must be a positive integer")
    normalized["max_evidence"] = max_evidence

    if expansion:
        weights = config["weights"]
        if not isinstance(weights, dict):
            raise TypeError("retrieval config weights must be an object")
        missing_weights = [key for key in _RETRIEVAL_WEIGHT_KEYS if key not in weights]
        if missing_weights:
            raise ValueError(
                "retrieval config weights missing required fields: "
                + ", ".join(missing_weights)
            )
        normalized_weights = {
            key: _finite_number(weights[key], f"weights.{key}")
            for key in _RETRIEVAL_WEIGHT_KEYS
        }
        if any(value < 0 for value in normalized_weights.values()):
            raise ValueError("retrieval config weights cannot be negative")
        if not any(value > 0 for value in normalized_weights.values()):
            raise ValueError("retrieval config requires at least one positive weight")
        normalized["weights"] = normalized_weights
        review_scores = config["review_state_scores"]
        if not isinstance(review_scores, dict):
            raise TypeError("review_state_scores must be an object")
        if "accepted" not in review_scores or "reviewed" not in review_scores:
            raise ValueError("review_state_scores must include accepted and reviewed")
        normalized["review_state_scores"] = {
            str(key): _finite_number(value, f"review_state_scores.{key}")
            for key, value in review_scores.items()
        }
    return normalized


def _query_values(query_signature: dict[str, Any], *names: str) -> tuple[str, ...]:
    value: Any = None
    for name in names:
        if name in query_signature:
            value = query_signature[name]
            break
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise TypeError(f"query field {names[0]} must be a string or sequence of strings")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"query field {names[0]} contains an invalid value")
        normalized.append(item)
    return tuple(dict.fromkeys(normalized))


def _normalize_query_signature(query_signature: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(query_signature, dict):
        raise TypeError("query_signature must be an object")
    return {
        "entity_ids": _query_values(query_signature, "entity_ids", "canonical_entity_ids", "entity_id"),
        "predicates": _query_values(query_signature, "predicates", "predicate"),
        "polarities": _query_values(query_signature, "polarities", "polarity"),
        "modalities": _query_values(query_signature, "modalities", "modality"),
        "stances": _query_values(
            query_signature,
            "desired_stances",
            "desired_stance",
            "useful_stances",
            "useful_stance",
            "stances",
            "stance",
        ),
    }


def _table_index(graph_tables: dict[str, Any], table_name: str, key: str) -> dict[str, dict[str, Any]]:
    rows = graph_tables.get(table_name, [])
    if not isinstance(rows, list):
        raise TypeError(f"graph table {table_name} must be a list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get(key), str):
            result[row[key]] = row
    return result


def _edge_indexes(
    graph_tables: dict[str, Any],
    node_ids: set[str],
    evidence_ids: set[str],
    claim_ids: set[str],
) -> tuple[dict[str, list[tuple[str, str, dict[str, Any]]]], dict[str, set[str]]]:
    edges = graph_tables.get("edges", graph_tables.get("Edges", []))
    if not isinstance(edges, list):
        raise TypeError("graph edges must be a list")
    adjacency: dict[str, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
    claims_by_evidence: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source_id = edge.get("source_id")
        target_id = edge.get("target_id")
        edge_type = edge.get("type")
        if not all(isinstance(value, str) for value in (source_id, target_id, edge_type)):
            continue
        if source_id not in node_ids or target_id not in node_ids:
            continue
        if edge_type not in EVIDENCE_STANCES | {"has_subject", "has_object", "evidenced_by"}:
            continue
        adjacency[source_id].append((target_id, edge_type, edge))
        adjacency[target_id].append((source_id, f"reverse:{edge_type}", edge))
        if edge_type in EVIDENCE_STANCES and source_id in evidence_ids and target_id in claim_ids:
            claims_by_evidence[source_id].add(target_id)
        elif edge_type == "evidenced_by" and target_id in evidence_ids and source_id in claim_ids:
            claims_by_evidence[target_id].add(source_id)
    for neighbors in adjacency.values():
        neighbors.sort(key=lambda item: (item[0], item[1]))
    return adjacency, claims_by_evidence


def _seed_evidence_id(
    hit: dict[str, Any], evidence_by_id: dict[str, dict[str, Any]], evidence_by_chunk: dict[tuple[str, str], str]
) -> str | None:
    if "evidence_chunk_id" in hit:
        evidence_id = hit["evidence_chunk_id"]
        if isinstance(evidence_id, str) and evidence_id in evidence_by_id:
            return evidence_id
        return None
    key = (hit.get("document_uid"), hit.get("chunk_id"))
    return evidence_by_chunk.get(key)


def _claim_entity_ids(claim: dict[str, Any]) -> set[str]:
    return {
        value
        for key in ("subject_entity_id", "object_entity_id")
        if isinstance((value := claim.get(key)), str) and value
    }


def _component(value: float, weight: float, role: str) -> dict[str, Any]:
    return {
        "value": float(value),
        "weight": float(weight),
        "contribution": float(value * weight),
        "role": role,
    }


def expand_graph_from_seeds(
    seed_hits: Iterable[dict[str, Any]],
    query_signature: dict[str, Any],
    graph_tables: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Expand accepted semantic evidence from dense/BM25 seeds with auditable traces.

    Query fields are canonical entity IDs, predicates, polarity, modality, and desired
    or useful stance. Seeds must already be produced by semantic dense/BM25 retrieval;
    this function only filters them and traverses explicit typed graph edges.
    """

    normalized_config = _validate_retrieval_config(config, expansion=True)
    query = _normalize_query_signature(query_signature)
    if not isinstance(graph_tables, dict):
        raise TypeError("graph_tables must be an object")
    evidence_by_id = _table_index(graph_tables, "EvidenceChunk", "evidence_chunk_id")
    document_by_id = _table_index(graph_tables, "Document", "document_uid")
    claim_by_id = _table_index(graph_tables, "Claim", "claim_id")
    evidence_by_chunk = {
        (row.get("document_uid"), row.get("chunk_id")): evidence_id
        for evidence_id, row in evidence_by_id.items()
        if isinstance(row.get("document_uid"), str) and isinstance(row.get("chunk_id"), str)
    }
    node_ids = set(evidence_by_id) | set(document_by_id) | set(claim_by_id)
    node_ids |= {
        str(row["entity_id"])
        for row in graph_tables.get("Entity", [])
        if isinstance(row, dict) and isinstance(row.get("entity_id"), str)
    }
    adjacency, claims_by_evidence = _edge_indexes(
        graph_tables, node_ids, set(evidence_by_id), set(claim_by_id)
    )

    retained_seeds_by_branch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hit in seed_hits or []:
        if not isinstance(hit, dict):
            raise TypeError("each seed hit must be an object")
        branch = hit.get("branch")
        if branch not in {"dense", "bm25"}:
            raise ValueError("seed branch must be exactly dense or bm25")
        rank = hit.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("seed rank must be a positive integer")
        raw_score = _finite_number(hit.get("raw_score"), "seed raw_score")
        if raw_score <= 0:
            continue
        if branch == "dense" and raw_score < normalized_config["minimum_dense_score"]:
            continue
        evidence_id = _seed_evidence_id(hit, evidence_by_id, evidence_by_chunk)
        if evidence_id is None:
            continue
        retained_seeds_by_branch[branch].append(
            {
                "branch": branch,
                "rank": rank,
                "raw_score": raw_score,
                "chunk_id": hit.get("chunk_id"),
                "document_uid": hit.get("document_uid"),
                "evidence_chunk_id": evidence_id,
                "source_trace": hit.get("source_trace", hit.get("trace")),
            }
        )
    seeds_by_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for branch in sorted(retained_seeds_by_branch):
        branch_seeds = retained_seeds_by_branch[branch]
        raw_scores = [seed["raw_score"] for seed in branch_seeds]
        minimum = min(raw_scores)
        maximum = max(raw_scores)
        spread = maximum - minimum
        for seed in branch_seeds:
            seed["normalized_score"] = (
                1.0 if spread == 0 else (seed["raw_score"] - minimum) / spread
            )
            seeds_by_evidence[seed["evidence_chunk_id"]].append(seed)
    if not seeds_by_evidence:
        return []

    paths_by_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seed_id in sorted(seeds_by_evidence):
        paths_by_evidence[seed_id].append({"nodes": [seed_id], "relations": [], "hop": 0})
        frontier = [(seed_id, (seed_id,), ())]
        while frontier:
            node_id, path, relations = frontier.pop(0)
            hop = len(relations)
            if hop >= normalized_config["max_hops"]:
                continue
            for neighbor, relation, _edge in adjacency.get(node_id, []):
                if neighbor in path:
                    continue
                next_path = path + (neighbor,)
                next_relations = relations + (relation,)
                if neighbor in evidence_by_id:
                    paths_by_evidence[neighbor].append(
                        {
                            "nodes": list(next_path),
                            "relations": list(next_relations),
                            "hop": hop + 1,
                        }
                    )
                frontier.append((neighbor, next_path, next_relations))

    results: list[dict[str, Any]] = []
    weights = normalized_config["weights"]
    review_scores = normalized_config["review_state_scores"]
    for evidence_id in sorted(paths_by_evidence):
        evidence = evidence_by_id[evidence_id]
        paths = paths_by_evidence[evidence_id]
        unique_paths: dict[str, dict[str, Any]] = {}
        for path in paths:
            marker = json.dumps(path, sort_keys=True, separators=(",", ":"))
            unique_paths[marker] = path
        graph_paths = sorted(unique_paths.values(), key=lambda path: (path["hop"], path["nodes"], path["relations"]))
        hop = min(path["hop"] for path in graph_paths)
        connected_claim_ids = sorted(claims_by_evidence.get(evidence_id, set()))
        claims = [claim_by_id[claim_id] for claim_id in connected_claim_ids if claim_id in claim_by_id]
        # A dense/BM25 seed is not a graph result by itself.  In particular,
        # claimless chunks must never acquire a positive score from authority,
        # review state, or hop-0 decay alone.
        if hop == 0 and not claims:
            continue
        entity_matches = any(_claim_entity_ids(claim) & set(query["entity_ids"]) for claim in claims)
        predicate_matches = any(claim.get("predicate") in query["predicates"] for claim in claims)
        polarity_matches = any(claim.get("polarity") in query["polarities"] for claim in claims)
        modality_matches = any(claim.get("modality") in query["modalities"] for claim in claims)
        # A hop-0 seed is not a semantic graph hit merely because an accepted
        # claim has authority/confidence. Require at least one requested
        # entity, predicate, polarity, or modality match.
        if hop == 0 and not (entity_matches or predicate_matches):
            continue
        stance_matches = any(
            claim.get("stance") in query["stances"] or claim.get("evidence_stance") in query["stances"]
            for claim in claims
        )
        review_state = str(evidence.get("status") or evidence.get("review_status") or "unknown")
        review_value = review_scores.get(review_state, review_scores.get("unknown", 0.0))
        document = document_by_id.get(evidence.get("document_uid"), {})
        authority = evidence.get("authority_score", document.get("authority_score", 0.0))
        authority_value = min(1.0, max(0.0, _finite_number(authority or 0.0, "authority_score")))
        confidence_value = max(
            [_finite_number(claim.get("model_confidence", 0.0), "model_confidence") for claim in claims] or [0.0]
        )
        seed_hits_for_evidence = sorted(
            seeds_by_evidence.get(evidence_id, []),
            key=lambda hit: (
                -hit["normalized_score"],
                -hit["raw_score"],
                str(hit.get("branch")),
                hit["rank"],
            ),
        )
        seed_value = seed_hits_for_evidence[0]["normalized_score"] if seed_hits_for_evidence else 0.0
        hop_value = normalized_config["hop_decay"] ** hop
        values = {
            "query_entity": 1.0 if entity_matches else 0.0,
            "predicate": 1.0 if predicate_matches else 0.0,
            "polarity": 1.0 if polarity_matches else 0.0,
            "modality": 1.0 if modality_matches else 0.0,
            "stance": 1.0 if stance_matches else 0.0,
            "review_state": review_value,
            "authority": authority_value,
            "extraction_confidence": confidence_value,
            "seed_score": seed_value,
            "hop_decay": hop_value,
        }
        roles = {
            "query_entity": "canonical entity ID match on a connected claim",
            "predicate": "predicate match on a connected claim",
            "polarity": "polarity qualifier match on a connected claim",
            "modality": "modality qualifier match on a connected claim",
            "stance": "desired/useful stance match on a connected claim",
            "review_state": "configured accepted/reviewed state score",
            "authority": "bounded source authority score",
            "extraction_confidence": "maximum validated claim confidence",
            "seed_score": "maximum retained per-branch normalized dense/BM25 seed score",
            "hop_decay": "configured decay raised to shortest graph hop",
        }
        components = {
            key: _component(values[key], weights[key], roles[key]) for key in _RETRIEVAL_WEIGHT_KEYS
        }
        graph_score = sum(item["contribution"] for item in components.values())
        if graph_score <= max(0.0, normalized_config["minimum_graph_score"]):
            continue
        relations = sorted({relation for path in graph_paths for relation in path["relations"]})
        stances = sorted(
            {
                str(claim.get("stance"))
                for claim in claims
                if isinstance(claim.get("stance"), str)
            }
        )
        trace = {
            "components": components,
            "total": graph_score,
            "relations": relations,
            "stances": stances,
            "seed_hits": seed_hits_for_evidence,
            "query_signature": query,
        }
        results.append(
            {
                **evidence,
                "candidate_id": evidence_id,
                "evidence_chunk_id": evidence_id,
                "graph_score": graph_score,
                "score": graph_score,
                "hop": hop,
                "graph_paths": graph_paths,
                "relations": relations,
                "stances": stances,
                "seed_hits": seed_hits_for_evidence,
                "trace": trace,
                "source_trace": trace,
            }
        )
    results.sort(key=lambda row: (-row["graph_score"], row["evidence_chunk_id"]))
    for rank, row in enumerate(results, start=1):
        row["branch"] = "graph"
        row["rank"] = rank
        row["raw_score"] = row["graph_score"]
    return results


def _fusion_candidate_key(hit: dict[str, Any]) -> str:
    for key in ("evidence_chunk_id", "candidate_id"):
        if isinstance(hit.get(key), str) and hit[key]:
            return hit[key]
    document_uid = hit.get("document_uid")
    chunk_id = hit.get("chunk_id")
    if isinstance(document_uid, str) and isinstance(chunk_id, str):
        return f"{document_uid}:{chunk_id}"
    if isinstance(chunk_id, str) and chunk_id:
        return chunk_id
    raise ValueError("fusion hit requires an evidence_chunk_id, candidate_id, or chunk_id")


def reciprocal_rank_fusion(
    branch_hits: dict[str, Iterable[dict[str, Any]]] | Iterable[dict[str, Any]],
    weights: dict[str, float],
    constant: float,
) -> list[dict[str, Any]]:
    """Fuse branches while retaining every rank, score, contribution, path, and stance."""

    if isinstance(branch_hits, dict):
        grouped = {str(branch): list(hits or []) for branch, hits in branch_hits.items()}
    else:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for hit in branch_hits or []:
            if not isinstance(hit, dict) or not isinstance(hit.get("branch"), str):
                raise TypeError("sequence fusion hits must include a branch")
            grouped[hit["branch"]].append(hit)
    if any(branch not in _RRF_BRANCH_NAMES for branch in grouped):
        raise ValueError("RRF branch names must be exactly dense, bm25, or graph")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("fusion weights must be a non-empty object")
    if any(str(branch) not in _RRF_BRANCH_NAMES for branch in weights):
        raise ValueError("RRF weight names must be exactly dense, bm25, or graph")
    constant_value = _finite_number(constant, "rrf constant")
    if constant_value <= 0:
        raise ValueError("rrf constant must be greater than 0")
    normalized_weights = {}
    for branch, weight in weights.items():
        normalized_weights[str(branch)] = _finite_number(weight, f"weights.{branch}")
        if normalized_weights[str(branch)] < 0:
            raise ValueError("fusion weights cannot be negative")
    candidates: dict[str, dict[str, Any]] = {}
    for branch in sorted(grouped):
        if branch not in normalized_weights:
            raise ValueError(f"missing fusion weight for branch: {branch}")
        best_hits: dict[str, dict[str, Any]] = {}
        for hit in grouped[branch]:
            if not isinstance(hit, dict):
                raise TypeError("each fusion hit must be an object")
            raw_score = _finite_number(hit.get("raw_score"), "fusion raw_score")
            if raw_score <= 0:
                continue
            graph_score = hit.get("graph_score")
            if branch == "graph":
                if (
                    isinstance(graph_score, bool)
                    or not isinstance(graph_score, (int, float))
                    or not math.isfinite(graph_score)
                    or graph_score <= 0
                ):
                    continue
            elif graph_score is not None and _finite_number(graph_score, "fusion graph_score") <= 0:
                continue
            rank = hit.get("rank")
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise ValueError("fusion rank must be a positive integer")
            key = _fusion_candidate_key(hit)
            previous = best_hits.get(key)
            ordering = (rank, -raw_score, json.dumps(hit, sort_keys=True, default=str))
            previous_ordering = (
                (previous["rank"], -previous["raw_score"], json.dumps(previous, sort_keys=True, default=str))
                if previous is not None
                else None
            )
            if previous is None or ordering < previous_ordering:
                best_hits[key] = {"hit": hit, "rank": rank, "raw_score": raw_score}
        for key in sorted(best_hits):
            hit = best_hits[key]["hit"]
            raw_score = best_hits[key]["raw_score"]
            rank = best_hits[key]["rank"]
            row = candidates.setdefault(
                key,
                {
                    "candidate_id": key,
                    "evidence_chunk_id": hit.get("evidence_chunk_id", key),
                    "chunk_id": hit.get("chunk_id"),
                    "document_uid": hit.get("document_uid"),
                    "rrf_score": 0.0,
                    "branch_traces": [],
                    "graph_paths": [],
                    "relations": [],
                    "stances": [],
                },
            )
            contribution = normalized_weights[branch] / (constant_value + rank)
            row["rrf_score"] += contribution
            graph_paths = hit.get("graph_paths", []) if isinstance(hit.get("graph_paths"), list) else []
            relations = hit.get("relations") if isinstance(hit.get("relations"), list) else [hit.get("relation")]
            stances = hit.get("stances") if isinstance(hit.get("stances"), list) else [hit.get("stance")]
            source_trace = hit.get("source_trace", hit.get("trace"))
            row["branch_traces"].append(
                {
                    "branch": branch,
                    "rank": rank,
                    "raw_score": raw_score,
                    "weight": normalized_weights[branch],
                    "weighted_contribution": contribution,
                    "score": contribution,
                    "graph_paths": list(graph_paths),
                    "relations": [relation for relation in relations if relation is not None],
                    "stances": [stance for stance in stances if stance is not None],
                    "source_trace": source_trace,
                }
            )
            for path in graph_paths:
                if path not in row["graph_paths"]:
                    row["graph_paths"].append(path)
            for relation in relations:
                if relation is not None and relation not in row["relations"]:
                    row["relations"].append(relation)
            for stance in stances:
                if stance is not None and stance not in row["stances"]:
                    row["stances"].append(stance)
    for row in candidates.values():
        row["branch_traces"].sort(key=lambda item: (item["branch"], item["rank"], item["raw_score"]))
        row["graph_paths"].sort(key=lambda path: json.dumps(path, sort_keys=True, default=str))
        row["relations"].sort()
        row["stances"].sort()
    return sorted(candidates.values(), key=lambda row: (-row["rrf_score"], row["candidate_id"]))


def _stable_sigmoid(logit: float) -> float:
    if logit >= 0:
        z = math.exp(-logit)
        return 1.0 / (1.0 + z)
    z = math.exp(logit)
    return z / (1.0 + z)


def select_evidence(
    candidates: Iterable[dict[str, Any]],
    rerank_scores: dict[str, float] | Iterable[float] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Convert reranker logits to probabilities and abstain below the configured floor."""

    normalized_config = _validate_retrieval_config(config, expansion=False)
    candidate_rows = list(candidates or [])
    for candidate in candidate_rows:
        if not isinstance(candidate, dict):
            raise TypeError("each evidence candidate must be an object")
    score_by_id: dict[str, float] = {}
    score_sequence: list[float] | None = None
    if isinstance(rerank_scores, dict):
        for key, value in rerank_scores.items():
            score_by_id[str(key)] = _finite_number(value, f"rerank_scores.{key}")
    elif rerank_scores is not None:
        score_sequence = [
            _finite_number(value, "rerank score") for value in list(rerank_scores)
        ]
        if len(score_sequence) != len(candidate_rows):
            raise ValueError("rerank score sequence must align with candidates")

    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidate_rows):
        candidate_id = candidate.get("evidence_chunk_id", candidate.get("candidate_id", candidate.get("chunk_id")))
        if score_sequence is not None:
            logit = score_sequence[index]
        elif isinstance(candidate.get("rerank_logit"), (int, float)) and not isinstance(candidate.get("rerank_logit"), bool):
            logit = _finite_number(candidate["rerank_logit"], "candidate rerank_logit")
        elif candidate_id in score_by_id:
            logit = score_by_id[candidate_id]
        else:
            raise ValueError(f"missing rerank score for candidate: {candidate_id}")
        probability = _stable_sigmoid(logit)
        row = dict(candidate)
        row["rerank_logit"] = logit
        row["rerank_probability"] = probability
        if probability >= normalized_config["minimum_rerank_probability"]:
            selected.append(row)
        else:
            rejected.append(row)
    selected.sort(
        key=lambda row: (-row["rerank_probability"], str(row.get("evidence_chunk_id", row.get("candidate_id", ""))))
    )
    selected = selected[: normalized_config["max_evidence"]]
    if not candidate_rows:
        return {"selected": [], "rejected": [], "abstained": True, "reason": "no_candidates"}
    if not selected:
        return {
            "selected": [],
            "rejected": rejected,
            "abstained": True,
            "reason": "all_candidates_below_minimum_rerank_probability",
        }
    return {"selected": selected, "rejected": rejected, "abstained": False, "reason": None}


def split_qwen_thinking(raw_output: Any) -> dict[str, Any]:
    """Separate Qwen-emitted thinking from the user-facing final content.

    Qwen-family generation can contain a complete ``<think>`` block, or only
    the closing tag when the opening tag was supplied by the chat template.
    An opening tag without a close is preserved and explicitly marked as a
    truncated generation instead of being mistaken for a final answer.
    """

    raw = str(raw_output or "")
    complete = re.search(r"<think>\s*(.*?)\s*</think>", raw, flags=re.DOTALL | re.IGNORECASE)
    if complete:
        return {
            "reasoning_content": complete.group(1).strip(),
            "final_content": (raw[0 : complete.start()] + raw[complete.end() :]).strip(),
            "thinking_status": "complete",
            "reasoning_truncated": False,
        }
    closing = re.search(r"</think>", raw, flags=re.IGNORECASE)
    if closing:
        return {
            "reasoning_content": raw[0 : closing.start()].removeprefix("<think>").strip(),
            "final_content": raw[closing.end() :].strip(),
            "thinking_status": "complete_closing_tag_only",
            "reasoning_truncated": False,
        }
    opening = re.search(r"<think>", raw, flags=re.IGNORECASE)
    if opening:
        return {
            "reasoning_content": raw[opening.end() :].strip(),
            "final_content": "",
            "thinking_status": "truncated",
            "reasoning_truncated": True,
        }
    return {
        "reasoning_content": "",
        "final_content": raw.strip(),
        "thinking_status": "not_emitted",
        "reasoning_truncated": False,
    }
