"""Pure, dependency-light evaluation and audit helpers.

The functions in this module deliberately return nullable values for missing
inputs and validate statistical inputs instead of silently coercing them.
They are also suitable for injecting verbatim into the generated notebook.
"""

from __future__ import annotations

import math
import numbers
import random
import unicodedata
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping, MutableMapping, Sequence
import hashlib
import re
from typing import Any


_CITATION_TOKEN_RE = __import__("re").compile(r"\[(E[1-9][0-9]*)\]")


class BoundedLRU(MutableMapping):
    """Small deterministic LRU cache with a hard memory bound."""

    def __init__(self, maxsize: int = 128):
        if isinstance(maxsize, bool) or not isinstance(maxsize, int) or maxsize < 1:
            raise ValueError("maxsize must be a positive integer")
        self.maxsize = maxsize
        self._data = OrderedDict()
        self.evictions = 0

    def __getitem__(self, key):
        value = self._data.pop(key)
        self._data[key] = value
        return value

    def __setitem__(self, key, value):
        if key in self._data:
            del self._data[key]
        self._data[key] = value
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)
            self.evictions += 1

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def peek(self, key, default=None):
        """Read without changing recency, useful for diagnostics."""
        return self._data.get(key, default)

    def snapshot(self) -> dict[str, Any]:
        """Return a detached JSON-safe cache diagnostic snapshot."""
        return {
            "capacity": self.maxsize,
            "size": len(self._data),
            "evictions": self.evictions,
            "items": [{"key": _json_safe_cache_value(key), "value": _json_safe_cache_value(value)} for key, value in self._data.items()],
        }


def _json_safe_cache_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe_cache_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_cache_value(item) for item in value]
    return str(value)


def derive_shard_run_name(base_name: Any, shard_index: Any = 0, shard_count: Any = 1) -> str:
    """Return a unique, validated run directory name for one shard."""

    base = normalize_optional_text(base_name)
    if base is None or "/" in base or "\\" in base or base in {".", ".."}:
        raise ValueError("shard run base name must be a non-empty path component")
    if isinstance(shard_index, bool) or not isinstance(shard_index, int):
        raise ValueError("shard index must be an integer")
    if isinstance(shard_count, bool) or not isinstance(shard_count, int):
        raise ValueError("shard count must be an integer")
    if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("invalid shard range")
    return f"{base}__shard-{shard_index}-of-{shard_count}"


def derive_effective_cache_capacity(base_capacity: Any, *, row_limit: Any, shard_count: Any = 1) -> int:
    """Derive a bounded cache capacity before config/run identity is hashed."""
    if isinstance(base_capacity, bool) or not isinstance(base_capacity, int) or base_capacity < 1:
        raise ValueError("cache capacity must be a positive integer")
    if isinstance(row_limit, bool) or not isinstance(row_limit, int) or row_limit < 0:
        raise ValueError("row limit must be a non-negative integer")
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count < 1:
        raise ValueError("shard count must be a positive integer")
    return max(base_capacity, (row_limit + shard_count - 1) // shard_count)


def identity_config(config: Mapping[str, Any], *, shard_index: Any = None, shard_count: Any = None) -> dict[str, Any]:
    """Build shard-aware identity while keeping split identity shard-independent.

    ``config_hash`` and ``split_identity_hash`` intentionally exclude shard
    assignment. ``run_identity_hash`` includes the validated shard identity so
    checkpoints and logs cannot be reused across workers.
    """

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    values = dict(config)
    configured_index = values.pop("shard_index", 0)
    configured_count = values.pop("shard_count", 1)
    index = configured_index if shard_index is None else shard_index
    count = configured_count if shard_count is None else shard_count
    if isinstance(index, bool) or not isinstance(index, int) or isinstance(count, bool) or not isinstance(count, int):
        raise ValueError("shard index/count must be integers")
    if count < 1 or index < 0 or index >= count:
        raise ValueError("invalid shard range")
    canonical = jsonable_identity(values)
    split_keys = {key: canonical[key] for key in sorted(canonical) if key not in {"shard_index", "shard_count", "run_name", "run_dir"}}
    shard = {"index": index, "count": count}
    config_hash = stable_identity_hash(canonical)
    split_identity_hash = stable_identity_hash(split_keys)
    run_identity_hash = stable_identity_hash({"config": canonical, "shard": shard})
    return {"config": canonical, "config_hash": config_hash, "split_identity_hash": split_identity_hash, "run_identity_hash": run_identity_hash, "shard": shard}


def jsonable_identity(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable_identity(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [jsonable_identity(item) for item in value]
    if hasattr(value, "as_posix"):
        return str(value)
    return value


def stable_identity_hash(value: Any) -> str:
    payload = __import__("json").dumps(jsonable_identity(value), ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_category(value: Any) -> str | None:
    """Normalize category labels before matching or stratification."""

    text = normalize_optional_text(value)
    return " ".join(text.split()).casefold() if text is not None else None


def filter_rows_by_category(rows: Sequence[Mapping[str, Any]], allowed_categories: Sequence[Any], *, category_key: str = "Category", id_key: str = "ID") -> dict[str, Any]:
    allowed = {normalize_category(value) for value in allowed_categories}
    allowed.discard(None)
    decisions = []
    reasons = []
    for row in rows:
        category = normalize_category(row.get(category_key) if isinstance(row, Mapping) else None)
        keep = category in allowed
        decisions.append(keep)
        reasons.append(None if keep else "category_not_allowed")
    result = filter_audit(rows, decisions, filter_name="category", reasons=reasons, id_key=id_key)
    result["normalized_allowed_categories"] = sorted(allowed)
    return {"rows": result.pop("kept_rows"), "manifest": result}


def quarantine_missing_references(rows: Sequence[Mapping[str, Any]], *, reference_key: str = "Counter Narrative", id_key: str = "ID") -> dict[str, Any]:
    """Keep all rows for generation while excluding missing references from scoring."""

    missing = [normalize_optional_text(row.get(reference_key) if isinstance(row, Mapping) else None) is None for row in rows]
    reasons = ["missing_reference" if flag else None for flag in missing]
    audit = filter_audit(rows, [not flag for flag in missing], filter_name="reference", reasons=reasons, id_key=id_key)
    quarantined = [{"record_id": str(row.get(id_key, index)), "row_index": index, "reason": "missing_reference", "row": row} for index, (row, flag) in enumerate(zip(rows, missing)) if flag]
    audit_rows = audit.pop("manifest_rows")
    return {"generation_rows": list(rows), "scorable_rows": audit.pop("kept_rows"), "quarantined_rows": quarantined, "manifest": {**audit, "rows": audit_rows, "purpose": "reference_metrics_only"}}


def validate_evidence_ledger(
    evidence_ledger: Sequence[Mapping[str, Any]],
    source_text_lookup: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Validate canonical evidence spans and return a detached copy.

    A ledger is an audit boundary: offsets, displayed text, and its digest
    must agree before any claim-level scoring can consume it.
    """

    if not isinstance(evidence_ledger, Sequence) or isinstance(evidence_ledger, (str, bytes)):
        raise ValueError("ledger must be a sequence")
    validated: list[dict[str, Any]] = []
    seen_evidence_ids: set[str] = set()
    for index, raw in enumerate(evidence_ledger):
        if not isinstance(raw, Mapping):
            raise ValueError(f"ledger row {index} is invalid")
        item = dict(raw)
        evidence_id = item.get("evidence_id")
        if evidence_id in seen_evidence_ids:
            raise ValueError(f"duplicate evidence_id: {evidence_id}")
        seen_evidence_ids.add(evidence_id)
        displayed = item.get("displayed_text")
        start = item.get("span_start")
        end = item.get("span_end")
        digest = item.get("displayed_text_sha256", item.get("text_sha256"))
        if not isinstance(evidence_id, str) or not _CITATION_TOKEN_RE.fullmatch(f"[{evidence_id}]"):
            raise ValueError(f"ledger evidence_id invalid: {evidence_id!r}")
        if not isinstance(displayed, str) or not displayed:
            raise ValueError(f"ledger displayed_text invalid for {evidence_id}")
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
            raise ValueError(f"ledger offsets invalid for {evidence_id}")
        expected_digest = __import__("hashlib").sha256(displayed.encode("utf-8")).hexdigest()
        if digest != expected_digest:
            raise ValueError(f"ledger displayed_text hash mismatch for {evidence_id}")
        source_text = item.get("source_text")
        source_key = item.get("source_text_key")
        if source_text is None and source_text_lookup is not None:
            if not isinstance(source_key, str) or source_key not in source_text_lookup:
                raise ValueError(f"ledger source text lookup missing for {evidence_id}")
            source_text = source_text_lookup[source_key]
        if source_text is not None and (not isinstance(source_text, str) or end > len(source_text)):
            raise ValueError(f"ledger offsets exceed source text for {evidence_id}")
        if source_text is not None and start >= end:
            raise ValueError(f"ledger offsets empty for {evidence_id}")
        if source_text is not None and item.get("source_text_sha256") != __import__("hashlib").sha256(source_text.encode("utf-8")).hexdigest():
            raise ValueError(f"ledger source text hash mismatch for {evidence_id}")
        evidence_text = item.get("evidence_text")
        if source_text is not None and (not isinstance(source_text, str) or source_text[start:end] != displayed):
            raise ValueError(f"ledger source span mismatch for {evidence_id}")
        if source_text is None and evidence_text is not None and evidence_text != displayed:
            raise ValueError(f"ledger evidence text mismatch for {evidence_id}")
        validated.append(item)
    return validated


def select_evidence_within_budget(
    evidence_ledger: Sequence[Mapping[str, Any]],
    char_budget: int,
) -> list[dict[str, Any]]:
    """Select a prefix of whole evidence spans without slicing any span."""

    if not isinstance(char_budget, int) or char_budget < 1:
        raise ValueError("evidence budget must be a positive integer")
    ledger = validate_evidence_ledger(evidence_ledger)
    selected: list[dict[str, Any]] = []
    used = 0
    for item in ledger:
        cost = len(item["displayed_text"])
        if used + cost > char_budget:
            if not selected:
                raise ValueError("evidence budget overflow: first span does not fit")
            break
        selected.append(item)
        used += cost
    return selected


def _sentence_spans(text: str) -> list[dict[str, Any]]:
    """Split text without losing source offsets or inter-sentence whitespace."""
    import re

    if not isinstance(text, str):
        text = str(text)
    if not text:
        return []
    spans: list[dict[str, Any]] = []
    start = 0
    # A sentence ends at terminal punctuation followed by whitespace or EOF.
    # Newlines also delimit prose when no terminal punctuation is present.
    boundary = re.compile(r"[.!?。！？।॥](?:[\"'»”’)]*)?(?=\s|$)|\n+")
    for match in boundary.finditer(text):
        end = match.end()
        if text[start:end].strip():
            spans.append({"text": text[start:end], "start_char": start, "end_char": end})
        start = end
    if start < len(text) and text[start:].strip():
        spans.append({"text": text[start:], "start_char": start, "end_char": len(text)})
    return spans


def sentence_aligned_windows(
    text: Any,
    max_chars: int,
    overlap: int = 0,
) -> list[dict[str, Any]]:
    """Return non-overlapping windows packed at sentence boundaries.

    Every returned ``text`` is an exact slice of the input, and therefore its
    ``start_char``/``end_char`` offsets can be used to audit what was shown to
    a model. ``overlap`` is accepted for call-site compatibility but is
    intentionally not applied: overlapping windows duplicate evidence and
    make citation necessity impossible to audit. A sentence longer than the
    limit is split only at whitespace (with an explicit ``oversized_sentence``
    marker), so the hard character bound is never violated.
    """

    if not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    if not isinstance(overlap, int) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer")
    source = "" if text is None else str(text)
    sentences = _sentence_spans(source)
    if not sentences:
        return []
    windows: list[dict[str, Any]] = []
    index = 0
    while index < len(sentences):
        sentence = sentences[index]
        if len(sentence["text"]) > max_chars:
            # Preserve the original span while making bounded word pieces.
            cursor = sentence["start_char"]
            limit = sentence["end_char"]
            while cursor < limit:
                proposed = min(limit, cursor + max_chars)
                if proposed < limit:
                    boundary = source.rfind(" ", cursor + 1, proposed + 1)
                    if boundary > cursor:
                        proposed = boundary
                if proposed <= cursor:
                    proposed = min(limit, cursor + max_chars)
                windows.append({
                    "text": source[cursor:proposed],
                    "start_char": cursor,
                    "end_char": proposed,
                    "sentence_start": sentence["start_char"],
                    "sentence_end": sentence["end_char"],
                    "sentence_aligned": False,
                    "split_reason": "oversized_sentence",
                })
                cursor = proposed
            index += 1
            continue
        end_index = index + 1
        while end_index < len(sentences):
            candidate_end = sentences[end_index]["end_char"]
            if candidate_end - sentence["start_char"] > max_chars:
                break
            end_index += 1
        final_end = sentences[end_index - 1]["end_char"]
        windows.append({
            "text": source[sentence["start_char"]:final_end],
            "start_char": sentence["start_char"],
            "end_char": final_end,
            "sentence_start": sentence["start_char"],
            "sentence_end": final_end,
            "sentence_aligned": True,
            "split_reason": None,
        })
        index = end_index
    return windows


def _claim_text(value: Any) -> str:
    import re

    value = _CITATION_TOKEN_RE.sub("", str(value or ""))
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"\s+", " ", value).strip().casefold()
    return value.rstrip(" .!?。！？।॥")


def build_claim_citation_records(
    narrative: Any,
    evidence_ledger: Sequence[Mapping[str, Any]],
    *,
    factual_claims: Sequence[Any] | None = None,
    abstention_validator: Callable[[str], bool] | None = None,
    safe_non_factual_validator: Callable[[str], bool] | None = None,
) -> list[dict[str, Any]]:
    """Create auditable sentence-level claim/citation records.

    An empty ``factual_claims`` declaration is not trusted on its own. It is a
    safe abstention only when the caller supplies and passes a deterministic
    ``abstention_validator``; otherwise every non-empty narrative sentence is
    treated as factual. Unmatched factual claims are added as uncited records
    so omissions cannot be hidden by sentence segmentation.
    """

    validated_ledger = validate_evidence_ledger(evidence_ledger)
    by_id = {str(item.get("evidence_id")): item for item in validated_ledger}
    supplied = None if factual_claims is None else [_claim_text(value) for value in factual_claims if _claim_text(value)]
    narrative_text = "" if narrative is None else str(narrative)
    safe_validator = safe_non_factual_validator or abstention_validator
    sentences = _sentence_spans(narrative_text)
    records: list[dict[str, Any]] = []
    matched: set[str] = set()
    for index, span in enumerate(sentences):
        raw_start = span["start_char"]
        raw_end = span["end_char"]
        while raw_start < raw_end and narrative_text[raw_start].isspace(): raw_start += 1
        while raw_end > raw_start and narrative_text[raw_end - 1].isspace(): raw_end -= 1
        claim = narrative_text[raw_start:raw_end]
        normalized = _claim_text(claim)
        declared = supplied is not None and normalized in supplied
        ids = _CITATION_TOKEN_RE.findall(claim)
        safe_non_factual = False
        if safe_validator is not None:
            try: safe_non_factual = bool(safe_validator(claim))
            except Exception: safe_non_factual = False
        factual = bool(normalized) and (declared or bool(ids) or not safe_non_factual)
        abstention_authorized = factual_claims == [] and safe_non_factual and safe_validator is not None
        if factual:
            matched.add(normalized)
        ids = _CITATION_TOKEN_RE.findall(claim)
        unknown = sorted(set(ids) - set(by_id))
        records.append({
            "claim_index": index,
            "claim": claim,
            "claim_text": _CITATION_TOKEN_RE.sub("", claim).strip(),
            "start_char": raw_start,
            "end_char": raw_end,
            "is_factual": factual,
            "factual_claim": factual,
            "abstention_authorized": abstention_authorized,
            "evidence_ids": ids,
            "citation_ids": ids,
            "unknown_evidence_ids": unknown,
            "citation_format_valid": not unknown and len(ids) == len(set(ids)),
            "evidence_spans": [str(by_id[item].get("displayed_text", by_id[item].get("text", ""))) for item in ids if item in by_id],
        })
    if supplied is not None:
        for claim in factual_claims or []:
            normalized = _claim_text(claim)
            if normalized and normalized not in matched:
                records.append({
                    "claim_index": len(records), "claim": str(claim).strip(), "claim_text": str(claim).strip(),
                    "start_char": None, "end_char": None, "is_factual": True, "factual_claim": True,
                    "abstention_authorized": False,
                    "evidence_ids": [], "citation_ids": [], "unknown_evidence_ids": [],
                    "citation_format_valid": True, "evidence_spans": [],
                })
                matched.add(normalized)
    return records


def aggregate_citation_support(
    records: Sequence[Mapping[str, Any]],
    *,
    evaluator: Callable[[str, str], float] | None,
    require_nli: bool = False,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Aggregate claim-level support with joint and leave-one-out entailment.

    The callback is deliberately tiny and injectable: ``(premise, hypothesis)
    -> score``. Format compliance is reported independently from entailment;
    resolving ``[E#]`` never counts as factual support.
    """

    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    factual = [row for row in records if bool(row.get("is_factual", row.get("factual_claim", False)))]
    all_ids = [item for row in records for item in row.get("evidence_ids", [])]
    valid_ids = [item for row in records for item in row.get("evidence_ids", []) if item not in set(row.get("unknown_evidence_ids", []))]
    format_result = {
        "valid": all(bool(row.get("citation_format_valid", False)) for row in records),
        "citation_count": len(all_ids),
        "valid_citation_count": len(valid_ids),
        "unknown_evidence_ids": sorted({str(item) for row in records for item in row.get("unknown_evidence_ids", [])}),
        "duplicate_citation_count": sum(max(0, len(row.get("evidence_ids", [])) - len(set(row.get("evidence_ids", [])))) for row in records),
    }
    abstention_authorized = bool(records) and all(bool(row.get("abstention_authorized", False)) for row in records)
    abstention = abstention_authorized and not all_ids and not factual
    base = {
        "format_compliance": format_result,
        "syntactic_citation_precision": (len(valid_ids) / len(all_ids)) if all_ids else None,
        "factual_claim_count": len(factual), "supported_claim_count": 0,
        "claim_citation_recall": 0.0 if factual else None, "necessary_citation_count": 0,
        "citation_recall": 0.0 if factual else None,
        "citation_necessity": None,
        "overcitation_count": 0, "citation_precision": None, "claim_results": [],
        "abstention": abstention, "entailment_mean": None, "citation_entailment": None,
        "evaluated_claim_count": 0, "incomplete_claim_count": 0, "evaluated_claim_entailment_mean": None,
    }
    if not factual:
        base.update({"entailment_status": "not_required" if abstention else "invalid_abstention", "pass": bool(format_result["valid"] and abstention)})
        return base
    if evaluator is None:
        if require_nli:
            base.update({"entailment_status": "unavailable", "incomplete_claim_count": len(factual), "pass": False})
        else:
            base.update({"entailment_status": "not_requested", "pass": bool(format_result["valid"])})
        return base
    scores: list[float] = []
    for row in factual:
        ids = list(row.get("evidence_ids", []))
        spans = list(row.get("evidence_spans", []))
        claim = str(row.get("claim_text", row.get("claim", ""))).strip()
        if not ids or len(spans) != len(ids) or row.get("unknown_evidence_ids"):
            base["claim_results"].append({"claim": claim, "evidence_ids": ids, "joint_score": None, "supported": False, "necessary_evidence_ids": [], "leave_one_out": {}})
            base["incomplete_claim_count"] += 1
            continue
        try:
            def checked_score(premise: str) -> float:
                score = float(evaluator(premise, claim))
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError("evaluator score must be finite and in [0, 1]")
                return score
            joint = checked_score("\n\n".join(spans))
            individual = [checked_score(span) for span in spans]
            leave_one_out = {}
            necessary_ids = []
            for index, evidence_id in enumerate(ids):
                remaining = "\n\n".join(spans[:index] + spans[index + 1:])
                leave_score = checked_score(remaining) if remaining else 0.0
                leave_one_out[evidence_id] = leave_score
                if joint >= threshold and leave_score < threshold:
                    necessary_ids.append(evidence_id)
        except Exception:
            base["evaluated_claim_entailment_mean"] = sum(scores) / len(scores) if scores else None
            base["incomplete_claim_count"] = len(factual) - len(scores)
            base.update({"entailment_status": "evaluator_error", "pass": False})
            return base
        supported = joint >= threshold
        scores.append(joint)
        base["evaluated_claim_count"] += 1
        base["claim_results"].append({"claim": claim, "evidence_ids": ids, "joint_score": joint, "individual_scores": individual, "supported": supported, "necessary_evidence_ids": necessary_ids, "leave_one_out": leave_one_out})
        base["supported_claim_count"] += int(supported)
        base["necessary_citation_count"] += len(necessary_ids)
        if supported:
            base["overcitation_count"] += max(0, len(ids) - len(necessary_ids))
        else:
            # With no joint entailment, none of the cited passages supports the
            # claim; every cited passage is therefore unnecessary.
            base["overcitation_count"] += len(ids)
    base["claim_citation_recall"] = base["supported_claim_count"] / len(factual)
    base["citation_recall"] = base["claim_citation_recall"]
    base["citation_precision"] = base["necessary_citation_count"] / len(all_ids) if all_ids else 0.0
    base["citation_necessity"] = base["necessary_citation_count"] / len(valid_ids) if valid_ids else 0.0
    base["evaluated_claim_entailment_mean"] = sum(scores) / len(scores) if scores else None
    if base["incomplete_claim_count"]:
        base["entailment_status"] = "scored_incomplete"
        base["pass"] = False
    else:
        base["entailment_mean"] = base["evaluated_claim_entailment_mean"]
        base["citation_entailment"] = base["entailment_mean"]
        base["entailment_status"] = "scored"
        base["pass"] = bool(format_result["valid"] and base["supported_claim_count"] == len(factual) and base["overcitation_count"] == 0)
    return base


def normalize_optional_text(value: Any) -> str | None:
    """Return normalized non-empty text, or ``None`` for a missing value.

    Only actual missing values are removed.  The literal string ``"nan"`` is
    valid user text and therefore is not treated as a missing reference.
    """

    if value is None:
        return None
    if isinstance(value, numbers.Real):
        try:
            if math.isnan(value):
                return None
        except (TypeError, ValueError):
            pass
    is_nan = getattr(value, "is_nan", None)
    if callable(is_nan) and is_nan():
        return None
    # pandas.NA and similar scalar sentinels cannot be used in a boolean test.
    try:
        if value is not value:  # NaN-like objects whose identity is stable.
            return None
    except Exception:
        pass
    if type(value).__name__ in {"NAType", "NaTType"}:
        return None
    # numpy datetime64/timedelta64 NaT values are not Real and expose no
    # pandas-style sentinel API; their canonical string representation is
    # stable and can be checked without importing numpy.
    if type(value).__name__ in {"datetime64", "timedelta64"} and str(value) == "NaT":
        return None
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
    text = unicodedata.normalize("NFKC", text).strip()
    return text or None


def fit_prompt_to_budget(
    prompt: Any,
    tokenizer: Any,
    max_input_tokens: int,
    *,
    schema_tail: Any = None,
    reserve_output_tokens: int = 0,
) -> str:
    """Preflight a rendered prompt without implicit tokenizer truncation.

    The helper never silently cuts prompt content. Callers must bound evidence
    explicitly before rendering; a schema tail, when supplied, is appended to
    the exact text being measured and therefore cannot disappear at the right
    edge of a tokenizer window.
    """

    if not isinstance(max_input_tokens, int) or isinstance(max_input_tokens, bool) or max_input_tokens < 1:
        raise ValueError("max_input_tokens must be a positive integer")
    if not isinstance(reserve_output_tokens, int) or isinstance(reserve_output_tokens, bool) or reserve_output_tokens < 0:
        raise ValueError("reserve_output_tokens must be a non-negative integer")
    if reserve_output_tokens >= max_input_tokens:
        raise ValueError("reserved output leaves no input budget")
    text = "" if prompt is None else str(prompt)
    tail = normalize_optional_text(schema_tail)
    if tail:
        text = text.rstrip() + "\n" + tail
    encoded = tokenizer(text=[text], return_tensors="pt", padding=False, truncation=False)
    mask = encoded.get("attention_mask") if isinstance(encoded, Mapping) else encoded["attention_mask"]
    if hasattr(mask, "sum") and not isinstance(mask, (list, tuple)):
        try:
            length = int(mask.sum().item())
        except AttributeError:
            length = int(mask.sum())
    else:
        length = sum(int(value) for value in mask[0])
    available = max_input_tokens - reserve_output_tokens
    if length > available:
        raise RuntimeError(f"prompt_token_budget_exceeded:{jsonable_identity({'prompt_tokens': length, 'input_budget': available, 'reserve_output_tokens': reserve_output_tokens, 'schema_tail_preserved': bool(tail)})}")
    return text


def fit_adaptive_prompt_with_evidence(
    evidence_ledger: Sequence[Mapping[str, Any]],
    *,
    prompt_builder: Callable[[Sequence[Mapping[str, Any]], str], str],
    payload_candidates: Sequence[str] = ("",),
    token_counter: Callable[[str], int],
    max_input_tokens: int,
    reserve_output_tokens: int = 0,
) -> dict[str, Any]:
    """Fit a prompt by dropping only whole ranked evidence spans.

    ``prompt_builder`` owns the immutable post/target/schema-tail text. The
    helper only varies the ranked evidence prefix and deterministic payload
    candidates, so it can never slice an evidence span or schema tail. A base
    prompt that cannot fit with no evidence and the smallest payload is
    returned as an explicit quarantine result instead of raising a whole-run
    overflow error.
    """

    if not isinstance(max_input_tokens, int) or isinstance(max_input_tokens, bool) or max_input_tokens < 1:
        raise ValueError("max_input_tokens must be a positive integer")
    if not isinstance(reserve_output_tokens, int) or isinstance(reserve_output_tokens, bool) or reserve_output_tokens < 0:
        raise ValueError("reserve_output_tokens must be a non-negative integer")
    if reserve_output_tokens >= max_input_tokens:
        raise ValueError("reserved output leaves no input budget")
    if not callable(prompt_builder) or not callable(token_counter):
        raise TypeError("prompt_builder and token_counter must be callable")
    ledger = validate_evidence_ledger(evidence_ledger)
    ranked = [
        item
        for _, item in sorted(
            enumerate(ledger),
            key=lambda pair: (
                float(pair[1].get("rank", pair[0])) if str(pair[1].get("rank", pair[0])).replace(".", "", 1).isdigit() else float(pair[0]),
                pair[0],
            ),
        )
    ]
    candidates = [str(payload) for payload in payload_candidates] or [""]
    available = max_input_tokens - reserve_output_tokens
    attempts: list[dict[str, Any]] = []
    for payload_index, payload in enumerate(candidates):
        for count in range(len(ranked), -1, -1):
            selected = ranked[:count]
            prompt = str(prompt_builder(selected, payload))
            prompt_tokens = int(token_counter(prompt))
            attempt = {"payload_index": payload_index, "selected_evidence_count": count, "prompt_tokens": prompt_tokens}
            attempts.append(attempt)
            if prompt_tokens <= available:
                selected_ids = [str(item["evidence_id"]) for item in selected]
                return {
                    "status": "fit",
                    "prompt": prompt,
                    "payload": payload,
                    "payload_index": payload_index,
                    "prompt_tokens": prompt_tokens,
                    "available_input_tokens": available,
                    "selected_evidence_ids": selected_ids,
                    "dropped_evidence_ids": [str(item["evidence_id"]) for item in ranked[count:]],
                    "attempts": attempts,
                }
    return {
        "status": "quarantine",
        "reason": "prompt_budget_irreducible",
        "prompt": None,
        "payload": candidates[-1],
        "payload_index": len(candidates) - 1,
        "prompt_tokens": attempts[-1]["prompt_tokens"] if attempts else None,
        "available_input_tokens": available,
        "selected_evidence_ids": [],
        "dropped_evidence_ids": [str(item["evidence_id"]) for item in ranked],
        "attempts": attempts,
    }


_SCRIPT_RANGES = {
    "tamil": ((0x0B80, 0x0BFF),),
    "devanagari": ((0x0900, 0x097F),),
    "bengali": ((0x0980, 0x09FF),),
    "telugu": ((0x0C00, 0x0C7F),),
    "kannada": ((0x0C80, 0x0CFF),),
    "malayalam": ((0x0D00, 0x0D7F),),
    "arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)),
    "cyrillic": ((0x0400, 0x04FF),),
    "greek": ((0x0370, 0x03FF),),
    "hebrew": ((0x0590, 0x05FF),),
    "thai": ((0x0E00, 0x0E7F),),
    "hangul": ((0x1100, 0x11FF), (0xAC00, 0xD7AF)),
    "han": ((0x3400, 0x4DBF), (0x4E00, 0x9FFF)),
    "japanese": ((0x3040, 0x30FF),),
}


def _script_for_char(char: str) -> str | None:
    codepoint = ord(char)
    if (0x41 <= codepoint <= 0x5A) or (0x61 <= codepoint <= 0x7A) or (
        0x00C0 <= codepoint <= 0x024F
    ):
        return "latin"
    for script, ranges in _SCRIPT_RANGES.items():
        if any(start <= codepoint <= end for start, end in ranges):
            return script
    return None


def script_bucket(text: Any) -> str:
    """Classify the scripts represented by alphabetic Unicode characters.

    ``mixed`` is intentional when a response contains more than one script;
    punctuation, digits, and whitespace do not make a script mixed.
    """

    normalized = normalize_optional_text(text)
    if normalized is None:
        return "unknown"
    scripts = {_script_for_char(char) for char in normalized}
    scripts.discard(None)
    if not scripts:
        return "unknown"
    if len(scripts) == 1:
        return next(iter(scripts))
    return "mixed"


_LANGUAGE_LABELS = {
    "en": "en", "eng": "en", "english": "en",
    "ta": "ta", "tam": "ta", "tamil": "ta",
    "hi": "hi", "hin": "hi", "hindi": "hi",
    "fr": "fr", "fra": "fr", "fre": "fr", "french": "fr",
    "es": "es", "spa": "es", "spanish": "es",
    "de": "de", "deu": "de", "ger": "de", "german": "de",
    "pt": "pt", "por": "pt", "portuguese": "pt",
    "it": "it", "ita": "it", "italian": "it",
    "ru": "ru", "rus": "ru", "russian": "ru",
    "zh": "zh", "zho": "zh", "chi": "zh", "chinese": "zh",
    "ja": "ja", "jpn": "ja", "japanese": "ja",
    "ko": "ko", "kor": "ko", "korean": "ko",
}


def _language_label(value: Any) -> str | None:
    normalized = normalize_optional_text(value)
    if normalized is None:
        return None
    return _LANGUAGE_LABELS.get(normalized.casefold())


def language_match(expected: Any, observed: Any) -> bool | None:
    """Compare trusted language labels, returning ``None`` for unknown labels.

    This deliberately does not infer language from script: many languages
    share Latin or Han scripts. Use :func:`script_match` for a script-only
    diagnostic, never as a language claim.
    """

    expected_label = _language_label(expected)
    observed_label = _language_label(observed)
    if expected_label is None or observed_label is None:
        return None
    return expected_label == observed_label


def script_match(expected: Any, observed: Any) -> bool | None:
    """Compare detected scripts for diagnostic reporting only."""

    expected_script = script_bucket(expected)
    observed_script = script_bucket(observed)
    if "unknown" in {expected_script, observed_script}:
        return None
    return expected_script == observed_script and observed_script != "mixed"


def filter_audit(
    rows: Sequence[Any],
    keep: Callable[[Any], bool] | Sequence[bool],
    *,
    filter_name: str = "filter",
    reasons: Sequence[str | None] | None = None,
    id_key: str = "ID",
) -> dict[str, Any]:
    """Apply a filter while returning rows and an explicit audit manifest."""

    if callable(keep):
        decisions = [bool(keep(row)) for row in rows]
    else:
        decisions = [bool(value) for value in keep]
        if len(decisions) != len(rows):
            raise ValueError("filter decisions must align with rows")
    if reasons is None:
        reasons = [None if decision else filter_name for decision in decisions]
    else:
        reasons = list(reasons)
        if len(reasons) != len(rows):
            raise ValueError("filter reasons must align with rows")
    kept_rows = [row for row, decision in zip(rows, decisions) if decision]
    dropped_rows = [row for row, decision in zip(rows, decisions) if not decision]
    manifest_rows = []
    for index, (row, decision, reason) in enumerate(zip(rows, decisions, reasons)):
        row_id = row.get(id_key) if isinstance(row, Mapping) else None
        manifest_rows.append({"index": index, "record_id": None if row_id is None else str(row_id), "filter": filter_name, "kept": bool(decision), "reason": None if decision else (reason or filter_name)})
    events = [item for item in manifest_rows if not item["kept"]]
    reason_counts = Counter(item["reason"] for item in events)
    return {
        "filter": filter_name,
        "input": len(rows),
        "kept": len(kept_rows),
        "dropped": len(dropped_rows),
        "kept_rows": kept_rows,
        "dropped_rows": dropped_rows,
        "events": events,
        "rows": manifest_rows,
        "manifest_rows": manifest_rows,
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def _clean_metric_text(value: Any) -> str | None:
    text = normalize_optional_text(value)
    return text.casefold() if text is not None else None


def unicode_chrf(reference: Any, hypothesis: Any, *, max_order: int = 6, beta: float = 2.0) -> float | None:
    """Compute a Unicode character n-gram F-score without an ASCII tokenizer."""

    if max_order < 1 or beta <= 0:
        raise ValueError("max_order must be positive and beta must be positive")
    ref = _clean_metric_text(reference)
    hyp = _clean_metric_text(hypothesis)
    if ref is None or hyp is None:
        return None
    # Whitespace is a boundary artifact, not a language character.  Keeping
    # all other code points makes this valid for Tamil, Devanagari, and emoji.
    ref = "".join(ref.split())
    hyp = "".join(hyp.split())
    if not ref or not hyp:
        return 0.0
    scores: list[float] = []
    beta_sq = beta * beta
    available_order = min(max_order, len(ref), len(hyp))
    for order in range(1, available_order + 1):
        ref_ngrams = Counter(ref[index : index + order] for index in range(len(ref) - order + 1))
        hyp_ngrams = Counter(hyp[index : index + order] for index in range(len(hyp) - order + 1))
        overlap = sum((ref_ngrams & hyp_ngrams).values())
        precision = overlap / sum(hyp_ngrams.values()) if hyp_ngrams else 0.0
        recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0.0
        score = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall) if precision and recall else 0.0
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


# Scientific-evaluation runtime contracts.  These helpers intentionally accept
# small injected callables so unit tests never need a model, a network, or a
# GPU.  Every metric is represented by ``{value, status, reason}``; a missing
# value is never converted to zero for an aggregate.
SUPPORTED_EVAL_LANGUAGES = frozenset({"en", "hi", "ta"})
ENGLISH_ONLY_METRICS = frozenset({"rouge_l", "meteor", "detoxify"})
BERTSCORE_MODEL_ID = "FacebookAI/xlm-roberta-large"
# A commit/revision, rather than ``main`` or an unpinned tag, is required for
# reproducible BERTScore.  Deployments may replace this with their audited
# revision through the explicit config argument.
BERTSCORE_MODEL_REVISION = "91703fe22dd9e5054634ccfb5b875f12f69158ec"
BERTSCORE_NUM_LAYERS = 24
NLI_MODEL_ID = "joeddav/xlm-roberta-large-xnli"
NLI_MODEL_REVISION = "07f8772bf0306314a97e4913cafde2cabf9814a9"
NLI_DATASET_PROVENANCE = {
    "en": {"dataset_id": "facebook/xnli", "dataset_revision": "072e4eb2b447bd887a772a7ab826ce0a7222b782", "config": "en", "split": "validation"},
    "hi": {"dataset_id": "mteb/IndicXnliPairClassification", "dataset_revision": "027e97b9afe84ea3447b57b7705b8864bb2b3a83", "config": "hi", "split": "test", "source_format": "parquet", "columns": {"sentence1": "premise", "sentence2": "hypothesis", "labels": "label"}},
    "ta": {"dataset_id": "mteb/IndicXnliPairClassification", "dataset_revision": "027e97b9afe84ea3447b57b7705b8864bb2b3a83", "config": "ta", "split": "test", "source_format": "parquet", "columns": {"sentence1": "premise", "sentence2": "hypothesis", "labels": "label"}},
}
DEFAULT_BERTSCORE_SETTINGS = {
    "model_id": BERTSCORE_MODEL_ID,
    "revision": BERTSCORE_MODEL_REVISION,
    "num_layers": BERTSCORE_NUM_LAYERS,
    "languages": {language: {"model_id": BERTSCORE_MODEL_ID, "revision": BERTSCORE_MODEL_REVISION} for language in sorted(SUPPORTED_EVAL_LANGUAGES)},
}
globals().update({"DEFAULT_BERTSCORE_CONFIG": DEFAULT_BERTSCORE_SETTINGS})


def load_nli_dataset_rows(dataset_loader: Callable[..., Any], provenance: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a pinned NLI split and normalize its data-only schema.

    IndicXNLI is intentionally consumed as the ``mteb`` parquet export.  This
    rejects the old executable dataset-script repository before any network
    loader is called, which is required for ``datasets>=4`` compatibility.
    The returned provenance includes a content digest for the exact rows used
    by calibration; callers should persist it with the calibration artifact.
    """

    if not isinstance(provenance, Mapping):
        raise ValueError("nli provenance must be a mapping")
    dataset_id = str(provenance.get("dataset_id") or "")
    if dataset_id.casefold() == "divyanshu/indicxnli":
        raise ValueError("legacy NLI dataset script is not allowed; use data-only parquet")
    revision = str(provenance.get("dataset_revision") or "")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("NLI dataset revision must be an immutable commit")
    config = str(provenance.get("config") or "")
    split = str(provenance.get("split") or "")
    if dataset_id == "mteb/IndicXnliPairClassification":
        if config not in {"hi", "ta"} or split != "test" or provenance.get("source_format") != "parquet":
            raise ValueError("IndicXNLI must use the pinned data-only parquet test split")
        columns = provenance.get("columns")
        expected_columns = {"sentence1": "premise", "sentence2": "hypothesis", "labels": "label"}
        if columns != expected_columns:
            raise ValueError("IndicXNLI parquet column mapping is invalid")
    dataset = dataset_loader(dataset_id, config, split=split, revision=revision)
    raw_rows = [dict(row) for row in dataset]
    required = ("sentence1", "sentence2", "labels") if dataset_id == "mteb/IndicXnliPairClassification" else ("premise", "hypothesis", "label")
    if any(not all(column in row for column in required) for row in raw_rows):
        raise ValueError("NLI dataset parquet columns are missing")
    rows = []
    for index, row in enumerate(raw_rows):
        if dataset_id == "mteb/IndicXnliPairClassification":
            premise, hypothesis, label = row["sentence1"], row["sentence2"], row["labels"]
        else:
            premise, hypothesis, label = row["premise"], row["hypothesis"], row["label"]
        if not isinstance(premise, str) or not isinstance(hypothesis, str):
            raise ValueError("NLI sentence columns must contain strings")
        try:
            label = int(label)
        except (TypeError, ValueError) as exc:
            raise ValueError("NLI labels must be integer class IDs") from exc
        if label not in {0, 1, 2}:
            raise ValueError("NLI labels must be in {0, 1, 2}")
        row_id = row.get("pairID", row.get("id", index))
        rows.append({"id": str(row_id), "premise": premise, "hypothesis": hypothesis, "label": label})
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("NLI example IDs must be unique")
    digest = normalized_dataset_content_hash(rows)
    enriched = dict(provenance)
    enriched["dataset_content_hash"] = digest
    enriched["selected_content_hash"] = digest
    enriched["row_count"] = len(rows)
    enriched["schema_revision"] = "nli-pair-normalization.v1"
    return rows, enriched


def multilingual_bertscore_config(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a detached, pinned BERTScore config for en/hi/ta."""

    config = {
        "model_id": DEFAULT_BERTSCORE_SETTINGS["model_id"],
        "revision": DEFAULT_BERTSCORE_SETTINGS["revision"],
        "num_layers": DEFAULT_BERTSCORE_SETTINGS["num_layers"],
        "languages": {key: dict(value) for key, value in DEFAULT_BERTSCORE_SETTINGS["languages"].items()},
    }
    if overrides:
        for key in ("model_id", "revision"):
            if key in overrides:
                value = normalize_optional_text(overrides[key])
                if value is None or value.casefold() in {"main", "latest", "none"}:
                    raise ValueError(f"BERTScore {key} must be pinned")
                config[key] = value
        if "languages" in overrides:
            for language, value in overrides["languages"].items():
                label = _language_label(language)
                if label not in SUPPORTED_EVAL_LANGUAGES or not isinstance(value, Mapping):
                    raise ValueError("BERTScore language config is invalid")
                model_id = normalize_optional_text(value.get("model_id", config["model_id"]))
                revision = normalize_optional_text(value.get("revision", config["revision"]))
                if model_id is None or revision is None or revision.casefold() in {"main", "latest", "none"}:
                    raise ValueError("BERTScore language config must be pinned")
                config["languages"][label] = {"model_id": model_id, "revision": revision}
        if "num_layers" in overrides:
            if isinstance(overrides["num_layers"], bool) or not isinstance(overrides["num_layers"], int) or overrides["num_layers"] < 1:
                raise ValueError("BERTScore num_layers must be positive")
            config["num_layers"] = overrides["num_layers"]
    return config


def _metric_result(value: Any = None, status: str = "unavailable", reason: str | None = None) -> dict[str, Any]:
    if status == "scored":
        try:
            if value is None or not math.isfinite(float(value)):
                return {"value": None, "status": "invalid_score", "reason": "nonfinite_metric_value"}
            value = float(value)
        except (TypeError, ValueError):
            return {"value": None, "status": "invalid_score", "reason": "non_numeric_metric_value"}
    return {"value": value, "status": status, "reason": reason}


def _detector_language(detector: Any, text: Any) -> str | None:
    if detector is None or normalize_optional_text(text) is None:
        return None
    try:
        raw = detector(text) if callable(detector) else detector.detect_language_of(str(text))
    except Exception:
        return None
    # lingua returns an enum whose name is e.g. ``TAMIL``; langdetect returns
    # a two-letter code.  Normalize both without treating script as language.
    raw_name = getattr(raw, "name", raw)
    raw_name = str(raw_name).split(".")[-1].replace("LANGUAGE", "").strip(" _-")
    return _language_label(raw_name) or _language_label(raw)


def load_language_detector(detector_factory: Callable[[], Any] | None = None) -> Callable[[str], str | None] | None:
    """Load an offline lingua detector, with no network/download fallback.

    A caller-supplied factory is used in tests and production wrappers.  If
    lingua is absent, ``None`` is returned and the notebook records language
    metrics as unavailable rather than silently guessing from script.
    """

    if detector_factory is not None:
        detector = detector_factory()
        if detector is None:
            return None
        return lambda text: _detector_language(detector, text)
    try:
        from lingua import Language, LanguageDetectorBuilder
        detector = LanguageDetectorBuilder.from_languages(
            Language.ENGLISH, Language.HINDI, Language.TAMIL
        ).build()
        return lambda text: _detector_language(detector, text)
    except Exception:
        return None


def evaluate_multilingual_record(
    record: Mapping[str, Any],
    *,
    language_detector: Callable[[str], Any] | None = None,
    rouge_scorer: Callable[..., Any] | None = None,
    meteor_scorer: Callable[..., Any] | None = None,
    bertscore_scorer: Callable[..., Any] | None = None,
    detoxify_scorer: Callable[..., Any] | None = None,
    bertscore_config: Mapping[str, Any] | None = None,
    reference_key: str = "Counter Narrative",
    response_key: str = "parsed_counter_narrative",
) -> dict[str, Any]:
    """Score one record with explicit language/status provenance.

    English-only ROUGE/METEOR/Detoxify callbacks are never invoked for Hindi,
    Tamil, or unknown languages.  BERTScore and Unicode chrF are multilingual;
    BERTScore still requires a configured scorer and a pinned model revision.
    """

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    reference = normalize_optional_text(record.get(reference_key))
    hypothesis = normalize_optional_text(record.get(response_key, record.get("response")))
    expected = _language_label(record.get("language", record.get("Language")))
    input_language = _detector_language(language_detector, record.get("Text"))
    reference_language = _detector_language(language_detector, reference)
    output_language = _detector_language(language_detector, hypothesis)
    language = expected or input_language
    detected = [value for value in (input_language, reference_language, output_language) if value is not None]
    if language is None or language not in SUPPORTED_EVAL_LANGUAGES:
        language_status = "unsupported_language" if language else "unavailable"
    elif language_detector is None or input_language is None:
        language_status = "unavailable"
    elif any(value != language for value in detected):
        language_status = "mismatch"
    else:
        language_status = "scored"
    result: dict[str, Any] = {
        "ID": str(record.get("ID", "")),
        "language": language,
        "language_status": language_status,
        "expected_language": expected,
        "input_language": input_language,
        "reference_language": reference_language,
        "output_language": output_language,
        "language_match": language_match(language, output_language),
        "input_language_match": language_match(language, input_language),
        "reference_language_match": language_match(language, reference_language),
        "script_match": script_match(record.get("Text"), hypothesis),
        "reference_available": reference is not None,
        "metrics": {},
    }
    missing_status = "excluded_missing_reference" if reference is None else None
    result["metrics"]["chrf"] = _metric_result(
        unicode_chrf(reference, hypothesis), "scored" if reference is not None and hypothesis is not None else (missing_status or "missing_hypothesis"),
        None if reference is not None and hypothesis is not None else ("missing_reference" if reference is None else "missing_hypothesis"),
    )
    config = multilingual_bertscore_config(bertscore_config)
    if language_status == "mismatch":
        result["metrics"]["bertscore"] = _metric_result(status="language_mismatch", reason="detected_language_mismatch")
    elif language_status != "scored" or language not in SUPPORTED_EVAL_LANGUAGES:
        result["metrics"]["bertscore"] = _metric_result(status="unsupported_language", reason="language_not_supported")
    elif reference is None:
        result["metrics"]["bertscore"] = _metric_result(status="excluded_missing_reference", reason="missing_reference")
    elif hypothesis is None:
        result["metrics"]["bertscore"] = _metric_result(status="missing_hypothesis", reason="missing_hypothesis")
    elif bertscore_scorer is None:
        result["metrics"]["bertscore"] = _metric_result(status="unavailable", reason="bertscore_scorer_unavailable")
    else:
        try:
            payload = bertscore_scorer(reference=reference, hypothesis=hypothesis, language=language, model_id=config["languages"][language]["model_id"], revision=config["languages"][language]["revision"])
            value = payload.get("f1") if isinstance(payload, Mapping) else payload
            result["metrics"]["bertscore"] = _metric_result(value, "scored")
        except Exception as exc:
            result["metrics"]["bertscore"] = _metric_result(status="error", reason=f"bertscore_error:{type(exc).__name__}")
    for name, scorer in (("rouge_l", rouge_scorer), ("meteor", meteor_scorer), ("detoxify", detoxify_scorer)):
        if language_status == "mismatch":
            result["metrics"][name] = _metric_result(status="language_mismatch", reason="detected_language_mismatch")
            continue
        if language_status != "scored" or language != "en":
            result["metrics"][name] = _metric_result(status="unsupported_language", reason="english_only_metric")
            continue
        if name != "detoxify" and (reference is None or hypothesis is None):
            result["metrics"][name] = _metric_result(status=missing_status or "missing_hypothesis", reason="missing_reference" if reference is None else "missing_hypothesis")
            continue
        if scorer is None:
            result["metrics"][name] = _metric_result(status="unavailable", reason=f"{name}_scorer_unavailable")
            continue
        try:
            if name == "detoxify":
                value = scorer(hypothesis)
            else:
                value = scorer(reference, hypothesis)
            if isinstance(value, Mapping):
                value = value.get("rougeL", value.get("meteor", value.get("score", value.get("value"))))
            result["metrics"][name] = _metric_result(value, "scored")
        except Exception as exc:
            result["metrics"][name] = _metric_result(status="error", reason=f"{name}_error:{type(exc).__name__}")
    return result


def evaluate_multilingual_records(records: Sequence[Mapping[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
    return [evaluate_multilingual_record(row, **kwargs) for row in records]


def summarize_metric_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return language/metric aggregates using only explicitly scored values."""

    grouped: dict[tuple[str, str], list[float]] = {}
    statuses: Counter[tuple[str, str, str]] = Counter()
    for row in records:
        language = row.get("language") or "unknown"
        for metric, item in (row.get("metrics") or {}).items():
            item = item if isinstance(item, Mapping) else {"value": item, "status": "scored"}
            status = str(item.get("status", "unavailable")); statuses[(str(language), metric, status)] += 1
            try:
                value = float(item.get("value"))
                if status == "scored" and math.isfinite(value): grouped.setdefault((str(language), metric), []).append(value)
            except (TypeError, ValueError):
                pass
    output = []
    keys = sorted(set(grouped) | {(language, metric) for language, metric, _ in statuses})
    for language, metric in keys:
        values = grouped.get((language, metric), [])
        status_counts = {status: count for (lang, met, status), count in statuses.items() if (lang, met) == (language, metric)}
        output.append({"language": language, "metric": metric, "n_scored": len(values), "mean": sum(values) / len(values) if values else None, "status_counts": dict(sorted(status_counts.items()))})
    return output


def summarize_language_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize detector outputs and match rates by expected language."""
    grouped: dict[str, dict[str, Any]] = {}
    for row in records:
        language = str(row.get("language") or row.get("expected_language") or "unknown")
        item = grouped.setdefault(language, {"language": language, "records": 0, "input_language_counts": Counter(), "reference_language_counts": Counter(), "output_language_counts": Counter(), "input_matches": 0, "reference_matches": 0, "output_matches": 0, "input_status": Counter(), "reference_status": Counter(), "output_status": Counter()})
        item["records"] += 1
        for field, count_key, status_key, match_key in (("input_language", "input_language_counts", "input_status", "input_language_match"), ("reference_language", "reference_language_counts", "reference_status", "reference_language_match"), ("output_language", "output_language_counts", "output_status", "language_match")):
            detected = row.get(field) or "unknown"; item[count_key][str(detected)] += 1
            match = row.get(match_key)
            item["%s_matches" % field.split("_")[0] if field == "input_language" else ("reference_matches" if field == "reference_language" else "output_matches")] += int(match is True)
            item[status_key]["matched" if match is True else "mismatched" if match is False else "unavailable"] += 1
    output = []
    for language, item in sorted(grouped.items()):
        records_count = item["records"]
        output.append({"language": language, "records": records_count, "input_language_counts": dict(item["input_language_counts"]), "reference_language_counts": dict(item["reference_language_counts"]), "output_language_counts": dict(item["output_language_counts"]), "input_match_rate": item["input_matches"] / records_count if records_count else None, "reference_match_rate": item["reference_matches"] / records_count if records_count else None, "output_match_rate": item["output_matches"] / records_count if records_count else None, "input_status": dict(item["input_status"]), "reference_status": dict(item["reference_status"]), "output_status": dict(item["output_status"])})
    return output


def validate_metric_rows_unique(rows: Sequence[Mapping[str, Any]], *, id_key: str = "ID", variant_key: str = "variant") -> bool:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get(id_key)), str(row.get(variant_key)))
        if key in seen:
            raise ValueError(f"duplicate metric row for ID/variant: {key[0]}/{key[1]}")
        seen.add(key)
    return True


def build_reference_metric_audit(
    rows: Sequence[Mapping[str, Any]], *, metrics: Sequence[str], language_key: str = "language"
) -> dict[str, Any]:
    """Report denominators and exclusions independently for every metric."""
    validate_metric_rows_unique(rows)
    output: dict[str, Any] = {}
    for metric in metrics:
        item: dict[str, Any] = {"input": len(rows), "scorable": 0, "excluded_by_reason": Counter(), "language_counts": Counter(), "coverage": None}
        for row in rows:
            language = str(row.get(language_key) or "unknown"); item["language_counts"][language] += 1
            status = str(row.get(f"{metric}_status") or "unavailable")
            value = row.get(metric)
            try: valid = status == "scored" and value is not None and math.isfinite(float(value))
            except (TypeError, ValueError): valid = False
            if valid: item["scorable"] += 1
            else:
                reason = str(row.get(f"{metric}_reason") or ("missing_reference" if row.get("reference_available") is False else status))
                item["excluded_by_reason"][reason] += 1
        item["coverage"] = item["scorable"] / item["input"] if item["input"] else None
        output[metric] = {"input": item["input"], "scorable": item["scorable"], "excluded_by_reason": dict(item["excluded_by_reason"]), "language_counts": dict(item["language_counts"]), "coverage": item["coverage"]}
    return output


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return BH-adjusted q-values in the original order."""

    values = [float(value) for value in p_values]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("p-values must be finite numbers in [0, 1]")
    count = len(values)
    if not count:
        return []
    ranked = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    adjusted = [0.0] * count
    running = 1.0
    for rank in range(count - 1, -1, -1):
        original_index, p_value = ranked[rank]
        running = min(running, p_value * count / (rank + 1))
        adjusted[original_index] = min(1.0, max(0.0, running))
    return adjusted


def _paired_values(
    rows: Any,
    variant_a: str,
    variant_b: str,
    id_key: str,
) -> tuple[list[str], list[float], list[float], dict[str, int]]:
    if isinstance(rows, Mapping):
        rows = [dict(row, **{id_key: key}) if isinstance(row, Mapping) else {id_key: key, variant_a: row[0], variant_b: row[1]} for key, row in rows.items()]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("rows must be a sequence of mappings")
    aligned: dict[str, tuple[float, float]] = {}
    seen_raw: set[tuple[str, str]] = set()
    seen_canonical: dict[str, tuple[str, str]] = {}
    exclusions: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, Mapping) or id_key not in row:
            raise ValueError(f"each row must contain {id_key!r}")
        raw_id = row[id_key]
        if _invalid_pair_id(raw_id):
            raise ValueError(f"invalid paired ID: {raw_id!r}")
        raw_token = (type(raw_id).__qualname__, repr(raw_id))
        identifier = str(raw_id)
        if raw_token in seen_raw:
            raise ValueError(f"duplicate paired ID: {identifier}")
        seen_raw.add(raw_token)
        prior_raw = seen_canonical.get(identifier)
        if prior_raw is not None and prior_raw != raw_token:
            raise ValueError(f"ID collision after canonicalization: {identifier}")
        seen_canonical[identifier] = raw_token
        left, right = row.get(variant_a), row.get(variant_b)
        left_status, right_status = _score_status(left), _score_status(right)
        if left_status == "nonfinite" or right_status == "nonfinite":
            exclusions["nonfinite"] += 1
            continue
        if left_status == "missing" or right_status == "missing":
            exclusions["incomplete"] += 1
            continue
        if left_status == "invalid" or right_status == "invalid":
            exclusions["invalid"] += 1
            continue
        left, right = float(left), float(right)
        aligned[identifier] = (left, right)
    identifiers = sorted(aligned)
    left = [aligned[key][0] for key in identifiers]
    right = [aligned[key][1] for key in identifiers]
    return identifiers, left, right, dict(exclusions)


def _invalid_pair_id(value: Any) -> bool:
    if value is None or type(value).__name__ in {"NAType", "NaTType"}:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, numbers.Number):
        try:
            if isinstance(value, numbers.Complex) and not isinstance(value, numbers.Real):
                return not (math.isfinite(float(value.real)) and math.isfinite(float(value.imag)))
            return not math.isfinite(float(value))
        except (TypeError, ValueError):
            return False
    if type(value).__name__ in {"datetime64", "timedelta64"}:
        return str(value) == "NaT"
    return False


def _score_status(value: Any) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return "missing"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "invalid"
    return "valid" if math.isfinite(number) else "nonfinite"


def paired_variant_tests(
    rows: Any,
    *,
    variant_a: str = "mp_kg_rag",
    variant_b: str = "kg_rag",
    id_key: str = "ID",
    seed: int = 0,
    permutations: int = 10_000,
) -> dict[str, Any]:
    """Compare two variants on the exact shared ID set using sign permutations."""

    if permutations < 1:
        raise ValueError("permutations must be positive")
    identifiers, left, right, exclusion_counts = _paired_values(rows, variant_a, variant_b, id_key)
    differences = [a - b for a, b in zip(left, right)]
    n = len(differences)
    mean_difference = sum(differences) / n if n else None
    tolerance = 1e-12
    wins = sum(value > tolerance for value in differences)
    ties = sum(abs(value) <= tolerance for value in differences)
    losses = n - wins - ties
    result: dict[str, Any] = {
        "ids": identifiers,
        "n": n,
        "mean_difference": mean_difference,
        "win_rate": wins / n if n else None,
        "tie_rate": ties / n if n else None,
        "loss_rate": losses / n if n else None,
        "p_value": None,
        "seed": seed,
        "permutations": permutations,
        "excluded_rows": sum(exclusion_counts.values()),
        "exclusion_counts": exclusion_counts,
    }
    if not n:
        result["p_value"] = None
        result["status"] = "insufficient_pairs"
        return result
    if not any(abs(value) > tolerance for value in differences):
        result["p_value"] = 1.0
        return result
    observed = abs(mean_difference)
    if n <= 16:
        signs = range(1 << n)
        exceedances = 0
        total = 0
        for mask in signs:
            signed_mean = sum(value if mask & (1 << index) else -value for index, value in enumerate(differences)) / n
            exceedances += abs(signed_mean) >= observed - 1e-15
            total += 1
        result["p_value"] = exceedances / total
    else:
        rng = random.Random(seed)
        exceedances = 0
        for _ in range(permutations):
            signed_mean = sum(value if rng.getrandbits(1) else -value for value in differences) / n
            exceedances += abs(signed_mean) >= observed - 1e-15
        result["p_value"] = (exceedances + 1) / (permutations + 1)
    return result


DEFAULT_VARIANT_COMPARISONS = (
    ("mp_kg_rag", "kg_rag"),
    ("mp_kg_rag", "qwen_zero_shot"),
    ("kg_rag", "qwen_zero_shot"),
)


def pairwise_metric_family(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
    comparisons: Sequence[tuple[str, str]] = DEFAULT_VARIANT_COMPARISONS,
    id_key: str = "ID",
    variant_key: str = "variant",
    seed: int = 0,
    permutations: int = 10_000,
    directions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run all declared paired variant/metric tests and one BH family.

    Rows may be long (``ID``, ``variant``, metric columns) or wide (``ID`` and
    variant columns).  Missing/unsupported metric values are excluded by the
    tested paired helper, never filled with zero.  Every p-value in the full
    metric-by-comparison grid participates in one correction family.
    """

    if not metrics:
        raise ValueError("at least one metric is required")
    validate_metric_rows_unique(rows, id_key=id_key, variant_key=variant_key)
    long_rows = [row for row in rows if isinstance(row, Mapping) and variant_key in row]
    outputs: list[dict[str, Any]] = []
    p_locations: list[tuple[int, str]] = []
    for left_variant, right_variant in comparisons:
        comparison_result = {"comparison": f"{left_variant}_vs_{right_variant}", "variant_a": left_variant, "variant_b": right_variant, "metrics": {}}
        for metric in metrics:
            wide: dict[str, dict[str, Any]] = {}
            if long_rows:
                for row in long_rows:
                    identifier = row.get(id_key)
                    if identifier is None or row.get(variant_key) not in {left_variant, right_variant}:
                        continue
                    value = row.get(metric)
                    status = row.get(f"{metric}_status")
                    # A metric status is authoritative when present.  In
                    # particular, non-RAG variants must remain null for
                    # citation support metrics instead of becoming artificial
                    # zero scores in a paired test.
                    if status is not None and str(status) != "scored":
                        value = None
                    wide.setdefault(str(identifier), {id_key: identifier})[str(row[variant_key])] = value
            else:
                wide = {}
                for row in rows:
                    if not isinstance(row, Mapping) or id_key not in row:
                        continue
                    detached = dict(row)
                    status = detached.get(f"{metric}_status")
                    if status is not None and str(status) != "scored":
                        for variant in (left_variant, right_variant):
                            if variant in detached:
                                detached[variant] = None
                    wide[str(row[id_key])] = detached
            paired_rows = [value for value in wide.values()]
            direction = (directions or {}).get(metric, "higher")
            if direction not in {"higher", "lower"}:
                raise ValueError(f"invalid metric direction: {metric}")
            if direction == "lower":
                paired_rows = [dict(row, **{left_variant: row.get(right_variant), right_variant: row.get(left_variant)}) for row in paired_rows]
            result = paired_variant_tests(paired_rows, variant_a=left_variant, variant_b=right_variant, id_key=id_key, seed=seed, permutations=permutations)
            result["direction"] = direction
            result["metric"] = metric
            comparison_result["metrics"][metric] = result
            if result.get("p_value") is not None:
                p_locations.append((len(outputs), metric))
        outputs.append(comparison_result)
    q_values = benjamini_hochberg([outputs[index]["metrics"][metric]["p_value"] for index, metric in p_locations]) if p_locations else []
    for q_value, (index, metric) in zip(q_values, p_locations):
        outputs[index]["metrics"][metric]["q_value"] = q_value
        outputs[index]["metrics"][metric]["bh_family"] = "declared_metric_x_comparison"
    # Make the correction auditable even when a test has no paired rows.
    for comparison in outputs:
        for metric, result in comparison["metrics"].items():
            result.setdefault("q_value", None)
            result.setdefault("bh_family", "declared_metric_x_comparison")
    return {"comparisons": outputs, "metrics": list(metrics), "declared_comparisons": [f"{a}_vs_{b}" for a, b in comparisons], "bh_family": "declared_metric_x_comparison", "bh_family_size": len(p_locations), "seed": seed, "permutations": permutations}


def bootstrap_mean_ci(
    values: Sequence[Any],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Return a seeded percentile bootstrap CI, excluding missing values."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    clean: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean.append(number)
    if not clean:
        return {
            "n": 0,
            "mean": None,
            "lower": None,
            "upper": None,
            "low": None,
            "high": None,
            "confidence": confidence,
            "seed": seed,
        }
    mean = sum(clean) / len(clean)
    rng = random.Random(seed)
    samples = []
    for _ in range(n_resamples):
        samples.append(sum(clean[rng.randrange(len(clean))] for _ in clean) / len(clean))
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    lower = samples[min(len(samples) - 1, max(0, int(math.floor(alpha * len(samples)))))]
    upper_index = min(len(samples) - 1, max(0, int(math.ceil((1.0 - alpha) * len(samples))) - 1))
    upper = samples[upper_index]
    return {
        "n": len(clean),
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "low": lower,
        "high": upper,
        "confidence": confidence,
        "seed": seed,
    }


def _rater_matrix(
    rows: Any,
    rater_columns: Sequence[str] | None = None,
    *,
    exact_two: bool = False,
) -> list[list[Any]]:
    if isinstance(rows, Mapping):
        if not rows:
            raise ValueError("at least two raters are required")
        columns = list(rater_columns or rows.keys())
        if len(columns) < 2:
            raise ValueError("at least two raters are required")
        if exact_two and len(columns) != 2:
            raise ValueError("exactly two raters are required")
        values = [rows[column] for column in columns]
        if not all(isinstance(value, Sequence) and not isinstance(value, (str, bytes)) for value in values):
            raise TypeError("rater mapping values must be sequences")
        if len({len(value) for value in values}) != 1:
            raise ValueError("rater columns must have equal lengths")
        return [list(row) for row in zip(*values)]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("rows must be a sequence")
    if rater_columns is not None:
        columns = list(rater_columns)
        if len(columns) < 2:
            raise ValueError("at least two raters are required")
        matrix = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("rater_columns require mapping rows")
            missing = [column for column in columns if column not in row]
            if missing:
                raise ValueError(f"missing rater column: {missing[0]}")
            matrix.append([row[column] for column in columns])
    else:
        matrix = [list(row.values()) if isinstance(row, Mapping) else list(row) for row in rows]
    if matrix and len(matrix[0]) < 2:
        raise ValueError("at least two raters are required")
    if exact_two and matrix and len(matrix[0]) != 2:
        raise ValueError("exactly two raters are required")
    if matrix and any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("each row must contain the same number of raters")
    if not matrix:
        raise ValueError("at least two raters are required")
    return matrix


def weighted_kappa_rows(rows: Any, *, rater_columns: Sequence[str] | None = None, weights: str = "quadratic") -> float:
    """Compute weighted Cohen kappa for exactly two rater columns."""

    matrix = _rater_matrix(rows, rater_columns, exact_two=True)
    pairs = [(row[0], row[1]) for row in matrix if row[0] is not None and row[1] is not None]
    if not pairs:
        return float("nan")
    if weights not in {"linear", "quadratic"}:
        raise ValueError("weights must be 'linear' or 'quadratic'")
    categories = sorted({value for pair in pairs for value in pair})
    if len(categories) == 1:
        return 1.0
    positions = {category: index for index, category in enumerate(categories)}
    size = len(categories) - 1
    def agreement(a: Any, b: Any) -> float:
        distance = abs(positions[a] - positions[b]) / size
        return 1.0 - (distance if weights == "linear" else distance * distance)
    total = len(pairs)
    observed = sum(agreement(a, b) for a, b in pairs) / total
    left_counts = Counter(a for a, _ in pairs)
    right_counts = Counter(b for _, b in pairs)
    expected = sum(
        (left_counts[a] / total) * (right_counts[b] / total) * agreement(a, b)
        for a in categories for b in categories
    )
    return 1.0 if math.isclose(expected, 1.0) else (observed - expected) / (1.0 - expected)


def krippendorff_alpha_ordinal(rows: Any, *, rater_columns: Sequence[str] | None = None) -> float:
    """Compute ordinal Krippendorff alpha using coincidence matrices.

    Each unit is normalized by its own valid-rater count. This is important
    when missing annotations leave different units with different numbers of
    ratings; averaging raw pair distances is not equivalent.
    """

    matrix = _rater_matrix(rows, rater_columns)
    units = [[value for value in row if value is not None] for row in matrix]
    units = [unit for unit in units if len(unit) >= 2]
    if not units:
        return float("nan")
    categories = sorted({value for unit in units for value in unit})
    positions = {category: index for index, category in enumerate(categories)}
    denominator = max(1, len(categories) - 1)

    def distance(a: Any, b: Any) -> float:
        delta = (positions[a] - positions[b]) / denominator
        return delta * delta

    # coincidence[c][k] is the expected number of ordered coincidences for
    # category c and k within units, with each unit divided by n_u - 1.
    coincidence: Counter[tuple[Any, Any]] = Counter()
    for unit in units:
        counts = Counter(unit)
        unit_size = len(unit)
        for left in categories:
            for right in categories:
                if left == right:
                    count = counts[left] * max(0, counts[left] - 1)
                else:
                    count = counts[left] * counts[right]
                coincidence[(left, right)] += count / (unit_size - 1)
    total_coincidences = sum(coincidence.values())
    if total_coincidences <= 0:
        return float("nan")

    observed = sum(value * distance(left, right) for (left, right), value in coincidence.items()) / total_coincidences
    marginal = Counter()
    for (left, _), value in coincidence.items():
        marginal[left] += value
    expected = 0.0
    for left in categories:
        for right in categories:
            if left == right:
                pair_mass = marginal[left] * max(0.0, marginal[left] - 1.0)
            else:
                pair_mass = marginal[left] * marginal[right]
            expected += pair_mass * distance(left, right)
    expected /= total_coincidences * max(1.0, total_coincidences - 1.0)
    if math.isclose(expected, 0.0):
        return 1.0 if math.isclose(observed, 0.0) else 0.0
    return 1.0 - observed / expected


def sample_annotation_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_ids: int,
    seed: int = 0,
    id_key: str = "ID",
    variant_key: str = "variant",
    stratify_key: str = "stratify_key",
) -> dict[str, Any]:
    """Sample IDs by stratum while retaining every variant for each ID."""

    if isinstance(max_ids, bool) or not isinstance(max_ids, int) or max_ids < 1:
        raise ValueError("max_ids must be a positive integer")
    groups: dict[str, list[tuple[str, list[Mapping[str, Any]]]]] = {}
    by_id: dict[str, list[Mapping[str, Any]]] = {}
    strata: dict[str, str] = {}
    raw_ids: dict[str, Any] = {}
    for row in rows:
        if not isinstance(row, Mapping) or id_key not in row:
            raise ValueError("annotation row must contain ID")
        identifier = str(row[id_key])
        if not identifier:
            raise ValueError("annotation ID must be non-empty")
        by_id.setdefault(identifier, []).append(row)
        raw_ids.setdefault(identifier, row[id_key])
        strata.setdefault(identifier, str(row.get(stratify_key, "unknown")))
    for identifier, values in by_id.items():
        groups.setdefault(strata[identifier], []).append((identifier, values))
    rng = random.Random(seed)
    for values in groups.values():
        values.sort(key=lambda item: item[0]); rng.shuffle(values)
    all_ids = sorted(by_id)
    selected: list[str] = []
    # Round-robin over shuffled strata gives every non-empty language/script
    # stratum a chance before filling the remainder from the global pool.
    cursors = {key: 0 for key in groups}
    while len(selected) < min(max_ids, len(all_ids)):
        progressed = False
        for stratum in sorted(groups):
            values = groups[stratum]
            cursor = cursors[stratum]
            if cursor < len(values) and values[cursor][0] not in selected:
                selected.append(values[cursor][0]); cursors[stratum] += 1; progressed = True
                if len(selected) >= max_ids: break
        if not progressed: break
    selected_set = set(selected)
    sampled_rows = [row for row in rows if str(row[id_key]) in selected_set]
    return {
        "rows": sampled_rows,
        "selected_ids": selected,
        "selected_id_count": len(selected),
        "input_id_count": len(all_ids),
        "max_ids": max_ids,
        "seed": seed,
        "stratify_key": stratify_key,
        "stratum_counts": {key: sum(1 for identifier in selected if strata[identifier] == key) for key in sorted(groups)},
        "variants_per_id": {identifier: sorted({str(row.get(variant_key)) for row in by_id[identifier]}) for identifier in selected},
    }


def human_agreement_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    id_key: str = "ID",
    rater_key: str = "rater",
    score_key: str = "score",
) -> dict[str, Any]:
    """Compute overlap/missingness, pairwise weighted kappa, and ordinal alpha."""

    if not rows:
        raise ValueError("at least two distinct raters are required")
    raters = sorted({str(row.get(rater_key, row.get("rater_id"))) for row in rows if row.get(rater_key, row.get("rater_id")) is not None})
    if len(raters) < 2:
        raise ValueError("at least two distinct raters are required")
    matrix: dict[str, dict[str, Any]] = {}
    for row in rows:
        if id_key not in row:
            raise ValueError("annotation row must contain ID")
        rater = row.get(rater_key, row.get("rater_id")); identifier = str(row[id_key])
        if rater is None:
            raise ValueError("annotation row must contain rater")
        rater = str(rater)
        if rater in matrix.setdefault(identifier, {}):
            raise ValueError(f"duplicate annotation for {identifier}/{rater}")
        matrix[identifier][rater] = row.get(score_key, row.get("rating"))
    ordered_rows = [[matrix[identifier].get(rater) for rater in raters] for identifier in sorted(matrix)]
    overlap_count = sum(sum(value is not None for value in row) >= 2 for row in ordered_rows)
    expected_cells = len(ordered_rows) * len(raters)
    observed_cells = sum(value is not None for row in ordered_rows for value in row)
    pairwise: dict[str, float] = {}
    for left_index, left in enumerate(raters):
        for right in raters[left_index + 1:]:
            key = f"{left}_vs_{right}"
            pairs = [{"left": row.get(left), "right": row.get(right)} for row in matrix.values()]
            value = weighted_kappa_rows(pairs, rater_columns=("left", "right"))
            pairwise[key] = None if isinstance(value, float) and math.isnan(value) else value
    alpha_value = krippendorff_alpha_ordinal(ordered_rows)
    alpha = None if isinstance(alpha_value, float) and math.isnan(alpha_value) else alpha_value
    return {
        "distinct_raters": len(raters), "raters": raters, "unit_count": len(ordered_rows),
        "overlap_count": overlap_count, "missingness_count": expected_cells - observed_cells,
        "observed_cells": observed_cells, "expected_cells": expected_cells,
        "weighted_kappa": pairwise if len(pairwise) != 1 else next(iter(pairwise.values())),
        "weighted_kappa_pairwise": pairwise, "krippendorff_alpha_ordinal": alpha,
        "status": "scored" if overlap_count else "insufficient_overlap",
    }


def build_nli_calibration_artifact(
    *, model_id: str, model_revision: str, dataset_id: str, dataset_revision: str,
    language: str, split: str, label_mapping: Mapping[str, Any], threshold: float,
    n_examples: int, example_ids: Sequence[Any], example_content_digest: str,
    dataset_content_hash: str | None = None,
    accuracy: float, per_label_stats: Mapping[str, Any], code_hash: str, eval_core_hash: str,
    calibration_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an authenticated held-out calibration record.

    The digest covers every provenance and quality field.  Verification must be
    rerun at startup; callers must not treat a hand-edited JSON file as a gate.
    """
    label = _language_label(language)
    required_text = (model_id, model_revision, dataset_id, dataset_revision, label, split, example_content_digest, code_hash, eval_core_hash)
    if any(normalize_optional_text(value) is None for value in required_text) or not isinstance(label_mapping, Mapping) or not label_mapping:
        raise ValueError("incomplete calibration provenance")
    if not isinstance(n_examples, int) or n_examples < 1 or len(example_ids) != n_examples:
        raise ValueError("calibration example count mismatch")
    if not 0.0 <= float(threshold) <= 1.0 or not 0.0 <= float(accuracy) <= 1.0:
        raise ValueError("calibration scores must be in [0, 1]")
    if any(str(key).startswith("LABEL_") for key in label_mapping):
        if not validate_nli_label_mapping(label_mapping, kind="model")["valid"]: raise ValueError("model label mapping mismatch")
    elif all(str(key) in {"0", "1", "2"} for key in label_mapping):
        if not validate_nli_label_mapping(label_mapping, kind="dataset")["valid"]: raise ValueError("dataset label mapping mismatch")
    payload = {
        "schema": "mpkg-nli-calibration.v2", "model_id": str(model_id), "model_revision": str(model_revision),
        "dataset_id": str(dataset_id), "dataset_revision": str(dataset_revision), "dataset_content_hash": str(dataset_content_hash or ""), "language": label, "split": str(split),
        "label_mapping": jsonable_identity(label_mapping), "threshold": float(threshold), "n_examples": n_examples,
        "example_ids": [str(value) for value in example_ids], "example_content_digest": str(example_content_digest),
        "accuracy": float(accuracy), "per_label_stats": jsonable_identity(per_label_stats), "code_hash": str(code_hash), "eval_core_hash": str(eval_core_hash),
    }
    if calibration_metadata is not None:
        payload["calibration"] = jsonable_identity(calibration_metadata)
    payload["artifact_digest"] = stable_identity_hash(payload)
    return payload


def verify_nli_calibration_artifact(
    artifact: Mapping[str, Any] | None, *, model_id: str, model_revision: str,
    dataset_id: str, dataset_revision: str, language: str, split: str | None = None, dataset_content_hash: str | None = None,
    dataset_examples: Sequence[Mapping[str, Any]] | None = None,
    label_mapping: Mapping[str, Any] | None = None,
    code_hash: str | None = None, eval_core_hash: str | None = None,
    fresh_predictor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    calibration_seed: int | None = None, calibration_min_support: int | None = None,
    calibration_bootstrap: int | None = None, calibration_criterion: str | None = None,
    calibration_entailment_label: int | None = None,
) -> dict[str, Any]:
    """Authenticate calibration provenance and reject stale/forged records."""
    label = _language_label(language)
    if not isinstance(artifact, Mapping) or artifact.get("language") != label:
        return {"enabled": False, "status": "unverified", "reason": "language_mismatch"}
    expected = {"model_id": model_id, "model_revision": model_revision, "dataset_id": dataset_id, "dataset_revision": dataset_revision}
    if split is not None: expected["split"] = split
    if dataset_content_hash is not None: expected["dataset_content_hash"] = dataset_content_hash
    if dataset_examples is not None and dataset_content_hash != normalized_dataset_content_hash(dataset_examples):
        return {"enabled": False, "status": "unverified", "reason": "dataset_content_digest_mismatch"}
    if dataset_examples is not None:
        try:
            metadata_reason = _validate_calibration_metadata(artifact, dataset_examples)
        except (TypeError, ValueError, KeyError):
            metadata_reason = "calibration_metadata_invalid"
        if metadata_reason:
            return {"enabled": False, "status": "unverified", "reason": metadata_reason}
        expected_ids = [str(row.get("id")) for row in dataset_examples]
        observed_ids = [str(value) for value in artifact.get("example_ids", [])]
        if observed_ids != expected_ids:
            return {"enabled": False, "status": "unverified", "reason": "calibration_example_ids_mismatch"}
        expected_example_digest = stable_identity_hash([{key: row.get(key) for key in ("id", "premise", "hypothesis", "label")} for row in dataset_examples])
        if artifact.get("example_content_digest") != expected_example_digest:
            return {"enabled": False, "status": "unverified", "reason": "calibration_example_digest_mismatch"}
    if label_mapping is not None and artifact.get("label_mapping") != jsonable_identity(label_mapping):
        return {"enabled": False, "status": "unverified", "reason": "label_mapping_mismatch"}
    if fresh_predictor is not None:
        if dataset_examples is None or None in (calibration_seed, calibration_min_support, calibration_bootstrap, calibration_criterion, calibration_entailment_label):
            return {"enabled": False, "status": "unverified", "reason": "fresh_calibration_configuration_missing"}
        try:
            fresh = calibrate_nli_threshold(dataset_examples, fresh_predictor, seed=int(calibration_seed), min_support=int(calibration_min_support), n_bootstrap=int(calibration_bootstrap), criterion=str(calibration_criterion), entailment_label=int(calibration_entailment_label))
            stored = artifact.get("calibration")
            if not isinstance(stored, Mapping):
                return {"enabled": False, "status": "unverified", "reason": "fresh_calibration_metadata_missing"}
            if abs(float(artifact.get("threshold")) - float(fresh["threshold"])) > 1e-12 or int(stored.get("seed")) != int(fresh["seed"]) or stored.get("criterion") != fresh["criterion"] or int(stored.get("entailment_label")) != int(fresh["entailment_label"]):
                return {"enabled": False, "status": "unverified", "reason": "fresh_calibration_threshold_or_config_mismatch"}
            if sorted(map(str, stored.get("calibration_ids", []))) != sorted(map(str, fresh["calibration_ids"])) or sorted(map(str, stored.get("audit_ids", []))) != sorted(map(str, fresh["audit_ids"])) or stored.get("support") != fresh["support"]:
                return {"enabled": False, "status": "unverified", "reason": "fresh_calibration_split_mismatch"}
            fresh_audit = fresh["metrics"]["audit"]; stored_audit = stored.get("metrics", {}).get("audit", {})
            for key in ("accuracy",):
                if abs(float(stored_audit.get(key)) - float(fresh_audit[key])) > 1e-12 or abs(float(artifact.get(key)) - float(fresh_audit[key])) > 1e-12:
                    return {"enabled": False, "status": "unverified", "reason": "fresh_calibration_accuracy_mismatch"}
            for key in ("accuracy_ci",):
                if stored_audit.get(key) != fresh_audit.get(key):
                    return {"enabled": False, "status": "unverified", "reason": "fresh_calibration_accuracy_ci_mismatch"}
            fresh_entailment = fresh_audit["entailment"]; stored_entailment = stored_audit.get("entailment", {})
            for key in ("precision", "recall", "f1", "precision_ci", "recall_ci", "f1_ci"):
                if stored_entailment.get(key) != fresh_entailment.get(key):
                    return {"enabled": False, "status": "unverified", "reason": f"fresh_calibration_{key}_mismatch"}
        except (TypeError, ValueError, KeyError, AttributeError):
            return {"enabled": False, "status": "unverified", "reason": "fresh_calibration_recompute_failed"}
    if code_hash is not None: expected["code_hash"] = code_hash
    if eval_core_hash is not None: expected["eval_core_hash"] = eval_core_hash
    if any(artifact.get(key) != value for key, value in expected.items()):
        return {"enabled": False, "status": "unverified", "reason": "calibration_provenance_mismatch"}
    digest_payload = {key: artifact.get(key) for key in artifact if key != "artifact_digest"}
    if artifact.get("artifact_digest") != stable_identity_hash(digest_payload):
        return {"enabled": False, "status": "unverified", "reason": "calibration_digest_mismatch"}
    try:
        if int(artifact.get("n_examples")) < 1 or not 0.0 <= float(artifact.get("threshold")) <= 1.0 or not normalize_optional_text(artifact.get("example_content_digest")):
            raise ValueError
        if len(artifact.get("example_ids", [])) != int(artifact["n_examples"]):
            raise ValueError
    except (TypeError, ValueError, KeyError):
        return {"enabled": False, "status": "unverified", "reason": "calibration_quality_invalid"}
    return {"enabled": True, "status": "calibrated", "dataset_id": artifact["dataset_id"], "dataset_revision": artifact["dataset_revision"], "language": label, "threshold": float(artifact["threshold"]), "n_examples": int(artifact["n_examples"]), "artifact_digest": artifact["artifact_digest"]}


def nli_calibration_quality(artifact: Mapping[str, Any], *, min_accuracy: float, min_entailment_precision: float, min_entailment_recall: float, min_per_label_support: int, min_accuracy_lower: float | None = None, min_entailment_precision_lower: float | None = None, min_entailment_recall_lower: float | None = None) -> dict[str, Any]:
    """Apply the configured held-out quality gate after provenance auth."""
    try:
        accuracy = float(artifact.get("accuracy")); stats = artifact.get("per_label_stats")
        entailment = stats.get("entailment") if isinstance(stats, Mapping) else None
        precision = float(entailment.get("precision")); recall = float(entailment.get("recall")); support = int(entailment.get("n_total"))
        positive_support = int(entailment.get("positive_support"))
        metadata = artifact.get("calibration"); audit_metrics = metadata.get("metrics", {}).get("audit", {}) if isinstance(metadata, Mapping) else {}
        accuracy_lower = float(audit_metrics.get("accuracy_ci", {}).get("lower")); precision_lower = float(audit_metrics.get("entailment", {}).get("precision_ci", {}).get("lower")); recall_lower = float(audit_metrics.get("entailment", {}).get("recall_ci", {}).get("lower"))
    except (TypeError, ValueError, AttributeError):
        return {"enabled": False, "status": "quality_unavailable", "reason": "missing_quality_statistics"}
    numeric_support = {str(key): int(value.get("n_total")) for key, value in stats.items() if str(key).isdigit() and isinstance(value, Mapping) and value.get("n_total") is not None}
    lower_bounds = {"accuracy": accuracy_lower, "entailment_precision": precision_lower, "entailment_recall": recall_lower}
    lower_thresholds = {"accuracy": min_accuracy if min_accuracy_lower is None else min_accuracy_lower, "entailment_precision": min_entailment_precision if min_entailment_precision_lower is None else min_entailment_precision_lower, "entailment_recall": min_entailment_recall if min_entailment_recall_lower is None else min_entailment_recall_lower}
    point_failed = accuracy < min_accuracy or precision < min_entailment_precision or recall < min_entailment_recall or support < min_per_label_support or positive_support < min_per_label_support or any(value < min_per_label_support for value in numeric_support.values())
    ci_failed = any(lower_bounds[key] < lower_thresholds[key] for key in lower_bounds)
    if point_failed or ci_failed:
        return {"enabled": False, "status": "quality_failed", "reason": "calibration_quality_below_threshold", "accuracy": accuracy, "entailment_precision": precision, "entailment_recall": recall, "entailment_support": support, "positive_support": positive_support, "per_label_support": numeric_support, "lower_bounds": lower_bounds, "lower_thresholds": lower_thresholds}
    return {"enabled": True, "status": "quality_passed", "accuracy": accuracy, "entailment_precision": precision, "entailment_recall": recall, "entailment_support": support, "positive_support": positive_support, "per_label_support": numeric_support, "lower_bounds": lower_bounds, "lower_thresholds": lower_thresholds}


MODEL_LABEL_MAPPING = {"LABEL_0": "contradiction", "LABEL_1": "neutral", "LABEL_2": "entailment"}
DATASET_LABEL_MAPPING = {"0": "entailment", "1": "neutral", "2": "contradiction"}


def validate_nli_label_mapping(mapping: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    expected = MODEL_LABEL_MAPPING if kind == "model" else DATASET_LABEL_MAPPING if kind == "dataset" else None
    if expected is None or not isinstance(mapping, Mapping):
        return {"valid": False, "reason": "unknown_label_mapping_kind"}
    normalized = {str(key): str(value).casefold() for key, value in mapping.items()}
    valid = normalized == {key: value for key, value in expected.items()}
    return {"valid": valid, "reason": None if valid else "label_mapping_mismatch", "expected": dict(expected), "observed": normalized}


def normalized_dataset_content_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    normalized = []
    for row in rows:
        normalized.append({str(key): unicodedata.normalize("NFKC", str(value)) if isinstance(value, str) else value for key, value in sorted(row.items(), key=lambda item: str(item[0]))})
    payload = __import__("json").dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_calibration_metadata(artifact: Mapping[str, Any], dataset_examples: Sequence[Mapping[str, Any]]) -> str | None:
    metadata = artifact.get("calibration")
    required = {"method", "seed", "criterion", "entailment_label", "calibration_ids", "audit_ids", "calibration_content_digest", "audit_content_digest", "support", "metrics", "dataset_content_hash"}
    if not isinstance(metadata, Mapping) or not required.issubset(metadata):
        return "calibration_metadata_missing_or_incomplete"
    expected_ids = [str(row.get("id")) for row in dataset_examples]
    if len(expected_ids) != len(set(expected_ids)):
        return "dataset_example_ids_not_unique"
    if metadata.get("method") != "stratified_calibration_audit_v1" or metadata.get("criterion") not in {"f1", "precision_recall"} or not isinstance(metadata.get("seed"), int):
        return "calibration_method_metadata_invalid"
    mapping = artifact.get("label_mapping")
    try:
        expected_entailment_label = int(next(key for key, value in mapping.items() if str(value).casefold() == "entailment"))
    except (AttributeError, StopIteration, TypeError, ValueError):
        return "calibration_label_mapping_invalid"
    if int(metadata.get("entailment_label")) != expected_entailment_label:
        return "calibration_entailment_label_mismatch"
    calibration_ids = metadata.get("calibration_ids"); audit_ids = metadata.get("audit_ids")
    if not isinstance(calibration_ids, list) or not isinstance(audit_ids, list) or calibration_ids != sorted(set(map(str, calibration_ids))) or audit_ids != sorted(set(map(str, audit_ids))):
        return "calibration_split_ids_invalid"
    calibration_ids = list(map(str, calibration_ids)); audit_ids = list(map(str, audit_ids))
    if set(calibration_ids) & set(audit_ids) or set(calibration_ids) | set(audit_ids) != set(expected_ids):
        return "calibration_split_coverage_invalid"
    if metadata.get("dataset_content_hash") != artifact.get("dataset_content_hash"):
        return "calibration_dataset_digest_mismatch"
    by_id = {str(row.get("id")): row for row in dataset_examples}
    calibration_rows = [by_id[identifier] for identifier in expected_ids if identifier in set(calibration_ids)]
    audit_rows = [by_id[identifier] for identifier in expected_ids if identifier in set(audit_ids)]
    if metadata.get("calibration_content_digest") != normalized_dataset_content_hash(calibration_rows) or metadata.get("audit_content_digest") != normalized_dataset_content_hash(audit_rows):
        return "calibration_split_content_digest_mismatch"
    support = metadata.get("support")
    if not isinstance(support, Mapping) or int(support.get("calibration", -1)) != len(calibration_rows) or int(support.get("audit", -1)) != len(audit_rows):
        return "calibration_support_mismatch"
    per_label = support.get("per_label")
    if not isinstance(per_label, Mapping):
        return "calibration_per_label_support_missing"
    for raw_label in sorted({str(row.get("label")) for row in dataset_examples}):
        expected_calibration = sum(str(row.get("label")) == raw_label for row in calibration_rows)
        expected_audit = sum(str(row.get("label")) == raw_label for row in audit_rows)
        observed = per_label.get(raw_label)
        if not isinstance(observed, Mapping) or int(observed.get("calibration", -1)) != expected_calibration or int(observed.get("audit", -1)) != expected_audit:
            return "calibration_per_label_support_mismatch"
    artifact_stats = artifact.get("per_label_stats")
    if not isinstance(artifact_stats, Mapping):
        return "calibration_per_label_metrics_missing"
    for raw_label in sorted({str(row.get("label")) for row in dataset_examples}):
        observed = artifact_stats.get(raw_label)
        if not isinstance(observed, Mapping) or int(observed.get("n_total", -1)) != sum(str(row.get("label")) == raw_label for row in audit_rows):
            return "calibration_artifact_label_support_mismatch"
    metrics = metadata.get("metrics"); audit_metrics = metrics.get("audit") if isinstance(metrics, Mapping) else None
    try:
        if float(audit_metrics.get("accuracy")) != float(artifact.get("accuracy")):
            return "calibration_accuracy_mismatch"
        if not isinstance(audit_metrics.get("accuracy_ci"), Mapping):
            return "calibration_accuracy_ci_missing"
        entailment = audit_metrics.get("entailment")
        for key in ("precision", "recall", "f1", "precision_ci", "recall_ci", "f1_ci"):
            if key not in entailment:
                return "calibration_entailment_metric_missing"
        stats = artifact.get("per_label_stats")
        if not isinstance(stats, Mapping) or not isinstance(stats.get("entailment"), Mapping):
            return "calibration_per_label_metrics_missing"
        stored_entailment = stats["entailment"]
        for key in ("precision", "recall"):
            if float(stored_entailment.get(key)) != float(entailment.get(key)):
                return "calibration_entailment_metric_mismatch"
        if float(stored_entailment.get("f1")) != float(entailment.get("f1")):
            return "calibration_entailment_metric_mismatch"
        total = sum(int(value.get("n_total")) for key, value in stats.items() if str(key).isdigit())
        correct = sum(int(value.get("correct")) for key, value in stats.items() if str(key).isdigit())
        if total < 1 or abs(correct / total - float(artifact.get("accuracy"))) > 1e-12:
            return "calibration_accuracy_not_reproducible"
        for key in ("precision", "recall", "f1"):
            value = float(entailment.get(key))
            if not 0.0 <= value <= 1.0:
                return "calibration_metric_out_of_range"
        for key in ("accuracy_ci", "precision_ci", "recall_ci", "f1_ci"):
            interval = audit_metrics.get(key) if key == "accuracy_ci" else entailment.get(key)
            if not isinstance(interval, Mapping) or not 0.0 <= float(interval.get("lower")) <= float(interval.get("upper")) <= 1.0:
                return "calibration_ci_invalid"
    except (TypeError, ValueError, AttributeError):
        return "calibration_metrics_invalid"
    return None


def _binary_metrics(gold: Sequence[int], scores: Sequence[float], threshold: float, *, entailment_label: int = 2) -> dict[str, Any]:
    truth = [int(value == entailment_label) for value in gold]; pred = [int(score >= threshold) for score in scores]
    tp = sum(a == b == 1 for a, b in zip(truth, pred)); fp = sum(a == 0 and b == 1 for a, b in zip(truth, pred)); fn = sum(a == 1 and b == 0 for a, b in zip(truth, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "support": sum(truth)}


def _bootstrap_binary_metric_ci(gold: Sequence[int], scores: Sequence[float], threshold: float, *, entailment_label: int, metric: str = "f1", n_resamples: int, seed: int) -> dict[str, float | None]:
    if not gold or n_resamples < 1:
        return {"lower": None, "upper": None}
    rng = random.Random(seed); values: list[float] = []; indices = list(range(len(gold)))
    for _ in range(n_resamples):
        sampled = [rng.choice(indices) for _ in indices]
        metric_result = _binary_metrics([gold[index] for index in sampled], [scores[index] for index in sampled], threshold, entailment_label=entailment_label)
        values.append(float(metric_result[metric]))
    values.sort()
    lower = values[min(len(values) - 1, int(0.025 * len(values)))]
    upper = values[min(len(values) - 1, int(0.975 * len(values)))]
    return {"lower": lower, "upper": upper}


def select_stratified_nli_rows(rows: Sequence[Mapping[str, Any]], *, max_rows: int, seed: int = 0) -> list[Mapping[str, Any]]:
    """Select a deterministic, label-stratified NLI sample for calibration.

    Rows are sorted by stable content before seeded shuffling, so dataset
    iteration order cannot silently change the calibration set.  The
    round-robin fill keeps every observed gold label represented.
    """
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("max_rows must be positive")
    groups: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        if "label" not in row:
            raise ValueError("NLI row missing label")
        groups.setdefault(int(row["label"]), []).append(row)
    rng = random.Random(seed)
    for label, values in groups.items():
        values.sort(key=lambda row: stable_identity_hash({key: row.get(key) for key in ("id", "premise", "hypothesis", "label")}))
        rng.shuffle(values)
    target = min(max_rows, sum(len(values) for values in groups.values()))
    selected: list[Mapping[str, Any]] = []
    cursor = {label: 0 for label in sorted(groups)}
    while len(selected) < target:
        progressed = False
        for label in sorted(groups):
            if cursor[label] < len(groups[label]):
                selected.append(groups[label][cursor[label]])
                cursor[label] += 1
                progressed = True
                if len(selected) >= target:
                    break
        if not progressed:
            break
    return selected


def calibrate_nli_threshold(rows: Sequence[Mapping[str, Any]], predictor: Callable[[Mapping[str, Any]], Mapping[str, Any]], *, seed: int = 0, min_support: int = 50, n_bootstrap: int = 2000, criterion: str = "f1", entailment_label: int = 2) -> dict[str, Any]:
    """Deterministically stratify, tune on calibration, and audit untouched data."""
    if criterion not in {"f1", "precision_recall"} or min_support < 1:
        raise ValueError("invalid calibration configuration")
    by_label: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        label = int(row["label"]); by_label.setdefault(label, []).append(row)
    rng = random.Random(seed); calibration: list[Mapping[str, Any]] = []; audit: list[Mapping[str, Any]] = []
    for label in sorted(by_label):
        values = sorted(by_label[label], key=lambda row: str(row.get("id"))); rng.shuffle(values)
        if len(values) < 2 * min_support:
            raise ValueError(f"insufficient_label_support:{label}")
        cut = len(values) // 2; calibration.extend(values[:cut]); audit.extend(values[cut:])
    def predicted(values):
        output = [predictor(row) for row in values]
        return [int(item["label"]) for item in output], [float(item["entailment_probability"]) for item in output]
    cal_pred_labels, cal_scores = predicted(calibration); audit_pred_labels, audit_scores = predicted(audit)
    cal_gold = [int(row["label"]) for row in calibration]; audit_gold = [int(row["label"]) for row in audit]
    candidates = sorted({0.05 * index for index in range(1, 20)} | {round(score, 6) for score in cal_scores if 0.0 <= score <= 1.0})
    def objective(value: float) -> float:
        metrics = _binary_metrics(cal_gold, cal_scores, value, entailment_label=entailment_label)
        return metrics["f1"] if criterion == "f1" else metrics["precision"] * metrics["recall"]
    threshold = max(candidates, key=lambda value: (objective(value), -value))
    binary = _binary_metrics(audit_gold, audit_scores, threshold, entailment_label=entailment_label)
    accuracy = sum(gold == prediction for gold, prediction in zip(audit_gold, audit_pred_labels)) / len(audit_gold)
    truth = [int(label == entailment_label) for label in audit_gold]
    pred = [int(score >= threshold) for score in audit_scores]
    accuracy_ci = bootstrap_mean_ci([float(gold == prediction) for gold, prediction in zip(audit_gold, audit_pred_labels)], n_resamples=n_bootstrap, seed=seed)
    return {"method": "stratified_calibration_audit_v1", "seed": seed, "threshold": threshold, "criterion": criterion, "entailment_label": entailment_label, "calibration_ids": {str(row["id"]) for row in calibration}, "audit_ids": {str(row["id"]) for row in audit}, "support": {"calibration": len(calibration), "audit": len(audit), "per_label": {str(label): {"calibration": sum(int(row["label"]) == label for row in calibration), "audit": sum(int(row["label"]) == label for row in audit)} for label in by_label}}, "metrics": {"audit": {"accuracy": accuracy, "accuracy_ci": accuracy_ci, "entailment": {**binary, "precision_ci": _bootstrap_binary_metric_ci(audit_gold, audit_scores, threshold, entailment_label=entailment_label, metric="precision", n_resamples=n_bootstrap, seed=seed + 1), "recall_ci": _bootstrap_binary_metric_ci(audit_gold, audit_scores, threshold, entailment_label=entailment_label, metric="recall", n_resamples=n_bootstrap, seed=seed + 2), "f1_ci": _bootstrap_binary_metric_ci(audit_gold, audit_scores, threshold, entailment_label=entailment_label, metric="f1", n_resamples=n_bootstrap, seed=seed + 3)}}}}


def validate_nli_calibration_artifact(artifact: Mapping[str, Any] | None, language: str) -> dict[str, Any]:
    """Require an authenticated per-language calibration artifact."""
    label = _language_label(language)
    entry = artifact.get(label) if isinstance(artifact, Mapping) and label else None
    if not isinstance(entry, Mapping):
        return {"enabled": False, "status": "uncalibrated", "reason": "missing_language_calibration"}
    required = {key: entry.get(key) for key in ("model_id", "model_revision", "dataset_id", "dataset_revision", "language")}
    return verify_nli_calibration_artifact(entry, model_id=required["model_id"], model_revision=required["model_revision"], dataset_id=required["dataset_id"], dataset_revision=required["dataset_revision"], language=label, split=entry.get("split"), code_hash=entry.get("code_hash"), eval_core_hash=entry.get("eval_core_hash"))


def load_nli_runtime_evaluator(
    language: str,
    *,
    calibration_artifact: Mapping[str, Any] | None,
    model_loader: Callable[..., Callable[[str, str], float]] | None = None,
    model_id: str = NLI_MODEL_ID,
    revision: str = NLI_MODEL_REVISION,
    calibration_provenance: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load multilingual NLI only after a per-language calibration gate."""

    label = _language_label(language)
    expected = calibration_provenance.get(label) if isinstance(calibration_provenance, Mapping) else None
    if expected is not None:
        entry = calibration_artifact.get(label) if isinstance(calibration_artifact, Mapping) else None
        if not isinstance(entry, Mapping):
            calibration = {"enabled": False, "status": "unverified", "reason": "missing_language_calibration"}
        else:
            calibration = verify_nli_calibration_artifact(entry, model_id=str(expected.get("model_id")), model_revision=str(expected.get("model_revision")), dataset_id=str(expected.get("dataset_id")), dataset_revision=str(expected.get("dataset_revision")), dataset_content_hash=expected.get("dataset_content_hash"), language=label, split=expected.get("split"), code_hash=expected.get("code_hash"), eval_core_hash=expected.get("eval_core_hash"), label_mapping=expected.get("dataset_label_mapping"))
    else:
        calibration = validate_nli_calibration_artifact(calibration_artifact, language)
    if not calibration["enabled"]:
        return {"evaluator": None, "model_id": model_id, "revision": revision, "language": _language_label(language), **calibration}
    if model_loader is None:
        return {"evaluator": None, "model_id": model_id, "revision": revision, "language": _language_label(language), "enabled": False, "status": "unavailable", "reason": "nli_model_loader_unavailable", **{key: value for key, value in calibration.items() if key != "enabled"}}
    try:
        evaluator = model_loader(model_id=model_id, revision=revision, language=_language_label(language), threshold=calibration["threshold"])
    except Exception as exc:
        return {"evaluator": None, "model_id": model_id, "revision": revision, "language": _language_label(language), "enabled": False, "status": "error", "reason": f"nli_model_load_error:{type(exc).__name__}"}
    if not callable(evaluator):
        return {"evaluator": None, "model_id": model_id, "revision": revision, "language": _language_label(language), "enabled": False, "status": "error", "reason": "nli_model_loader_invalid"}
    return {"evaluator": evaluator, "model_id": model_id, "revision": revision, "language": _language_label(language), **calibration}
