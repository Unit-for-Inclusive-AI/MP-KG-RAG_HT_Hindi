"""Faithful ports of the production prompt/ledger layer.

Sources (read-only references):
  outputs/remote_vm_qwen35_mpkg_rag_runtime.py lines 2456, 2626, 2682, 3065,
  4905, 5097, 5205, 5235, 5548-5700.

One deliberate divergence: build_evidence_ledger supports a per-row
displayed-span fallback when the production source-text memory
(SOURCE_TEXT_BY_KEY) is unavailable. Rows record source_resolution so the
divergence is auditable; everything else (span invariant when a source text
is resolved, whole-span budgets, fitting) is mirrored exactly.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Callable, Mapping, Sequence

PERSPECTIVES = [
    ("fact_checking", "Correct only directly supported claims."),
    ("cultural_context", "Give respectful context."),
    ("harm_reduction", "Avoid amplifying harmful language."),
    ("legal_rights", "Use only source-grounded rights claims."),
    ("persuasion", "Suggest an empathetic response."),
]

ACCEPTED_PARSE_STATUSES = frozenset({"initial", "repair"})

_CITATION_TOKEN_RE = re.compile(r"\[(E[1-9][0-9]*)\]")

PERSPECTIVE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "perspective",
        "rationale",
        "claims_to_address",
        "supported_evidence_ids",
        "response_guidance",
        "risk_flags",
        "confidence",
    ],
    "properties": {
        "perspective": {"type": "string"},
        "rationale": {"type": "string"},
        "claims_to_address": {"type": "array", "items": {"type": "string"}},
        "supported_evidence_ids": {
            "type": "array",
            "items": {"type": "string", "pattern": "^E[1-9][0-9]*$"},
        },
        "response_guidance": {"type": "array", "items": {"type": "string"}},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "claim_focus",
        "selected_evidence_ids",
        "response_steps",
        "tone",
        "factual_constraints",
        "safety_constraints",
    ],
    "properties": {
        "claim_focus": {"type": "string"},
        "selected_evidence_ids": {
            "type": "array",
            "items": {"type": "string", "pattern": "^E[1-9][0-9]*$"},
        },
        "response_steps": {"type": "array", "items": {"type": "string"}},
        "tone": {"type": "string"},
        "factual_constraints": {"type": "array", "items": {"type": "string"}},
        "safety_constraints": {"type": "array", "items": {"type": "string"}},
    },
}

FINAL_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["counter_narrative", "cited_evidence_ids", "factual_claims", "safety_notes"],
    "properties": {
        "counter_narrative": {"type": "string"},
        "cited_evidence_ids": {
            "type": "array",
            "items": {"type": "string", "pattern": "^E[1-9][0-9]*$"},
        },
        "factual_claims": {"type": "array", "items": {"type": "string"}},
        "safety_notes": {"type": "array", "items": {"type": "string"}},
    },
}


def canonical_page(value: Any) -> Any:
    """Mirror runtime L4905 without a pandas dependency."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def validate_evidence_ledger(
    evidence_ledger: Sequence[Mapping[str, Any]],
    source_text_lookup: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Port of runtime L2626: offsets, displayed text, and digest must agree."""
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
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
        ):
            raise ValueError(f"ledger offsets invalid for {evidence_id}")
        expected_digest = hashlib.sha256(displayed.encode("utf-8")).hexdigest()
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
        if source_text is not None and item.get("source_text_sha256") != hashlib.sha256(
            source_text.encode("utf-8")
        ).hexdigest():
            raise ValueError(f"ledger source text hash mismatch for {evidence_id}")
        evidence_text = item.get("evidence_text")
        if source_text is not None and source_text[start:end] != displayed:
            raise ValueError(f"ledger source span mismatch for {evidence_id}")
        if source_text is None and evidence_text is not None and evidence_text != displayed:
            raise ValueError(f"ledger evidence text mismatch for {evidence_id}")
        validated.append(item)
    return validated


def select_evidence_within_budget(
    evidence_ledger: Sequence[Mapping[str, Any]],
    char_budget: int,
) -> list[dict[str, Any]]:
    """Port of runtime L2682: prefix of whole spans, never slice a span."""
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


def fit_adaptive_prompt_with_evidence(
    evidence_ledger: Sequence[Mapping[str, Any]],
    *,
    prompt_builder: Callable[[Sequence[Mapping[str, Any]], str], str],
    payload_candidates: Sequence[str] = ("",),
    token_counter: Callable[[str], int],
    max_input_tokens: int,
    reserve_output_tokens: int = 0,
) -> dict[str, Any]:
    """Port of runtime L3065: drop whole ranked spans until the prompt fits."""
    if (
        not isinstance(max_input_tokens, int)
        or isinstance(max_input_tokens, bool)
        or max_input_tokens < 1
    ):
        raise ValueError("max_input_tokens must be a positive integer")
    if (
        not isinstance(reserve_output_tokens, int)
        or isinstance(reserve_output_tokens, bool)
        or reserve_output_tokens < 0
    ):
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
                float(pair[1].get("rank", pair[0]))
                if str(pair[1].get("rank", pair[0])).replace(".", "", 1).isdigit()
                else float(pair[0]),
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
            attempt = {
                "payload_index": payload_index,
                "selected_evidence_count": count,
                "prompt_tokens": prompt_tokens,
            }
            attempts.append(attempt)
            if prompt_tokens <= available:
                return {
                    "status": "fit",
                    "prompt": prompt,
                    "payload": payload,
                    "payload_index": payload_index,
                    "prompt_tokens": prompt_tokens,
                    "available_input_tokens": available,
                    "selected_evidence_ids": [str(item["evidence_id"]) for item in selected],
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


def build_evidence_ledger(
    evidence: Sequence[Mapping[str, Any]],
    source_lookup: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Adapted port of runtime L5549.

    Rows resolve their source text when `source_lookup` provides the key and
    the recorded span slices exactly to the displayed text. Otherwise the
    displayed span is used and `source_resolution` records why.
    """
    rows: list[dict[str, Any]] = []
    for i, x in enumerate(evidence):
        displayed = str(x.get("text", ""))
        source_key = x.get("source_text_key")
        start = int(x.get("span_start", 0) or 0)
        end = int(x.get("span_end", len(displayed)) or 0)
        resolved: str | None = None
        resolution = "displayed_span_fallback"
        if source_lookup is None:
            resolution = "displayed_span_fallback_no_lookup"
        elif not source_key:
            resolution = "displayed_span_fallback_missing_key"
        elif source_key not in source_lookup:
            resolution = "displayed_span_fallback_lookup_miss"
        else:
            candidate = source_lookup[source_key]
            if (
                isinstance(candidate, str)
                and start >= 0
                and start < end
                and end <= len(candidate)
                and candidate[start:end] == displayed
            ):
                resolved = candidate
                resolution = "source_text_resolved"
            else:
                resolution = "displayed_span_fallback_slice_mismatch"
        if resolved is not None:
            source_text_sha256 = hashlib.sha256(resolved.encode("utf-8")).hexdigest()
        else:
            if not (start >= 0 and end >= start) or end - start != len(displayed):
                raise RuntimeError("evidence_span_invariant_failed")
            source_text_sha256 = x.get("source_text_sha256") or hashlib.sha256(
                displayed.encode("utf-8")
            ).hexdigest()
        row = {
            "evidence_id": f"E{i + 1}",
            "ledger_id": f"E{i + 1}",
            "rank": x.get("rank", i),
            "evidence_chunk_id": x["evidence_chunk_id"],
            "chunk_id": x.get("chunk_id"),
            "document_uid": x.get("document_uid"),
            "source_id": x.get("source_id"),
            "locator": f"p. {canonical_page(x.get('page'))}"
            if canonical_page(x.get("page")) is not None
            else f"section: {x.get('section', 'document')}",
            "displayed_text": displayed,
            "evidence_text": displayed,
            "source_text_key": source_key or f"inline-{i}",
            "source_text_sha256": source_text_sha256,
            "span_start": start,
            "span_end": end,
            "displayed_text_sha256": hashlib.sha256(displayed.encode("utf-8")).hexdigest(),
            "source_resolution": resolution,
        }
        if resolved is not None:
            row["source_text"] = resolved
        rows.append(row)
    return validate_evidence_ledger(rows)


def validate_evidence_ids(ids: Any, ledger: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Port of runtime L5558."""
    if not isinstance(ids, list):
        return {
            "valid": False,
            "unknown": [str(ids)] if ids is not None else [],
            "evidence_ids": [],
            "quarantine": True,
            "reason": "evidence_ids_not_list",
        }
    values = list(ids)
    allowed = {x["evidence_id"] for x in ledger}
    unknown = [
        str(value) for value in values if not isinstance(value, str) or value not in allowed
    ]
    valid = (
        all(isinstance(value, str) for value in values)
        and len(values) == len(set(values))
        and not unknown
        and all(re.fullmatch(r"E[1-9][0-9]*", value) for value in values)
    )
    return {"valid": valid, "unknown": unknown, "evidence_ids": values, "quarantine": not valid}


def evidence_for_prompt(
    evidence: Sequence[Mapping[str, Any]], char_budget: int | None = None
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    """Port of runtime L5561."""
    ledger = build_evidence_ledger(evidence)
    if char_budget is None:
        return list(evidence), ledger
    selected = select_evidence_within_budget(ledger, int(char_budget))
    selected_ids = {x["evidence_id"] for x in selected}
    return [
        item for item, entry in zip(evidence, ledger) if entry["evidence_id"] in selected_ids
    ], selected


def _ledger_block(ledger: Sequence[Mapping[str, Any]]) -> str:
    return "\n\n".join(
        f"[{x['evidence_id']}] source={x['source_id']} {x['locator']} "
        f"document_uid={x['document_uid']} span={x['span_start']}:{x['span_end']}\n"
        f"{x['displayed_text']}"
        for x in ledger
    )


def evidence_block(evidence: Sequence[Mapping[str, Any]], limit: int | None = None) -> str:
    """Port of runtime L5566."""
    _, ledger = evidence_for_prompt(evidence, limit)
    return _ledger_block(ledger)


def prompt_token_count(tokenizer: Any, prompt: str) -> int:
    """Mirror runtime _prompt_token_count L5571 for an injected tokenizer."""
    encoded = tokenizer(text=[str(prompt)], return_tensors="pt", padding=False, truncation=False)
    mask = encoded["attention_mask"]
    return int(mask.sum().item()) if hasattr(mask.sum(), "item") else int(mask.sum())


def apply_chat_template(tokenizer: Any, messages: Sequence[Mapping[str, str]], enable_thinking: bool) -> str:
    """Mirror runtime _apply_chat_template L5097."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _unfitted_result(ledger: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "status": "skipped_no_tokenizer",
        "prompt": None,
        "payload": "",
        "payload_index": 0,
        "prompt_tokens": None,
        "available_input_tokens": None,
        "selected_evidence_ids": [str(item["evidence_id"]) for item in ledger],
        "dropped_evidence_ids": [],
        "attempts": [],
    }


def adaptive_messages(
    evidence: Sequence[Mapping[str, Any]],
    message_builder: Callable[[Sequence[Mapping[str, Any]], str], list[dict[str, str]]],
    payload_candidates: Sequence[str],
    max_new_tokens: int,
    *,
    tokenizer: Any | None = None,
    max_seq_length: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Port of runtime _adaptive_messages L5577 with an injectable tokenizer.

    Without a tokenizer the full-ledger candidate is returned with an
    explicit `skipped_no_tokenizer` fit record (the generation stage, which
    owns the real tokenizer, performs the fitting).
    """
    ledger = build_evidence_ledger(evidence)
    if tokenizer is None:
        fit = _unfitted_result(ledger)
        return message_builder(ledger, fit["payload"]), fit

    def rendered_builder(selected: Sequence[Mapping[str, Any]], payload: str) -> str:
        return apply_chat_template(
            tokenizer, message_builder(selected, payload), enable_thinking=False
        )

    fit = fit_adaptive_prompt_with_evidence(
        ledger,
        prompt_builder=rendered_builder,
        payload_candidates=payload_candidates,
        token_counter=lambda prompt: prompt_token_count(tokenizer, prompt),
        max_input_tokens=int(max_seq_length),
        reserve_output_tokens=int(max_new_tokens),
    )
    if fit.get("status") != "fit":
        return message_builder(ledger, fit.get("payload", "")), fit
    selected_ids = set(fit.get("selected_evidence_ids", []))
    selected = [item for item in ledger if item["evidence_id"] in selected_ids]
    return message_builder(selected, fit.get("payload", "")), fit


def adaptive_perspective_messages(
    post: str,
    target: str,
    category: str,
    evidence: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any | None = None,
    max_seq_length: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Port of runtime L5586 returning {perspective, goal, messages, rendered, fit}."""
    prompts: list[dict[str, Any]] = []
    schema_tail = "JSON schema (must be followed exactly): " + json.dumps(
        PERSPECTIVE_SCHEMA, ensure_ascii=False, sort_keys=True
    )
    for name, goal in PERSPECTIVES:

        def builder(
            selected: Sequence[Mapping[str, Any]], payload: str, name: str = name, goal: str = goal
        ) -> list[dict[str, str]]:
            return [
                {
                    "role": "system",
                    "content": "Think carefully, then return exactly one JSON object after the thinking block.",
                },
                {
                    "role": "user",
                    "content": f"Perspective: {name}\nGoal: {goal}\nPost: {post}\nTarget: {target}\nCategory: {category}\nEvidence (complete ledger spans only):\n{_ledger_block(selected)}\nReturn perspective, rationale, claims_to_address, supported_evidence_ids, response_guidance, risk_flags, confidence. Use only E1...En IDs.\n{schema_tail}",
                },
            ]

        messages, fit = adaptive_messages(
            evidence,
            builder,
            ("",),
            max_new_tokens,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        prompts.append(
            {
                "perspective": name,
                "goal": goal,
                "messages": messages,
                "rendered": fit.get("prompt"),
                "fit": fit,
            }
        )
    return prompts


def perspective_messages(
    post: str,
    target: str,
    category: str,
    evidence: Sequence[Mapping[str, Any]],
    *,
    char_budget: int | None,
) -> list[list[dict[str, str]]]:
    """Port of runtime L5605 (non-adaptive full-span variant)."""
    prompt_evidence, _ = evidence_for_prompt(evidence, char_budget)
    return [
        [
            {"role": "system", "content": "Return exactly one JSON object and no reasoning trace."},
            {
                "role": "user",
                "content": f"Perspective: {name}\nGoal: {goal}\nPost: {post}\nTarget: {target}\nCategory: {category}\nEvidence (full exact displayed spans):\n{evidence_block(prompt_evidence)}\nReturn perspective, rationale, claims_to_address, supported_evidence_ids, response_guidance, risk_flags, confidence. Use only E1...En IDs.",
            },
        ]
        for name, goal in PERSPECTIVES
    ]


def plan_messages(
    post: str,
    target: str,
    evidence: Sequence[Mapping[str, Any]],
    perspective_outputs: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any | None = None,
    max_seq_length: int,
    max_new_tokens: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Port of the builder inside runtime synthesize_plan L5629 (no generation)."""
    compact = [
        {
            k: p["structured_output"].get(k, [])
            for k in [
                "perspective",
                "rationale",
                "claims_to_address",
                "supported_evidence_ids",
                "response_guidance",
                "risk_flags",
            ]
        }
        for p in perspective_outputs
        if p["structured_output"].get("parse_status") in ACCEPTED_PARSE_STATUSES
    ]
    compact_minimal = [
        {
            "perspective": p.get("perspective"),
            "supported_evidence_ids": p.get("supported_evidence_ids", []),
        }
        for p in compact
    ]
    payload_candidates = (
        json.dumps(compact, ensure_ascii=False, sort_keys=True),
        json.dumps(compact_minimal, ensure_ascii=False, sort_keys=True),
    )
    schema_tail = "JSON schema (must be followed exactly): " + json.dumps(
        PLAN_SCHEMA, ensure_ascii=False, sort_keys=True
    )

    def builder(selected: Sequence[Mapping[str, Any]], payload: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "Return one JSON evidence plan only."},
            {
                "role": "user",
                "content": f"Post: {post}\nTarget: {target}\nEvidence (complete ledger spans only):\n{_ledger_block(selected)}\nPerspectives: {payload}\nReturn claim_focus, selected_evidence_ids, response_steps, tone, factual_constraints, safety_constraints. Use only E IDs.\n{schema_tail}",
            },
        ]

    return adaptive_messages(
        evidence,
        builder,
        payload_candidates,
        max_new_tokens,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )


def final_messages(
    post: str,
    target: str,
    evidence: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    *,
    tokenizer: Any | None = None,
    max_seq_length: int,
    max_new_tokens: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Port of the builder inside runtime generate_final_counter_narrative L5639."""
    compact_plan = {
        key: plan.get(key)
        for key in ["claim_focus", "selected_evidence_ids", "response_steps", "tone"]
        if key in plan
    }
    payload_candidates = (
        json.dumps(plan, ensure_ascii=False, sort_keys=True),
        json.dumps(compact_plan, ensure_ascii=False, sort_keys=True),
    )
    schema_tail = "JSON schema (must be followed exactly): " + json.dumps(
        FINAL_RESPONSE_SCHEMA, ensure_ascii=False, sort_keys=True
    )

    def builder(selected: Sequence[Mapping[str, Any]], payload: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "Write one safe evidence-grounded counter-narrative as JSON only.",
            },
            {
                "role": "user",
                "content": f"Post: {post}\nTarget: {target}\nPlan: {payload}\nEvidence (complete ledger spans only):\n{_ledger_block(selected)}\nFactual claims require inline [E1] citations. If there are no factual claims, return exactly one approved safe-abstention template and set factual_claims to []. Return counter_narrative, cited_evidence_ids, factual_claims, safety_notes. Use only E IDs; chunk IDs are metadata only.\n{schema_tail}",
            },
        ]

    return adaptive_messages(
        evidence,
        builder,
        payload_candidates,
        max_new_tokens,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )


def grounding_repair_messages(
    post: str,
    target: str,
    evidence: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    failure_reasons: Sequence[Any],
    *,
    tokenizer: Any | None = None,
    max_seq_length: int,
    max_new_tokens: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Port of the builder inside runtime grounding_repair_response L5650."""
    payload_candidates = (
        json.dumps({"plan": plan, "failures": failure_reasons}, ensure_ascii=False, sort_keys=True),
        json.dumps(
            {"failures": [str(item) for item in failure_reasons[:3]]},
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    schema_tail = "JSON schema (must be followed exactly): " + json.dumps(
        FINAL_RESPONSE_SCHEMA, ensure_ascii=False, sort_keys=True
    )

    def builder(selected: Sequence[Mapping[str, Any]], payload: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "Repair one failed evidence-grounded response as strict JSON.",
            },
            {
                "role": "user",
                "content": f"Post: {post}\nTarget: {target}\nRepair context: {payload}\nEvidence (complete ledger spans only):\n{_ledger_block(selected)}\nReturn counter_narrative, cited_evidence_ids, factual_claims, safety_notes. Cite only supported E IDs.\n{schema_tail}",
            },
        ]

    return adaptive_messages(
        evidence,
        builder,
        payload_candidates,
        max_new_tokens,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
