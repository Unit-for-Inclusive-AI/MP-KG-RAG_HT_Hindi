#!/usr/bin/env python3
"""Resumable MLX extraction runner for the v3 quote-first claims contract and
mention discovery.

Implements .audit/claim_repair_20261019/METHOD_FIX_quote_first.md:

  * claims (prompt semantic-claims.v3-quote-first, schema quote-first-claims.v3):
    the model returns only verbatim quotes ({"claims":[{"quote", "candidate_ids"?}]});
    the pipeline locates each quote with work/quote_locate.py (offsets by
    construction), enforces gates G2 (min length), G3 (claim-scoped candidate
    link from candidates.json for eligibility) and G4 (dedupe by span and by
    normalized quote), performs exactly one locate-failure retry with a feedback
    prompt, then rejects remaining quotes into quote_rejection_queue.jsonl.
    In addition, when a claims call cannot be parsed as schema-valid JSON or is
    truncated at the token budget, the runner performs exactly one JSON-repair
    retry with the failed raw output and the parse error before processing
    quotes; this is separate from the single locate retry.
  * mentions (prompt/schema mention-discovery.v2-quote-first): the model
    returns ordered {text, type} surfaces only (no ids, no offsets, at most 80
    items, compact JSON); the runner locates every surface by exact then
    normalized matching, assigns mention ids and offsets, builds the frozen
    mention-discovery.v1 records and validates them with
    core.validate_mentions, so mention offsets are exact by construction (the
    same quote-first move as claims v3); repeated surfaces are collapsed to one
    mention per distinct surface string. Budget 2048, thinking off. The model
    call is repaired in one of three ways: one JSON-repair retry on parse
    failure, a salvage of complete items from a budget-truncated array, and one
    empty-list nudge retry when a chunked response contains no usable mention.
    The v1 revision is kept in work/mpkg_rag_core.py for the shared validator;
    this runner no longer calls build_mention_prompt.

Batching (optional):
  --batch-size N >= 2 runs N chunk prompts through mlx_lm's BatchGenerator in
  one batch_generate-style call (greedy sampler with the same top_p/top_k/min_p
  as the streaming path). Per-row outputs are decoded and parsed independently;
  a row that fails parsing gets its own JSON repair (batched with the other
  failed rows). Rows are still written, flushed and checkpointed one chunk at a
  time, so --resume behaves exactly as in sequential mode. Batch generation is
  optional and defaults to 1 (sequential).

Outputs (append-only JSONL, flushed per chunk; run_manifest.json rewritten
atomically after every chunk so --resume works after interrupts):

  claims_v3.jsonl                 one row per chunk
  mentions.jsonl                  one row per chunk
  quote_rejection_queue.jsonl     one row per rejected quote
  timings.jsonl                   one row per model call (batch rows carry
                                  batch_size and wall_s; elapsed_s is the
                                  per-row share of the batch wall)
  run_manifest.json               config + identity + counters + failures

Usage:
  python3 run_extraction_mlx.py --chunks CHUNKS.json --candidates CANDIDATES.json \
      --out DIR --mode both [--limit N] [--resume] [--chunk-ids IDS] \
      [--thinking 0|1] [--seed 42] [--batch-size N]

Flags added for hardening (backwards compatible; all default to previous
behaviour except the token budgets, now 1024 for both modes):
  --batch-size N   prompts per batched generation call (default 1 = sequential)

Notes:
  - --chunk-ids accepts a JSON array file or a newline-delimited list; its order
    is preserved. Chunk ids missing from the chunks file are reported and skipped.
  - --resume skips chunk_ids already present in the corresponding output jsonl.
    A crash between the rejection append and the chunk row can duplicate queue
    rows on resume; rows are otherwise append-only and never rewritten.
  - Offline only: the pinned model revision must already exist in the HF cache.
  - The JSON-repair retry replaces the MLX engine's internal schema repair:
    calls go through MLXGenerator._stream_once (one generation, no hidden
    second generation), so repair cost and recovery are explicit in timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
import uuid
from pathlib import Path

RUNNER_PATH = Path(__file__).resolve()
PROJECT = RUNNER_PATH.parents[1]
SNAPSHOT_PARENT = PROJECT / "outputs/2026-10-17-mlx-corrected/test_pilot200_mixed_graph_w08_thr005_schema3_cot512_v1_seed42"
SNAPSHOT_DIR = SNAPSHOT_PARENT / "source_snapshot"

sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(SNAPSHOT_PARENT))

from work import mpkg_rag_core as core  # noqa: E402
from work import quote_locate as ql  # noqa: E402
from source_snapshot.mlx_engine import _project_schema_object, _schema_valid, split_thinking  # noqa: E402

MODEL_ID = "mlx-community/Qwen3.5-4B-4bit"
MODEL_REVISION = "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"
PROMPT_REVISION_CLAIMS = "semantic-claims.v3-quote-first"
PROMPT_REVISION_MENTIONS = "mention-discovery.v2.1-quote-first"
CLAIMS_BUDGET = 1024
MENTIONS_BUDGET = 2048
MENTION_V2_MAX_ITEMS = 80
MIN_QUOTE_CHARS = 40
MIN_QUOTE_WORDS = 5
MAX_PROMPT_CANDIDATES = 40
MAX_REPAIR_RAW_CHARS = 12_000
JSON_REPAIR_LIMIT = 1
LOCATE_RETRY_LIMIT = 1

CLAIMS_V3_SCHEMA = {
    "version": "quote-first-claims.v3",
    "type": "object",
    "additionalProperties": False,
    "required": ["claims"],
    "properties": {
        "claims": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["quote"],
                "properties": {
                    "quote": {"type": "string", "minLength": MIN_QUOTE_CHARS},
                    "candidate_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
                },
            },
        }
    },
}

CLAIMS_SYSTEM = (
    "You extract evidence claims from one supplied passage. Return at most 4 claims "
    "explicitly stated in the passage.\n"
    "Each claim is a VERBATIM QUOTE copied character-for-character from the passage: "
    "copy exactly, do not summarize, merge, correct, or paraphrase; keep the original "
    "punctuation and spacing. Never return offsets; offsets are produced by the pipeline. "
    "Your quote will be located by exact matching: quotes that cannot be located exactly "
    "are rejected after one retry. If you cannot copy a claim exactly, omit it.\n"
    "candidate_ids is optional and may only use ids from the candidate list supplied for "
    "this chunk; never invent entity ids. Attach a candidate only when its mentioned text "
    "lies inside your quote.\n"
    "Return the JSON object directly: no markdown code fences, no commentary, nothing "
    "before or after the JSON."
)

MENTION_V2_SCHEMA = {
    "version": "mention-discovery.v2-quote-first",
    "type": "object",
    "additionalProperties": False,
    "required": ["mentions"],
    "properties": {
        "mentions": {
            "type": "array",
            "maxItems": MENTION_V2_MAX_ITEMS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text"],
                "properties": {
                    "text": {"type": "string"},
                    "type": {"type": "string"},
                },
            },
        }
    },
}

MENTIONS_SYSTEM_V2 = (
    "You discover entity mentions in one supplied passage. Return at most 80 mention "
    "objects in order of appearance.\n"
    "Each mention text is the EXACT SURFACE STRING copied character-for-character from "
    "the passage: copy exactly, do not normalize, translate, expand, or paraphrase; keep "
    "the original case, spacing and punctuation. Prefer the shortest exact surface that "
    "names the entity. Never return offsets and never return mention ids; offsets are "
    "produced by the pipeline, which locates every surface by exact matching.\n"
    "List only named entities: persons, organizations, locations, laws, policies, "
    "events, works and similar proper names. Skip pronouns, generic nouns, and bare "
    "dates, years or numbers. List each distinct surface string at most once, even "
    "when it appears several times in the passage.\n"
    "When you have listed every named entity you can find, close the array and stop "
    "immediately. Never pad the list with repeated items. Return an empty mentions "
    "list only when the passage contains no named entity.\n"
    "Each item has text (the exact surface string) and type (a short tag such as PERSON, "
    "ORG, GPE, LOC, LAW, EVENT or WORK).\n"
    "Return the JSON object directly: no markdown code fences, no commentary, nothing "
    "before or after the JSON. Use compact single-line JSON."
)

MENTION_V2_EMPTY_NUDGE = (
    "Your previous answer contained no mentions, but this passage does contain entity "
    "mentions. List the entity mention surfaces you can copy character-for-character "
    "from the passage (persons, organizations, locations, laws and other named "
    "entities), at most 80, in order of appearance, each distinct surface at most "
    "once. Keep the same JSON object shape {\"mentions\": [{\"text\": ..., \"type\": "
    "...}]}. Return an empty mentions array only if the passage truly contains no "
    "entity mention."
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def word_count(text: str) -> int:
    return len(text.split())


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl_chunk_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not Path(path).exists():
        return ids
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and isinstance(row.get("chunk_id"), str):
                ids.add(row["chunk_id"])
    return ids


def load_chunk_ids_file(path: Path) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
    except ValueError:
        value = None
    if isinstance(value, list):
        ids = [str(item) for item in value]
    elif isinstance(value, dict) and isinstance(value.get("chunk_ids"), list):
        ids = [str(item) for item in value["chunk_ids"]]
    else:
        ids = [line.strip() for line in raw.splitlines() if line.strip()]
    seen: set[str] = set()
    ordered: list[str] = []
    for chunk_id in ids:
        if chunk_id not in seen:
            seen.add(chunk_id)
            ordered.append(chunk_id)
    return ordered


def iter_batches(ids: list[str], size: int):
    size = max(1, int(size))
    for start in range(0, len(ids), size):
        yield ids[start:start + size]


FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def tolerant_json(raw) -> dict | None:
    """Recover a JSON object from raw model text when a strict parse failed
    (e.g. markdown code fences, which json.loads rejects)."""

    text = str(raw or "")
    if not text:
        return None
    candidates = []
    if "[repair]" in text:
        candidates.append(text.rsplit("[repair]", 1)[-1])
    candidates.append(text)
    for candidate in candidates:
        stripped = candidate.strip()
        fence = FENCE_RE.search(stripped)
        if fence:
            stripped = fence.group(1).strip()
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            value = json.loads(stripped[start:end + 1])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _bounded_raw(raw: str) -> str:
    """Bound repair context while retaining both ends of the raw output."""

    value = str(raw or "")
    if len(value) <= MAX_REPAIR_RAW_CHARS:
        return value
    marker = "\n...[raw output truncated for repair context]...\n"
    keep = max(1, (MAX_REPAIR_RAW_CHARS - len(marker)) // 2)
    return value[:keep] + marker + value[-keep:]


def _with_schema_contract(messages, schema) -> list[dict]:
    """Append the same output-contract message the MLX engine used, so prompt
    content stays frozen across the single and batch paths."""

    return list(messages) + [{
        "role": "user",
        "content": (
            "Output contract: return exactly one JSON object matching this schema. "
            "Include every required key and no additional keys. Do not describe the schema.\n"
            + json.dumps(schema, ensure_ascii=False, sort_keys=True)
        ),
    }]


def build_repair_messages(messages, schema, raw, truncated, parse_error) -> list[dict]:
    """One JSON-repair turn: failed raw output + explicit failure reason.

    The reason is the parse error, or - when the previous call ran out of token
    budget - the truncation notice required by the hardening spec.
    """

    bounded = _bounded_raw(raw)
    if truncated:
        reason = "output was truncated; return only valid compact JSON matching the schema"
    else:
        reason = f"JSON parse error ({parse_error or 'output was not valid JSON'})"
    instruction = (
        f"Repair the previous output. The previous output failed: {reason}. "
        "Return exactly one JSON object and nothing else: no markdown fences, no commentary, "
        "no explanation.\n"
        "Required schema:\n"
        + json.dumps(schema, ensure_ascii=False, sort_keys=True)
    )
    return list(messages) + [
        {"role": "assistant", "content": bounded or "(empty output)"},
        {"role": "user", "content": instruction},
    ]


def parse_raw(raw, schema) -> tuple[dict | None, str, str | None]:
    """Parse one raw generation into a schema-valid payload.

    Returns (payload, status, parse_error) with status in
    {"ok", "recovered", "failed"} - "recovered" means the strict parse failed
    but a tolerant parse (fences / surrounding text) produced a valid object.
    """

    text = str(raw or "")
    if not text.strip():
        return None, "failed", "empty_output"
    try:
        value = json.loads(split_thinking(text)["final_content"])
    except (TypeError, ValueError):
        value = None
    if value is not None:
        projected = _project_schema_object(value, schema)
        if projected is not None and _schema_valid(projected, schema):
            return projected, "ok", None
        return None, "failed", "schema_invalid"
    recovered = tolerant_json(text)
    if recovered is not None:
        projected = _project_schema_object(recovered, schema)
        if projected is not None and _schema_valid(projected, schema):
            return projected, "recovered", None
        return None, "failed", "schema_invalid"
    return None, "failed", "json_parse_error"


def _iter_json_objects(text: str, start: int):
    """Yield (start, end) spans of complete top-level brace-balanced objects.

    Scanning starts at ``start`` and stops at the first object that never closes
    (truncated model output). String literals are skipped so braces inside
    mention surfaces do not confuse the balance.
    """

    index = start
    length = len(text)
    while index < length:
        if text[index] != "{":
            index += 1
            continue
        depth = 0
        in_string = False
        escaped = False
        cursor = index
        while cursor < length:
            char = text[cursor]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
            cursor += 1
        if cursor >= length:
            return
        yield (index, cursor + 1)
        index = cursor + 1


def _salvage_mentions_items(text: str) -> list[dict]:
    """Recover complete mention items from a budget-truncated JSON array."""

    field = text.find('"mentions"')
    if field < 0:
        return []
    array_start = text.find("[", field)
    if array_start < 0:
        return []
    items: list[dict] = []
    for start, end in _iter_json_objects(text, array_start + 1):
        try:
            item = json.loads(text[start:end])
        except ValueError:
            break
        if isinstance(item, dict):
            items.append(item)
        if len(items) >= 512:
            break
    return items


def _normalize_mention_v2_items(items) -> list[dict]:
    normalized: list[dict] = []
    if not isinstance(items, list):
        return normalized
    for item in items:
        if not isinstance(item, dict):
            continue
        surface = item.get("text")
        if not isinstance(surface, str) or not surface.strip():
            continue
        tag = item.get("type")
        normalized.append({
            "text": surface.strip(),
            "type": tag.strip() if isinstance(tag, str) and tag.strip() else "unknown",
        })
    return normalized


def _mention_items_from(value) -> list[dict] | None:
    """Return normalized mention items when ``value`` carries a mentions array."""

    if isinstance(value, dict):
        if isinstance(value.get("mentions"), list):
            return _normalize_mention_v2_items(value["mentions"])
        nested = value.get("properties")
        if isinstance(nested, dict) and isinstance(nested.get("mentions"), list):
            return _normalize_mention_v2_items(nested["mentions"])
    if isinstance(value, list):
        return _normalize_mention_v2_items(value)
    return None


def parse_mentions_v2(raw) -> tuple[dict | None, str, str | None]:
    """Parse a mention-discovery.v2-quote-first response.

    The payload is {"mentions": [{"text", "type"?}]} with no ids and no offsets
    (those are produced by locating each surface in the passage). Statuses:
    "ok" (strict JSON), "recovered" (tolerant parse), "salvaged" (complete items
    recovered from a budget-truncated array), "failed".
    """

    text = str(raw or "")
    if not text.strip():
        return None, "failed", "empty_output"
    value = None
    try:
        value = json.loads(split_thinking(text)["final_content"])
    except (TypeError, ValueError):
        value = None
    items = _mention_items_from(value) if value is not None else None
    if items is not None:
        return {"mentions": items}, "ok", None
    if value is None:
        recovered = tolerant_json(text)
        if recovered is not None:
            recovered_items = _mention_items_from(recovered)
            if recovered_items is not None:
                return {"mentions": recovered_items}, "recovered", None
    salvaged = _salvage_mentions_items(text)
    if salvaged:
        return {"mentions": _normalize_mention_v2_items(salvaged)}, "salvaged", None
    if value is not None or tolerant_json(text) is not None:
        return None, "failed", "schema_invalid"
    return None, "failed", "json_parse_error"


def mlx_memory() -> dict:
    try:
        import mlx.core as mx
    except Exception:
        return {"peak_bytes": None, "active_bytes": None}
    try:
        peak = int(mx.metal.get_peak_memory())
    except Exception:
        peak = None
    try:
        active = int(mx.get_active_memory())
    except Exception:
        active = None
    return {"peak_bytes": peak, "active_bytes": active}


class Manifest:
    """Atomic run_manifest.json writer (config, identity, counters, failures)."""

    def __init__(self, path: Path, config: dict, identity: dict,
                 processed_modes: list[str] | None = None, resume: bool = False) -> None:
        self.path = path
        existing = None
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                existing = None
        self.payload = {
            "created_at": now_iso(),
            "updated_at": None,
            "config": config,
            "identity": identity,
            "counters": {
                "claims": {"attempted": 0, "ok": 0, "failed": [], "quotes_returned": 0,
                           "quotes_located": 0, "quotes_retained": 0, "quotes_rejected": 0,
                           "quotes_eligible": 0, "retry_chunks": 0, "json_repairs": 0,
                           "json_repair_recoveries": 0, "truncations": 0},
                "mentions": {"attempted": 0, "ok": 0, "failed": [], "mentions_returned": 0,
                             "mentions_located": 0, "mentions_accepted": 0, "mentions_quarantined": 0,
                             "unlocated_mentions": 0, "mentions_over_cap": 0, "repeated_response_items": 0,
                             "semantic_invalid": 0,
                             "json_repairs": 0, "json_repair_recoveries": 0, "empty_retry_chunks": 0,
                             "empty_retry_recoveries": 0, "truncations": 0},
            },
            "peak_memory_bytes": None,
        }
        if existing:
            self.payload["created_at"] = existing.get("created_at") or self.payload["created_at"]
            self.payload["peak_memory_bytes"] = existing.get("peak_memory_bytes")
            for mode in ("claims", "mentions"):
                if mode not in (processed_modes or []) or resume:
                    prior = (existing.get("counters") or {}).get(mode)
                    if isinstance(prior, dict):
                        self.payload["counters"][mode] = prior
        self.write()

    def finalize(self, recounts: dict) -> None:
        for mode, counters in recounts.items():
            prior_failed = self.payload["counters"].get(mode, {}).get("failed") or []
            counters["failed"] = prior_failed
            self.payload["counters"][mode] = counters
        self.write()

    def write(self) -> None:
        self.payload["updated_at"] = now_iso()
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def bump(self, mode: str, **deltas) -> None:
        counters = self.payload["counters"][mode]
        for key, value in deltas.items():
            if isinstance(value, list):
                counters.setdefault(key, []).extend(value)
            else:
                counters[key] = counters.get(key, 0) + value

    def note_failure(self, mode: str, chunk_id: str, error: str) -> None:
        self.payload["counters"][mode]["failed"].append(
            {"chunk_id": chunk_id, "error": error[:500], "at": now_iso()}
        )


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out = Path(args.out).expanduser().resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.chunks_by_id: dict[str, dict] = {}
        for chunk in load_json(Path(args.chunks).expanduser()):
            self.chunks_by_id[str(chunk.get("chunk_id"))] = chunk
        self.candidates_by_chunk: dict[str, list[dict]] = {}
        if args.candidates:
            for candidate in load_json(Path(args.candidates).expanduser()):
                self.candidates_by_chunk.setdefault(str(candidate.get("chunk_id")), []).append(candidate)
            for rows in self.candidates_by_chunk.values():
                rows.sort(key=lambda row: (int(row.get("start", 0)), int(row.get("end", 0)), str(row.get("candidate_id"))))
        self.model = None
        self.timings: list[dict] = []
        self._timings_flushed = 0
        modes = ["claims", "mentions"] if args.mode == "both" else [args.mode]
        self.processed_modes = modes
        self.manifest = Manifest(
            self.out / "run_manifest.json",
            processed_modes=modes,
            resume=bool(args.resume),
            config={
                "chunks": str(Path(args.chunks).expanduser().resolve()),
                "candidates": str(Path(args.candidates).expanduser().resolve()) if args.candidates else None,
                "out": str(self.out),
                "mode": args.mode,
                "limit": args.limit,
                "resume": bool(args.resume),
                "chunk_ids_file": str(Path(args.chunk_ids).expanduser().resolve()) if args.chunk_ids else None,
                "thinking": int(args.thinking),
                "seed": int(args.seed),
                "temperature": 0.0,
                "batch_size": int(args.batch_size),
                "budgets": {"claims": CLAIMS_BUDGET, "mentions": MENTIONS_BUDGET},
                "repair": {"json_repair_retry_limit": JSON_REPAIR_LIMIT,
                           "locate_retry_limit": LOCATE_RETRY_LIMIT,
                           "max_repair_raw_chars": MAX_REPAIR_RAW_CHARS},
                "gates": {"min_quote_chars": MIN_QUOTE_CHARS, "min_quote_words": MIN_QUOTE_WORDS,
                          "max_prompt_candidates": MAX_PROMPT_CANDIDATES, "retry_limit": LOCATE_RETRY_LIMIT,
                          "max_mention_items": MENTION_V2_MAX_ITEMS, "empty_retry_limit": 1},
            },
            identity={
                "runner": str(RUNNER_PATH),
                "runner_sha256": sha256_file(RUNNER_PATH),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "prompt_revisions": {"claims": PROMPT_REVISION_CLAIMS, "mentions": PROMPT_REVISION_MENTIONS},
                "schema_versions": {"claims": CLAIMS_V3_SCHEMA["version"],
                                    "mentions": MENTION_V2_SCHEMA["version"],
                                    "mentions_validation": core.MENTION_DISCOVERY_SCHEMA["version"]},
                "source_hashes": {
                    "mpkg_rag_core.py": sha256_file(Path(core.__file__)),
                    "quote_locate.py": sha256_file(Path(ql.__file__)),
                },
                "host": {"platform": platform.platform(), "machine": platform.machine(),
                         "python": sys.version.split()[0]},
            },
        )
        if args.chunk_ids:
            ordered = load_chunk_ids_file(Path(args.chunk_ids).expanduser())
        else:
            ordered = list(self.chunks_by_id)
        missing = [chunk_id for chunk_id in ordered if chunk_id not in self.chunks_by_id]
        if missing:
            print(json.dumps({"warning": "chunk_ids_missing_from_chunks_file", "count": len(missing),
                              "sample": missing[:5]}), flush=True)
        ordered = [chunk_id for chunk_id in ordered if chunk_id in self.chunks_by_id]
        self.selected_ids = ordered

    def ensure_model(self):
        if self.model is None:
            random.seed(self.args.seed)
            try:
                import mlx.core as mx

                mx.random.seed(int(self.args.seed))
            except Exception:
                pass
            from source_snapshot.config import DemoConfig
            from source_snapshot.mlx_engine import MLXGenerator

            print(json.dumps({"status": "loading_model", "model": MODEL_ID, "revision": MODEL_REVISION,
                              "offline": True}), flush=True)
            cfg = DemoConfig.load(PROJECT / "exports/MP_KG_RAG_Gradio_UI_Windows_Claude_Port")
            cfg = cfg.__class__(**{**cfg.__dict__, "qwen_model": MODEL_ID,
                                   "qwen_revision": MODEL_REVISION, "offline": True})
            self.model = MLXGenerator.load(cfg)
        return self.model

    def flush_timings(self) -> None:
        path = self.out / "timings.jsonl"
        if self._timings_flushed < len(self.timings):
            with path.open("a", encoding="utf-8") as handle:
                for item in self.timings[self._timings_flushed:]:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
            self._timings_flushed = len(self.timings)
        peaks = [item.get("peak_memory_bytes") for item in self.timings if item.get("peak_memory_bytes")]
        if peaks:
            best = max(peaks)
            if best > (self.manifest.payload.get("peak_memory_bytes") or 0):
                self.manifest.payload["peak_memory_bytes"] = int(best)
                self.manifest.write()

    # --------------------------------------------------------------- generation

    @staticmethod
    def _parse_result(mode: str, raw: str, schema) -> tuple[dict | None, str, str | None]:
        """Route mentions modes through the v2 quote-first parser; claims stay on
        the schema-validated raw parser."""

        if str(mode).startswith("mentions"):
            return parse_mentions_v2(raw)
        return parse_raw(raw, schema)

    def _generate_singleton(self, req: dict, budget: int, thinking: bool,
                            timings: list[dict], mode: str, attempt: int,
                            batch_failed: bool = False) -> dict:
        generator = self.ensure_model()
        messages = _with_schema_contract(req["messages"], req["schema"])
        started = time.perf_counter()
        final = None
        try:
            for event in generator._stream_once(messages, int(budget), 0.0, None,
                                                enable_thinking=bool(thinking)):
                final = event
        except Exception as exc:
            timings.append({"ts": now_iso(), "chunk_id": req["chunk_id"], "mode": mode, "attempt": attempt,
                            "thinking": bool(thinking), "elapsed_s": round(time.perf_counter() - started, 3),
                            "error": f"{type(exc).__name__}: {exc}"[:500], "batch_size": 1,
                            "batch_fallback": batch_failed})
            raise
        elapsed = time.perf_counter() - started
        metrics = dict(getattr(final, "metrics", None) or {})
        raw = str(getattr(final, "raw_trace", "") or "")
        finish_reason = final.finish_reason if final else None
        generated = metrics.get("generated_tokens")
        truncated = finish_reason == "length" or (
            isinstance(generated, int) and int(budget) and generated >= int(budget))
        payload, parse_status, parse_error = self._parse_result(mode, raw, req["schema"])
        memory = mlx_memory()
        timings.append({
            "ts": now_iso(), "chunk_id": req["chunk_id"], "mode": mode, "attempt": attempt,
            "thinking": bool(thinking), "elapsed_s": round(elapsed, 3),
            "prompt_tokens": metrics.get("prompt_tokens"), "generated_tokens": generated,
            "tokens_per_second": round(metrics.get("tokens_per_second"), 2) if metrics.get("tokens_per_second") else None,
            "finish_reason": finish_reason,
            "error": None if parse_status != "failed" else parse_error,
            "tolerant_parse": parse_status == "recovered", "parse_status": parse_status,
            "truncated": bool(truncated), "batch_size": 1, "wall_s": round(elapsed, 3),
            "batch_fallback": batch_failed, "batch_failed": batch_failed,
            "peak_memory_bytes": memory.get("peak_bytes"), "active_memory_bytes": memory.get("active_bytes"),
        })
        return {"chunk_id": req["chunk_id"], "raw": raw, "payload": payload,
                "parse_status": parse_status, "parse_error": parse_error,
                "finish_reason": finish_reason, "generated_tokens": generated,
                "truncated": bool(truncated), "elapsed_s": round(elapsed, 3)}

    def _generate_batch(self, requests: list[dict], budget: int, thinking: bool,
                        timings: list[dict], mode: str, attempt: int) -> list[dict]:
        from mlx_lm.generate import BatchGenerator
        from mlx_lm.sample_utils import make_sampler

        generator = self.ensure_model()
        tokenizer = generator.tokenizer
        cfg = generator.config
        sampler = make_sampler(
            temp=0.0,
            top_p=float(getattr(cfg, "top_p", 0.9) if cfg else 0.9),
            top_k=int(getattr(cfg, "top_k", 0) if cfg else 0),
            min_p=float(getattr(cfg, "min_p", 0.0) if cfg else 0.0),
        )
        prompt_token_lists = []
        for req in requests:
            messages = _with_schema_contract(req["messages"], req["schema"])
            tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   enable_thinking=bool(thinking))
            prompt_token_lists.append(list(tokens))
        started = time.perf_counter()
        batch = BatchGenerator(generator.model,
                               stop_tokens=[[t] for t in tokenizer.eos_token_ids],
                               sampler=sampler)
        try:
            uids = batch.insert(prompt_token_lists, [int(budget)] * len(requests))
            pieces: dict[int, list[int]] = {uid: [] for uid in uids}
            finish: dict[int, str] = {}
            with batch.stats() as stats:
                while responses := batch.next_generated():
                    for response in responses:
                        if response.finish_reason is not None:
                            finish[response.uid] = response.finish_reason
                        if response.finish_reason != "stop":
                            pieces[response.uid].append(response.token)
        finally:
            batch.close()
        wall = time.perf_counter() - started
        memory = mlx_memory()
        results = []
        for req, uid in zip(requests, uids):
            raw = tokenizer.decode(pieces[uid])
            payload, parse_status, parse_error = self._parse_result(mode, raw, req["schema"])
            finish_reason = finish.get(uid)
            generated = len(pieces[uid])
            truncated = finish_reason == "length" or generated >= int(budget)
            timings.append({
                "ts": now_iso(), "chunk_id": req["chunk_id"], "mode": mode, "attempt": attempt,
                "thinking": bool(thinking), "elapsed_s": round(wall / max(1, len(requests)), 3),
                "wall_s": round(wall, 3), "batch_size": len(requests),
                "prompt_tokens": None, "generated_tokens": generated,
                "batch_prompt_tokens": getattr(stats, "prompt_tokens", None),
                "batch_generation_tokens": getattr(stats, "generation_tokens", None),
                "tokens_per_second": round(getattr(stats, "generation_tps", 0.0), 2) if getattr(stats, "generation_tps", None) else None,
                "finish_reason": finish_reason,
                "error": None if parse_status != "failed" else parse_error,
                "tolerant_parse": parse_status == "recovered", "parse_status": parse_status,
                "truncated": bool(truncated),
                "peak_memory_bytes": memory.get("peak_bytes"), "active_memory_bytes": memory.get("active_bytes"),
            })
            results.append({"chunk_id": req["chunk_id"], "raw": raw, "payload": payload,
                            "parse_status": parse_status, "parse_error": parse_error,
                            "finish_reason": finish_reason, "generated_tokens": generated,
                            "truncated": bool(truncated), "elapsed_s": round(wall / max(1, len(requests)), 3)})
        return results

    def _generate(self, mode: str, requests: list[dict], budget: int, thinking: bool,
                  attempt: int, timings: list[dict]) -> list[dict]:
        if not requests:
            return []
        if len(requests) == 1:
            return [self._generate_singleton(requests[0], budget, thinking, timings, mode, attempt)]
        try:
            return self._generate_batch(requests, budget, thinking, timings, mode, attempt)
        except Exception as exc:
            print(json.dumps({"warning": "batch_generate_failed_fallback_to_singletons", "mode": mode,
                              "batch_size": len(requests),
                              "error": f"{type(exc).__name__}: {exc}"[:300]}), flush=True)
            return [self._generate_singleton(req, budget, thinking, timings, mode, attempt, batch_failed=True)
                    for req in requests]

    def _complete_requests(self, mode: str, requests: list[dict], budget: int,
                           thinking: bool, timings: list[dict]) -> dict[str, dict]:
        """Initial call per request plus exactly one JSON-repair retry for rows
        whose output failed to parse or was truncated at the budget."""

        results: dict[str, dict] = {}
        if not requests:
            return results
        initial = self._generate(mode, requests, budget, thinking, 1, timings)
        failed: list[dict] = []
        for req, res in zip(requests, initial):
            results[req["chunk_id"]] = {
                "payload": res["payload"], "raw": res["raw"], "truncated": res["truncated"],
                "json_status": res["parse_status"], "parse_error": res["parse_error"],
                "repair_used": False, "repair_recovered": False, "finish_reason": res["finish_reason"],
                "gen_elapsed_s": res["elapsed_s"],
            }
            if res["payload"] is None:
                failed.append(req)
        if not failed:
            return results
        repair_requests = []
        for req in failed:
            prior = results[req["chunk_id"]]
            repair_requests.append({
                "chunk_id": req["chunk_id"],
                "schema": req["schema"],
                "messages": build_repair_messages(req["messages"], req["schema"], prior["raw"],
                                                  prior["truncated"], prior["parse_error"]),
            })
        repaired = self._generate(f"{mode}_json_repair", repair_requests, budget, False, 2, timings)
        for req, res in zip(repair_requests, repaired):
            row = results[req["chunk_id"]]
            row["repair_used"] = True
            row["truncated"] = bool(row["truncated"] or res["truncated"])
            row["gen_elapsed_s"] = row.get("gen_elapsed_s", 0.0) + res["elapsed_s"]
            row["raw"] = (row["raw"] + "\n[repair]\n" + res["raw"]) if row["raw"] else res["raw"]
            if res["payload"] is not None:
                row["payload"] = res["payload"]
                row["json_status"] = "repair"
                row["repair_recovered"] = True
                row["finish_reason"] = res["finish_reason"]
            else:
                row["json_status"] = "failed"
        return results

    # ------------------------------------------------------------------ claims

    def candidate_prompt_lines(self, chunk_id: str) -> tuple[list[str], set[str]]:
        rows = self.candidates_by_chunk.get(chunk_id, [])[:MAX_PROMPT_CANDIDATES]
        lines = [f"{row.get('candidate_id')} | {row.get('label')} | span[{row.get('start')}:{row.get('end')}]"
                 for row in rows]
        return lines, {str(row.get("candidate_id")) for row in rows}

    def build_claims_messages(self, chunk: dict, candidate_lines: list[str]) -> list[dict]:
        text = str(chunk.get("text", ""))
        context = {"document_uid": chunk.get("document_uid"), "chunk_id": chunk.get("chunk_id"),
                   "source_id": chunk.get("source_id"), "page": chunk.get("page")}
        user = f"Source context: {json.dumps(context, ensure_ascii=False, sort_keys=True)}\n"
        if candidate_lines:
            user += "Candidate list for this chunk (candidate_id | label | span[start:end]):\n"
            user += "\n".join(candidate_lines) + "\n"
        else:
            user += "Candidate list: (none supplied for this chunk; omit candidate_ids)\n"
        user += f"Passage (do not normalize; copy quotes character-for-character from this text):\n{text}"
        return [{"role": "system", "content": CLAIMS_SYSTEM}, {"role": "user", "content": user}]

    def _count_occurrences(self, text: str, quote: str) -> int:
        if not quote:
            return 0
        count = 0
        pos = text.find(quote)
        while pos != -1:
            count += 1
            pos = text.find(quote, pos + 1)
        return count

    def _process_quotes(self, payload, attempt: int, chunk: dict) -> tuple[list[dict], list[dict]]:
        text = str(chunk.get("text", ""))
        located: list[dict] = []
        failures: list[dict] = []
        items = payload.get("claims") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return located, failures
        for item in items:
            if not isinstance(item, dict):
                failures.append({"quote": "", "reason": "missing_quote", "attempt": attempt, "candidate_ids": []})
                continue
            chosen = [value for value in (item.get("candidate_ids") or []) if isinstance(value, str)]
            raw_quote = item.get("quote")
            if not isinstance(raw_quote, str) or not raw_quote.strip():
                failures.append({"quote": "", "reason": "missing_quote", "attempt": attempt, "candidate_ids": chosen})
                continue
            quote = raw_quote.strip()
            if len(quote) < MIN_QUOTE_CHARS or word_count(quote) < MIN_QUOTE_WORDS:
                failures.append({"quote": quote, "reason": "below_min_length", "attempt": attempt,
                                 "candidate_ids": chosen})
                continue
            hit = ql.locate_quote(text, quote, min_chars=MIN_QUOTE_CHARS, min_words=MIN_QUOTE_WORDS, hint=None)
            if hit is None:
                failures.append({"quote": quote, "reason": "not_located", "attempt": attempt,
                                 "candidate_ids": chosen})
                continue
            located.append({
                "quote": hit["quote"],
                "response_quote": quote,
                "span_start": hit["start"],
                "span_end": hit["end"],
                "span_method": hit["method"],
                "score": hit["score"],
                "occurrence_count": self._count_occurrences(text, quote),
                "candidate_ids_raw": chosen,
                "attempt": attempt,
            })
        return located, failures

    def _apply_g3_g4(self, located: list[dict], supplied_ids: set[str], chunk_id: str) -> list[dict]:
        spans = {str(row.get("candidate_id")): (int(row.get("start", 0)), int(row.get("end", 0)))
                 for row in self.candidates_by_chunk.get(chunk_id, [])}
        kept: list[dict] = []
        for index, record in enumerate(located):
            record["claim_index"] = index
            kept.append(record)
        by_span: dict[tuple, dict] = {}
        for record in kept:
            key = (record["span_start"], record["span_end"])
            current = by_span.get(key)
            if current is None or ((record["span_end"] - record["span_start"],
                                    -record["claim_index"]) > (current["span_end"] - current["span_start"],
                                                               -current["claim_index"])):
                by_span[key] = record
        span_deduped = sorted(by_span.values(), key=lambda item: item["claim_index"])
        by_norm: dict[str, dict] = {}
        for record in span_deduped:
            norm, _ = ql.normalize_with_index_map(record["quote"])
            if norm not in by_norm:
                by_norm[norm] = record
        final: list[dict] = []
        for record in sorted(by_norm.values(), key=lambda item: item["claim_index"]):
            valid_ids: list[str] = []
            for candidate_id in dict.fromkeys(record["candidate_ids_raw"]):
                span = spans.get(candidate_id)
                if candidate_id in supplied_ids and span is not None \
                        and span[0] >= record["span_start"] and span[1] <= record["span_end"]:
                    valid_ids.append(candidate_id)
            final.append({
                "quote": record["quote"],
                "span_start": record["span_start"],
                "span_end": record["span_end"],
                "span_method": record["span_method"],
                "graph_retrieval_eligible": bool(valid_ids),
                "candidate_ids_valid": valid_ids,
                "candidate_ids_raw": record["candidate_ids_raw"],
                "occurrence_count": record["occurrence_count"],
                "locate_score": record["score"],
                "attempt": record["attempt"],
            })
        return final

    def _claims_locate_retry_messages(self, item: dict) -> list[dict]:
        chunk = item["chunk"]
        text = str(chunk.get("text", ""))
        previous_raw = str(item["comp"]["raw"] or "")[:4000]
        feedback_lines = []
        for i, failure in enumerate(item["retry_failures"], 1):
            if failure["reason"] == "below_min_length":
                guidance = ("too short: copy a longer exact span, at least 40 characters and 5 words, "
                            "that contains this text; otherwise omit it")
            else:
                guidance = "not located: copy it character-for-character from the passage; otherwise omit it"
            feedback_lines.append(f'{i}. {json.dumps(failure["quote"])} — {guidance}')
        feedback = (
            "The previous response contained quotes that were not retained. For each failed quote, "
            "follow the guidance and return a corrected claim, or omit it. Do not paraphrase and do "
            "not add offsets.\nFailed quotes:\n"
            + "\n".join(feedback_lines)
            + f"\nPassage (do not normalize):\n{text}"
        )
        return item["messages"] + [
            {"role": "assistant", "content": previous_raw or json.dumps(item["comp"]["payload"] or {})},
            {"role": "user", "content": feedback},
        ]

    def _finalize_claims_chunk(self, item: dict, retry_result: dict | None) -> tuple[dict, list[dict]]:
        chunk = item["chunk"]
        chunk_id = item["chunk_id"]
        request_id = item["request_id"]
        comp = item["comp"]
        located = list(item["located"])
        failures = item["failures"]
        rejections: list[dict] = []
        retry_used = bool(item["retry_failures"])
        retry_returned = 0
        if retry_used:
            retry_failures_left: list[dict] = []
            recovered: set[str] = set()
            if retry_result is not None:
                retry_payload = retry_result["payload"]
                retry_returned = len(retry_payload.get("claims", [])) \
                    if isinstance(retry_payload, dict) and isinstance(retry_payload.get("claims"), list) else 0
                retry_located, retry_failures_left = self._process_quotes(retry_payload, 2, chunk)
                located.extend(retry_located)
                recovered = {entry["response_quote"] for entry in retry_located}
            still_failing = {entry["quote"] for entry in retry_failures_left}
            for failure in retry_failures_left:
                rejections.append({"chunk_id": chunk_id, "request_id": request_id, "quote": failure["quote"],
                                   "reason": failure["reason"], "attempt": 2, "created_at": now_iso()})
            for failure in item["retry_failures"]:
                if failure["quote"] in recovered or failure["quote"] in still_failing:
                    continue
                rejections.append({"chunk_id": chunk_id, "request_id": request_id, "quote": failure["quote"],
                                   "reason": failure["reason"], "attempt": 2, "created_at": now_iso()})
            for failure in [entry for entry in failures if not entry["quote"]]:
                rejections.append({"chunk_id": chunk_id, "request_id": request_id, "quote": "",
                                   "reason": failure["reason"], "attempt": 1, "created_at": now_iso()})
        else:
            for failure in failures:
                rejections.append({"chunk_id": chunk_id, "request_id": request_id, "quote": failure["quote"],
                                   "reason": failure["reason"], "attempt": 1, "created_at": now_iso()})

        claims = self._apply_g3_g4(located, item["supplied_ids"], chunk_id)
        memory = mlx_memory()
        row = {
            "chunk_id": chunk_id,
            "request_id": request_id,
            "document_uid": chunk.get("document_uid"),
            "claims": claims,
            "parse_status": comp["json_status"],
            "prompt_revision": PROMPT_REVISION_CLAIMS,
            "schema_version": CLAIMS_V3_SCHEMA["version"],
            "thinking": bool(self.args.thinking),
            "retry_used": retry_used,
            "json_repair_used": comp["repair_used"],
            "json_repair_recovered": comp["repair_recovered"],
            "truncated": bool(comp["truncated"]),
            "candidates_supplied": len(item["supplied_ids"]),
            "quotes_returned": item["first_returned"] + retry_returned,
            "quotes_located_raw": len(located),
            "quotes_retained": len(claims),
            "quotes_rejected": len(rejections),
            "quotes_eligible": sum(1 for claim in claims if claim["graph_retrieval_eligible"]),
            "elapsed_seconds": None,
            "peak_memory_bytes": memory.get("peak_bytes"),
            "active_memory_bytes": memory.get("active_bytes"),
        }
        return row, rejections

    def run_claims_batch(self, chunks: list[dict]) -> list[str]:
        output_path = self.out / "claims_v3.jsonl"
        queue_path = self.out / "quote_rejection_queue.jsonl"
        requests = []
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id"))
            candidate_lines, _ = self.candidate_prompt_lines(chunk_id)
            requests.append({"chunk_id": chunk_id,
                             "messages": self.build_claims_messages(chunk, candidate_lines),
                             "schema": CLAIMS_V3_SCHEMA})
        completed = self._complete_requests("claims", requests, CLAIMS_BUDGET,
                                            bool(self.args.thinking), self.timings)
        prepared = []
        for chunk, req in zip(chunks, requests):
            chunk_id = str(chunk.get("chunk_id"))
            _, supplied_ids = self.candidate_prompt_lines(chunk_id)
            comp = completed[chunk_id]
            payload = comp["payload"]
            located, failures = self._process_quotes(payload, 1, chunk)
            first_returned = len(payload.get("claims", [])) \
                if isinstance(payload, dict) and isinstance(payload.get("claims"), list) else 0
            prepared.append({"chunk": chunk, "chunk_id": chunk_id,
                             "request_id": f"claims-{uuid.uuid4().hex[:16]}",
                             "messages": req["messages"], "comp": comp,
                             "located": located, "failures": failures,
                             "first_returned": first_returned, "supplied_ids": supplied_ids,
                             "retry_failures": [entry for entry in failures if entry["quote"]]})
        retry_requests = []
        for item in prepared:
            if not item["retry_failures"]:
                continue
            retry_requests.append({"chunk_id": item["chunk_id"], "schema": CLAIMS_V3_SCHEMA,
                                   "messages": self._claims_locate_retry_messages(item)})
        retry_results = {res["chunk_id"]: res for res in
                         self._generate("claims_retry", retry_requests, CLAIMS_BUDGET, False, 2,
                                        self.timings)}
        written: list[str] = []
        for item in prepared:
            try:
                t0 = time.perf_counter()
                retry_result = retry_results.get(item["chunk_id"])
                row, rejections = self._finalize_claims_chunk(item, retry_result)
                processing = time.perf_counter() - t0
                retry_gen = retry_result["elapsed_s"] if (item["retry_failures"] and retry_result) else 0.0
                row["elapsed_seconds"] = round(
                    item["comp"].get("gen_elapsed_s", 0.0) + retry_gen + processing, 3)
                with output_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                if rejections:
                    with queue_path.open("a", encoding="utf-8") as handle:
                        for entry in rejections:
                            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        handle.flush()
                self.manifest.bump("claims", attempted=1, ok=1,
                                   quotes_returned=row["quotes_returned"],
                                   quotes_located=row["quotes_located_raw"],
                                   quotes_retained=row["quotes_retained"],
                                   quotes_rejected=row["quotes_rejected"],
                                   quotes_eligible=row["quotes_eligible"],
                                   retry_chunks=1 if row["retry_used"] else 0,
                                   json_repairs=1 if row["json_repair_used"] else 0,
                                   json_repair_recoveries=1 if row["json_repair_recovered"] else 0,
                                   truncations=1 if row["truncated"] else 0)
                print(json.dumps({"chunk": item["chunk_id"][:16], "mode": "claims",
                                  "parse_status": row["parse_status"],
                                  "returned": row["quotes_returned"],
                                  "retained": row["quotes_retained"],
                                  "rejected": row["quotes_rejected"],
                                  "eligible": row["quotes_eligible"],
                                  "retry": row["retry_used"],
                                  "json_repair": row["json_repair_used"],
                                  "truncated": row["truncated"],
                                  "elapsed": row["elapsed_seconds"]}), flush=True)
                self.manifest.write()
                self.flush_timings()
                written.append(item["chunk_id"])
            except Exception as exc:
                self.manifest.bump("claims", attempted=1)
                self.manifest.note_failure("claims", item["chunk_id"], f"{type(exc).__name__}: {exc}")
                print(json.dumps({"chunk": item["chunk_id"][:16], "mode": "claims",
                                  "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:300]}),
                      flush=True)
        return written

    # ---------------------------------------------------------------- mentions

    def build_mention_v2_messages(self, text: str, context: dict) -> list[dict]:
        user = (f"Source context: {json.dumps(context, ensure_ascii=False, sort_keys=True)}\n"
                f"Passage (do not normalize; copy mention surfaces character-for-character from this text):\n{text}")
        return [{"role": "system", "content": MENTIONS_SYSTEM_V2}, {"role": "user", "content": user}]

    @staticmethod
    def _payload_items(payload) -> list[dict]:
        if isinstance(payload, dict) and isinstance(payload.get("mentions"), list):
            return [item for item in payload["mentions"] if isinstance(item, dict)]
        return []

    def _locate_mention_items(self, items: list[dict], text: str) -> tuple[list[dict], list[dict], int, int]:
        """Locate model-supplied surfaces; assign ordered ids and exact offsets.

        Surfaces are matched exact-first (preferring the first unused occurrence
        at or after the previous match so mentions stay in order), then with the
        quote_locate normalization pass. Surfaces that repeat an earlier surface
        are collapsed (one mention per distinct surface string). Located records
        are returned as mention-discovery.v1 records for core.validate_mentions;
        a located record cannot have a wrong offset because the offset is
        produced by the match itself.
        """

        located: list[dict] = []
        misses: list[dict] = []
        used: set[tuple[int, int]] = set()
        seen_surfaces: set[str] = set()
        over_cap = 0
        repeated = 0
        cursor = 0
        normalized_text: str | None = None
        normalized_map: list[int] | None = None
        for index, item in enumerate(items):
            if index >= MENTION_V2_MAX_ITEMS:
                over_cap += 1
                continue
            surface = str(item.get("text") or "").strip()
            if not surface:
                continue
            if surface in seen_surfaces:
                repeated += 1
                continue
            seen_surfaces.add(surface)
            occurrences: list[tuple[int, int]] = []
            position = text.find(surface)
            while position != -1:
                occurrences.append((position, position + len(surface)))
                position = text.find(surface, position + 1)
            chosen = None
            method = "exact"
            for start, end in occurrences:
                if start >= cursor and (start, end) not in used:
                    chosen = (start, end)
                    break
            if chosen is None:
                for start, end in occurrences:
                    if (start, end) not in used:
                        chosen = (start, end)
                        break
            if chosen is None:
                if normalized_text is None:
                    normalized_text, normalized_map = ql.normalize_with_index_map(text)
                normalized_surface = ql.normalize_with_index_map(surface)[0]
                if normalized_surface:
                    spans: list[tuple[int, int]] = []
                    position = normalized_text.find(normalized_surface)
                    while position != -1:
                        spans.append((normalized_map[position],
                                      normalized_map[position + len(normalized_surface)]))
                        position = normalized_text.find(normalized_surface, position + 1)
                    for start, end in spans:
                        if start >= cursor and (start, end) not in used:
                            chosen, method = (start, end), "normalized"
                            break
                    if chosen is None:
                        for start, end in spans:
                            if (start, end) not in used:
                                chosen, method = (start, end), "normalized"
                                break
            if chosen is None:
                misses.append({"surface": surface[:80],
                               "reason": "duplicate_occurrences" if occurrences else "not_located"})
                continue
            start, end = chosen
            used.add((start, end))
            cursor = end
            located.append({
                "mention_id": str(len(located) + 1),
                "text": text[start:end],
                "start": start,
                "end": end,
                "mention_type": str(item.get("type") or "unknown"),
                "match_method": method,
                "response_surface": surface,
            })
        return located, misses, over_cap, repeated

    def run_mentions_batch(self, chunks: list[dict]) -> list[str]:
        output_path = self.out / "mentions.jsonl"
        requests = []
        contexts: dict[str, dict] = {}
        request_by_id: dict[str, dict] = {}
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id"))
            text = str(chunk.get("text", ""))
            context = {key: chunk.get(key) for key in (
                "document_uid", "chunk_id", "source_id", "source_type", "authority_score",
                "status", "content_sha256", "document_sha256", "text_sha256") if chunk.get(key) is not None}
            contexts[chunk_id] = context
            request = {"chunk_id": chunk_id,
                       "messages": self.build_mention_v2_messages(text, context),
                       "schema": MENTION_V2_SCHEMA}
            requests.append(request)
            request_by_id[chunk_id] = request
        completed = self._complete_requests("mentions", requests, MENTIONS_BUDGET, False, self.timings)
        empty_ids = [chunk_id for chunk_id in request_by_id
                     if isinstance(completed[chunk_id]["payload"], dict)
                     and not self._payload_items(completed[chunk_id]["payload"])]
        retry_results: dict[str, dict] = {}
        if empty_ids:
            nudge_requests = [{
                "chunk_id": chunk_id,
                "schema": MENTION_V2_SCHEMA,
                "messages": request_by_id[chunk_id]["messages"] + [
                    {"role": "assistant", "content": '{"mentions": []}'},
                    {"role": "user", "content": MENTION_V2_EMPTY_NUDGE},
                ],
            } for chunk_id in empty_ids]
            retry_results = {result["chunk_id"]: result for result in
                             self._generate("mentions_empty_retry", nudge_requests,
                                            MENTIONS_BUDGET, False, 2, self.timings)}
        written: list[str] = []
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id"))
            try:
                t0 = time.perf_counter()
                text = str(chunk.get("text", ""))
                context = contexts[chunk_id]
                comp = completed[chunk_id]
                payload = comp["payload"]
                empty_retry_used = chunk_id in retry_results
                empty_retry_recovered = False
                retry = retry_results.get(chunk_id)
                if retry is not None and self._payload_items(retry["payload"]):
                    payload = retry["payload"]
                    comp = {**comp, "payload": payload,
                            "raw": (comp["raw"] + "\n[empty_retry]\n" + retry["raw"]) if comp["raw"] else retry["raw"],
                            "json_status": "empty_retry",
                            "truncated": bool(comp["truncated"] or retry["truncated"]),
                            "gen_elapsed_s": comp.get("gen_elapsed_s", 0.0) + retry["elapsed_s"]}
                    empty_retry_recovered = True
                items = self._payload_items(payload)
                mentions: list[dict] = []
                quarantined = 0
                validation_status = None
                located: list[dict] = []
                misses: list[dict] = []
                over_cap = 0
                repeated = 0
                if items:
                    located, misses, over_cap, repeated = self._locate_mention_items(items, text)
                    v1_shaped = {
                        "schema_version": core.MENTION_DISCOVERY_SCHEMA["version"],
                        "mentions": [{key: record[key] for key in
                                      ("mention_id", "text", "start", "end", "mention_type")}
                                     for record in located],
                    }
                    validation = core.validate_mentions(v1_shaped, text, context)
                    mentions = list(validation.get("accepted", []))
                    quarantined = len(validation.get("quarantined", []))
                    validation_status = validation.get("status")
                elif payload is not None:
                    validation = core.validate_mentions(
                        {"schema_version": core.MENTION_DISCOVERY_SCHEMA["version"], "mentions": []},
                        text, context)
                    validation_status = validation.get("status")
                memory = mlx_memory()
                row = {
                    "document_uid": chunk.get("document_uid"),
                    "chunk_id": chunk_id,
                    "text_sha256": sha256_text(text),
                    "text": text,
                    "mentions": mentions,
                    "mentions_returned": len(items),
                    "mentions_located": len(located),
                    "unlocated_mentions": len(misses),
                    "unlocated_surfaces": [miss["surface"] for miss in misses[:5]],
                    "mentions_over_cap": over_cap,
                    "repeated_response_items": repeated,
                    "parse_status": comp["json_status"],
                    "validation_status": validation_status,
                    "quarantined_mentions": quarantined,
                    "json_repair_used": comp["repair_used"],
                    "json_repair_recovered": comp["repair_recovered"],
                    "empty_retry_used": empty_retry_used,
                    "empty_retry_recovered": empty_retry_recovered,
                    "truncated": bool(comp["truncated"]),
                    "prompt_revision": PROMPT_REVISION_MENTIONS,
                    "schema_version": MENTION_V2_SCHEMA["version"],
                    "raw_output": comp["raw"],
                    "elapsed_seconds": round(comp.get("gen_elapsed_s", 0.0)
                                             + (time.perf_counter() - t0), 3),
                    "peak_memory_bytes": memory.get("peak_bytes"),
                    "active_memory_bytes": memory.get("active_bytes"),
                }
                with output_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                self.manifest.bump("mentions", attempted=1, ok=1,
                                   mentions_returned=len(items),
                                   mentions_located=len(located),
                                   mentions_accepted=len(mentions),
                                   mentions_quarantined=quarantined,
                                   unlocated_mentions=len(misses),
                                   mentions_over_cap=over_cap,
                                   repeated_response_items=repeated,
                                   semantic_invalid=1 if quarantined > 0 else 0,
                                   json_repairs=1 if row["json_repair_used"] else 0,
                                   json_repair_recoveries=1 if row["json_repair_recovered"] else 0,
                                   empty_retry_chunks=1 if empty_retry_used else 0,
                                   empty_retry_recoveries=1 if empty_retry_recovered else 0,
                                   truncations=1 if row["truncated"] else 0)
                print(json.dumps({"chunk": chunk_id[:16], "mode": "mentions",
                                  "parse_status": row["parse_status"],
                                  "validation_status": row["validation_status"],
                                  "returned": len(items),
                                  "located": len(located),
                                  "accepted": len(mentions),
                                  "unlocated": len(misses),
                                  "repeated": repeated,
                                  "quarantined": quarantined,
                                  "empty_retry": empty_retry_used,
                                  "json_repair": row["json_repair_used"],
                                  "truncated": row["truncated"]}), flush=True)
                self.manifest.write()
                self.flush_timings()
                written.append(chunk_id)
            except Exception as exc:
                self.manifest.bump("mentions", attempted=1)
                self.manifest.note_failure("mentions", chunk_id, f"{type(exc).__name__}: {exc}")
                print(json.dumps({"chunk": chunk_id[:16], "mode": "mentions",
                                  "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:300]}),
                      flush=True)
        return written

    # -------------------------------------------------------------------- main

    def _recount_from_outputs(self, mode: str) -> dict | None:
        path = {"claims": self.out / "claims_v3.jsonl", "mentions": self.out / "mentions.jsonl"}.get(mode)
        if path is None or not path.exists():
            return None
        rows: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        if mode == "claims":
            return {"attempted": len(rows), "ok": len(rows),
                    "quotes_returned": sum(r.get("quotes_returned", 0) or 0 for r in rows),
                    "quotes_located": sum(r.get("quotes_located_raw", 0) or 0 for r in rows),
                    "quotes_retained": sum(r.get("quotes_retained", 0) or 0 for r in rows),
                    "quotes_rejected": sum(r.get("quotes_rejected", 0) or 0 for r in rows),
                    "quotes_eligible": sum(r.get("quotes_eligible", 0) or 0 for r in rows),
                    "retry_chunks": sum(1 for r in rows if r.get("retry_used")),
                    "json_repairs": sum(1 for r in rows if r.get("json_repair_used")),
                    "json_repair_recoveries": sum(1 for r in rows if r.get("json_repair_recovered")),
                    "truncations": sum(1 for r in rows if r.get("truncated"))}
        return {"attempted": len(rows), "ok": len(rows),
                "mentions_returned": sum(r.get("mentions_returned", 0) or 0 for r in rows),
                "mentions_located": sum(r.get("mentions_located", 0) or 0 for r in rows),
                "mentions_accepted": sum(len(r.get("mentions") or []) for r in rows),
                "mentions_quarantined": sum(r.get("quarantined_mentions", 0) or 0 for r in rows),
                "unlocated_mentions": sum(r.get("unlocated_mentions", 0) or 0 for r in rows),
                "mentions_over_cap": sum(r.get("mentions_over_cap", 0) or 0 for r in rows),
                "repeated_response_items": sum(r.get("repeated_response_items", 0) or 0 for r in rows),
                "semantic_invalid": sum(1 for r in rows if (r.get("quarantined_mentions") or 0) > 0),
                "json_repairs": sum(1 for r in rows if r.get("json_repair_used")),
                "json_repair_recoveries": sum(1 for r in rows if r.get("json_repair_recovered")),
                "empty_retry_chunks": sum(1 for r in rows if r.get("empty_retry_used")),
                "empty_retry_recoveries": sum(1 for r in rows if r.get("empty_retry_recovered")),
                "truncations": sum(1 for r in rows if r.get("truncated"))}

    def run(self) -> int:
        args = self.args
        modes = ["claims", "mentions"] if args.mode == "both" else [args.mode]
        selected = self.selected_ids[: args.limit] if args.limit else self.selected_ids
        done_by_mode: dict[str, set[str]] = {}
        pending_by_mode: dict[str, list[str]] = {}
        for mode in modes:
            done = load_jsonl_chunk_ids(
                self.out / f"{'claims_v3' if mode == 'claims' else 'mentions'}.jsonl")
            done_by_mode[mode] = done
            pending_by_mode[mode] = [cid for cid in selected if not (args.resume and cid in done)]
        print(json.dumps({"status": "start", "selected": len(selected),
                          "pending": {mode: len(pending_by_mode[mode]) for mode in modes},
                          "modes": modes, "resume": bool(args.resume),
                          "batch_size": int(args.batch_size),
                          "already_done": {mode: len(done_by_mode[mode]) for mode in modes}}), flush=True)

        started_wall = time.perf_counter()
        processed = 0
        for mode in modes:
            for batch_ids in iter_batches(pending_by_mode[mode], args.batch_size):
                chunks = [self.chunks_by_id[cid] for cid in batch_ids]
                try:
                    if mode == "claims":
                        written = self.run_claims_batch(chunks)
                    else:
                        written = self.run_mentions_batch(chunks)
                except KeyboardInterrupt:
                    self.flush_timings()
                    self.manifest.write()
                    print(json.dumps({"status": "interrupted", "mode": mode,
                                      "batch": [cid[:12] for cid in batch_ids]}), flush=True)
                    return 130
                except Exception as exc:
                    already = load_jsonl_chunk_ids(
                        self.out / f"{'claims_v3' if mode == 'claims' else 'mentions'}.jsonl")
                    for chunk_id in batch_ids:
                        if chunk_id in already:
                            continue
                        self.manifest.bump(mode, attempted=1)
                        self.manifest.note_failure(mode, chunk_id, f"{type(exc).__name__}: {exc}")
                    print(json.dumps({"status": "batch_failed", "mode": mode,
                                      "batch": [cid[:12] for cid in batch_ids],
                                      "error": f"{type(exc).__name__}: {exc}"[:300]}), flush=True)
                    self.flush_timings()
                    self.manifest.write()
                    continue
                processed += len(written)

        wall = time.perf_counter() - started_wall
        if processed:
            self.manifest.payload["wall_seconds"] = round(wall, 3)
        self.flush_timings()
        recounts = {}
        for mode in self.processed_modes:
            counters = self._recount_from_outputs(mode)
            if counters is not None:
                recounts[mode] = counters
        self.manifest.finalize(recounts)
        print(json.dumps({"status": "done", "chunks_processed": processed, "wall_seconds": round(wall, 3),
                          "peak_memory_bytes": self.manifest.payload.get("peak_memory_bytes")}), flush=True)
        return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable MLX extraction runner (v3 quote-first claims + mention discovery).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--chunks", required=True, help="path to chunks.json")
    parser.add_argument("--candidates", default=None, help="optional path to candidates.json")
    parser.add_argument("--out", required=True, help="output directory for jsonl + manifest")
    parser.add_argument("--mode", choices=["claims", "mentions", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="max chunks to process")
    parser.add_argument("--resume", action="store_true",
                        help="skip chunk_ids already present in the mode's out jsonl")
    parser.add_argument("--chunk-ids", default=None,
                        help="optional subset file: JSON array, {chunk_ids:[...]}, or newline list")
    parser.add_argument("--thinking", type=int, choices=[0, 1], default=0,
                        help="enable model thinking for claims calls (mentions always non-thinking)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="prompts generated per batched MLX call; 1 = sequential (batch is optional)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    runner = Runner(args)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
