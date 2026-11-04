"""Lazy MLX-LM lifecycle and structured streaming for the local demo.

Nothing in this module imports MLX at module import time.  This is deliberate:
the history/retrieval tests also run on Linux and on machines without Metal.
"""

from __future__ import annotations

import json
import os
import platform
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from .config import DemoConfig, QWEN_MODEL, QWEN_REVISION


_MAX_REPAIR_RAW_CHARS = 16_384


def _bounded_repair_raw(raw: Any) -> str:
    """Keep repair context bounded while retaining both ends of raw output."""

    value = str(raw or "")
    if len(value) <= _MAX_REPAIR_RAW_CHARS:
        return value
    marker = "\n...[raw output truncated for repair context]...\n"
    keep = max(1, (_MAX_REPAIR_RAW_CHARS - len(marker)) // 2)
    return value[:keep] + marker + value[-keep:]


def _repair_prompt(raw: Any, schema: Mapping[str, Any], error: str | None = None) -> str:
    """Build a bounded, non-thinking repair instruction with the failed output."""

    failure = str(error or "invalid JSON or schema validation failure")
    return (
        "Repair the previous output. Return exactly one JSON object, no reasoning. "
        "The previous output failed validation; preserve valid information where possible "
        "and emit only the repaired JSON object.\n"
        f"Validation error: {failure}\n"
        "Required schema:\n"
        f"{json.dumps(schema, sort_keys=True)}\n"
        "Previous raw output (verbatim, bounded):\n"
        "<previous_raw_output>\n"
        f"{_bounded_repair_raw(raw)}\n"
        "</previous_raw_output>"
    )


@dataclass(frozen=True)
class GenerationEvent:
    """UI-neutral generation update; the final event carries parsed JSON."""

    text: str = ""
    raw_trace: str = ""
    structured: Mapping[str, Any] | None = None
    done: bool = False
    metrics: Mapping[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    error: str | None = None
    repaired: bool = False


def split_thinking(raw: str) -> dict[str, Any]:
    """Split Qwen's optional reasoning block without losing the raw output."""

    value = str(raw or "")
    match = re.search(r"<think>\s*(.*?)\s*</think>", value, flags=re.I | re.S)
    if match:
        return {"reasoning_content": match.group(1).strip(), "final_content": (value[: match.start()] + value[match.end() :]).strip(), "thinking_status": "complete", "reasoning_truncated": False}
    closing = re.search(r"</think>", value, flags=re.I)
    if closing:
        return {"reasoning_content": value[: closing.start()].removeprefix("<think>").strip(), "final_content": value[closing.end() :].strip(), "thinking_status": "complete_closing_tag_only", "reasoning_truncated": False}
    opening = re.search(r"<think>", value, flags=re.I)
    if opening:
        return {"reasoning_content": value[opening.end() :].strip(), "final_content": "", "thinking_status": "truncated", "reasoning_truncated": True}
    return {"reasoning_content": "", "final_content": value.strip(), "thinking_status": "not_emitted", "reasoning_truncated": False}


def _schema_valid(value: Any, schema: Mapping[str, Any]) -> bool:
    if schema.get("type") == "object":
        if not isinstance(value, Mapping) or schema.get("additionalProperties") is False and set(value) - set(schema.get("properties", {})):
            return False
        if any(key not in value for key in schema.get("required", ())):
            return False
        return all(_schema_valid(value[key], spec) for key, spec in schema.get("properties", {}).items() if key in value)
    if schema.get("type") == "array":
        return isinstance(value, list) and all(_schema_valid(item, schema.get("items", {})) for item in value)
    expected = schema.get("type")
    if isinstance(expected, list):
        return any(_schema_valid(value, {"type": item}) for item in expected)
    if expected == "string":
        return isinstance(value, str) and (not schema.get("pattern") or re.fullmatch(schema["pattern"], value) is not None)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and float(schema.get("minimum", float("-inf"))) <= float(value) <= float(schema.get("maximum", float("inf")))
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _project_schema_object(value: Any, schema: Mapping[str, Any]) -> Any:
    """Drop surplus top-level keys while retaining the verbatim raw trace."""

    if isinstance(value, Mapping) and schema.get("type") == "object":
        nested = value.get("properties")
        required = set(schema.get("required", ()))
        # Small local models sometimes echo the JSON-schema envelope and put
        # their actual values inside ``properties``. Unwrap that deterministic
        # shape before applying ordinary validation.
        if isinstance(nested, Mapping) and required <= set(nested):
            value = nested
    if isinstance(value, Mapping) and schema.get("type") == "object" and schema.get("additionalProperties") is False:
        allowed = schema.get("properties", {})
        return {key: item for key, item in value.items() if key in allowed}
    return value


class MLXGenerator:
    """Reusable one-model MLX-LM generator.

    ``load`` is the only method that imports optional model dependencies. Tests
    can instantiate this class with fake model/tokenizer/stream functions.
    """

    def __init__(self, model: Any, tokenizer: Any, stream_generate: Any, *, config: DemoConfig | None = None, sampler_factory: Any | None = None, peak_memory_reader: Any | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._stream_generate = stream_generate
        self.config = config
        self._sampler_factory = sampler_factory
        self._peak_memory_reader = peak_memory_reader

    @classmethod
    def load(cls, config: DemoConfig) -> "MLXGenerator":
        if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
            raise RuntimeError("the local MLX demo requires an Apple-silicon macOS host with Metal")
        try:
            from huggingface_hub import snapshot_download
            from mlx_lm import load, stream_generate
        except ImportError as exc:
            raise RuntimeError("MLX-LM and huggingface-hub are required on Apple silicon; install the local demo requirements") from exc
        model_id = config.qwen_model or QWEN_MODEL
        revision = config.qwen_revision or QWEN_REVISION
        cache_dir = getattr(config, "model_cache_dir", None)
        offline = bool(getattr(config, "offline", False)) or os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        if offline:
            # ``local_files_only`` is the Hub's no-network code path.  Do not
            # catch its failure and retry online: a pinned local snapshot is
            # an explicit prerequisite for offline operation.
            snapshot = snapshot_download(repo_id=model_id, revision=revision, local_files_only=True, **({"cache_dir": str(cache_dir)} if cache_dir else {}))
            snapshot_path = __import__("pathlib").Path(snapshot)
            if not snapshot_path.is_dir() or snapshot_path.name != revision:
                raise RuntimeError(f"offline model snapshot is not the pinned revision {revision}: {snapshot_path}")
        else:
            kwargs = {"repo_id": model_id, "revision": revision}
            if cache_dir:
                kwargs["cache_dir"] = str(cache_dir)
            snapshot = snapshot_download(**kwargs)
        model, tokenizer = load(snapshot)
        return cls(model, tokenizer, stream_generate, config=config)

    def _peak_memory(self) -> int:
        if self._peak_memory_reader is not None:
            try:
                return max(0, int(self._peak_memory_reader()))
            except (TypeError, ValueError, RuntimeError):
                return 0
        # Injected/fake generators intentionally omit config.  Do not import
        # MLX merely to collect optional metrics in ordinary non-Metal tests.
        if self.config is None:
            return 0
        try:
            import mlx.core as mx

            return max(0, int(mx.metal.get_peak_memory()))
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
            return 0

    def _prompt(self, messages: Any, *, enable_thinking: bool = True) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        return "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages)

    @staticmethod
    def _response_text(response: Any) -> str:
        if isinstance(response, str):
            return response
        return str(getattr(response, "text", getattr(response, "delta", "")) or "")

    @staticmethod
    def _response_metric(response: Any, name: str, default: Any = None) -> Any:
        value = getattr(response, name, default)
        return value() if callable(value) else value

    def _sampler(self, temperature: float) -> Any:
        factory = self._sampler_factory
        if factory is None:
            # A dependency-free fake generator can stream without a sampler;
            # real MLX instances are always loaded with a DemoConfig.
            if self.config is None:
                return None
            try:
                from mlx_lm.sample_utils import make_sampler
            except ImportError as exc:
                raise RuntimeError("mlx-lm sample_utils.make_sampler is required for local generation") from exc
            factory = make_sampler
        config = self.config
        kwargs = {"temp": float(temperature), "top_p": float(getattr(config, "top_p", 0.9) if config else 0.9), "top_k": int(getattr(config, "top_k", 0) if config else 0), "min_p": float(getattr(config, "min_p", 0.0) if config else 0.0)}
        return factory(**kwargs)

    def _stream_once(self, messages: Any, budget: int, temperature: float, cancel: Any = None, *, enable_thinking: bool = True) -> Iterator[GenerationEvent]:
        if cancel is not None and cancel.is_set():
            yield GenerationEvent(done=True, error="cancelled")
            return
        prompt = self._prompt(messages, enable_thinking=enable_thinking)
        started = time.perf_counter()
        pieces: list[str] = []
        prompt_tokens = 0
        generated_tokens = 0
        peak_memory_bytes = 0
        finish_reason = None
        # MLX-LM's stream_generate accepts a callable sampler.  Passing a
        # ``temperature`` keyword is a runtime error in current releases.
        iterator = self._stream_generate(self.model, self.tokenizer, prompt, max_tokens=int(budget), sampler=self._sampler(temperature))
        for response in iterator:
            if cancel is not None and cancel.is_set():
                break
            text = self._response_text(response)
            pieces.append(text)
            prompt_tokens = int(self._response_metric(response, "prompt_tokens", prompt_tokens) or prompt_tokens)
            generated_tokens = int(self._response_metric(response, "generation_tokens", generated_tokens) or generated_tokens)
            if not generated_tokens:
                generated_tokens = int(self._response_metric(response, "generated_tokens", generated_tokens) or generated_tokens)
            if not generated_tokens:
                generated_tokens = len(pieces)
            finish_reason = self._response_metric(response, "finish_reason", finish_reason)
            peak_memory_bytes = max(peak_memory_bytes, self._peak_memory())
            raw = "".join(pieces)
            yield GenerationEvent(text=text, raw_trace=raw, metrics={"prompt_tokens": prompt_tokens, "generated_tokens": generated_tokens, "peak_memory_bytes": peak_memory_bytes, "elapsed_seconds": time.perf_counter() - started}, finish_reason=finish_reason)
        raw = "".join(pieces)
        elapsed = max(time.perf_counter() - started, 1e-9)
        parsed_text = split_thinking(raw)["final_content"]
        try:
            structured = json.loads(parsed_text)
            if not isinstance(structured, Mapping):
                structured = None
        except (TypeError, ValueError):
            structured = None
        if not prompt_tokens:
            try:
                prompt_tokens = len(self.tokenizer.encode(prompt))
            except (AttributeError, TypeError, ValueError):
                prompt_tokens = 0
        yield GenerationEvent(text="", raw_trace=raw, structured=structured, done=True, finish_reason=finish_reason or "stop", metrics={"prompt_tokens": prompt_tokens, "generated_tokens": generated_tokens or len(pieces), "tokens_per_second": (generated_tokens or len(pieces)) / elapsed, "peak_memory_bytes": peak_memory_bytes, "elapsed_seconds": elapsed})

    def stream_structured(self, messages: Any, schema: Mapping[str, Any], budget: int, temperature: float, *, cancel: Any = None, enable_thinking: bool = True) -> Iterator[GenerationEvent]:
        """Stream one JSON generation, with one non-thinking repair on failure."""

        # MLX-LM does not provide constrained decoding here, so the schema must
        # be part of the initial prompt. Previously Qwen only saw it after a
        # failed generation, causing perspective-dependent ad-hoc key names.
        schema_messages = list(messages) + [{
            "role": "user",
            "content": (
                "Output contract: return exactly one JSON object matching this schema. "
                "Include every required key and no additional keys. Do not describe the schema.\n"
                + json.dumps(schema, ensure_ascii=False, sort_keys=True)
            ),
        }]
        initial: list[GenerationEvent] = []
        for event in self._stream_once(schema_messages, budget, temperature, cancel, enable_thinking=enable_thinking):
            initial.append(event)
            if not event.done:
                yield event
        final = initial[-1] if initial else GenerationEvent(done=True)
        initial_structured = _project_schema_object(final.structured, schema)
        if initial_structured is not None and _schema_valid(initial_structured, schema):
            yield GenerationEvent(text="", raw_trace=final.raw_trace, structured=initial_structured, done=True, repaired=final.repaired, finish_reason=final.finish_reason, metrics=final.metrics)
            return
        if cancel is not None and cancel.is_set():
            yield GenerationEvent(raw_trace=final.raw_trace, done=True, error="cancelled", metrics=final.metrics)
            return
        repair_messages = schema_messages + [{"role": "user", "content": _repair_prompt(final.raw_trace, schema, final.error)}]
        repaired: list[GenerationEvent] = []
        for event in self._stream_once(repair_messages, budget, 0.0, cancel, enable_thinking=False):
            repaired.append(event)
            if not event.done:
                yield event
        repair_final = repaired[-1] if repaired else GenerationEvent(done=True)
        combined_raw = final.raw_trace + ("\n[repair]\n" + repair_final.raw_trace if repair_final.raw_trace else "")
        repaired_structured = _project_schema_object(repair_final.structured, schema)
        if repaired_structured is not None and _schema_valid(repaired_structured, schema):
            yield GenerationEvent(text="", raw_trace=combined_raw, structured=repaired_structured, done=True, repaired=True, finish_reason=repair_final.finish_reason, metrics=repair_final.metrics)
        else:
            yield GenerationEvent(text="", raw_trace=combined_raw, done=True, error="schema_invalid", finish_reason=repair_final.finish_reason, metrics=repair_final.metrics, repaired=True)


__all__ = ["GenerationEvent", "MLXGenerator", "split_thinking"]
