"""Ollama HTTP generation adapter for the local demo (Windows, Linux, macOS).

This is a drop-in replacement for :class:`MLXGenerator`: it exposes the same
``stream_structured`` generator of :class:`GenerationEvent` objects, applies
the same output contract and one non-thinking repair, and never fabricates a
response.  Only the standard library is used; the model runs in a separately
installed Ollama server reached over HTTP.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Iterator, Mapping

from .config import DemoConfig
from .mlx_engine import GenerationEvent, _project_schema_object, _repair_prompt, _schema_valid, split_thinking


class OllamaError(RuntimeError):
    """Raised when the Ollama server cannot complete a request."""


def normalize_ollama_url(value: str | None) -> str:
    """Accept Ollama's ``OLLAMA_HOST`` forms (``host``, ``host:port``, URL)."""

    url = str(value or "").strip().rstrip("/")
    if not url:
        return "http://localhost:11434"
    if "://" not in url:
        url = "http://" + url
    scheme, rest = url.split("://", 1)
    host = rest.split("/", 1)[0]
    # ``ollama serve`` listens on all interfaces for 0.0.0.0; clients connect locally.
    if host.startswith("0.0.0.0"):
        rest = "localhost" + rest[len("0.0.0.0"):]
        host = rest.split("/", 1)[0]
    if ":" not in host.rsplit("]", 1)[-1]:
        rest = host + ":11434" + rest[len(host):]
    return f"{scheme}://{rest}"


class OllamaGenerator:
    """Stream structured JSON from one model served by Ollama's ``/api/chat``."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        *,
        config: DemoConfig | None = None,
        timeout: float = 600.0,
        num_ctx: int = 8192,
        keep_alive: str = "30m",
        opener: Callable[..., Any] | None = None,
    ) -> None:
        if not str(model or "").strip():
            raise ValueError("an Ollama model name is required")
        self.model = str(model).strip()
        self.base_url = normalize_ollama_url(base_url)
        self.config = config
        self.timeout = float(timeout)
        self.num_ctx = int(num_ctx)
        self.keep_alive = keep_alive
        self._open = opener or urllib.request.urlopen
        self._send_think = True

    @classmethod
    def load(cls, config: DemoConfig) -> "OllamaGenerator":
        """Connect to the configured server and require the model to be pulled."""

        generator = cls(
            config.ollama_model,
            config.ollama_url,
            config=config,
            timeout=config.ollama_timeout,
            num_ctx=config.ollama_num_ctx,
            keep_alive=config.ollama_keep_alive,
        )
        generator.check_ready()
        return generator

    # -- HTTP helpers -------------------------------------------------------

    def _request(self, path: str, payload: Mapping[str, Any] | None = None, *, timeout: float | None = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            return self._open(request, timeout=self.timeout if timeout is None else timeout)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = json.loads(exc.read().decode("utf-8", "replace")).get("error", "")
            except (ValueError, AttributeError, OSError):
                pass
            raise OllamaError(f"Ollama returned HTTP {exc.code} for {path}: {detail or exc.reason}") from exc
        except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            raise OllamaError(
                f"cannot reach Ollama at {self.base_url} ({reason}); start the Ollama app or run `ollama serve`"
            ) from exc

    def _get_json(self, path: str, *, timeout: float = 10.0) -> Any:
        with self._request(path, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def check_ready(self) -> dict[str, Any]:
        """Fail loudly unless the server answers and the model is installed."""

        version = self._get_json("/api/version").get("version")
        models = self._get_json("/api/tags").get("models", [])
        names = {str(item.get(key)) for item in models for key in ("name", "model") if item.get(key)}
        wanted = {self.model} if ":" in self.model else {self.model, self.model + ":latest"}
        if not wanted & names:
            available = ", ".join(sorted(names)) or "none"
            raise OllamaError(
                f"Ollama model '{self.model}' is not installed at {self.base_url} (available: {available}); "
                f"run `ollama pull {self.model}` or set MPKG_OLLAMA_MODEL"
            )
        return {"version": version, "model": self.model, "base_url": self.base_url}

    # -- generation ---------------------------------------------------------

    def _options(self, budget: int, temperature: float) -> dict[str, Any]:
        config = self.config
        options: dict[str, Any] = {
            "num_predict": int(budget),
            "num_ctx": self.num_ctx,
            "temperature": float(temperature),
            "top_p": float(getattr(config, "top_p", 0.9) if config else 0.9),
            "min_p": float(getattr(config, "min_p", 0.0) if config else 0.0),
        }
        top_k = int(getattr(config, "top_k", 0) if config else 0)
        if top_k > 0:
            options["top_k"] = top_k
        return options

    def _open_chat(self, payload: dict[str, Any]) -> Any:
        try:
            return self._request("/api/chat", payload)
        except OllamaError as exc:
            # Models without a thinking template reject the ``think`` field;
            # retry once without it and remember the capability.
            if "think" in payload and "think" in str(exc).casefold():
                self._send_think = False
                payload = {key: value for key, value in payload.items() if key != "think"}
                return self._request("/api/chat", payload)
            raise

    def _stream_once(self, messages: Any, schema: Mapping[str, Any] | None, budget: int, temperature: float, cancel: Any = None, *, enable_thinking: bool = True) -> Iterator[GenerationEvent]:
        if cancel is not None and cancel.is_set():
            yield GenerationEvent(done=True, error="cancelled")
            return
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": self._options(budget, temperature),
        }
        if schema is not None:
            # Ollama compiles the JSON schema into a decoding grammar, the
            # equivalent of constrained decoding on the CUDA target.
            payload["format"] = dict(schema)
        if self._send_think:
            payload["think"] = bool(enable_thinking)
        started = time.perf_counter()
        thinking: list[str] = []
        content: list[str] = []
        prompt_tokens = generated_tokens = 0
        eval_seconds = 0.0
        finish_reason: str | None = None

        def raw_trace() -> str:
            reasoning = "".join(thinking).strip()
            return (f"<think>\n{reasoning}\n</think>\n\n" if reasoning else "") + "".join(content)

        def metrics() -> dict[str, Any]:
            elapsed = max(time.perf_counter() - started, 1e-9)
            return {
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "tokens_per_second": generated_tokens / (eval_seconds or elapsed) if generated_tokens else 0.0,
                "peak_memory_bytes": 0,
                "elapsed_seconds": elapsed,
            }

        response = self._open_chat(payload)
        try:
            for line in response:
                if cancel is not None and cancel.is_set():
                    yield GenerationEvent(raw_trace=raw_trace(), done=True, error="cancelled", metrics=metrics())
                    return
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError as exc:
                    raise OllamaError(f"Ollama sent a malformed stream line: {line[:200]!r}") from exc
                if chunk.get("error"):
                    raise OllamaError(f"Ollama generation failed: {chunk['error']}")
                message = chunk.get("message") or {}
                if message.get("thinking"):
                    thinking.append(str(message["thinking"]))
                delta = str(message.get("content") or "")
                if chunk.get("done"):
                    if delta:
                        content.append(delta)
                    prompt_tokens = int(chunk.get("prompt_eval_count") or 0)
                    generated_tokens = int(chunk.get("eval_count") or 0)
                    eval_seconds = float(chunk.get("eval_duration") or 0) / 1e9
                    finish_reason = "length" if chunk.get("done_reason") == "length" else str(chunk.get("done_reason") or "stop")
                    break
                if delta:
                    content.append(delta)
                    generated_tokens += 1
                    yield GenerationEvent(text=delta, raw_trace=raw_trace(), metrics=metrics())
            else:
                raise OllamaError("Ollama closed the stream before the response finished")
        except (socket.timeout, TimeoutError) as exc:
            raise OllamaError(f"Ollama did not respond within {self.timeout:.0f}s; raise MPKG_OLLAMA_TIMEOUT for slow CPUs") from exc
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        raw = raw_trace()
        try:
            structured = json.loads(split_thinking(raw)["final_content"])
            if not isinstance(structured, Mapping):
                structured = None
        except (TypeError, ValueError):
            structured = None
        yield GenerationEvent(text="", raw_trace=raw, structured=structured, done=True, finish_reason=finish_reason or "stop", metrics=metrics())

    def stream_structured(self, messages: Any, schema: Mapping[str, Any], budget: int, temperature: float, *, cancel: Any = None, enable_thinking: bool = True) -> Iterator[GenerationEvent]:
        """Stream one JSON generation, with one non-thinking repair on failure."""

        # Same output contract as the MLX adapter, so prompts are identical
        # across backends; Ollama additionally enforces the schema grammar.
        schema_messages = list(messages) + [{
            "role": "user",
            "content": (
                "Output contract: return exactly one JSON object matching this schema. "
                "Include every required key and no additional keys. Do not describe the schema.\n"
                + json.dumps(schema, ensure_ascii=False, sort_keys=True)
            ),
        }]
        initial: list[GenerationEvent] = []
        for event in self._stream_once(schema_messages, schema, budget, temperature, cancel, enable_thinking=enable_thinking):
            initial.append(event)
            if not event.done:
                yield event
        final = initial[-1] if initial else GenerationEvent(done=True)
        if final.error == "cancelled":
            yield final
            return
        initial_structured = _project_schema_object(final.structured, schema)
        if initial_structured is not None and _schema_valid(initial_structured, schema):
            yield GenerationEvent(text="", raw_trace=final.raw_trace, structured=initial_structured, done=True, finish_reason=final.finish_reason, metrics=final.metrics)
            return
        if cancel is not None and cancel.is_set():
            yield GenerationEvent(raw_trace=final.raw_trace, done=True, error="cancelled", metrics=final.metrics)
            return
        error = "output truncated at the token budget" if final.finish_reason == "length" else final.error
        repair_messages = schema_messages + [{"role": "user", "content": _repair_prompt(final.raw_trace, schema, error)}]
        repaired: list[GenerationEvent] = []
        for event in self._stream_once(repair_messages, schema, budget, 0.0, cancel, enable_thinking=False):
            repaired.append(event)
            if not event.done:
                yield event
        repair_final = repaired[-1] if repaired else GenerationEvent(done=True)
        combined_raw = final.raw_trace + ("\n[repair]\n" + repair_final.raw_trace if repair_final.raw_trace else "")
        repaired_structured = _project_schema_object(repair_final.structured, schema)
        if repaired_structured is not None and _schema_valid(repaired_structured, schema):
            yield GenerationEvent(text="", raw_trace=combined_raw, structured=repaired_structured, done=True, repaired=True, finish_reason=repair_final.finish_reason, metrics=repair_final.metrics)
        else:
            yield GenerationEvent(text="", raw_trace=combined_raw, done=True, error=repair_final.error or "schema_invalid", finish_reason=repair_final.finish_reason, metrics=repair_final.metrics, repaired=True)


__all__ = ["OllamaError", "OllamaGenerator", "normalize_ollama_url"]
