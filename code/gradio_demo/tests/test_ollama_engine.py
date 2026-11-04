"""Ollama adapter tests: no model or running server required."""

from __future__ import annotations

import io
import json
import socket
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from local_mpkg_demo.config import DemoConfig
from local_mpkg_demo.ollama_engine import OllamaError, OllamaGenerator, normalize_ollama_url
from local_mpkg_demo.pipeline import FINAL_SCHEMA, load_generator

VALID_FINAL = {"counter_narrative": "Respect everyone [E1].", "cited_evidence_ids": ["E1"], "factual_claims": [], "safety_notes": []}


def stream_lines(text: str, *, thinking: str = "", done_reason: str = "stop") -> list[bytes]:
    lines = []
    if thinking:
        lines.append({"message": {"role": "assistant", "content": "", "thinking": thinking}, "done": False})
    for index in range(0, len(text), 7):
        lines.append({"message": {"role": "assistant", "content": text[index:index + 7]}, "done": False})
    lines.append({"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": done_reason, "prompt_eval_count": 42, "eval_count": 9, "eval_duration": 3_000_000_000})
    return [json.dumps(line).encode() + b"\n" for line in lines]


class FakeResponse:
    def __init__(self, lines: list[bytes] | None = None, body: dict | None = None, raise_at: int | None = None) -> None:
        self._lines = lines or []
        self._body = body
        self._raise_at = raise_at
        self.closed = False

    def __iter__(self):
        for index, line in enumerate(self._lines):
            if self._raise_at is not None and index == self._raise_at:
                raise socket.timeout("timed out")
            yield line

    def read(self) -> bytes:
        return json.dumps(self._body).encode()

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class FakeOpener:
    """Records requests and replays queued responses or exceptions."""

    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, request, timeout=None):
        body = json.loads(request.data.decode()) if request.data else None
        self.requests.append({"url": request.full_url, "body": body, "timeout": timeout})
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def collect(iterator):
    events = list(iterator)
    return events, events[-1]


@pytest.mark.parametrize(("value", "expected"), [
    (None, "http://localhost:11434"),
    ("", "http://localhost:11434"),
    ("localhost", "http://localhost:11434"),
    ("127.0.0.1:11500", "http://127.0.0.1:11500"),
    ("0.0.0.0", "http://localhost:11434"),
    ("0.0.0.0:11434", "http://localhost:11434"),
    ("http://gpu-box:11434/", "http://gpu-box:11434"),
    ("https://ollama.example", "https://ollama.example:11434"),
])
def test_normalize_ollama_url(value, expected):
    assert normalize_ollama_url(value) == expected


def test_stream_structured_parses_schema_json_and_sends_grammar():
    opener = FakeOpener(FakeResponse(stream_lines(json.dumps(VALID_FINAL))))
    generator = OllamaGenerator("qwen3:4b", opener=opener, num_ctx=4096)
    events, final = collect(generator.stream_structured([{"role": "user", "content": "hi"}], FINAL_SCHEMA, 128, 0.6, enable_thinking=False))
    assert final.done and final.error is None and final.repaired is False
    assert final.structured == VALID_FINAL
    assert "".join(event.text for event in events[:-1]) == json.dumps(VALID_FINAL)
    assert final.metrics["prompt_tokens"] == 42 and final.metrics["generated_tokens"] == 9
    assert final.metrics["tokens_per_second"] == pytest.approx(3.0)
    request = opener.requests[0]
    assert request["url"] == "http://localhost:11434/api/chat"
    assert request["body"]["format"] == FINAL_SCHEMA
    assert request["body"]["think"] is False and request["body"]["stream"] is True
    assert request["body"]["options"]["num_ctx"] == 4096 and request["body"]["options"]["num_predict"] == 128
    assert "Output contract" in request["body"]["messages"][-1]["content"]


def test_thinking_is_kept_in_raw_trace_but_not_parsed():
    opener = FakeOpener(FakeResponse(stream_lines(json.dumps(VALID_FINAL), thinking="weigh the evidence")))
    _, final = collect(OllamaGenerator("qwen3:4b", opener=opener).stream_structured([], FINAL_SCHEMA, 64, 0.6))
    assert final.structured == VALID_FINAL
    assert final.raw_trace.startswith("<think>\nweigh the evidence\n</think>")


def test_invalid_output_gets_exactly_one_repair():
    opener = FakeOpener(FakeResponse(stream_lines("not json")), FakeResponse(stream_lines(json.dumps(VALID_FINAL))))
    _, final = collect(OllamaGenerator("qwen3:4b", opener=opener).stream_structured([], FINAL_SCHEMA, 64, 0.6))
    assert final.repaired is True and final.structured == VALID_FINAL
    assert "[repair]" in final.raw_trace and "not json" in final.raw_trace
    repair = opener.requests[1]["body"]
    assert repair["options"]["temperature"] == 0.0 and repair["think"] is False
    assert "<previous_raw_output>" in repair["messages"][-1]["content"]


def test_failed_repair_reports_schema_invalid_without_fake_output():
    opener = FakeOpener(FakeResponse(stream_lines("{}")), FakeResponse(stream_lines('{"counter_narrative": 3}')))
    _, final = collect(OllamaGenerator("qwen3:4b", opener=opener).stream_structured([], FINAL_SCHEMA, 64, 0.6))
    assert final.structured is None and final.error == "schema_invalid" and final.repaired is True
    assert len(opener.requests) == 2


def test_truncated_output_is_named_in_repair_prompt():
    opener = FakeOpener(FakeResponse(stream_lines('{"counter_narr', done_reason="length")), FakeResponse(stream_lines(json.dumps(VALID_FINAL))))
    _, final = collect(OllamaGenerator("qwen3:4b", opener=opener).stream_structured([], FINAL_SCHEMA, 8, 0.6))
    assert final.structured == VALID_FINAL
    assert "truncated at the token budget" in opener.requests[1]["body"]["messages"][-1]["content"]


def test_connection_refused_raises_clear_error():
    opener = FakeOpener(urllib.error.URLError(ConnectionRefusedError(10061, "refused")))
    with pytest.raises(OllamaError, match=r"cannot reach Ollama at http://localhost:11434.*ollama serve"):
        list(OllamaGenerator("qwen3:4b", opener=opener).stream_structured([], FINAL_SCHEMA, 64, 0.6))


def test_read_timeout_raises_and_closes_stream():
    response = FakeResponse(stream_lines(json.dumps(VALID_FINAL)), raise_at=1)
    opener = FakeOpener(response)
    with pytest.raises(OllamaError, match="MPKG_OLLAMA_TIMEOUT"):
        list(OllamaGenerator("qwen3:4b", opener=opener, timeout=5).stream_structured([], FINAL_SCHEMA, 64, 0.6))
    assert response.closed


def test_stream_error_line_is_surfaced():
    opener = FakeOpener(FakeResponse([b'{"error": "model requires more system memory"}\n']))
    with pytest.raises(OllamaError, match="more system memory"):
        list(OllamaGenerator("qwen3:4b", opener=opener).stream_structured([], FINAL_SCHEMA, 64, 0.6))


def test_stream_ending_without_done_is_an_error():
    opener = FakeOpener(FakeResponse(stream_lines(json.dumps(VALID_FINAL))[:-1]))
    with pytest.raises(OllamaError, match="closed the stream"):
        list(OllamaGenerator("qwen3:4b", opener=opener).stream_structured([], FINAL_SCHEMA, 64, 0.6))


def test_models_without_thinking_retry_once_without_think_field():
    rejected = urllib.error.HTTPError("http://localhost:11434/api/chat", 400, "Bad Request", {}, io.BytesIO(b'{"error": "\\"llama3\\" does not support thinking"}'))
    opener = FakeOpener(rejected, FakeResponse(stream_lines(json.dumps(VALID_FINAL))))
    generator = OllamaGenerator("llama3", opener=opener)
    _, final = collect(generator.stream_structured([], FINAL_SCHEMA, 64, 0.6, enable_thinking=True))
    assert final.structured == VALID_FINAL
    assert "think" in opener.requests[0]["body"] and "think" not in opener.requests[1]["body"]
    assert generator._send_think is False


def test_cancel_stops_without_repair():
    cancel = threading.Event()
    opener = FakeOpener(FakeResponse(stream_lines(json.dumps(VALID_FINAL))))
    events = []
    for event in OllamaGenerator("qwen3:4b", opener=opener).stream_structured([], FINAL_SCHEMA, 64, 0.6, cancel=cancel):
        events.append(event)
        cancel.set()
    assert events[-1].error == "cancelled" and len(opener.requests) == 1


def test_check_ready_requires_pulled_model():
    opener = FakeOpener(FakeResponse(body={"version": "0.40.2"}), FakeResponse(body={"models": [{"name": "llama3:8b", "model": "llama3:8b"}]}))
    with pytest.raises(OllamaError, match=r"ollama pull qwen3:4b"):
        OllamaGenerator("qwen3:4b", opener=opener).check_ready()


def test_untagged_model_matches_latest_tag():
    opener = FakeOpener(FakeResponse(body={"version": "0.40.2"}), FakeResponse(body={"models": [{"name": "mymodel:latest"}]}))
    assert OllamaGenerator("mymodel", opener=opener).check_ready()["model"] == "mymodel"


class _OllamaHandler(BaseHTTPRequestHandler):
    """A minimal Ollama look-alike that exercises real urllib streaming."""

    def log_message(self, *args) -> None:  # keep pytest output clean
        pass

    def _json(self, value) -> None:
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/version":
            self._json({"version": "test"})
        elif self.path == "/api/tags":
            self._json({"models": [{"name": "qwen3:4b", "model": "qwen3:4b"}]})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.chat_requests.append(request)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for line in stream_lines(json.dumps(VALID_FINAL)):
            self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")


@pytest.fixture()
def ollama_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OllamaHandler)
    server.chat_requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def test_load_generator_talks_to_a_real_http_server(tmp_path: Path, ollama_server):
    url = f"http://127.0.0.1:{ollama_server.server_address[1]}"
    config = DemoConfig.load(tmp_path, {"MPKG_GENERATOR": "ollama", "MPKG_OLLAMA_URL": url, "MPKG_OLLAMA_MODEL": "qwen3:4b"})
    generator = load_generator(config)
    assert isinstance(generator, OllamaGenerator) and generator.base_url == url
    events, final = collect(generator.stream_structured([{"role": "user", "content": "x"}], FINAL_SCHEMA, 64, 0.6, enable_thinking=False))
    assert final.structured == VALID_FINAL and len(events) > 2
    assert ollama_server.chat_requests[0]["model"] == "qwen3:4b"


def test_load_generator_fails_closed_when_server_is_down(tmp_path: Path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = DemoConfig.load(tmp_path, {"MPKG_GENERATOR": "ollama", "MPKG_OLLAMA_URL": f"http://127.0.0.1:{port}"})
    with pytest.raises(OllamaError, match="cannot reach Ollama"):
        load_generator(config)
