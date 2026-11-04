"""End-to-end pipeline run through the real Ollama HTTP client (fake server)."""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from local_mpkg_demo.config import DemoConfig, EvidenceRecord, PERSPECTIVES, RetrievalResult
from local_mpkg_demo.history import HistoryStore
from local_mpkg_demo.pipeline import MPKGRAGPipeline, RunRequest, _validate_final_citations


def answer_for(request: dict) -> dict:
    """Return a schema-shaped answer the way a compliant model would."""

    required = set(request["format"]["required"])
    prompt = "\n".join(message["content"] for message in request["messages"])
    if "counter_narrative" in required:
        return {"counter_narrative": "Being gay is not an illness [E1].", "cited_evidence_ids": ["E1"], "factual_claims": ["Being gay is not an illness."], "safety_notes": []}
    if "claim_focus" in required:
        return {"claim_focus": "illness myth", "selected_evidence_ids": ["E1"], "response_steps": ["correct"], "tone": "calm", "factual_constraints": [], "safety_constraints": []}
    if "perspective" in required:
        name = re.search(r"Perspective: (\w+)", prompt).group(1)
        return {"perspective": name, "rationale": "grounded", "claims_to_address": ["illness"], "supported_evidence_ids": ["E1"], "response_guidance": ["be kind"], "risk_flags": [], "confidence": 0.8}
    return {"target": "Gay Men", "category": "Homophobic"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:
        body = json.dumps({"version": "test"} if self.path == "/api/version" else {"models": [{"name": "qwen3:4b"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(request)
        text = json.dumps(answer_for(request))
        lines = [{"message": {"content": text}, "done": False}, {"message": {"content": ""}, "done": True, "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 5, "eval_duration": 1_000_000_000}]
        body = b"".join(json.dumps(line).encode() + b"\n" for line in lines)
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class StubRetriever:
    def retrieve(self, query, limit=5):
        record = EvidenceRecord(evidence_id="E1", text="WHO removed homosexuality from its list of mental disorders in 1990.", source_id="SRC001", page=3)
        return RetrievalResult(query=query, evidence=(record,))


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.requests = []
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def test_full_mp_kg_rag_run_reaches_validated_final_answer(tmp_path: Path, server):
    url = f"http://127.0.0.1:{server.server_address[1]}"
    config = DemoConfig.load(tmp_path, {"MPKG_GENERATOR": "ollama", "MPKG_OLLAMA_URL": url})
    history = HistoryStore(tmp_path / "history.sqlite3", safe_export_dir=tmp_path / "exports")
    pipeline = MPKGRAGPipeline(config, retriever=StubRetriever(), history=history)
    events = list(pipeline.run(RunRequest(text="gay log bimar hote hain, they need treatment")))
    assert not [event for event in events if event.status == "failed"], [event.error for event in events if event.status == "failed"]
    completed = [event.stage for event in events if event.status == "completed"]
    assert completed == ["metadata", "retrieval", *PERSPECTIVES, "plan", "final", "validation"]
    final = next(event for event in events if event.stage == "final" and event.status == "completed")
    assert _validate_final_citations(final.payload, {"E1"}) == (True, "")
    assert events[-1].payload == {"valid": True, "cited_evidence_ids": ["E1"], "unknown_evidence_ids": []}
    # metadata + five perspectives + plan + final, each with the stage schema as grammar
    assert len(server.requests) == 8 and all("format" in request for request in server.requests)
    conversation = history.list_conversations("")[0]
    assert history.get_conversation(conversation.id).run_status == "completed"


def test_run_fails_visibly_when_ollama_is_down(tmp_path: Path):
    config = DemoConfig.load(tmp_path, {"MPKG_GENERATOR": "ollama", "MPKG_OLLAMA_URL": "http://127.0.0.1:9"})
    history = HistoryStore(tmp_path / "history.sqlite3", safe_export_dir=tmp_path / "exports")
    pipeline = MPKGRAGPipeline(config, retriever=StubRetriever(), history=history)
    events = list(pipeline.run(RunRequest(text="gay people are sick", target="Gay Men", category="Homophobic")))
    failed = [event for event in events if event.status == "failed"]
    assert failed and "cannot reach Ollama" in str(failed[-1].error)
    assert not any(event.stage == "final" and event.status == "completed" for event in events)
