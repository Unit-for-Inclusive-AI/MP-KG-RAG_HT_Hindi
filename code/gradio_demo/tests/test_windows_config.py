"""Windows-safe configuration, device selection, and on-disk index behaviour."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from local_mpkg_demo import backends, index_store
from local_mpkg_demo.config import OLLAMA_MODEL, DemoConfig, QueryInput, generator_settings
from local_mpkg_demo.history import HistoryStore

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "semantic_graph_source"


def test_defaults_select_ollama_off_apple_silicon(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Windows")
    settings = generator_settings({})
    assert settings["generator_backend"] == "ollama"
    assert settings["ollama_url"] == "http://localhost:11434"
    assert settings["ollama_model"] == OLLAMA_MODEL
    assert settings["retrieval_device"] == "auto"
    assert settings["ollama_num_ctx"] == 8192 and settings["ollama_timeout"] == 600.0


def test_environment_overrides_and_precedence():
    settings = generator_settings({
        "MPKG_GENERATOR": "OLLAMA", "OLLAMA_HOST": "0.0.0.0", "MPKG_OLLAMA_MODEL": "qwen3:8b",
        "MPKG_OLLAMA_NUM_CTX": "4096", "MPKG_OLLAMA_TIMEOUT": "90", "MPKG_RETRIEVAL_DEVICE": "CPU",
    })
    assert settings["ollama_url"] == "http://localhost:11434"
    assert settings["ollama_model"] == "qwen3:8b" and settings["ollama_num_ctx"] == 4096
    assert settings["ollama_timeout"] == 90.0 and settings["retrieval_device"] == "cpu"
    assert generator_settings({"OLLAMA_HOST": "a:1", "MPKG_OLLAMA_URL": "http://b:2"})["ollama_url"] == "http://b:2"


@pytest.mark.parametrize("environ", [
    {"MPKG_GENERATOR": "vllm"},
    {"MPKG_RETRIEVAL_DEVICE": "tpu"},
    {"MPKG_OLLAMA_NUM_CTX": "lots"},
    {"MPKG_OLLAMA_TIMEOUT": "-1"},
])
def test_invalid_settings_fail_loudly(environ):
    with pytest.raises(ValueError):
        generator_settings(environ)


def test_config_paths_are_native_and_rooted(tmp_path: Path):
    config = DemoConfig.load(tmp_path, {"MPKG_GENERATOR": "ollama", "MPKG_OLLAMA_MODEL": "qwen3:4b"})
    for path in (config.index_path, config.history_path, config.safe_export_dir, config.chunks_path):
        assert isinstance(path, Path) and path.is_absolute()
    assert config.index_path.parent == tmp_path.resolve()
    assert config.generator_label == "ollama:qwen3:4b"


def _fake_torch(cuda: bool, mps: bool = False):
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
    )


def test_retrieval_device_defaults_to_cpu_and_cuda_is_opt_in(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True))
    assert backends._device("auto") == "cpu"
    assert backends._device("cpu") == "cpu"
    assert backends._device("cuda") == "cuda"
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    with pytest.raises(RuntimeError, match="no CUDA device"):
        backends._device("cuda")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, mps=True))
    assert backends._device("auto") == "mps"


def test_directory_fsync_is_skipped_on_windows(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(index_store.os, "name", "nt")
    monkeypatch.setattr(index_store.os, "open", lambda *a, **k: pytest.fail("directory opened on Windows"))
    index_store._fsync_directory(tmp_path)


class HashEncoder:
    """Deterministic stand-in for BGE-M3 with the production dimension."""

    name = "test-hash-encoder"

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=32):
        single = isinstance(texts, str)
        rows = []
        for text in [texts] if single else texts:
            vector = np.zeros(1024, dtype=np.float32)
            for token in str(text).casefold().split():
                vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % 1024] += 1.0
            norm = np.linalg.norm(vector)
            rows.append(vector / norm if norm else vector)
        return np.asarray(rows[0] if single else rows)


class OverlapReranker:
    name = "test-overlap-reranker"

    def predict(self, pairs, batch_size=16):
        return [len(set(q.casefold().split()) & set(t.casefold().split())) / 5.0 - 0.5 for q, t in pairs]


@pytest.fixture(scope="module")
def corpus_slice(tmp_path_factory) -> Path:
    if not (SOURCE / "chunks.json").exists():
        pytest.skip("semantic_graph_source is not present")
    out = tmp_path_factory.mktemp("slice")
    chunks = json.loads((SOURCE / "chunks.json").read_text(encoding="utf-8"))[:30]
    ids = {row["chunk_id"] for row in chunks}
    ledger = json.loads((SOURCE / "evidence_ledger.json").read_text(encoding="utf-8"))
    ledger["records"] = [row for row in ledger["records"] if row["chunk_id"] in ids]
    extractions = [row for row in json.loads((SOURCE / "semantic_extractions.json").read_text(encoding="utf-8")) if row["chunk_id"] in ids]
    candidates = [row for row in json.loads((SOURCE / "candidates.json").read_text(encoding="utf-8")) if row.get("chunk_id") in ids]
    for name, value in (("chunks", chunks), ("evidence_ledger", ledger), ("semantic_extractions", extractions), ("candidates", candidates)):
        (out / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    (out / "manifest.json").write_bytes((SOURCE / "manifest.json").read_bytes())
    return out


def _slice_config(tmp_path: Path, source: Path) -> DemoConfig:
    base = DemoConfig.load(tmp_path, {"MPKG_GENERATOR": "ollama"})
    return replace(
        base, semantic_graph_source=source, chunks_path=source / "chunks.json",
        evidence_ledger_path=source / "evidence_ledger.json", semantic_extractions_path=source / "semantic_extractions.json",
        candidates_path=source / "candidates.json", source_manifest_path=source / "manifest.json",
    )


def test_index_build_and_reuse_on_this_filesystem(tmp_path: Path, corpus_slice: Path):
    config = _slice_config(tmp_path, corpus_slice)
    encoder, reranker = HashEncoder(), OverlapReranker()
    manifest = index_store.build_or_validate_index(config, dense_encoder=encoder, reranker=reranker)
    current = (config.index_path / "CURRENT").read_text(encoding="utf-8")
    generation = config.index_path / "generations" / current
    assert manifest.row_counts["evidence"] == 30 and generation.is_dir()
    assert manifest.model_revisions["generator"] == "ollama:qwen3:4b"
    # No half-built temporary generation may be left behind.
    assert [path.name for path in (config.index_path / "generations").iterdir()] == [current]
    # A second call validates and reuses the same immutable generation.
    again = index_store.build_or_validate_index(config, dense_encoder=encoder, reranker=reranker)
    assert again.identity["generation_id"] == current
    assert index_store.resolve_current_generation(config) == generation
    retriever = index_store.LocalRetriever(config, dense_encoder=encoder, reranker=reranker)
    result = retriever.retrieve(QueryInput(text="hate speech detection", target="LGBTQ+ Community", category="Homophobic"))
    assert all(record.evidence_id for record in result.evidence)


def test_history_export_publishes_atomically(tmp_path: Path):
    store = HistoryStore(tmp_path / "history.sqlite3", safe_export_dir=tmp_path / "exports")
    conversation = store.create_conversation("Windows export", input_text="hello")
    destination = tmp_path / "exports" / "conversation.md"
    store.export_conversation(conversation.id, "markdown", destination=destination)
    assert "Windows export" in destination.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        store.export_conversation(conversation.id, "markdown", destination=destination)
