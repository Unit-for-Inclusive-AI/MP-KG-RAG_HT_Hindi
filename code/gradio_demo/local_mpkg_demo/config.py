"""Configuration and small, serialisable contracts for the local demo."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

QWEN_MODEL = "mlx-community/Qwen3.5-4B-4bit"
QWEN_REVISION = "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"
# Ollama is the default generator everywhere except Apple silicon, where the
# original pinned MLX model remains available.  Override with environment
# variables (see README-windows.md) instead of editing code.
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:4b"
GENERATOR_BACKENDS = ("ollama", "mlx")
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
PERSPECTIVES = ("fact_checking", "cultural_context", "harm_reduction", "legal_rights", "persuasion")


def _default_generator_backend() -> str:
    apple_silicon = platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}
    return "mlx" if apple_silicon else "ollama"


def generator_settings(environ: Mapping[str, str]) -> dict[str, Any]:
    """Read generator/device overrides from environment variables.

    ``MPKG_OLLAMA_URL`` wins over Ollama's own ``OLLAMA_HOST`` so a client can
    target a server without changing how that server listens.
    """

    from .ollama_engine import normalize_ollama_url

    backend = str(environ.get("MPKG_GENERATOR") or _default_generator_backend()).strip().casefold()
    if backend not in GENERATOR_BACKENDS:
        raise ValueError(f"MPKG_GENERATOR must be one of {', '.join(GENERATOR_BACKENDS)}; got {backend!r}")
    device = str(environ.get("MPKG_RETRIEVAL_DEVICE") or "auto").strip().casefold()
    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"MPKG_RETRIEVAL_DEVICE must be auto, cpu, cuda or mps; got {device!r}")

    def number(name: str, default: float, cast: Any) -> Any:
        raw = environ.get(name)
        if raw is None or not str(raw).strip():
            return cast(default)
        try:
            value = cast(str(raw).strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be a number; got {raw!r}") from exc
        if value <= 0:
            raise ValueError(f"{name} must be positive; got {raw!r}")
        return value

    return {
        "generator_backend": backend,
        "ollama_url": normalize_ollama_url(environ.get("MPKG_OLLAMA_URL") or environ.get("OLLAMA_HOST") or OLLAMA_URL),
        "ollama_model": str(environ.get("MPKG_OLLAMA_MODEL") or OLLAMA_MODEL).strip(),
        "ollama_num_ctx": number("MPKG_OLLAMA_NUM_CTX", 8192, int),
        "ollama_timeout": number("MPKG_OLLAMA_TIMEOUT", 600.0, float),
        "ollama_keep_alive": str(environ.get("MPKG_OLLAMA_KEEP_ALIVE") or "30m").strip(),
        "retrieval_device": device,
    }


@dataclass(frozen=True)
class DemoConfig:
    project_root: Path
    semantic_graph_source: Path
    chunks_path: Path
    evidence_ledger_path: Path
    semantic_extractions_path: Path
    candidates_path: Path
    source_manifest_path: Path
    index_path: Path
    history_path: Path
    safe_export_dir: Path | None = None
    # Model loading controls.  ``offline`` is deliberately explicit so an
    # offline launch can never silently turn into a Hub request.
    offline: bool = False
    model_cache_dir: Path | None = None
    embedding_backend: Any | None = None
    reranker_backend: Any | None = None
    qwen_model: str = QWEN_MODEL
    qwen_revision: str = QWEN_REVISION
    generator_backend: str = "ollama"
    ollama_url: str = OLLAMA_URL
    ollama_model: str = OLLAMA_MODEL
    ollama_num_ctx: int = 8192
    ollama_timeout: float = 600.0
    ollama_keep_alive: str = "30m"
    retrieval_device: str = "auto"
    embedding_model: str = EMBEDDING_MODEL
    embedding_revision: str = EMBEDDING_REVISION
    reranker_model: str = RERANKER_MODEL
    reranker_revision: str = RERANKER_REVISION
    embedding_dimension: int = 1024
    retrieval_max_hops: int = 2
    minimum_dense_score: float = 0.25
    minimum_graph_score: float = 0.0
    minimum_authority: float = 0.60
    minimum_rerank_probability: float = 0.55
    rrf_weights: tuple[tuple[str, float], ...] = (("dense", 1.1), ("bm25", 1.0), ("graph", 0.8))
    rrf_constant: float = 60.0
    dense_top_k: int = 12
    bm25_top_k: int = 12
    graph_top_k: int = 8
    rerank_top_k: int = 5
    max_evidence: int = 5
    perspectives: tuple[str, ...] = PERSPECTIVES
    top_p: float = 0.9
    top_k: int = 0
    min_p: float = 0.0

    @classmethod
    def load(cls, project_root: Path, environ: Mapping[str, str] | None = None) -> "DemoConfig":
        env = os.environ if environ is None else environ
        root = Path(project_root).expanduser().resolve()
        source = root / "semantic_graph_source"
        # The checked-out demo worktree may be alongside the main checkout,
        # while the completed (large, ignored) semantic artifacts live in the
        # main checkout.  Prefer a local source directory, then reuse that
        # sibling artifact set without copying hundreds of megabytes.
        if not source.exists():
            sibling_source = root.parent.parent / "semantic_graph_source"
            if sibling_source.exists():
                source = sibling_source
        return cls(
            project_root=root,
            semantic_graph_source=source,
            chunks_path=source / "chunks.json",
            evidence_ledger_path=source / "evidence_ledger.json",
            semantic_extractions_path=source / "semantic_extractions.json",
            candidates_path=source / "candidates.json",
            source_manifest_path=source / "manifest.json",
            index_path=root / "local_mpkg_demo_index",
            history_path=root / "local_mpkg_demo_history.sqlite3",
            safe_export_dir=root / "exports",
            **generator_settings(env),
        )

    @property
    def generator_label(self) -> str:
        """Human-readable identity of the configured generation model."""

        if self.generator_backend == "mlx":
            return f"{self.qwen_model}@{self.qwen_revision}"
        return f"ollama:{self.ollama_model}"

    @property
    def retrieval_config(self) -> dict[str, Any]:
        return {
            "max_hops": self.retrieval_max_hops,
            "minimum_dense_score": self.minimum_dense_score,
            "minimum_graph_score": self.minimum_graph_score,
            "hop_decay": 0.75,
            "minimum_rerank_probability": self.minimum_rerank_probability,
            "max_evidence": self.max_evidence,
            "dense_top_k": self.dense_top_k,
            "bm25_top_k": self.bm25_top_k,
            "graph_top_k": self.graph_top_k,
            "rerank_top_k": self.rerank_top_k,
            "minimum_authority": self.minimum_authority,
            "weights": {
                "query_entity": 2.0, "predicate": 1.0, "polarity": 0.5,
                "modality": 0.4, "stance": 0.8, "review_state": 0.8,
                "authority": 0.8, "extraction_confidence": 0.0,
                "seed_score": 1.0, "hop_decay": 0.6,
            },
            "review_state_scores": {"accepted": 1.0, "reviewed": 0.85, "unknown": 0.0},
            "rrf_weights": {key: value for key, value in self.rrf_weights},
        }


@dataclass(frozen=True, init=False)
class QueryInput:
    """Normalized user query. ``post`` is accepted as a compatibility alias for text."""

    text: str
    target: str
    category: str
    query_signature: Mapping[str, Any]

    def __init__(self, text: str = "", target: str = "", category: str = "", query_signature: Mapping[str, Any] | None = None, *, post: str | None = None):
        object.__setattr__(self, "text", str(post if post is not None and not text else text))
        object.__setattr__(self, "target", str(target or ""))
        object.__setattr__(self, "category", str(category or ""))
        object.__setattr__(self, "query_signature", dict(query_signature or {}))

    @property
    def post(self) -> str:
        return self.text


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    text: str
    displayed_text: str = ""
    document_uid: str = ""
    chunk_id: str = ""
    source_id: str = ""
    source_type: str = ""
    locator: str = ""
    page: int | None = None
    span_start: int | None = None
    span_end: int | None = None
    integrity_hash: str = ""
    routes: tuple[str, ...] = ()
    scores: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], **updates: Any) -> "EvidenceRecord":
        data = dict(row)
        data.update(updates)
        text = str(data.get("text") or data.get("displayed_text") or "")
        return cls(
            evidence_id=str(data.get("evidence_id") or data.get("evidence_chunk_id") or data.get("id") or ""),
            text=text,
            displayed_text=str(data.get("displayed_text") or text),
            document_uid=str(data.get("document_uid") or ""),
            chunk_id=str(data.get("chunk_id") or ""),
            source_id=str(data.get("source_id") or ""),
            source_type=str(data.get("source_type") or ""),
            locator=str(data.get("locator") or data.get("source_locator") or ""),
            page=data.get("page") if isinstance(data.get("page"), int) else data.get("page_number"),
            span_start=data.get("span_start") if isinstance(data.get("span_start"), int) else data.get("start_char"),
            span_end=data.get("span_end") if isinstance(data.get("span_end"), int) else data.get("end_char"),
            integrity_hash=str(data.get("integrity_hash") or data.get("displayed_text_sha256") or data.get("source_text_sha256") or data.get("text_sha256") or data.get("chunk_sha256") or data.get("sha256") or ""),
            routes=tuple(str(item) for item in data.get("routes", ()) if item),
            scores={str(k): float(v) for k, v in (data.get("scores") or {}).items() if isinstance(v, (int, float))},
            metadata=data,
        )


@dataclass(frozen=True)
class RetrievalResult:
    query: QueryInput
    evidence: tuple[EvidenceRecord, ...]
    candidates: tuple[Any, ...] = ()
    trace: Mapping[str, Any] = field(default_factory=dict)
    abstained: bool = False
    reason: str | None = None

    @property
    def selected(self) -> tuple[EvidenceRecord, ...]:
        return self.evidence

    @property
    def selected_ledger(self) -> tuple[EvidenceRecord, ...]:
        return self.evidence

    @property
    def ledger(self) -> tuple[EvidenceRecord, ...]:
        return self.evidence


@dataclass(frozen=True)
class IndexManifest:
    identity: Mapping[str, Any]
    source_hashes: Mapping[str, str]
    model_revisions: Mapping[str, str]
    dimensions: Mapping[str, int]
    row_counts: Mapping[str, int]
    index_path: Path
    created_at: str = ""
    readiness_warnings: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class Conversation:
    id: str
    title: str
    created_at: str
    updated_at: str
    status: str = "new"
    input_text: str = ""
    target: str = ""
    category: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    stages: tuple[Any, ...] = ()
    run_status: str | None = None
    run_payload: Mapping[str, Any] = field(default_factory=dict)
    run_error: str | None = None
    run_started_at: str | None = None
    run_ended_at: str | None = None


@dataclass(frozen=True)
class StageRecord:
    id: str
    run_id: str
    stage_name: str
    payload: Mapping[str, Any]
    raw_trace: str
    created_at: str
    sequence: int = 0
    status: str = "completed"
    metrics: Mapping[str, Any] = field(default_factory=dict)
    partial_text: str = ""
    error: str | None = None
