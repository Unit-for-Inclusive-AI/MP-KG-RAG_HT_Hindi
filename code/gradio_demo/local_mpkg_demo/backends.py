"""Lazy local retrieval backends for the demo (CPU by default; MPS or CUDA opt-in).

The optional SentenceTransformers dependency is imported only when a real
production index or query is requested.  This keeps fixture tests and UI
construction usable on hosts without Metal or the ML packages installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _offline(config: Any) -> bool:
    return bool(getattr(config, "offline", False)) or os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def _device(requested: str = "auto") -> str:
    """Resolve the retrieval device: CPU unless an accelerator is requested.

    ``auto`` keeps the original Apple-silicon behaviour (MPS when present) and
    otherwise uses CPU, so CUDA is never required.  ``cuda`` opts in to an
    NVIDIA GPU and fails clearly when PyTorch cannot see one.
    """

    requested = str(requested or "auto").casefold()
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if requested in {"cuda", "mps"}:
            raise RuntimeError(f"retrieval device '{requested}' requested but PyTorch is not installed")
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("MPKG_RETRIEVAL_DEVICE=cuda but PyTorch has no CUDA device; install a CUDA torch build or use cpu")
        return "cuda"
    try:
        mps = getattr(torch, "backends", None) is not None and torch.backends.mps.is_available()
    except (AttributeError, RuntimeError):
        mps = False
    if requested == "mps" and not mps:
        raise RuntimeError("MPKG_RETRIEVAL_DEVICE=mps but PyTorch has no MPS device")
    return "mps" if mps else "cpu"


def load_retrieval_backends(config: Any) -> tuple[Any | None, Any | None]:
    """Load the pinned BGE-M3 encoder and reranker on first real use.

    Explicitly supplied backends always win; this is the seam used by the
    small deterministic fixture tests.  The model revision and offline mode
    are passed through to Hugging Face so a demo index records reproducible
    model identities and never silently goes online in offline mode.
    """

    embedding = getattr(config, "embedding_backend", None)
    reranker = getattr(config, "reranker_backend", None)
    if embedding is not None and reranker is not None:
        return embedding, reranker

    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "SentenceTransformers is required for BGE-M3 retrieval; install requirements-windows.txt (or requirements-local-demo.txt on macOS)"
        ) from exc

    model_cache_dir = getattr(config, "model_cache_dir", None)
    common: dict[str, Any] = {
        "device": _device(getattr(config, "retrieval_device", "auto")),
        "local_files_only": _offline(config),
    }
    if model_cache_dir:
        common["cache_folder"] = str(Path(model_cache_dir))

    if embedding is None:
        embedding = SentenceTransformer(
            getattr(config, "embedding_model"),
            revision=getattr(config, "embedding_revision"),
            **common,
        )
        # Source material is already split into retrieval chunks. Keeping the
        # encoder at BGE-M3's very large default context wastes substantial
        # time and unified memory on a 16 GB Mac without adding usable text.
        embedding.max_seq_length = 512
    if reranker is None:
        reranker_kwargs = {
            "device": common["device"],
            "local_files_only": common["local_files_only"],
        }
        if model_cache_dir:
            reranker_kwargs["cache_folder"] = str(Path(model_cache_dir))
        reranker = CrossEncoder(
            getattr(config, "reranker_model"),
            revision=getattr(config, "reranker_revision"),
            max_length=512,
            **reranker_kwargs,
        )
    return embedding, reranker


__all__ = ["load_retrieval_backends"]
