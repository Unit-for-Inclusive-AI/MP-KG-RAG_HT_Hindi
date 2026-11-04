"""Opt-in real local MP-KG-RAG smoke query.

The default invocation is a no-op readiness message.  ``--run-model`` is
required before this script checks the generator (Ollama server and model, or
MLX Metal on Apple silicon), downloads BGE artifacts, or executes inference.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path


def _require_metal() -> None:
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise RuntimeError("real local smoke requires Apple-silicon macOS")
    try:
        import mlx.core as mx

        # This call both validates the Metal namespace and warms the backend.
        mx.metal.get_active_memory()
    except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
        raise RuntimeError("real local smoke requires a working MLX Metal backend") from exc


def _require_generator(config) -> str:
    """Fail at the generator gate before any retrieval work starts."""

    if config.generator_backend == "mlx":
        _require_metal()
        return config.generator_label
    from local_mpkg_demo.ollama_engine import OllamaGenerator

    status = OllamaGenerator.load(config).check_ready()
    return f"ollama:{status['model']} (server {status['version']} at {status['base_url']})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-model", action="store_true", help="download/load pinned models and run one real query")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--query", default="What does the evidence say about LGBTQ rights and safety?")
    parser.add_argument("--offline", action="store_true", help="use only locally cached model snapshots")
    args = parser.parse_args(argv)
    if not args.run_model:
        print("Smoke skipped: pass --run-model to load models and execute inference.")
        return 0

    # Running the script puts work\ on sys.path; the demo imports from the root.
    sys.path.insert(0, str(args.project_root.resolve()))
    from dataclasses import replace

    from local_mpkg_demo.backends import load_retrieval_backends
    from local_mpkg_demo.config import DemoConfig
    from local_mpkg_demo.index_store import LocalRetriever, build_or_validate_index
    from local_mpkg_demo.pipeline import MPKGRAGPipeline, RunRequest, _validate_final_citations

    config = DemoConfig.load(args.project_root)
    if args.offline:
        config = replace(config, offline=True)
    print(f"Generator: {_require_generator(config)}")
    started = time.perf_counter()
    embedding, reranker = load_retrieval_backends(config)
    manifest = build_or_validate_index(
        config,
        progress=lambda message, fraction: print(f"[{fraction:5.1%}] {message}"),
        dense_encoder=embedding,
        reranker=reranker,
    )
    print(f"Index ready: {manifest.index_path} ({time.perf_counter() - started:.1f}s)")
    pipeline = MPKGRAGPipeline(
        config,
        retriever=LocalRetriever(config, dense_encoder=embedding, reranker=reranker),
    )
    events = []
    stage_started = time.perf_counter()
    peak_memory = 0
    for event in pipeline.run(RunRequest(text=args.query), __import__("threading").Event()):
        if event.stage not in {item.stage for item in events}:
            print(f"stage {event.stage}: {time.perf_counter() - stage_started:.1f}s")
            stage_started = time.perf_counter()
        events.append(event)
        peak_memory = max(peak_memory, int(event.metrics.get("peak_memory_bytes", 0) or 0))
        if event.status == "failed":
            raise RuntimeError(f"pipeline failed at {event.stage}: {event.error}")

    final_event = next((event for event in reversed(events) if event.stage == "final" and event.structured_payload), None)
    validation_event = next((event for event in reversed(events) if event.stage == "validation" and event.structured_payload), None)
    if final_event is None or not str(final_event.structured_payload.get("counter_narrative", "")).strip():
        raise RuntimeError("smoke produced an empty final answer")
    evidence_ids = {record.evidence_id for record in final_event.evidence}
    valid, reason = _validate_final_citations(final_event.structured_payload, evidence_ids)
    if not valid:
        raise RuntimeError(f"smoke citation validation failed: {reason}")
    if validation_event is None or validation_event.structured_payload.get("valid") is not True:
        raise RuntimeError("smoke did not emit a valid citation validation stage")
    print(f"Final answer: {final_event.structured_payload['counter_narrative']}")
    print(f"Evidence IDs cited: {', '.join(final_event.structured_payload.get('cited_evidence_ids', [])) or 'none'}")
    memory = f"; peak Metal memory {peak_memory / (1024 ** 3):.2f} GiB" if config.generator_backend == "mlx" else ""
    print(f"Stages: {len(events)} events; total {time.perf_counter() - started:.1f}s{memory}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Smoke blocked: {exc}", file=sys.stderr)
        raise SystemExit(2)
