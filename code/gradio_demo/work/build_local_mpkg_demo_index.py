"""Build/validate the disk-backed index used by the local demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ``python work\build_local_mpkg_demo_index.py`` puts work\ on sys.path; the
# demo package and ``work.mpkg_rag_core`` both resolve from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_mpkg_demo.backends import load_retrieval_backends
from local_mpkg_demo.config import DemoConfig
from local_mpkg_demo.index_store import build_or_validate_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--offline", action="store_true", help="Use only already-cached BGE model snapshots")
    parser.add_argument("--model-cache-dir", type=Path, default=None, help="Optional Hugging Face model cache directory")
    args = parser.parse_args()
    config = DemoConfig.load(args.project_root)
    if args.offline or args.model_cache_dir:
        from dataclasses import replace

        config = replace(config, offline=args.offline, model_cache_dir=args.model_cache_dir)
    fixture = False
    try:
        fixture = json.loads(config.source_manifest_path.read_text(encoding="utf-8")).get("identity", {}).get("stage_hash") == "tiny"
    except (OSError, TypeError, ValueError):
        pass
    embedding, reranker = (None, None) if fixture else load_retrieval_backends(config)
    manifest = build_or_validate_index(
        config,
        progress=lambda message, fraction: print(f"[{fraction:5.1%}] {message}"),
        dense_encoder=embedding,
        reranker=reranker,
    )
    print(f"Index: {manifest.index_path}")
    counts = manifest.row_counts
    print(
        "Rows: "
        f"evidence={counts.get('evidence', 0)} "
        f"documents={counts.get('documents', 0)} "
        f"entities={counts.get('entities', 0)} "
        f"mentions={counts.get('mentions', 0)} "
        f"graph_edges={counts.get('graph_edges', 0)}"
    )


if __name__ == "__main__":
    main()
