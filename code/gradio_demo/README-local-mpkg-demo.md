# Local MP-KG-RAG Pipeline Lab

> **Windows:** see [README-windows.md](README-windows.md). There, generation
> runs through Ollama and retrieval runs on the CPU. The instructions below
> are for the original Apple-silicon MLX setup.

This is a single-user, local demonstration for an Apple-silicon Mac with
approximately 16 GB unified memory. It uses the complete semantic artifacts,
real BGE-M3 retrieval/reranking, and one pinned four-bit Qwen3.5 MLX model.
Expect roughly 2–4 minutes for a complete query after the first-load work.

## Setup and launch

From this checkout:

```bash
python3 -m venv .venv-local-demo
source .venv-local-demo/bin/activate
python -m pip install -r requirements-local-demo.txt

# One-time BGE index build; downloads BGE-M3 and BGE-reranker-v2-M3.
python work/build_local_mpkg_demo_index.py

# Launch the local Gradio UI.
python -m local_mpkg_demo.app
```

The completed `semantic_graph_source/` directory is reused in place. In the
standard worktree layout, the launcher also finds artifacts in the adjacent
main checkout, so no large artifact copy is required. The first model run
downloads the pinned `mlx-community/Qwen3.5-4B-4bit` snapshot (about 3 GB).

After the snapshots and index exist, offline reuse is explicit:

```bash
python work/build_local_mpkg_demo_index.py --offline
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m local_mpkg_demo.app
```

Run the opt-in real smoke check with:

```bash
python work/smoke_local_mpkg_demo.py --run-model
```

The smoke command requires Apple silicon and Metal. Normal unit tests and UI
construction do not import MLX or download models.

## Local state and reset

Conversation history is stored in `local_mpkg_demo_history.sqlite3`; exports
are written under `exports/`. The immutable retrieval index is under
`local_mpkg_demo_index/` and can be rebuilt if source artifacts change.

To reset this demo's generated state, stop the app and remove those three
paths, then rerun the index build:

```bash
rm -rf local_mpkg_demo_index local_mpkg_demo_history.sqlite3 exports
python work/build_local_mpkg_demo_index.py
```

Use the UI Export control for a portable JSON or Markdown conversation copy.
