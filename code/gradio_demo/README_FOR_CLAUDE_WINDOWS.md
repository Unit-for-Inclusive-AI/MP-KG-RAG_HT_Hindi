# MP-KG-RAG Gradio demo export for Windows porting

This bundle contains the current local Gradio UI, its retrieval pipeline and semantic graph artifacts, a short demo recording, and a Claude handoff prompt. The included app is the original Apple Silicon version: it uses MLX and does not run on Windows unchanged. The dependency file is preserved as the source version and includes Apple-only packages.

## Use with Claude on Windows

1. Extract this folder to a writable path, such as `C:\MP-KG-RAG-Gradio`.
2. Open that folder in Claude Code (or give Claude the folder and the prompt file).
3. Ask Claude to follow `CLAUDE_PROMPT_WINDOWS_PORT.md` and make the port in this extracted copy.
4. The prompt asks Claude to connect generation to a local Ollama server, prepare Windows dependencies and launch the Gradio UI for a real one-query smoke run.

The Ollama model and the BGE retrieval models are not included. They must be downloaded on the target computer. The semantic corpus is included.

## Bundle contents

- `local_mpkg_demo/` — Gradio interface, orchestration, retrieval index, persistence, and original MLX inference adapter.
- `work/mpkg_rag_core.py` — graph/retrieval core imported by the demo.
- `work/build_local_mpkg_demo_index.py` — builds the local retrieval index.
- `semantic_graph_source/` — input data needed to build that index.
- `gradio_demo_5_cases.mp4` — visual reference for the existing UI/demo behavior.
- `requirements-local-demo.txt` — original macOS/MLX requirements, provided for reference; replace for Windows as directed in the prompt.
- `CLAUDE_PROMPT_WINDOWS_PORT.md` — detailed Windows conversion and run request.

This export intentionally excludes cached model weights, generated indexes, local conversation history, and unrelated project files.
