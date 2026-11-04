# Claude task: port and run the MP-KG-RAG Gradio demo on Windows

You are working in an extracted copy of this folder on a Windows PC. Convert this existing Gradio application so it can be installed and run on Windows, then perform an end-to-end smoke run. Do not change the original project outside this extracted folder.

## Start by inspecting

- Read `README_FOR_CLAUDE_WINDOWS.md`, `README-local-mpkg-demo.md`, and the full `local_mpkg_demo/` package.
- Inspect `gradio_demo_5_cases.mp4` for the expected UI layout and flow.
- Trace the current `MLXGenerator` integration and the app launch path before editing.
- Identify Windows-incompatible paths, shell commands, dependencies, and assumptions.

## Required implementation

1. Preserve the existing Gradio screens and the MP-KG-RAG stages, evidence display, citation validation, conversation history, and export behavior. Make only changes needed for Windows portability and clearer setup.
2. Replace the MLX-only generation adapter with a small configurable Ollama HTTP client. Default to `http://localhost:11434`, make the model name configurable, handle connection errors/timeouts cleanly, and parse the model response using the existing schemas and validation/retry behavior where possible. Do not silently fall back to fake generation.
3. Keep `semantic_graph_source/` and the existing retrieval/indexing behavior. Run retrieval on CPU by default; allow a supported CUDA device when available, but do not make CUDA required. Remove MLX packages from the Windows requirements.
4. Add Windows setup instructions and a PowerShell launch script. The instructions must cover Python environment creation, dependency installation, starting Ollama, pulling/configuring a compatible chat model, building the BGE retrieval index, launching Gradio, and troubleshooting common failures. Keep model identifiers and Ollama URL in configuration or environment variables.
5. Do not add training or fine-tuning. This is an inference demo using a locally served Ollama model. Do not claim that the model was trained for MP-KG-RAG unless the repository contains a verified fine-tuned checkpoint (none is included in this bundle).

## Validation required

- Run the project's relevant unit tests and add focused tests for the new Ollama adapter and Windows-safe configuration.
- Start Ollama and use an actually available model; build or validate the retrieval index and submit one real query through the Gradio pipeline. Confirm it reaches a final answer with evidence IDs that pass citation validation.
- Launch the Gradio server and confirm its local URL responds. Capture a screenshot of the running UI and save it in this folder.
- Report exact Windows version, Python version, model identifier, commands, test results, the real query result status, and any limitations. If Ollama/model access or required hardware is unavailable, stop at that gate and report it without claiming the end-to-end run succeeded.

## Delivery

Keep changes in this extracted folder. Provide a concise summary of changed files and exact commands the user should run after the port. Do not modify, delete, or overwrite `semantic_graph_source/` or the supplied demo video.
