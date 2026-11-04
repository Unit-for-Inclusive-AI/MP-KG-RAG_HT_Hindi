# Local MP-KG-RAG Pipeline Lab on Windows

This is the Windows port of the local Gradio demo. The screens, the MP-KG-RAG
stages (metadata, retrieval, five perspectives, plan, final, validation), the
evidence ledger, citation validation, conversation history and export all work
as in the original. Only the model runtime changed:

- **Generation** runs in [Ollama](https://ollama.com), a local model server,
  instead of MLX (which only runs on Apple-silicon Macs). The app talks to it
  over HTTP. If Ollama or the model is missing, the run fails with a clear
  message. It never falls back to fake output.
- **Retrieval** (BGE-M3 embeddings, BGE reranker v2-M3, BM25 and the semantic
  graph) runs on the CPU by default. An NVIDIA GPU is optional.

No model here is trained or fine-tuned for MP-KG-RAG. The demo prompts a
general instruction model served by Ollama, and grounds it in retrieved
evidence.

## What you need

- Windows 10 or 11, 64-bit, with about 16 GB RAM (24 GB is comfortable) and
  about 15 GB free disk for models and the index.
- Python 3.11, 64-bit, from python.org. Tick "Add python.exe to PATH".
- Ollama for Windows: https://ollama.com/download, or
  `winget install Ollama.Ollama`.

## Setup (one time)

Open PowerShell in this folder.

1. Allow local scripts for this window only:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```

2. Create the Python environment and install dependencies:

   ```powershell
   .\setup_windows.ps1
   ```

   This creates `.venv-local-demo` with Python 3.11 and installs
   `requirements-windows.txt` (CPU PyTorch, Gradio, sentence-transformers).
   To do it by hand instead:

   ```powershell
   py -3.11 -m venv .venv-local-demo
   .\.venv-local-demo\Scripts\python.exe -m pip install -r requirements-windows.txt
   ```

3. Start Ollama and pull the chat model. Ollama usually starts with Windows
   and shows a tray icon. If it isn't running, start the Ollama app or run
   `ollama serve` in a separate window. Then:

   ```powershell
   ollama pull qwen3:4b
   ollama run qwen3:4b "Reply with the word ok."
   ```

4. Build the retrieval index. This downloads BGE-M3 and the reranker (about
   4.5 GB) and embeds all 6,431 evidence chunks:

   ```powershell
   .\.venv-local-demo\Scripts\python.exe work\build_local_mpkg_demo_index.py
   ```

   This is slow on a laptop CPU. A benchmark measured about 1.3 s per chunk
   on a Ryzen 5, which works out to about 2.5 hours. The real build on the
   same laptop took 11 hours, because the CPU slows down under sustained
   load. Keep the laptop plugged in and stop it from sleeping (Settings >
   System > Power). The build runs once; the index is reused while
   `semantic_graph_source/` is unchanged.

## Launch

```powershell
.\run_demo.ps1
```

The script checks that Ollama answers and the model is installed (it starts
`ollama serve` if needed), checks that the index exists, and then starts
Gradio at http://127.0.0.1:7860. Open that address in a browser.

Useful options:

```powershell
.\run_demo.ps1 -Model qwen3:8b           # another pulled Ollama model
.\run_demo.ps1 -Port 7861                # another port
.\run_demo.ps1 -Device cuda              # retrieval on an NVIDIA GPU
.\run_demo.ps1 -OllamaUrl http://192.168.1.20:11434   # Ollama on another PC
.\run_demo.ps1 -Offline                  # use only cached BGE models
```

Without the script:

```powershell
.\.venv-local-demo\Scripts\python.exe -m local_mpkg_demo.app --project-root . --host 127.0.0.1 --port 7860
```

A full MP-KG-RAG run makes eight model calls (metadata inference when target
or category is empty, five perspectives, plan, final). On a CPU-only laptop
with `qwen3:4b`, expect several minutes per query.

## Configuration

All settings are environment variables. `run_demo.ps1` sets the common ones
from its parameters.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MPKG_GENERATOR` | `ollama` (`mlx` on Apple silicon) | Generation backend |
| `MPKG_OLLAMA_URL` | `http://localhost:11434` | Ollama server; falls back to `OLLAMA_HOST` |
| `MPKG_OLLAMA_MODEL` | `qwen3:4b` | Ollama model tag (must already be pulled) |
| `MPKG_OLLAMA_NUM_CTX` | `8192` | Context window requested from Ollama |
| `MPKG_OLLAMA_TIMEOUT` | `600` | Seconds to wait for the server per read |
| `MPKG_OLLAMA_KEEP_ALIVE` | `30m` | How long Ollama keeps the model loaded |
| `MPKG_RETRIEVAL_DEVICE` | `auto` (CPU; MPS on a Mac) | `cpu`, `cuda`, `mps` or `auto` |
| `HF_HUB_OFFLINE=1` | unset | Use only cached Hugging Face models |

Choosing a model: any Ollama chat model that supports structured (JSON
schema) output works. A 4B model at 4-bit fits easily in 16 GB alongside the
retrieval models. An 8B model is slower and tighter. The original Mac demo
used Qwen3.5-4B (4-bit); `qwen3:4b` is the closest Ollama equivalent.
Ollama applies each stage's JSON schema as a decoding constraint, and the
app still validates every result and makes at most one repair attempt.

Retrieval on CUDA: install a CUDA build of PyTorch first (for example
`pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128`),
then use `-Device cuda`. The index identity doesn't depend on the device, so
an index built on the CPU stays usable. Query vectors can differ from CPU ones
by tiny floating-point amounts.

## Smoke test and unit tests

```powershell
.\.venv-local-demo\Scripts\python.exe -m pytest tests -q
.\.venv-local-demo\Scripts\python.exe work\smoke_local_mpkg_demo.py --run-model
```

The smoke test checks Ollama and the model, loads the index, runs one real
query through the full pipeline, and fails unless the final answer cites
evidence IDs that pass citation validation.

## Local state and reset

- `local_mpkg_demo_index\`: the retrieval index (rebuild with the build script)
- `local_mpkg_demo_history.sqlite3`: conversation history
- `exports\`: JSON/Markdown exports from the UI

To reset, stop the app and run:

```powershell
Remove-Item -Recurse -Force local_mpkg_demo_index, local_mpkg_demo_history.sqlite3, exports
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `cannot reach Ollama at http://localhost:11434` | Start the Ollama app or `ollama serve`; check `MPKG_OLLAMA_URL`. |
| `Ollama model 'qwen3:4b' is not installed` | `ollama pull qwen3:4b`, or set `-Model` to a tag shown by `ollama list`. |
| `Ollama did not respond within 600s` | The CPU is slow or busy. Close heavy apps, or set `MPKG_OLLAMA_TIMEOUT=1200`. |
| `model requires more system memory` | Use a smaller model, close other apps, or lower `MPKG_OLLAMA_NUM_CTX` to 4096. |
| `Index: not ready` in the UI | Run the index build, or `.\run_demo.ps1 -BuildIndex`. |
| `running scripts is disabled on this system` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same window. |
| `py` is not recognized | Install Python 3.11 from python.org with the launcher option. |
| `MPKG_RETRIEVAL_DEVICE=cuda but PyTorch has no CUDA device` | Install a CUDA PyTorch build, or use `-Device cpu`. |
| BGE download fails behind a proxy | Set `HTTPS_PROXY`, or copy a populated Hugging Face cache and use `-Offline`. |
| Port 7860 is busy | `.\run_demo.ps1 -Port 7861`. |
| A run fails at a perspective stage with `schema_invalid` | The model returned invalid JSON twice. Retry, or use a larger model. |
