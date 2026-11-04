# MP-KG-RAG Hindi Hinglish Counter Narrative Resources

**Content warning** This repository has examples of hate speech content. These are kept only for research on counter speech.

This project builds resources for studying counter narratives against hate speech written in Hindi Hinglish mixed language. It gives three main resources.

1. **Dataset** 1,550 human reviewed Hindi Hinglish hate speech posts paired with counter narrative replies, including earlier generated edited versions from each of five rounds.
2. **Knowledge graph** 14,723 claims taken from 111 trusted documents. Each claim is an exact quote of its source passage, stored with character position.
3. **MP-KG-RAG framework** retrieval system, five value based agents, a planner, a cited writer, baselines, benchmark outputs, scoring scripts, plus a Gradio demo that runs on a normal laptop CPU.

## Contents

| Path | What it holds |
| --- | --- |
| `data/hindi_hinglish_counter_narratives_1550.csv` / `.jsonl` | Final dataset with ID, Text, Category, Target, Counter Narrative columns |
| `data/versions/` | Seed version plus rounds v2 to v5, review sheets showing original and edited text (`*_m.xlsx`, `m2.xlsx`), metrics file for each round (`metrics.xlsx`) |
| `data/eval_test141.jsonl`, `data/eval_test141_ids.txt` | Evaluation set of 141 posts used for testing |
| `kg/source_manifest.csv` | List of 111 source documents with file name, source type, authority score, SHA-256 hash, passage count |
| `kg/claims.jsonl`, `entities.jsonl`, `mentions.jsonl`, `documents.jsonl`, `edges.jsonl` | Knowledge graph files |
| `kg/index_manifest.json` | Details of model versions used for retrieval index |
| `results/generations_141_4arms.jsonl` | All 564 benchmark replies (4 systems x 141 posts) with prompts and raw output |
| `results/generation_benchmark_results.json` | Benchmark summary |
| `results/retrieval/` | Retrieval diagnostic files with rankings, sample, metrics |
| `code/production_pipeline/` | Core library (`work/mpkg_rag_core.py`, `work/mpkg_eval_core.py`), notebook builders, config, tests |
| `code/benchmarks/`, `code/eval_v2/` | Benchmark runner (llama.cpp), extraction runners, evaluation tooling |
| `code/corpus_rebuild_tools/` | Scripts that rebuild passages, index from source PDFs |
| `code/retrieval_table10_codex/` | Retrieval experiment, annotation scoring |
| `code/gradio_demo/` | Interactive demo (Ollama / MLX back ends), see README-windows.md |
| `DATA_CARD.md`, `ANNOTATION_GUIDELINES.md` | Documentation of data, review process |
| `examples/load_data.py` | Minimal loading example |

## Quick start

```
pip install pandas
python examples/load_data.py
```

To run full pipeline on a laptop (Windows, macOS or Linux), follow `code/gradio_demo/README-windows.md`. It installs BGE-M3 retriever, starts a local Ollama model, launches Gradio interface.

## Source documents

111 source documents are not shared again in this repository. `kg/source_manifest.csv` lists each file with its SHA-256 hash. `code/corpus_rebuild_tools/` rebuilds passages, retrieval index from PDFs.

## Licence

Dataset, graph, documentation are released under CC BY-NC-SA 4.0 licence for research, non commercial use (see LICENSE). Seed comments come from LT-EDI shared task on homophobia, transphobia, these stay under terms of that task.

## Citation

If you use this project in your work, please give credit by citing related publication listed on this repository page.

## Maintenance

Maintainers keep updating this repository. New additions such as typed entities, verified claims, human relevance labels come out as tagged releases. Please report problems through issue tracker.
