# MP-KG-RAG: Hindi–Hinglish Counter-Narrative Resources

> **Content warning:** this repository contains examples of homophobic hate
> speech, included only for research on counter-speech.

This repository accompanies the ECIR 2027 resource paper *"MP-KG-RAG: A
Hindi–Hinglish Counter-Narrative Dataset, Knowledge Graph and Multi-Agent
Framework against Homophobic Hate Speech"*. It releases three resources:

1. **Dataset:** 1,550 human-reviewed Hindi–Hinglish hate-speech and
   counter-narrative pairs, with the generated and edited version of every
   pair from each of the five expansion rounds.
2. **Knowledge graph:** 14,723 claims from 111 authoritative documents. Every
   claim is an exact quote of its source passage, stored with character
   offsets.
3. **MP-KG-RAG framework:** retrieval, five value-oriented agents, a planner
   and a cited writer, plus baselines, benchmark outputs, scoring scripts and
   a Gradio demo that runs on a laptop CPU.

## Contents

| Path | What it holds |
| --- | --- |
| `data/hindi_hinglish_counter_narratives_1550.csv` / `.jsonl` | Final dataset: `ID, Text, Category, Target, Counter Narrative` |
| `data/versions/` | Seed (v1) and rounds v2–v5, review sheets with original vs edited text (`*_m.xlsx`, `m2.xlsx`) and per-round metrics (`metrics.xlsx`) |
| `data/eval_test141.jsonl`, `data/eval_test141_ids.txt` | The 141-post evaluation set used in the paper |
| `kg/source_manifest.csv` | The 111 source documents: file name, source type, authority score, SHA-256, passage count |
| `kg/claims.jsonl`, `entities.jsonl`, `mentions.jsonl`, `documents.jsonl`, `edges.jsonl` | The knowledge graph |
| `kg/index_manifest.json` | Model revisions and identity of the retrieval index |
| `results/generations_141_4arms.jsonl` | All 564 benchmark replies (4 systems × 141 posts) with prompts and raw output |
| `results/generation_benchmark_results.json` | Benchmark summary |
| `results/retrieval/` | Retrieval diagnostic: rankings, sample, metrics |
| `code/production_pipeline/` | Core library (`work/mpkg_rag_core.py`, `work/mpkg_eval_core.py`), notebook builders, config and tests |
| `code/benchmarks/`, `code/eval_v2/` | Benchmark runner (llama.cpp), extraction runners, evaluation tooling |
| `code/corpus_rebuild_tools/` | Scripts that rebuild passages and the index from the source PDFs |
| `code/retrieval_table10_codex/` | Retrieval experiment and annotation scoring |
| `code/gradio_demo/` | Interactive demo (Ollama / MLX back-ends); see `README-windows.md` |
| `DATA_CARD.md`, `ANNOTATION_GUIDELINES.md` | Documentation of the data and the review process |
| `examples/load_data.py` | Minimal loading example |

## Quick start

```bash
pip install pandas
python examples/load_data.py
```

To run the full pipeline on a laptop (Windows, macOS or Linux), follow
`code/gradio_demo/README-windows.md`. It installs the BGE-M3 retriever, starts
a local Ollama model and launches the Gradio interface.

## Source documents

The 111 source documents are not redistributed. `kg/source_manifest.csv` lists
each file with its SHA-256 hash, and `code/corpus_rebuild_tools/` rebuilds the
passages and the retrieval index from the PDFs.

## Licence

The dataset, graph and documentation are released under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) for
research and non-commercial use (see `LICENSE`). The seed comments come from
the LT-EDI shared task on homophobia and transphobia and remain subject to the
terms of that task.

## Citation

```bibtex
@inproceedings{chiman2027mpkgrag,
  title     = {MP-KG-RAG: A Hindi--Hinglish Counter-Narrative Dataset, Knowledge
               Graph and Multi-Agent Framework against Homophobic Hate Speech},
  author    = {Chiman, Tanvi and Prabakaran, Amritha and Kumaresan, Prasanna Kumar
               and Rajalakshmi, Ratnavel and Durairaj, Thenmozhi
               and Chakravarthi, Bharathi Raja},
  booktitle = {Advances in Information Retrieval (ECIR 2027)},
  year      = {2027}
}
```

## Maintenance

The authors maintain this repository. Extensions such as typed entities,
verified claims and human relevance labels are published as tagged releases.
Please report problems through the issue tracker.
