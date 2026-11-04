# eval_v2 — production-faithful MP-arm re-run tooling

Purpose: re-run the production MP-arm retrieval + prompt pipeline over Test IDs
against an index generation, recompute the paper's auto-tables from archived
outputs, and calibrate the recomputation against the published numbers.

## Modules

- `config_v2.py` — production constants (retrieval contract, RRF weights,
  prompt budgets), generation loading, retrieval divergence check, corpus
  binding verification, code hashes.
- `bm25_vendored.py` — rank_bm25 0.2.2 `BM25Okapi` port + production tokenizer.
- `production_index.py` — read-only generation loader (id_map/embeddings/
  sqlite/graph) + dense, BM25, graph primitives + reranker + target-anchor
  entity resolution. **Never calls `build_or_validate_index`.**
- `production_retrieve.py` — exact mirror of the runtime `retrieve_evidence`
  (gate → dense/bm25/graph → RRF(1.1/1.0/0.8 @ 60.0) → full-pool rerank →
  `select_evidence(θ=0.55, max=5)`).
- `production_prompts.py` — ledger validation/selection/fitting and the
  perspective/plan/final/repair prompt builders (ports of runtime L2626,
  L2682, L3065, L5548-5700).
- `records_v2.py` — audit records: score_corrected-compatible keys plus
  candidates with rerank logits, validated ledger, prompt fit audit.
- `run_mp_arm_v2.py` — the driver.
- `composition_v2.py` — deterministic table-value assemblers (five-system
  metric table, retrieval summaries incl. cross-script aggregates, F1@5 from
  assumed P/R, BERTScore means, checkpoint aggregates). Pure functions.
- `recompute_tables.py` — calibration CLI: re-runs the frozen scorer over the
  archived 141-ID inputs, recomputes BERTScore/retrieval/checkpoint values,
  diffs everything against the recorded artifacts, and writes
  `calibration_report.{json,md}`. Exit 3 on any mismatch.

## Safety invariants

1. Read-only against the index; loads the five generation files directly.
2. Never flips `CURRENT`, never creates a generation, never edits the paper,
   `semantic_graph_source/`, or the decision log.
3. Generation (Qwen decode) is out of scope here; records that reach the
   prompt stage carry `generation_status: pending_generation` with fully
   assembled prompts.

## Usage

```bash
cd /Users/pika/Downloads/MP_KG_RAG_v2
python3 -m eval_v2.run_mp_arm_v2 \
  --out /path/to/out \
  --ids-file outputs/2026-10-16-local-mlx-test155/test_homophobic.jsonl \
  --generation-id 20261016T175419796064Z-23a9c32e \
  --max-ids 5 \
  [--ids 623,385,1287] [--resume] [--signatures-file sigs.jsonl] \
  [--tokenizer auto|none|<hf-id>] [--device cpu]
```

Outputs `<out>/mp_kg_rag.jsonl` (one audit record per ID) and
`<out>/manifest.json` (generation identity, corpus binding, code hashes,
constants, divergences, environment).

Environment defaults applied by the driver: `MPKG_RETRIEVAL_DEVICE=cpu`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` (override with `--online`).

## Calibration

```bash
cd /Users/pika/Downloads/MP_KG_RAG_v2
python3 -m eval_v2.recompute_tables \
  [--diagnostic-dir ...] [--retrieval-dir ...] [--checkpoint-dir ...] \
  [--no-scorer-rerun] [--bootstrap-draws 200] [--seed 42] [--out ...]
```

Reads archived artifacts only; writes `calibration_report.json` and
`calibration_report.md` into `--out` (default `.audit/eval_v2_20261019/`).

## Known divergences (recorded per run)

See `.audit/eval_v2_20261019/report.md` §5: ledger source-resolution fallback,
deterministic target-anchor signatures without `--signatures-file`, fitting
skipped without a tokenizer, RRF constant pinned from the runtime call site.
