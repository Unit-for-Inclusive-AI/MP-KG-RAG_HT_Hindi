#!/usr/bin/env python3
"""vLLM (CUDA) port of run_extraction_mlx.py for the v3 quote-first claims
contract and mention discovery.

The entire orchestration (wave driver, JSON-repair retry, locate retry, G2/G3/G4
gates, mention locating/validation, outputs, manifest, resume) is inherited
unchanged from work/run_extraction_mlx.py. Only the engine boundary is
replaced:

  * ensure_model()        -> vLLM LLM (Qwen base fp16, tensor parallel) instead
                             of MLXGenerator over mlx-community/Qwen3.5-4B-4bit
  * _generate_singleton() -> one-prompt vLLM call
  * _generate_batch()     -> batched vLLM call (continuous batching)

Prompts, schemas, budgets, seeds (greedy temperature 0.0), repair/nudge
semantics and output JSONL row shapes are identical to the MLX runner, so the
two engines are drop-in interchangeable at the artifact level. Engine identity
(model repo, dtype, vLLM version) is recorded in run_manifest.json; numeric
drift between MLX 4-bit and CUDA fp16 is expected and documented there.

Requires the same directory layout as the MLX runner (PROJECT with work/ and
source_snapshot/ on sys.path); source_snapshot.mlx_engine only needs the three
pure helpers imported by the MLX runner (split_thinking, _schema_valid,
_project_schema_object), so a small CUDA-safe shim is sufficient.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

PORT_PATH = Path(__file__).resolve()
PROJECT = PORT_PATH.parents[1]
sys.path.insert(0, str(PROJECT))

from work import run_extraction_mlx as base  # noqa: E402

ENGINE_NAME = "vllm"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
# The MLX run tokenized prompts with the mlx-community tokenizer, whose merges
# differ from Qwen/Qwen3.5-4B on Devanagari text (same vocab, different splits:
# e.g. 24 vs 39 tokens for a Hindi sample; ASCII identical). Tokenizing with
# anything else sends materially different prompt tokens for Hindi docs.
TOKENIZER_ID = "mlx-community/Qwen3.5-4B-4bit"
TOKENIZER_REVISION = "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"


class VLLMRunner(base.Runner):
    """Same runner contract; vLLM engine."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._tokenizer = None
        identity = self.manifest.payload["identity"]
        identity["runner"] = str(PORT_PATH)
        identity["runner_sha256"] = base.sha256_file(PORT_PATH)
        identity["model_id"] = MODEL_ID
        identity["model_revision"] = MODEL_REVISION
        identity["engine"] = {"name": ENGINE_NAME, "version": None, "dtype": args.dtype,
                              "tensor_parallel_size": int(args.tp_size),
                              "max_model_len": int(args.max_model_len),
                              "gpu_memory_utilization": float(args.gpu_memory_utilization),
                              "tokenizer_id": TOKENIZER_ID,
                              "tokenizer_revision": TOKENIZER_REVISION}
        identity["port_of"] = {
            "mlx_runner": str(base.RUNNER_PATH),
            "mlx_runner_sha256": base.sha256_file(base.RUNNER_PATH),
            "note": ("identical prompts/schemas/gates/parsers; engine MLX 4-bit -> vLLM fp16, "
                     "expect numeric drift in generated text"),
        }
        self.manifest.write()

    def ensure_model(self):
        if self.model is None:
            random.seed(self.args.seed)
            from vllm import LLM
            from vllm import __version__ as vllm_version

            print(json.dumps({"status": "loading_model", "engine": ENGINE_NAME,
                              "version": vllm_version, "model": MODEL_ID,
                              "revision": MODEL_REVISION,
                              "tensor_parallel_size": int(self.args.tp_size),
                              "dtype": self.args.dtype,
                              "max_model_len": int(self.args.max_model_len)}), flush=True)
            self.model = LLM(
                model=MODEL_ID,
                revision=MODEL_REVISION,
                tokenizer=TOKENIZER_ID,
                tokenizer_revision=TOKENIZER_REVISION,
                tensor_parallel_size=int(self.args.tp_size),
                dtype=self.args.dtype,
                max_model_len=int(self.args.max_model_len),
                gpu_memory_utilization=float(self.args.gpu_memory_utilization),
                enforce_eager=bool(self.args.enforce_eager),
                seed=int(self.args.seed),
                disable_log_stats=True,
            )
            self._tokenizer = self.model.get_tokenizer()
            identity = self.manifest.payload["identity"]
            identity["engine"]["version"] = vllm_version
            self.manifest.payload["config"]["engine"] = dict(identity["engine"])
            self.manifest.write()
        return self.model

    # --------------------------------------------------------------- generation

    def _vllm_generate(self, requests: list[dict], budget: int, thinking: bool,
                       timings: list[dict], mode: str, attempt: int,
                       batch_failed: bool = False) -> list[dict]:
        from vllm import SamplingParams

        llm = self.ensure_model()
        tokenizer = self._tokenizer
        prompts = []
        for req in requests:
            messages = base._with_schema_contract(req["messages"], req["schema"])
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=bool(thinking)))
        params = SamplingParams(temperature=0.0, max_tokens=int(budget))
        started = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        wall = time.perf_counter() - started
        peak_bytes = active_bytes = None
        try:
            import torch

            if torch.cuda.is_available():
                peak_bytes = int(torch.cuda.max_memory_allocated())
                active_bytes = int(torch.cuda.memory_allocated())
        except Exception:
            pass

        results: list[dict] = []
        per_row: list[dict] = []
        total_prompt = total_generated = 0
        for req, out in zip(requests, outputs):
            completion = out.outputs[0] if out.outputs else None
            raw = completion.text if completion else ""
            finish_reason = completion.finish_reason if completion else None
            generated = len(completion.token_ids) if completion else 0
            prompt_tokens = len(out.prompt_token_ids or [])
            total_prompt += prompt_tokens
            total_generated += generated
            payload, parse_status, parse_error = self._parse_result(mode, raw, req["schema"])
            truncated = finish_reason == "length" or (int(budget) and generated >= int(budget))
            per_row.append({"prompt_tokens": prompt_tokens, "generated": int(generated)})
            results.append({"chunk_id": req["chunk_id"], "raw": raw, "payload": payload,
                            "parse_status": parse_status, "parse_error": parse_error,
                            "finish_reason": finish_reason, "generated_tokens": int(generated),
                            "truncated": bool(truncated),
                            "elapsed_s": round(wall / max(1, len(requests)), 3)})

        batch = len(requests) > 1
        tps = round(total_generated / wall, 2) if wall > 0 and total_generated else None
        for req, res, row in zip(requests, results, per_row):
            timing = {
                "ts": base.now_iso(), "chunk_id": req["chunk_id"], "mode": mode, "attempt": attempt,
                "thinking": bool(thinking),
                "elapsed_s": round(wall / max(1, len(requests)), 3),
                "wall_s": round(wall, 3), "batch_size": len(requests),
                "prompt_tokens": row["prompt_tokens"], "generated_tokens": row["generated"],
                "tokens_per_second": (round(row["generated"] / wall, 2) if not batch and wall > 0
                                      and row["generated"] else tps),
                "finish_reason": res["finish_reason"],
                "error": None if res["parse_status"] != "failed" else res["parse_error"],
                "tolerant_parse": res["parse_status"] == "recovered",
                "parse_status": res["parse_status"],
                "truncated": res["truncated"],
                "engine": ENGINE_NAME,
                "peak_memory_bytes": peak_bytes, "active_memory_bytes": active_bytes,
            }
            if batch:
                timing["batch_prompt_tokens"] = total_prompt
                timing["batch_generation_tokens"] = total_generated
            else:
                timing["batch_fallback"] = batch_failed
                timing["batch_failed"] = batch_failed
            timings.append(timing)
        return results

    def _generate_singleton(self, req: dict, budget: int, thinking: bool,
                            timings: list[dict], mode: str, attempt: int,
                            batch_failed: bool = False) -> dict:
        return self._vllm_generate([req], budget, thinking, timings, mode, attempt,
                                   batch_failed=batch_failed)[0]

    def _generate_batch(self, requests: list[dict], budget: int, thinking: bool,
                        timings: list[dict], mode: str, attempt: int) -> list[dict]:
        return self._vllm_generate(requests, budget, thinking, timings, mode, attempt)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vLLM (CUDA) port of the v3 quote-first extraction runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--chunks", required=True, help="path to chunks.json")
    parser.add_argument("--candidates", default=None, help="optional path to candidates.json")
    parser.add_argument("--out", required=True, help="output directory for jsonl + manifest")
    parser.add_argument("--mode", choices=["claims", "mentions", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="max chunks to process")
    parser.add_argument("--resume", action="store_true",
                        help="skip chunk_ids already present in the mode's out jsonl")
    parser.add_argument("--chunk-ids", default=None,
                        help="optional subset file: JSON array, {chunk_ids:[...]}, or newline list")
    parser.add_argument("--thinking", type=int, choices=[0, 1], default=0,
                        help="enable model thinking for claims calls (mentions always non-thinking)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="prompts per wave; >1 uses vLLM continuous batching")
    parser.add_argument("--tp-size", type=int, default=2, help="vLLM tensor parallel size")
    parser.add_argument("--dtype", default="float16", help="vLLM dtype (T4: float16)")
    parser.add_argument("--max-model-len", type=int, default=8192, help="vLLM context length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="vLLM KV cache fraction per GPU")
    parser.add_argument("--enforce-eager", action="store_true", help="disable CUDA graphs")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    runner = VLLMRunner(args)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
