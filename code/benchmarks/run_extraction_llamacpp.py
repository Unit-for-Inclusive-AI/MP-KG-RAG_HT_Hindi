#!/usr/bin/env python3
"""llama.cpp (CUDA) port of run_extraction_mlx.py for the v3 quote-first claims
contract and mention discovery.

The entire orchestration (wave driver, JSON-repair retry, locate retry, G2/G3/G4
gates, mention locating/validation, outputs, manifest, resume) is inherited
unchanged from work/run_extraction_mlx.py. Only the engine boundary is
replaced:

  * ensure_model()        -> local llama.cpp llama-server over HTTP (Q4_K_M
                             GGUF, 2x T4 layer split, continuous batching via
                             -np slots) instead of MLXGenerator over
                             mlx-community/Qwen3.5-4B-4bit
  * _generate_singleton() -> one-prompt /completion call
  * _generate_batch()     -> concurrent /completion calls, one per slot

Prompts, schemas, budgets, seeds (greedy temperature 0.0), repair/nudge
semantics and output JSONL row shapes are identical to the MLX runner, so the
two engines are drop-in interchangeable at the artifact level. Prompts are
rendered with the Qwen/Qwen3.5-4B chat template (tokenize=False); the GGUF's
embedded tokenizer tokenizes server-side. Engine identity (GGUF path + sha256,
server version, slots, context, flash attention) is recorded in
run_manifest.json; numeric drift between MLX 4-bit and llama.cpp Q4_K_M is
expected and documented there.

Requires the same directory layout as the MLX runner (PROJECT with work/ and
source_snapshot/ on sys.path); source_snapshot.mlx_engine only needs the three
pure helpers imported by the MLX runner (split_thinking, _schema_valid,
_project_schema_object), so a small CUDA-safe shim is sufficient.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PORT_PATH = Path(__file__).resolve()
PROJECT = PORT_PATH.parents[1]
sys.path.insert(0, str(PROJECT))

from work import run_extraction_mlx as base  # noqa: E402

ENGINE_NAME = "llamacpp"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
GGUF_PATH = "/kaggle/working/mpkg/models/Qwen3.5-4B-Q4_K_M.gguf"
LLAMA_SERVER_BIN = "/kaggle/working/llama.cpp/build/bin/llama-server"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8081


class LlamaCppRunner(base.Runner):
    """Same runner contract; llama.cpp llama-server engine."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._tokenizer = None
        self._server = None
        identity = self.manifest.payload["identity"]
        identity["runner"] = str(PORT_PATH)
        identity["runner_sha256"] = base.sha256_file(PORT_PATH)
        identity["model_id"] = MODEL_ID
        identity["model_revision"] = MODEL_REVISION
        identity["engine"] = {"name": ENGINE_NAME, "version": None, "gguf": args.gguf,
                              "gguf_sha256": None, "slots": int(args.slots),
                              "ctx": int(args.ctx), "flash_attn": bool(args.flash_attn),
                              "tensor_split": "1,1", "n_gpu_layers": 99,
                              "server_bin": args.server_bin, "external": False}
        identity["port_of"] = {
            "mlx_runner": str(base.RUNNER_PATH),
            "mlx_runner_sha256": base.sha256_file(base.RUNNER_PATH),
            "note": ("identical prompts/schemas/gates/parsers; engine MLX 4-bit -> llama.cpp Q4_K_M "
                     "GGUF, numeric drift in generated text expected"),
        }
        if getattr(args, "ports", None):
            self.ports = [int(p.strip()) for p in args.ports.split(",") if p.strip()]
        else:
            self.ports = [int(args.port)]
        self.manifest.write()

    def _health(self, port: int | None = None, timeout: float = 3.0) -> bool:
        p = port if port is not None else self.ports[0]
        url = f"http://{SERVER_HOST}:{p}/health"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return int(getattr(response, "status", 200)) == 200
        except Exception:
            return False

    def _log_tail(self, lines: int = 20) -> str:
        path = Path(self.args.out) / "llama_server.log"
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"(server log unavailable: {exc})"
        return "\n".join(content.splitlines()[-lines:])

    def _server_version(self) -> str | None:
        try:
            proc = subprocess.run([self.args.server_bin, "--version"], capture_output=True,
                                  text=True, timeout=60, check=False)
        except Exception:
            return None
        for stream in (proc.stdout, proc.stderr):
            for line in (stream or "").splitlines():
                if line.strip():
                    return line.strip()
        return None

    @staticmethod
    def _gguf_sha256(path: Path) -> str | None:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
        except OSError:
            return None
        return digest.hexdigest()

    def _props(self) -> dict | None:
        url = f"http://{SERVER_HOST}:{self.args.port}/props"
        try:
            with urllib.request.urlopen(url, timeout=3.0) as response:
                if int(getattr(response, "status", 200)) != 200:
                    return None
                value = json.loads(response.read().decode("utf-8"))
        except Exception:
            return None
        return value if isinstance(value, dict) else None

    def _verify_external(self, props: dict) -> None:
        expected_slots = int(self.args.slots)
        expected_ctx = int(self.args.ctx) // max(1, expected_slots)
        model_path = props.get("model_path")
        if not isinstance(model_path, str) or not model_path:
            raise RuntimeError("external llama-server /props has no model_path; refusing to reuse")
        if Path(model_path).resolve() != Path(self.args.gguf).resolve():
            raise RuntimeError(f"external llama-server model mismatch: {model_path} != {self.args.gguf}")
        server_slots = props.get("total_slots")
        if server_slots is None:
            raise RuntimeError("external llama-server /props has no total_slots; refusing to reuse")
        if int(server_slots) != expected_slots:
            raise RuntimeError(f"external llama-server slots mismatch: {server_slots} != {expected_slots}")
        settings = props.get("default_generation_settings")
        slot_ctx = settings.get("n_ctx") if isinstance(settings, dict) else None
        total_ctx = props.get("n_ctx")
        if slot_ctx is None and total_ctx is None:
            raise RuntimeError("external llama-server /props has no n_ctx; refusing to reuse")
        if slot_ctx is not None and int(slot_ctx) != expected_ctx:
            raise RuntimeError(f"external llama-server per-slot ctx mismatch: {slot_ctx} != {expected_ctx}")
        if total_ctx is not None and int(total_ctx) != int(self.args.ctx):
            raise RuntimeError(f"external llama-server ctx mismatch: {total_ctx} != {int(self.args.ctx)}")

    def _abort_server(self, reason: str):
        proc = self._server
        if proc is not None and proc != "external":
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        log = getattr(self, "_server_log", None)
        if log is not None:
            try:
                log.close()
            except Exception:
                pass
            self._server_log = None
        self._server = None
        raise RuntimeError(reason)

    def ensure_model(self):
        all_healthy = all(self._health(p) for p in self.ports)
        if self._server == "external" and all_healthy and self.model is not None and self._tokenizer is not None:
            return self.model
        if self._server == "external":
            if not all_healthy:
                raise RuntimeError("external llama-server is no longer healthy on all ports")
        elif all_healthy and len(self.ports) > 1:
            self._server = "external"
        elif self._server is not None:
            exit_code = self._server.poll()
            if exit_code is not None or not self._health():
                self._abort_server(
                    f"managed llama-server is not healthy (exit={exit_code})\n" + self._log_tail())
        elif self._health():
            props = self._props()
            if props is None:
                raise RuntimeError("listener returned 200 on /health but no /props; refusing to reuse")
            self._verify_external(props)
            self._server = "external"
            self._external_props = props
        if self._server is None:
            log_path = Path(self.args.out) / "llama_server.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("ab")
            self._server_log = log
            self._server = subprocess.Popen(
                [self.args.server_bin, "-m", self.args.gguf,
                 "--host", SERVER_HOST, "--port", str(self.args.port),
                 "-ngl", "99", "--tensor-split", "1,1",
                 "-np", str(self.args.slots), "-c", str(self.args.ctx),
                 "-fa", ("on" if self.args.flash_attn else "off"),
                 "--no-webui"],
                stdout=log, stderr=subprocess.STDOUT)
            print(json.dumps({"status": "loading_model", "engine": ENGINE_NAME,
                              "gguf": self.args.gguf, "slots": int(self.args.slots),
                              "ctx": int(self.args.ctx), "port": int(self.args.port)}), flush=True)
            started = time.time()
            while not self._health():
                exit_code = self._server.poll()
                if exit_code is not None:
                    self._abort_server(f"llama-server exited with code {exit_code}\n"
                                       + self._log_tail())
                if time.time() - started >= 900:
                    self._abort_server("llama-server did not become healthy within 900s\n"
                                       + self._log_tail())
                time.sleep(2)
        if self.model is None or self._tokenizer is None:
            identity = self.manifest.payload["identity"]
            if self._server == "external":
                props = getattr(self, "_external_props", None) or {}
                build = props.get("build_info")
                identity["engine"]["version"] = str(build) if build is not None else None
                identity["engine"]["external"] = True
            else:
                identity["engine"]["version"] = self._server_version()
            identity["engine"]["gguf_sha256"] = self._gguf_sha256(Path(self.args.gguf))
            self.manifest.payload["config"]["engine"] = dict(identity["engine"])
            self.manifest.write()
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
            self.model = "llamacpp-server"
        return self.model

    # --------------------------------------------------------------- generation

    def _llamacpp_generate(self, requests: list[dict], budget: int, thinking: bool,
                           timings: list[dict], mode: str, attempt: int,
                           batch_failed: bool = False) -> list[dict]:
        self.ensure_model()
        prompts = []
        for req in requests:
            messages = base._with_schema_contract(req["messages"], req["schema"])
            prompts.append(self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=bool(thinking)))

        def call(prompt, port):
            body = {
                "prompt": prompt,
                "n_predict": int(budget),
                "temperature": 0.0,
                "top_k": 0,
                "top_p": 1.0,
                "repeat_penalty": 1.0,
                "seed": int(self.args.seed),
                "cache_prompt": True,
                "stream": False,
            }
            request = urllib.request.Request(
                f"http://{SERVER_HOST}:{port}/completion",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=1800) as response:
                    data = json.loads(response.read().decode("utf-8"))
                elapsed = time.perf_counter() - started
                raw = data.get("content", "")
                meta = data.get("timings", {})
                prompt_n = meta.get("prompt_n", 0)
                predicted_n = meta.get("predicted_n", data.get("tokens_predicted", 0))
                finish_reason = "length" if predicted_n >= int(budget) else "stop"
                return raw, int(prompt_n), int(predicted_n), finish_reason, elapsed, None
            except Exception as exc:
                elapsed = time.perf_counter() - started
                return "", 0, 0, "error", elapsed, str(exc)

        started = time.perf_counter()
        workers = max(1, min(len(requests), int(self.args.slots) * len(self.ports)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(call, prompt, self.ports[i % len(self.ports)])
                       for i, prompt in enumerate(prompts)]
            responses = [future.result() for future in futures]
        wall = time.perf_counter() - started

        results: list[dict] = []
        per_row: list[dict] = []
        total_prompt = total_generated = 0
        for req, (raw, prompt_n, predicted_n, finish_reason, elapsed, err) in zip(requests, responses):
            total_prompt += prompt_n
            total_generated += predicted_n
            if err is not None:
                payload, parse_status, parse_error = None, "failed", err
                truncated = False
            else:
                payload, parse_status, parse_error = self._parse_result(mode, raw, req["schema"])
                truncated = finish_reason == "length" or (int(budget) and predicted_n >= int(budget))
            per_row.append({"prompt_tokens": prompt_n, "generated": predicted_n,
                            "elapsed": elapsed})
            results.append({"chunk_id": req["chunk_id"], "raw": raw, "payload": payload,
                            "parse_status": parse_status, "parse_error": parse_error,
                            "finish_reason": finish_reason, "generated_tokens": int(predicted_n),
                            "truncated": bool(truncated),
                            "elapsed_s": round(elapsed, 3)})

        batch = len(requests) > 1
        for req, res, row in zip(requests, results, per_row):
            elapsed = row["elapsed"]
            timing = {
                "ts": base.now_iso(), "chunk_id": req["chunk_id"], "mode": mode, "attempt": attempt,
                "thinking": bool(thinking), "elapsed_s": round(elapsed, 3),
                "wall_s": round(wall, 3), "batch_size": len(requests),
                "prompt_tokens": row["prompt_tokens"], "generated_tokens": row["generated"],
                "tokens_per_second": (round(row["generated"] / elapsed, 2) if elapsed > 0 else None),
                "finish_reason": res["finish_reason"],
                "error": None if res["parse_status"] != "failed" else res["parse_error"],
                "tolerant_parse": res["parse_status"] == "recovered",
                "parse_status": res["parse_status"],
                "truncated": res["truncated"],
                "engine": ENGINE_NAME,
                "peak_memory_bytes": None, "active_memory_bytes": None,
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
        return self._llamacpp_generate([req], budget, thinking, timings, mode, attempt,
                                       batch_failed=batch_failed)[0]

    def _generate_batch(self, requests: list[dict], budget: int, thinking: bool,
                        timings: list[dict], mode: str, attempt: int) -> list[dict]:
        return self._llamacpp_generate(requests, budget, thinking, timings, mode, attempt)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="llama.cpp (CUDA) port of the v3 quote-first extraction runner.",
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
                        help="prompts per wave; >1 issues concurrent HTTP requests")
    parser.add_argument("--slots", type=int, default=8,
                        help="llama-server parallel slots (-np); also max concurrent HTTP requests")
    parser.add_argument("--ctx", type=int, default=65536,
                        help="llama-server total context (-c); per-slot = ctx/slots")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="llama-server HTTP port")
    parser.add_argument("--ports", type=str, default=None,
                        help="comma-separated ports for multi-server parallelization (e.g. 8081,8082)")
    parser.add_argument("--server-bin", default=LLAMA_SERVER_BIN, help="path to llama-server")
    parser.add_argument("--gguf", default=GGUF_PATH, help="path to the Q4_K_M GGUF")
    parser.add_argument("--flash-attn", type=int, choices=[0, 1], default=1,
                        help="llama-server flash attention (-fa)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    runner = LlamaCppRunner(args)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
