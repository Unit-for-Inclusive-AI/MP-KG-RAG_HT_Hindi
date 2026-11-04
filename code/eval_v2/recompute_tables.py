"""Recompute the paper's auto-table values from archived outputs/scores.

Calibration-only: reads archived artifacts, re-runs the frozen scorer over the
archived 141-ID inputs, recomputes BERTScore means, retrieval summaries, and
the production-checkpoint aggregates, then diffs every value against what was
recorded. Writes `.audit`-style reports only; never touches the archives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
HATE_ROOT = REPO_ROOT.parent / "Hate_Speech_Counter_Narrative_RAG_Project"

from . import composition_v2 as comp

SCORER_SHA_EXPECTED = "d770e9e28dc391b8a0bc3be044fd605e0e9503b7060fb5ad295c4f3641c451ee"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize_score_json(doc: dict[str, Any]) -> dict[str, Any]:
    d = json.loads(json.dumps(doc))
    d["run_dir"] = "<run-dir>"
    if "provenance" in d and isinstance(d["provenance"].get("input_file_sha256"), dict):
        vals = sorted(d["provenance"]["input_file_sha256"].values())
        d["provenance"]["input_file_sha256"] = {"<sorted>": vals}
    return d


def deep_diff(a: Any, b: Any, path: str = "") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append({"path": f"{path}/{k}", "a": a.get(k), "b": b.get(k), "kind": "missing"})
            else:
                out.extend(deep_diff(a[k], b[k], f"{path}/{k}"))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append({"path": path, "a": f"len={len(a)}", "b": f"len={len(b)}", "kind": "len"})
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out.extend(deep_diff(x, y, f"{path}[{i}]"))
    else:
        same = a == b
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            same = abs(a - b) <= 0.0
        if not same:
            out.append({"path": path, "a": a, "b": b, "kind": "value"})
    return out


def section_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    d = Path(args.checkpoint_dir)
    audit = d / "quality_audit_rows.jsonl"
    inputs = comp.load_json(d / "evaluation_inputs.json")
    recorded_sha = inputs["provenance"]["quality_audit_rows_sha256"]
    actual_sha = sha256_file(audit)
    fresh = comp.checkpoint_aggregates(audit)
    rows = comp.compare_checkpoint_aggregates(fresh, inputs)
    consistency = comp.adequacy_reason_consistency(audit)
    return {
        "audit_rows_sha256": actual_sha,
        "recorded_sha256": recorded_sha,
        "sha_match": actual_sha == recorded_sha,
        "recomputed_variants": fresh,
        "comparisons": rows,
        "adequacy_reason_consistency": consistency,
        "recomputed_metrics": sorted(comp.CHECKPOINT_METRICS),
        "not_recomputed": {
            "per_row_rubric": "adequacy_pass and the per-metric row booleans are inputs from the remote evaluator (code not in this repo); this run re-aggregates them and verifies pass<->reasons consistency",
            "metrics_hash": "the evaluator's identity-hash preimage is not documented offline",
        },
    }


def section_retrieval(args: argparse.Namespace) -> dict[str, Any]:
    d = Path(args.retrieval_dir)
    fresh = comp.retrieval_summaries(d / "ranked_passages.csv")
    recorded = comp.parse_retrieval_metrics_csv(d / "metrics.csv")
    rows = comp.compare_retrieval_summaries(fresh, recorded)
    paper = comp.compare_retrieval_paper_claims(fresh)
    return {
        "recomputed": {f"{k[0]} :: {k[1]}": v for k, v in fresh.items()},
        "comparisons": rows,
        "paper_claims": paper,
    }


def section_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    d = Path(args.diagnostic_dir)
    archived = comp.load_json(d / "score_corrected.json")
    md_claims = comp.parse_score_corrected_md((d / "score_corrected.md").read_text())

    result: dict[str, Any] = {
        "archived_scorer_sha256": archived["provenance"]["scorer_sha256"],
        "md_claims_count": len(md_claims),
    }

    fresh = archived
    if not args.no_scorer_rerun:
        scorer = Path(args.scorer)
        actual_sha = sha256_file(scorer)
        tmp = Path(tempfile.mkdtemp(prefix="ev2_recompute_", dir=args.tmp_root))
        try:
            for name in ("manifest.json", "kg_rag.jsonl", "mp_kg_rag.jsonl",
                         "qwen_cot.jsonl", "qwen_few_shot.jsonl", "qwen_zero_shot.jsonl"):
                shutil.copy2(d / name, tmp / name)
            cmd = [sys.executable, str(scorer), "--run-dir", str(tmp),
                   "--bootstrap-draws", str(args.bootstrap_draws), "--seed", str(args.seed)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            result["scorer_rerun"] = {
                "cmd": " ".join(cmd),
                "returncode": proc.returncode,
                "scorer_sha256": actual_sha,
                "scorer_sha256_matches_archive": actual_sha == archived["provenance"]["scorer_sha256"],
                "scorer_sha256_matches_pin": actual_sha == SCORER_SHA_EXPECTED,
            }
            if proc.returncode != 0:
                result["scorer_rerun"]["stderr_tail"] = proc.stderr[-2000:]
                result["scorer_rerun"]["ok"] = False
                return result
            result["scorer_rerun"]["ok"] = True
            fresh = comp.load_json(tmp / "score_corrected.json")
            diffs = deep_diff(normalize_score_json(archived), normalize_score_json(fresh))
            result["archive_vs_fresh_diff_count"] = len(diffs)
            result["archive_vs_fresh_diffs"] = diffs[:50]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    table_fresh = comp.five_system_table(fresh)
    result["fresh_table"] = table_fresh
    result["md_comparisons"] = comp.compare_five_system_table(table_fresh, md_claims)

    bert = comp.load_json(d / "bertscore.json")
    bert_fresh = comp.bertscore_means(bert)
    bert_rows = []
    for system, vals in bert_fresh.items():
        stored = bert["systems"].get(system, {}).get("mean_f1")
        bert_rows.append({
            "system": system,
            "recomputed_mean_f1": vals["mean_f1"],
            "recorded_mean_f1": stored,
            "rows": vals["scored_f1_rows"],
            "match": stored is not None and abs(vals["mean_f1"] - stored) <= 1e-12,
        })
    result["bertscore"] = bert_rows
    return result


def summarize(section_rows: list[dict[str, Any]]) -> dict[str, int]:
    matched = sum(1 for r in section_rows if r.get("match") is True)
    unmatched = sum(1 for r in section_rows if r.get("match") is not True)
    return {"matched": matched, "unmatched": unmatched, "total": len(section_rows)}


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = ["# Calibration report — recomputation vs recorded values", ""]
    ckpt = report["checkpoint"]
    lines.append("## A. Production checkpoint (1550 rows, quality_audit_rows.jsonl)")
    lines.append("")
    lines.append(f"- audit-rows sha256 match: **{ckpt['sha_match']}**")
    lines.append(f"- row-consistency (adequacy_pass == failure_reasons empty): checked {ckpt['adequacy_reason_consistency']['checked_rows']}, mismatches {ckpt['adequacy_reason_consistency']['mismatch_count']}")
    s = summarize(ckpt["comparisons"])
    lines.append(f"- aggregate comparisons: {s['matched']}/{s['total']} matched")
    unm = [r for r in ckpt["comparisons"] if not r.get("match")]
    if unm:
        lines.append("")
        lines.append("| variant | metric | recomputed | recorded |")
        lines.append("|---|---|---:|---:|")
        for r in unm:
            lines.append(f"| {r['variant']} | {r['metric']} | {r['recomputed']!r} | {r['recorded']!r} |")
    lines.append("")
    lines.append("Not recomputed: per-row rubric booleans (evaluator code lived on the remote VM); metrics hash preimage.")
    lines.append("")

    ret = report["retrieval"]
    lines.append("## B. Retrieval table (100-post local run)")
    lines.append("")
    s = summarize(ret["comparisons"])
    lines.append(f"- ranked_passages.csv -> metrics.csv recompute: {s['matched']}/{s['total']} matched")
    pc = ret["paper_claims"]["measured_vs_paper"]
    s2 = summarize(pc)
    lines.append(f"- measured Auth./Cov. vs paper (rounded claims): {s2['matched']}/{s2['total']} matched")
    f1 = ret["paper_claims"]["f1_from_assumed_pr"]
    for r in f1:
        lines.append(f"- F1@5 from assumed P/R {r['retriever']}: recomputed {r['recomputed']:.4f} vs paper {r['paper']} -> {'match' if r['match'] else 'MISMATCH'}")
    lines.append("")
    lines.append("Script rows: paper values are author assumptions for the full retriever; the local run measured:")
    lines.append("")
    lines.append("| script | measured Auth. | measured Cov.% | paper (assumed) |")
    lines.append("|---|---:|---:|---|")
    for r in ret["paper_claims"]["script_measured_vs_assumed"]:
        if r["metric"] == "auth":
            continue
        auth_r = next(x for x in ret["paper_claims"]["script_measured_vs_assumed"]
                      if x["script"] == r["script"] and x["metric"] == "auth")
        lines.append(f"| {r['script']} | {auth_r['recomputed']:.4f} | {r['recomputed']:.2f} | Auth. {auth_r['paper_assumed']}, Cov. {r['paper_assumed']}% |")
    lines.append("")

    diag = report["diagnostic"]
    lines.append("## C. 141-ID diagnostic (five-system metrics + BERTScore)")
    lines.append("")
    if "scorer_rerun" in diag:
        rr = diag["scorer_rerun"]
        lines.append(f"- frozen scorer rerun: ok={rr.get('ok')} rc={rr['returncode']} sha-matches-archive={rr['scorer_sha256_matches_archive']} sha-matches-pin={rr['scorer_sha256_matches_pin']}")
        if "archive_vs_fresh_diff_count" in diag:
            lines.append(f"- archived vs fresh score_corrected.json diffs (exact-compare, path-normalized): **{diag['archive_vs_fresh_diff_count']}**")
    s = summarize(diag["md_comparisons"])
    lines.append(f"- fresh scores vs score_corrected.md table (4dp tolerance): {s['matched']}/{s['total']} matched")
    unm = [r for r in diag["md_comparisons"] if not r.get("match")]
    if unm:
        lines.append("")
        lines.append("| system | metric | recomputed | recorded |")
        lines.append("|---|---|---:|---:|")
        for r in unm:
            lines.append(f"| {r['system']} | {r['metric']} | {r['recomputed']} | {r['recorded']} |")
    s = summarize(diag["bertscore"])
    lines.append(f"- BERTScore mean-F1 recompute from row_scores: {s['matched']}/{s['total']} matched (1e-12)")
    lines.append("")
    lines.append("## Scope notes")
    lines.append("")
    lines.append("- The manuscript draft (sigconf.tex) currently contains none of the five-system 141-ID metrics; the recorded reference is `score_corrected.md`.")
    lines.append("- MP-arm diagnostic values are from the mixed pilot (theta 0.05, rerank_pool 64, max_evidence 3, concise variant) — exploratory, not the production contract.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path,
                        default=HATE_ROOT / ".audit/table9_test141_diagnostic")
    parser.add_argument("--retrieval-dir", type=Path,
                        default=HATE_ROOT / "outputs/2026-10-16-retrieval-table10")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=REPO_ROOT / "production_checkpoint_remote")
    parser.add_argument("--scorer", type=Path,
                        default=REPO_ROOT / "outputs/2026-10-17-mlx-corrected/score_corrected.py")
    parser.add_argument("--out", type=Path, default=HATE_ROOT / ".audit/eval_v2_20261019")
    parser.add_argument("--bootstrap-draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-scorer-rerun", action="store_true")
    parser.add_argument("--tmp-root", type=Path, default=None)
    args = parser.parse_args()

    report = {
        "generated_by": "eval_v2/recompute_tables.py",
        "checkpoint": section_checkpoint(args),
        "retrieval": section_retrieval(args),
        "diagnostic": section_diagnostic(args),
    }

    all_rows: list[dict[str, Any]] = []
    all_rows += report["checkpoint"]["comparisons"]
    all_rows += report["retrieval"]["comparisons"]
    all_rows += report["retrieval"]["paper_claims"]["measured_vs_paper"]
    all_rows += report["retrieval"]["paper_claims"]["f1_from_assumed_pr"]
    all_rows += report["diagnostic"].get("md_comparisons", [])
    all_rows += report["diagnostic"].get("bertscore", [])
    if "scorer_rerun" in report["diagnostic"] and report["diagnostic"]["scorer_rerun"].get("ok"):
        report["diagnostic"]["archive_vs_fresh_diff_count"] = report["diagnostic"].get("archive_vs_fresh_diff_count", 0)
    report["summary"] = summarize(all_rows)
    if "archive_vs_fresh_diff_count" in report["diagnostic"]:
        report["summary"]["archive_vs_fresh_diffs"] = report["diagnostic"]["archive_vs_fresh_diff_count"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (out / "calibration_report.md").write_text(render_markdown(report))
    print(json.dumps(report["summary"], indent=2))
    unmatched = report["summary"]["unmatched"]
    if report["summary"].get("archive_vs_fresh_diffs"):
        unmatched += report["summary"]["archive_vs_fresh_diffs"]
    if report["checkpoint"]["adequacy_reason_consistency"]["mismatch_count"]:
        unmatched += report["checkpoint"]["adequacy_reason_consistency"]["mismatch_count"]
    sys.exit(3 if unmatched else 0)


if __name__ == "__main__":
    main()
