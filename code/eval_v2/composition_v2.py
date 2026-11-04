"""Deterministic table-value assemblers for the MP-arm results tables.

Pure functions only: no model calls, no file writes, no network. Consumes the
archived score artifacts (score JSON, BERTScore JSON, retrieval CSVs, audit
JSONL) and produces the values that populate the paper's auto-tables, in the
same formatting conventions used by the archived outputs.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

SYSTEMS_FIVE = ("qwen_zero_shot", "qwen_few_shot", "qwen_cot", "kg_rag", "mp_kg_rag")

SCORE_MD_COLUMNS = [
    ("bleu1_raw", "bleu1_raw_sacrebleu_0_100"),
    ("bleu1_norm", "bleu1_normalized_sacrebleu_0_100"),
    ("rouge_l", "rouge_l_mean_f1_0_1"),
    ("meteor_exact", "meteor_exact_mean_0_1"),
    ("chrfpp", "chrfpp_sacrebleu_0_100"),
    ("distinct1", "distinct1"),
    ("distinct2", "distinct2"),
    ("self_bleu4", "self_bleu4_mean_sacrebleu_0_100"),
    ("mean_raw_words", "raw_word_length_mean"),
]

SCORE_MD_COUNT_COLUMNS = [
    ("rows", "rows"),
    ("success", "success"),
    ("citation_errors", "citation_validation_failures"),
    ("strict_success", "citation_valid_success"),
    ("failure", "failure"),
    ("abstention", "abstention"),
    ("coverage", "row_coverage"),
]


def harmonic_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def parse_score_corrected_md(text: str) -> dict[str, dict[str, float]]:
    """Extract the per-system metric table recorded in score_corrected.md."""
    out: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 17 or cells[0] not in SYSTEMS_FIVE:
            continue
        system = cells[0]
        row: dict[str, float] = {}
        for idx, (name, _) in enumerate(SCORE_MD_COUNT_COLUMNS, start=1):
            row[name] = float(cells[idx])
        for idx, (name, _) in enumerate(SCORE_MD_COLUMNS, start=8):
            row[name] = float(cells[idx])
        out[system] = row
    return out


def five_system_table(score_json: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Recompute the published 5-system table values from a score_corrected.json."""
    table: dict[str, dict[str, float]] = {}
    for system in SYSTEMS_FIVE:
        info = score_json["systems"][system]
        row: dict[str, float] = {}
        for name, key in SCORE_MD_COUNT_COLUMNS:
            row[name] = float(info["counts"][key])
        for name, key in SCORE_MD_COLUMNS:
            row[name] = float(info["all_successful_metrics"][key])
        table[system] = row
    return table


def compare_five_system_table(
    fresh: dict[str, dict[str, float]],
    recorded: dict[str, dict[str, float]],
    *,
    tol_counts: float = 0.0,
    tol_metrics: float = 5e-5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_names = {name for name, _ in SCORE_MD_COLUMNS}
    for system in SYSTEMS_FIVE:
        for name in list(dict(SCORE_MD_COUNT_COLUMNS)) + list(dict(SCORE_MD_COLUMNS)):
            a = fresh.get(system, {}).get(name)
            b = recorded.get(system, {}).get(name)
            if a is None or b is None:
                rows.append({"system": system, "metric": name, "recomputed": a, "recorded": b,
                             "match": False, "note": "missing value"})
                continue
            tol = tol_metrics if name in metric_names else tol_counts
            rows.append({"system": system, "metric": name, "recomputed": a, "recorded": b,
                         "abs_diff": abs(a - b), "match": abs(a - b) <= tol})
    return rows


def bertscore_means(bertscore_json: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Recompute per-system mean F1 from the row_scores cache (scorer's own filter:
    rows with a numeric f1)."""
    out: dict[str, dict[str, float]] = {}
    for system, rows in bertscore_json["row_scores"].items():
        values = [float(r["f1"]) for r in rows.values() if r.get("f1") is not None]
        out[system] = {
            "mean_f1": sum(values) / len(values) if values else float("nan"),
            "scored_f1_rows": float(len(values)),
        }
    return out


RETRIEVAL_GROUP_ORDER = (
    ("BM25", "All"),
    ("BM25", "Romanized Hinglish"),
    ("BM25", "Devanagari Hindi"),
    ("Dense (BGE-M3)", "All"),
    ("Dense (BGE-M3)", "Romanized Hinglish"),
    ("Dense (BGE-M3)", "Devanagari Hindi"),
    ("Dense + BM25", "All"),
    ("Dense + BM25", "Romanized Hinglish"),
    ("Dense + BM25", "Devanagari Hindi"),
    ("+ Local mention graph", "All"),
    ("+ Local mention graph", "Romanized Hinglish"),
    ("+ Local mention graph", "Devanagari Hindi"),
)


def _summarize_passages(grow: list[dict[str, str]]) -> dict[str, float]:
    posts: dict[str, list[dict[str, str]]] = {}
    for r in grow:
        posts.setdefault(r["post_id"], []).append(r)
    auth = sum(float(r["authority"]) for r in grow) / len(grow)
    retained = [r for r in grow if r["above_theta"].strip() == "True"]
    covered = sum(1 for pg in posts.values() if any(r["above_theta"].strip() == "True" for r in pg))
    return {
        "n_posts": float(len(posts)),
        "auth_top5_before_threshold": auth,
        "cov_theta": covered / len(posts),
        "covered_posts": float(covered),
        "retrieved_passages": float(len(grow)),
        "retained_passages": float(len(retained)),
    }


def retrieval_summaries(ranked_passages_csv: Path, *, theta: float = 0.55) -> dict[tuple[str, str], dict[str, float]]:
    """Recompute the 12 authority/coverage summaries from ranked passage rows,
    including the cross-script 'All' aggregate per retriever."""
    rows = list(csv.DictReader(ranked_passages_csv.open()))
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for r in rows:
        groups.setdefault((r["retriever"], r["script"]), []).append(r)
    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, grow in groups.items():
        out[key] = _summarize_passages(grow)
    for retriever in sorted({k[0] for k in groups}):
        grow = [r for r in rows if r["retriever"] == retriever]
        out[(retriever, "All")] = _summarize_passages(grow)
    return out


def parse_retrieval_metrics_csv(metrics_csv: Path) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    for r in csv.DictReader(metrics_csv.open()):
        out[(r["retriever"], r["script"])] = {
            "auth_top5_before_threshold": float(r["Auth_top5_before_threshold"]),
            "cov_theta": float(r["Cov_theta_0.55"]),
            "covered_posts": float(r["covered_posts"]),
            "retrieved_passages": float(r["retrieved_passages"]),
            "retained_passages": float(r["retained_passages"]),
        }
    return out


def compare_retrieval_summaries(
    fresh: dict[tuple[str, str], dict[str, float]],
    recorded: dict[tuple[str, str], dict[str, float]],
    *,
    tol: float = 1e-9,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in RETRIEVAL_GROUP_ORDER:
        for metric in ("auth_top5_before_threshold", "cov_theta", "covered_posts",
                       "retrieved_passages", "retained_passages"):
            a = fresh.get(key, {}).get(metric)
            b = recorded.get(key, {}).get(metric)
            rows.append({
                "retriever": key[0], "script": key[1], "metric": metric,
                "recomputed": a, "recorded": b,
                "abs_diff": abs(a - b) if a is not None and b is not None else None,
                "match": a is not None and b is not None and abs(a - b) <= tol,
            })
    return rows


PAPER_RETRIEVAL_HEADLINE = {
    "BM25": {"auth": 0.879, "cov_pct": 3.0},
    "Dense (BGE-M3)": {"auth": 0.909, "cov_pct": 12.0},
    "Dense + BM25": {"auth": 0.909, "cov_pct": 12.0},
}

PAPER_RETRIEVAL_ASSUMED_F1 = {
    ("BM25", 0.41, 0.38): 0.394,
    ("Dense (BGE-M3)", 0.49, 0.45): 0.469,
    ("Dense + BM25", 0.55, 0.51): 0.529,
    ("+ Graph (full)", 0.57, 0.53): 0.549,
}

PAPER_RETRIEVAL_ASSUMED_SCRIPT = {
    "Romanized Hinglish": {"auth": 0.905, "cov_pct": 5.0},
    "Devanagari Hindi": {"auth": 0.925, "cov_pct": 34.0},
}


def compare_retrieval_paper_claims(
    fresh: dict[tuple[str, str], dict[str, float]],
    *,
    tol_auth: float = 5e-4,
    tol_cov_pct: float = 5e-2,
) -> dict[str, list[dict[str, Any]]]:
    measured: list[dict[str, Any]] = []
    for retriever, claim in PAPER_RETRIEVAL_HEADLINE.items():
        a = fresh.get((retriever, "All"), {})
        if not a:
            continue
        measured.append({"retriever": retriever, "metric": "auth", "recomputed": a["auth_top5_before_threshold"],
                         "paper": claim["auth"], "match": abs(a["auth_top5_before_threshold"] - claim["auth"]) <= tol_auth})
        cov_pct = a["cov_theta"] * 100.0
        measured.append({"retriever": retriever, "metric": "cov_pct", "recomputed": cov_pct,
                         "paper": claim["cov_pct"], "match": abs(cov_pct - claim["cov_pct"]) <= tol_cov_pct})
    f1_rows: list[dict[str, Any]] = []
    for (name, p, r), claimed in PAPER_RETRIEVAL_ASSUMED_F1.items():
        v = harmonic_f1(p, r)
        f1_rows.append({"retriever": name, "metric": "F1_at_5_from_assumed_PR", "recomputed": v,
                        "paper": claimed, "match": abs(v - claimed) <= 5e-4})
    script_rows: list[dict[str, Any]] = []
    for script, claim in PAPER_RETRIEVAL_ASSUMED_SCRIPT.items():
        a = fresh.get(("+ Local mention graph", script), {})
        if not a:
            continue
        script_rows.append({"retriever": "+ Local mention graph", "script": script, "metric": "auth",
                            "recomputed": a["auth_top5_before_threshold"], "paper_assumed": claim["auth"],
                            "note": "paper value is an author assumption for the full retriever, not this local run"})
        script_rows.append({"retriever": "+ Local mention graph", "script": script, "metric": "cov_pct",
                            "recomputed": a["cov_theta"] * 100.0, "paper_assumed": claim["cov_pct"],
                            "note": "paper value is an author assumption for the full retriever, not this local run"})
    return {"measured_vs_paper": measured, "f1_from_assumed_pr": f1_rows, "script_measured_vs_assumed": script_rows}


CHECKPOINT_METRICS = ("adequacy_pass", "citation_validity", "grounding", "perspective_coverage", "chrf")


def checkpoint_aggregates(audit_rows_path: Path) -> dict[str, dict[str, float]]:
    """Recompute per-variant metric means from the per-row audit JSONL."""
    sums: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    with audit_rows_path.open() as f:
        for line in f:
            row = json.loads(line)
            variant = row["variant"]
            acc = sums.setdefault(variant, {m: 0.0 for m in CHECKPOINT_METRICS})
            counts[variant] = counts.get(variant, 0) + 1
            for m in CHECKPOINT_METRICS:
                acc[m] += float(row[m])
    out: dict[str, dict[str, float]] = {}
    for variant, acc in sums.items():
        out[variant] = {m: acc[m] / counts[variant] for m in CHECKPOINT_METRICS}
        out[variant]["rows"] = float(counts[variant])
    return out


def adequacy_reason_consistency(audit_rows_path: Path, *, limit: int | None = None) -> dict[str, Any]:
    """Verify adequacy_pass == (failure_reasons == []) row by row."""
    checked = 0
    mismatches: list[str] = []
    with audit_rows_path.open() as f:
        for line in f:
            row = json.loads(line)
            reasons = row["adequacy_failure_reasons"]
            if isinstance(reasons, str):
                reasons = json.loads(reasons)
            expected = not bool(reasons)
            if bool(row["adequacy_pass"]) != expected:
                mismatches.append(row["ID"])
            checked += 1
            if limit and checked >= limit:
                break
    return {"checked_rows": checked, "mismatches": mismatches[:20], "mismatch_count": len(mismatches)}


CHECKPOINT_VARIANT_ALIASES = {
    "zero-shot": "zero-shot",
    "few-shot": "few-shot",
    "KG-RAG": "KG-RAG",
    "MP-KG-RAG": "MP-KG-RAG",
}


def compare_checkpoint_aggregates(
    fresh: dict[str, dict[str, float]],
    recorded_inputs: dict[str, Any],
    *,
    tol: float = 1e-12,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, alias in CHECKPOINT_VARIANT_ALIASES.items():
        rec = recorded_inputs["variants"].get(alias)
        if rec is None:
            rows.append({"variant": alias, "metric": "*", "recomputed": None, "recorded": None,
                         "match": False, "note": "variant missing from evaluation_inputs.json"})
            continue
        for metric in CHECKPOINT_METRICS:
            if metric not in rec:
                continue
            a = fresh.get(variant, {}).get(metric)
            b = float(rec[metric]["value"])
            rows.append({"variant": alias, "metric": metric, "recomputed": a, "recorded": b,
                         "abs_diff": abs(a - b) if a is not None else None,
                         "match": a is not None and abs(a - b) <= tol})
        n = rec.get("composite", {}).get("denominator")
        a_rows = fresh.get(variant, {}).get("rows")
        rows.append({"variant": alias, "metric": "rows", "recomputed": a_rows, "recorded": float(n) if n else None,
                     "match": n is not None and a_rows == float(n)})
    return rows


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


_WS = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()
