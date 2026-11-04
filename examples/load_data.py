"""Minimal example: load the dataset, the evaluation set and the graph."""

import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Dataset
pairs = pd.read_csv(ROOT / "data" / "hindi_hinglish_counter_narratives_1550.csv")
print(f"{len(pairs)} pairs")
print(pairs["Category"].value_counts().to_string(), "\n")
print("Example pair:")
print("  Text:", pairs.loc[0, "Text"])
print("  Counter narrative:", pairs.loc[0, "Counter Narrative"], "\n")

# Evaluation set used in the paper
eval_ids = (ROOT / "data" / "eval_test141_ids.txt").read_text(encoding="utf-8").split()
print(f"{len(eval_ids)} evaluation posts")

# Knowledge graph: claims with exact source offsets
claims = [json.loads(line) for line in open(ROOT / "kg" / "claims.jsonl", encoding="utf-8")]
print(f"{len(claims)} claims; first: {claims[0]['text'][:80]!r}")
edges = Counter(json.loads(line)["type"] for line in open(ROOT / "kg" / "edges.jsonl", encoding="utf-8"))
print("edge types:", dict(edges))

# Source documents
manifest = pd.read_csv(ROOT / "kg" / "source_manifest.csv")
print(manifest.groupby("source_type").size().to_string())
