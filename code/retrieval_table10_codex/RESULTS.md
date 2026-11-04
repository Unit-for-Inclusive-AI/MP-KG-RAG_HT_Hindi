# Local retrieval results, 16 October 2026

A new diagnostic run on 100 saved test IDs, with 62 Latin-script and 38 Devanagari records. This is not the original planned human-annotated Table 10.

| Configuration / group | Auth. top five | Coverage at θ = 0.55 | Covered posts |
|---|---:|---:|---:|
| BM25 | 0.8787 | 3.00% | 3/100 |
| Dense (BGE-M3) | 0.9094 | 12.00% | 12/100 |
| Dense + BM25 | 0.9091 | 12.00% | 12/100 |
| + Local mention graph | 0.9091 | 12.00% | 12/100 |
| Local graph: Romanized Hinglish | 0.9056 | 1.61% | 1/62 |
| Local graph: Devanagari Hindi | 0.9147 | 28.95% | 11/38 |

Authority averages the five retrieved candidates before thresholding. Only 3 BM25 and 12 dense/hybrid candidate occurrences pass the reranker threshold in this run. `metrics.csv` also reports authority after thresholding.

**P@5, R@5, and nDCG@5 are not measured.** The 990 pooled post/passage pairs in `annotation_pool.csv` need two independent relevance judgments, with disagreements adjudicated. `score_annotations.py` calculates those metrics once real labels exist.

The local mention graph has zero structured claim nodes. Its top-five lists are identical to hybrid retrieval for all 100 posts, despite a nonempty graph branch on 11 posts. These graph results cannot replace the paper’s full claim-graph row or its script breakdown.

The experiment uses existing local demo settings: weighted RRF and direct input/target/category queries, with 512-token encoder/reranker inputs. These differ from parts of the draft protocol. See `README.md` for exact versions, sample selection, scoring definitions, and limitations.

The low coverage is measured under this exact query, candidate, and threshold protocol. It is not retrieval precision, recall, factuality, or a human quality judgment.

Verification: independently recomputed all 12 authority and coverage summaries from 2,000 ranked passage occurrences; checked sigmoid conversion, sample IDs, annotation-pool completeness, and blank-label protection.
