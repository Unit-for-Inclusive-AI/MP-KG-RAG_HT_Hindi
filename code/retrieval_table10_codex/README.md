# Local retrieval evaluation for Table 10

This is a **new local diagnostic evaluation**, not recovery of the paper's original
100 annotated posts. No passage relevance grades have been invented.

## Sample

- `sample_100.json` freezes 100 IDs from the 141 Homophobic records in the saved
  `Test` sheet of `dataset_split_trans_replaced.xlsx`.
- Seed 42, proportional stratification by script and target, with largest-remainder
  allocation. 62 Latin-script records and 38 records containing Devanagari.
- The script buckets use a Unicode rule, not human language identification.
  Latin-script records are labelled Romanized Hinglish to match the requested table.
- 90 unique input texts occur among these 100 dataset IDs. Duplicate text records
  remain because the unit of the saved split is a dataset ID.
- Targets: Lesbians 23, Bisexual Individuals 18, Gay Men 16, Pride Community 16,
  Same-sex Marriage 16, LGBTQ+ Community 11.
- Split-workbook SHA-256:
  `146adcc0574d5fb34e66a523af3f62038cdbccec128bf5a65a83291f5d78f2b8`.

## Retrieval protocol

Reuse the immutable local generation
`20260831T184337874072Z-f068d13b`, containing 90 documents and 6,431 passages.
Validate its manifest, source hashes, and structural integrity before use.
The index uses canonical `chunks.json`, because its manifest reports a stale,
misaligned older evidence ledger. Do not join to that older ledger by row position.

Queries concatenate the saved input text, target, and category without LLM query
rewriting. Dense retrieval uses pinned BGE-M3
`5617a9f61b028005a4858fdac845db406aefb181` with normalized embeddings and 512-token
maximum input length. Corpus embeddings already exist. BM25 uses SQLite FTS5's
BM25 implementation and its existing tokenizer. The dense candidate floor is
0.25. Candidate counts are 12 for dense, 12 for BM25, and 8 for the graph branch.

The local index configuration uses weighted RRF, constant 60 and weights dense
1.1, BM25 1.0, graph 0.8. This differs from the equal-weight equation in the paper.
The four configurations retain the native or fused top five candidates. Those
candidates receive common cross-encoder scores from pinned BGE-reranker-v2-M3
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, with 512-token maximum pair length.
Always apply sigmoid to raw reranker logits, including logits between zero and one.

Coverage is the fraction of the sample with at least one of those five candidates
having reranker probability >= 0.55. Scores on other branches, cosine similarities,
and unnormalized BM25 scores are not compared to this threshold.

Authority is supplied by the stored source-type policy; it is not a human judgment
of factual correctness. Report both the mean authority of the retrieved top five
before thresholding and of the retained passages after thresholding. These have
different denominators. Each passage occurrence across posts contributes equally.

## Graph limitation

The available local index has entity and mention nodes but **zero structured claim
nodes**. Its implemented graph branch links exact canonical entity names in the
query to mention/evidence paths within two hops. In this sample only `lgbtq` links,
in 11 posts. It is labelled `+ Local mention graph` throughout. Its results must
not replace the paper's `+ Graph (full)` claim-graph results. The weighted fusion,
query construction, and index limitations must also be disclosed for other rows
if reporting this new experiment in the paper.

## Files and remaining annotations

- `bm25_native_top5.csv`: BM25 top-five ranks and source authority, available
  independently of neural model downloads.
- `metrics.csv`: automatic authority and threshold coverage results.
- `ranked_passages.csv`: auditable ranks, IDs, authority, raw logits, probabilities,
  and threshold decisions.
- `rankings.json`: native/fused rank order before reranker thresholding.
- `annotation_pool.csv`: deduplicated post/passage pairs, with passage text and
  blank columns for two independent annotators and adjudication. Contains no
  retriever identity or reranker scores to reduce annotation bias.
- `run_manifest.json`: versions, identities, protocol, and execution metadata.

For each pooled passage assign 0 = irrelevant, 1 = related to the topic,
2 = directly relevant to the specific claim. Two annotators must independently
grade every pair. Resolve disagreements in `adjudicated_grade`. Then run:

```bash
python score_annotations.py annotation_pool.csv
```

The scorer refuses blanks, disagreements without adjudication, duplicate keys,
and an incomplete or mismatched pool. It computes macro P@5 using grade 2,
pooled R@5 relative to all grade-2 passages in the pool, and nDCG@5 using gains
`2**grade - 1` with log2 rank discount. Zero relevant/positive pool denominators
produce zero scores and are counted separately. This is pooled recall, not recall
against every relevant passage in the complete corpus. Saved rankings are native
or fused top five, so nDCG is explicitly measured on that order, not a later
reranked ordering.

Without those real annotations, P@5, R@5, and nDCG@5 remain **not measured**.
Do not use reranker scores or source authority as replacement relevance labels.

## Run again

```bash
python run_retrieval.py
```

The generator LLM is not loaded or called. Existing source artifacts and the
immutable retrieval index are not rebuilt. All new evaluation files are stored
in this output directory; the paper is not edited by these scripts.
