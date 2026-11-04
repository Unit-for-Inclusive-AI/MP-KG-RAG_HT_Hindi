# rebuild_v2 corpus build report

- tool: `source_refresh/rebuild_v2/build_corpus_v2.py`
- output: `source_refresh/rebuild_v2/semantic_graph_source_v2/` (chunks.json, pages.json, evidence_ledger.json, manifest.json)
- stage_hash: `b4320c599df10b78dcb692eb23ff4adc496eccba32a351bc97cfcf718bee0191`
- stage_hash definition: `sha256(json.dumps({"stage": "rebuild_v2", "chunks.json": H(chunks.json), "pages.json": H(pages.json)}, sort_keys=True))`
- counts: 8,848 chunks / 111 documents / 6,787 pages
- chunker: `bge_T512_O64_snapON` — BGE-M3 tokens, min 500 / target 512 / max 900 / overlap 64, sentence-split with boundary snapping, hard-split of oversized sentences, tail repair (merge or move, both bounded by max 900)
- fidelity: the built chunks are identical (units, page_first/page_last, hard_split, sentence counts, merged_tail flag, sentence-boundary text) to `simulate_rechunk.py` output for all 111 documents; the build run reports no problems.

## Corpus composition

90 baseline documents -> 14 dropped -> 76 kept, plus 35 added documents = 111.

- Dropped (uid): a533552e15d8eaad5f1a, e73ce899d0d017b75b3f, e5cf65fda072f471c943, 5870aa0b966171a20605, 9840d21271d20ff290b3, e9175600c16216faf150, 49677fd1dc33dc2136f1, 5a2c409735f50d18d1f8, f7acb60dab5c6650cd64; dropped (source_id): SRC094, SRC057, SRC035, SRC021, SRC014.
- Added: SRC101-SRC109 (staged batch-1 documents), SRC110-SRC126 (17 raw documents, ordered by manifest id), SRC127-SRC135 (9 v3 documents, alphabetical). ICD-11 chapter is included as the HTML source; the alternate_format_duplicate DOCX version is excluded. The Supriyo document is included.

## Declared conventions

- Text-slice convention (declared overlap convention): for same-page chunks, `text == page_text[char_start:char_end]` (page-local offsets). For cross-page chunks, `text == "\n".join([first_page_text[a:]] + middle_page_texts + [last_page_text[:b]])` — the `all_segments_newline` variant of the acceptance gate. All 5,452 cross-page chunks match this variant; all 3,396 same-page chunks match their page slice.
- Offset mapping: `span_start`/`span_end` (and `doc_char_start`/`doc_char_end`) are document-global; `char_start`/`char_end`/`start_char`/`end_char` are page-local to `page_first`. Evidence-ledger `span_start`/`span_end` equal the chunk values.
- Token fields: `token_start = <BGE tokens of page_first_text[:char_start]>` (page-scoped, mirroring the baseline word-space convention), `token_count` = chunk BGE units, `token_end = token_start + token_count`; `start_token`/`end_token` equal `token_start`/`token_end`.
- `locator = source_locator = relative_path#page=<page_first>`.
- `page_id = sha256(uid + "\x00" + str(page) + "\x00" + sha256(page_text))`. The original page_id formula was not recoverable from the pipeline; this deterministic formula is declared here. `page_sha256 = page_text_sha256 = source_text_sha256 = sha256(page text)`; `source_text_key = "source-text-" + page_id[:20]`.
- Page-number gaps: baseline `pages.json` omits empty pages (16 kept documents have at least one numbering gap). The rebuild inserts empty-text page records for missing interior page numbers so cross-page slices and middle-page lookups are well defined. No chunk references an empty page as `page_first`/`page_last`.
- Duplicates: no duplicate chunk_ids; no duplicate (document_uid, text_sha256) pairs; no duplicate chunk texts. Adjacent-chunk overlap is the declared snap overlap (bounded by 64 units); there are no two chunks that are declared duplicates, and the only known content alternates are the ICD-11 HTML/DOCX pair (DOCX excluded, see above).

## Chunker statistics

- in 500-900 (chunker units): 99.19% [gate recount of stored text: 98.56%]; 460-960: 99.46% [recount: 98.96%]
- median chunk size: 521 units; mid-sentence end rate: 6.76% [recount: 5.93%]
- cross-page chunks: 5,452; hard-split chunks: 98 (78 of them in the data-table document SRC113); tail-merged chunks: 73
- chunks by source_type: court 1,077 / education 28 / government 597 / ngo_report 3,492 / official 1,014 / peer_reviewed 1,464 / survey 111 / un_agency 1,065

## Document-count note for acceptance C8

Documents by source_type: court 5, education 2, government 3, ngo_report 19, official 12, peer_reviewed 49, survey 5, un_agency 16 (total 111, inside the expected [104, 115] window). Versus the source_refresh swap-plan projection (total 104) the per-type deltas are peer_reviewed +6 (bonus/alternate sources acquired after the plan), official -1, un_agency +1, court +1; all other types match. These are classification/availability drifts after the plan was written, not missing or duplicated documents.

## Reproduce

```
cd /Users/pika/Downloads/MP_KG_RAG_v2/source_refresh/rebuild_v2
/opt/homebrew/Caskroom/miniconda/base/bin/python3 build_corpus_v2.py
/opt/homebrew/Caskroom/miniconda/base/bin/python3 accept_corpus_v2.py \
  --corpus /Users/pika/Downloads/MP_KG_RAG_v2/source_refresh/rebuild_v2/semantic_graph_source_v2 \
  --full \
  --out /Users/pika/Downloads/Hate_Speech_Counter_Narrative_RAG_Project/.audit/rebuild_v2_acceptance_20261019
```
