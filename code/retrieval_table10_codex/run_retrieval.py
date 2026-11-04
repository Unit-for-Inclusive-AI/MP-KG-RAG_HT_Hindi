"""Reproducible local retrieval diagnostics, with no invented relevance labels."""
import csv
import hashlib
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

OUT = Path(__file__).resolve().parent
ROOT = Path('/Users/pika/Downloads/MP_KG_RAG_v2')
CODE = ROOT / '.worktrees/local-mpkg-gradio'
sys.path.insert(0, str(CODE))
from local_mpkg_demo.config import DemoConfig, QueryInput
from local_mpkg_demo.index_store import LocalRetriever, resolve_current_generation
from work.mpkg_rag_core import reciprocal_rank_fusion


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.time()
    torch.set_num_threads(4)
    config = DemoConfig.load(ROOT)
    generation = resolve_current_generation(config)
    manifest = json.loads((generation / 'index_manifest.json').read_text())
    sample = json.loads((OUT / 'sample_100.json').read_text())
    assert len(sample) == len({str(r['ID']) for r in sample}) == 100
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print('Loading pinned BGE-M3 on', device, flush=True)
    encoder = SentenceTransformer(config.embedding_model, revision=config.embedding_revision,
                                  device=device, local_files_only=True)
    encoder.max_seq_length = 512
    if device == 'mps':
        encoder.half()
    queries = [' '.join((r['Text'], r['Target'], r['Category'])) for r in sample]
    qvectors = encoder.encode(queries, batch_size=8, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    np.save(OUT / 'query_embeddings.npy', qvectors)
    del encoder
    if device == 'mps':
        torch.mps.empty_cache()

    # Reuse the immutable, validated generation without rebuilding or mutating it.
    retriever = LocalRetriever.__new__(LocalRetriever)
    retriever.config = config
    retriever.index_path = generation
    retriever.vectors = np.load(generation / 'embeddings.npy', mmap_mode='r')
    retriever.evidence_ids = json.loads((generation / 'id_map.json').read_text())
    retriever._graph = json.loads((generation / 'graph.json').read_text())
    retriever._graph_tables = retriever._graph['tables']
    retriever._adjacency = retriever._graph['adjacency']
    retriever._test_fixture = False
    with sqlite3.connect(f'file:{generation / "evidence.sqlite3"}?mode=ro', uri=True) as db:
        retriever._records = {eid: json.loads(raw) for eid, raw in db.execute('SELECT evidence_id,row_json FROM evidence')}
        assert 'fts5' in db.execute("SELECT sql FROM sqlite_master WHERE name='evidence_fts'").fetchone()[0].lower()
    assert len(retriever._records) == len(retriever.evidence_ids) == 6431
    norms = np.linalg.norm(retriever.vectors, axis=1)
    assert np.isfinite(norms).all() and np.allclose(norms, 1, atol=0.005)
    entities = retriever._graph_tables.get('Entity', [])
    # This is an explicit lexical entity linker for the local mention graph,
    # not the paper's LLM-extracted claim/entity signature.
    def signature(text):
        lower = text.casefold()
        return {'entity_ids': [e['entity_id'] for e in entities
                              if len(e.get('canonical_name', '')) >= 3
                              and re.search(r'(?<!\w)' + re.escape(e['canonical_name'].casefold()) + r'(?!\w)', lower)]}

    rankings = []
    raw_rows = []
    pairs = []
    pair_keys = []
    for i, (row, text, vec) in enumerate(zip(sample, queries, qvectors)):
        bm25 = retriever._bm25_search(text, config.bm25_top_k)
        scores = np.asarray(retriever.vectors @ vec)
        order = sorted(range(len(scores)), key=lambda j: (-float(scores[j]), retriever.evidence_ids[j]))[:config.dense_top_k]
        dense = [{**retriever._records[retriever.evidence_ids[j]],
                  'evidence_chunk_id': retriever.evidence_ids[j], 'branch': 'dense',
                  'rank': rank, 'raw_score': float(scores[j])}
                 for rank, j in enumerate(order, 1)
                 if float(scores[j]) >= config.minimum_dense_score and retriever._allowed(retriever._records[retriever.evidence_ids[j]])]
        query = QueryInput(row['Text'], row['Target'], row['Category'], signature(text))
        graph = retriever._graph_search([*bm25, *dense], query, config.graph_top_k)
        hybrid = reciprocal_rank_fusion({'dense': dense, 'bm25': bm25}, dict(config.rrf_weights), config.rrf_constant)
        full = reciprocal_rank_fusion({'dense': dense, 'bm25': bm25, 'graph': graph}, dict(config.rrf_weights), config.rrf_constant)
        configs = {'BM25': bm25[:5], 'Dense (BGE-M3)': dense[:5],
                   'Dense + BM25': hybrid[:5], '+ Local mention graph': full[:5]}
        seen = set()
        for name, hits in configs.items():
            rankings.append({'ID': str(row['ID']), 'script': row['script'], 'retriever': name,
                             'hits': [h['evidence_chunk_id'] for h in hits],
                             'graph_hits': len(graph), 'linked_entity_ids': query.query_signature['entity_ids']})
            for rank, hit in enumerate(hits, 1):
                eid = hit['evidence_chunk_id']
                record = retriever._records[eid]
                raw_rows.append({'post_id': str(row['ID']), 'script': row['script'], 'retriever': name,
                                 'rank': rank, 'evidence_id': eid, 'source_id': record['source_id'],
                                 'authority': record['authority_score'], 'reranker_logit': None,
                                 'reranker_probability': None, 'above_theta': None})
                if eid not in seen:
                    pairs.append((text, record['text']))
                    pair_keys.append((str(row['ID']), eid))
                    seen.add(eid)
        if (i + 1) % 10 == 0:
            print('Ranked', i + 1, '/100; graph branch hits', len(graph), flush=True)
    dump('rankings.json', rankings)
    write_csv('rankings_unscored.csv', raw_rows)
    from huggingface_hub.constants import HF_HUB_CACHE
    reranker_weights = Path(HF_HUB_CACHE) / 'models--BAAI--bge-reranker-v2-m3' / 'snapshots' / config.reranker_revision / 'model.safetensors'
    if not reranker_weights.exists():
        print('Rankings saved. Waiting for verified reranker weights.', flush=True)
        while not reranker_weights.exists():
            time.sleep(5)
    print('Loading pinned reranker for', len(pairs), 'unique post/passage pairs', flush=True)
    reranker = CrossEncoder(config.reranker_model, revision=config.reranker_revision,
                           device=device, local_files_only=True, max_length=512)
    if device == 'mps':
        reranker.model.half()
    # Identity guarantees raw logits; always apply sigmoid, even for logits in [0,1].
    logits = np.asarray(reranker.predict(pairs, batch_size=8, show_progress_bar=True,
                                         activation_fn=torch.nn.Identity()), dtype=np.float64).reshape(-1)
    assert len(logits) == len(pairs) and np.isfinite(logits).all()
    lookup = dict(zip(pair_keys, logits))
    for r in raw_rows:
        logit = float(lookup[(r['post_id'], r['evidence_id'])])
        r['reranker_logit'] = logit
        r['reranker_probability'] = float(1 / (1 + np.exp(-np.clip(logit, -60, 60))))
        r['above_theta'] = r['reranker_probability'] >= config.minimum_rerank_probability
    write_csv('ranked_passages.csv', raw_rows)
    pool = []
    for (post_id, eid), logit in lookup.items():
        post = next(r for r in sample if str(r['ID']) == post_id)
        evidence = retriever._records[eid]
        pool.append({'post_id': post_id, 'script': post['script'], 'post_text': post['Text'],
                     'target': post['Target'], 'evidence_id': eid, 'source_id': evidence['source_id'],
                     'passage_text': evidence['text'], 'annotator_1_grade': '',
                     'annotator_2_grade': '', 'adjudicated_grade': ''})
    write_csv('annotation_pool.csv', pool)
    metrics = []
    for name in configs:
        for group in ['All', 'Romanized Hinglish', 'Devanagari Hindi']:
            ids = {str(r['ID']) for r in sample if group == 'All' or r['script'] == group}
            rows = [r for r in raw_rows if r['retriever'] == name and r['post_id'] in ids]
            selected = [r for r in rows if r['above_theta']]
            covered = {r['post_id'] for r in selected}
            metrics.append({'retriever': name, 'script': group, 'n_posts': len(ids),
                            'P@5': '', 'R@5': '', 'nDCG@5': '',
                            'Auth_top5_before_threshold': float(np.mean([r['authority'] for r in rows])) if rows else None,
                            'Auth_retained_after_threshold': float(np.mean([r['authority'] for r in selected])) if selected else None,
                            'Cov_theta_0.55': len(covered) / len(ids), 'covered_posts': len(covered),
                            'retrieved_passages': len(rows), 'retained_passages': len(selected)})
    write_csv('metrics.csv', metrics)
    dump('run_manifest.json', {'sample_seed': 42, 'sample_size': 100,
         'sampling': 'proportional script x target stratification, largest remainder, from 141 Homophobic rows of saved Test split',
         'sample_sha256': hashlib.sha256((OUT/'sample_100.json').read_bytes()).hexdigest(),
         'index_manifest': manifest, 'device': device, 'max_length': 512,
         'threshold': 0.55, 'query': 'Text + Target + Category, without LLM rewriting',
         'ranking': 'native BM25/dense, weighted RRF for hybrids; score native/fused top 5 for common threshold coverage',
         'graph': 'local mention graph, exact canonical-name linking; not paper structured claim graph',
         'structured_claim_nodes': len(retriever._graph_tables.get('Claim', [])),
         'graph_branch_post_count': sum(r['graph_hits'] > 0 for r in rankings if r['retriever'] == '+ Local mention graph'),
         'metrics_missing': 'P@5, R@5 and nDCG@5 await real relevance grades',
         'unique_annotation_pairs': len(pool), 'elapsed_seconds': time.time()-started})
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
