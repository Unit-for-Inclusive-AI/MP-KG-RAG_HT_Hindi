#!/usr/bin/env python3
"""Build the complete browser-ready semantic PDF knowledge graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COLORS = {
    "document": "#4f8cff",
    "page": "#9b7bff",
    "chunk": "#34d6c7",
    "claim": "#ffb454",
    "entity": "#ff5fa2",
}
ENTITY_PALETTE = ("#ff5fa2", "#ff7866", "#ffc857", "#73e2a7", "#61c9ff", "#b692ff", "#f06cff")


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _stable_unit(value: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / (2**64 - 1)


def _entity_key(candidate: dict[str, Any]) -> str:
    entity_type = _slug(candidate.get("entity_type") or candidate.get("candidate_type") or candidate.get("type") or "unknown")
    text = _slug(candidate.get("normalized_text") or candidate.get("label") or candidate.get("text"))
    return f"{entity_type}|{text}" if text else f"unresolved|{candidate.get('candidate_id', '')}"


def _entity_id(key: str) -> str:
    return "entity:" + hashlib.sha1(key.encode()).hexdigest()[:20]


def _doc_title(chunk: dict[str, Any], source_id: str) -> str:
    path = str(chunk.get("relative_path") or chunk.get("source_locator") or chunk.get("locator") or source_id)
    name = Path(path).stem
    prefix = source_id + "_"
    if name.startswith(prefix):
        name = name[len(prefix):]
    return name.replace("_", " ") or source_id


def build_graph(
    extractions: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    pages: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic graph while retaining every extracted claim."""
    chunks_by_id = {str(row["chunk_id"]): row for row in chunks}
    if len(chunks_by_id) != len(chunks):
        raise ValueError("duplicate chunk identifiers")
    page_rows = pages.get("records", []) if isinstance(pages, dict) else pages
    page_lookup = {(str(row.get("source_id")), int(row.get("page", row.get("page_number", 0)))): row for row in page_rows}
    candidate_by_id = {str(row["candidate_id"]): row for row in candidates}

    source_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        source_chunks[str(chunk.get("source_id") or "UNKNOWN")].append(chunk)
    sources = sorted(source_chunks)
    phi = math.pi * (3 - math.sqrt(5))
    document_positions: dict[str, tuple[float, float]] = {}
    for index, source_id in enumerate(sources):
        radius = 170 * math.sqrt(index + 1)
        document_positions[source_id] = (radius * math.cos(index * phi), radius * math.sin(index * phi))

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    edge_counter = 0

    def add_edge(source: str, target: str, kind: str) -> None:
        nonlocal edge_counter
        edge_counter += 1
        edges.append({"id": f"edge:{edge_counter}", "source": source, "target": target, "kind": kind})

    for source_id in sources:
        doc_id = f"document:{source_id}"
        sample = sorted(source_chunks[source_id], key=lambda x: str(x.get("chunk_id")))[0]
        x, y = document_positions[source_id]
        nodes[doc_id] = {
            "id": doc_id, "kind": "document", "label": _doc_title(sample, source_id),
            "source_id": source_id, "path": sample.get("relative_path"), "x": x, "y": y,
            "size": 18, "color": COLORS["document"],
        }

    page_keys = sorted({(str(c.get("source_id") or "UNKNOWN"), int(c.get("page", c.get("page_number", 0)))) for c in chunks})
    pages_by_source: dict[str, list[int]] = defaultdict(list)
    for source_id, page in page_keys:
        pages_by_source[source_id].append(page)
    for source_id, page in page_keys:
        doc_id, page_id = f"document:{source_id}", f"page:{source_id}:{page}"
        ordinal = pages_by_source[source_id].index(page)
        count = len(pages_by_source[source_id])
        angle = 2 * math.pi * ordinal / max(1, count) + _stable_unit(source_id, "page-angle")
        ring = 38 + 4.5 * math.sqrt(count)
        dx, dy = document_positions[source_id]
        meta = page_lookup.get((source_id, page), {})
        nodes[page_id] = {
            "id": page_id, "kind": "page", "label": f"{source_id} · page {page}",
            "source_id": source_id, "page": page, "path": meta.get("relative_path"),
            "x": dx + ring * math.cos(angle), "y": dy + ring * math.sin(angle),
            "size": 4.5, "color": COLORS["page"],
        }
        add_edge(doc_id, page_id, "document_has_page")

    for chunk in sorted(chunks, key=lambda row: str(row["chunk_id"])):
        chunk_id = str(chunk["chunk_id"])
        source_id = str(chunk.get("source_id") or "UNKNOWN")
        page = int(chunk.get("page", chunk.get("page_number", 0)))
        page_id, node_id = f"page:{source_id}:{page}", f"chunk:{chunk_id}"
        parent = nodes[page_id]
        angle = 2 * math.pi * _stable_unit(chunk_id, "chunk-angle")
        radius = 8 + 12 * _stable_unit(chunk_id, "chunk-radius")
        text = str(chunk.get("text") or "")
        nodes[node_id] = {
            "id": node_id, "kind": "chunk", "label": f"Chunk {chunk_id[:10]}", "chunk_id": chunk_id,
            "source_id": source_id, "page": page, "path": chunk.get("relative_path"),
            "text": text, "authority_score": chunk.get("authority_score"),
            "evidence_id": chunk.get("evidence_id"), "evidence_text_sha256": chunk.get("evidence_text_sha256"),
            "x": parent["x"] + radius * math.cos(angle), "y": parent["y"] + radius * math.sin(angle),
            "size": 2.6, "color": COLORS["chunk"],
        }
        add_edge(page_id, node_id, "page_has_chunk")

    entity_mentions: Counter[str] = Counter()
    entity_labels: dict[str, Counter[str]] = defaultdict(Counter)
    entity_types: dict[str, str] = {}
    entity_claim_positions: dict[str, list[tuple[float, float]]] = defaultdict(list)
    entity_claim_ids: dict[str, set[str]] = defaultdict(set)
    pending_mentions: list[tuple[str, str]] = []
    claim_count = 0
    invalid_claim_spans = 0
    for extraction in sorted(extractions, key=lambda row: str(row.get("chunk_id"))):
        chunk_id = str(extraction.get("chunk_id"))
        if chunk_id not in chunks_by_id:
            raise ValueError(f"semantic extraction references missing chunk: {chunk_id}")
        chunk_node = nodes[f"chunk:{chunk_id}"]
        source_id = str(extraction.get("source_id") or chunks_by_id[chunk_id].get("source_id") or "UNKNOWN")
        page = int(extraction.get("page", chunks_by_id[chunk_id].get("page", 0)))
        for claim_index, claim in enumerate(extraction.get("claims") or []):
            claim_count += 1
            claim_id = f"claim:{chunk_id}:{claim_index}"
            angle = 2 * math.pi * _stable_unit(claim_id, "claim-angle")
            radius = 4 + 10 * _stable_unit(claim_id, "claim-radius")
            candidate_ids = [str(value) for value in claim.get("candidate_ids") or []]
            start, end = claim.get("span_start"), claim.get("span_end")
            chunk_text = str(chunks_by_id[chunk_id].get("text") or "")
            span_valid = (
                isinstance(start, int) and not isinstance(start, bool)
                and isinstance(end, int) and not isinstance(end, bool)
                and 0 <= start < end <= len(chunk_text)
            )
            invalid_claim_spans += not span_valid
            nodes[claim_id] = {
                "id": claim_id, "kind": "claim", "label": str(claim.get("claim") or "Untitled claim")[:110],
                "claim": str(claim.get("claim") or ""), "chunk_id": chunk_id, "source_id": source_id,
                "page": page, "span_start": start, "span_end": end, "span_valid": span_valid,
                "path": chunks_by_id[chunk_id].get("relative_path"),
                "evidence_id": chunks_by_id[chunk_id].get("evidence_id"),
                "evidence_text_sha256": chunks_by_id[chunk_id].get("evidence_text_sha256"),
                "candidate_ids": candidate_ids,
                "x": chunk_node["x"] + radius * math.cos(angle), "y": chunk_node["y"] + radius * math.sin(angle),
                "size": 1.8 + min(3, math.log2(len(candidate_ids) + 1)), "color": COLORS["claim"],
            }
            add_edge(f"chunk:{chunk_id}", claim_id, "chunk_asserts_claim")
            seen_entity_ids: set[str] = set()
            for candidate_id in candidate_ids:
                candidate = candidate_by_id.get(candidate_id, {"candidate_id": candidate_id, "entity_type": "unresolved", "label": candidate_id})
                key = _entity_key(candidate)
                entity_id = _entity_id(key)
                if entity_id in seen_entity_ids:
                    continue
                seen_entity_ids.add(entity_id)
                entity_mentions[entity_id] += 1
                entity_labels[entity_id][str(candidate.get("label") or candidate.get("text") or candidate_id)] += 1
                entity_types[entity_id] = str(candidate.get("entity_type") or candidate.get("candidate_type") or "unknown")
                entity_claim_positions[entity_id].append((nodes[claim_id]["x"], nodes[claim_id]["y"]))
                entity_claim_ids[entity_id].add(claim_id)
                pending_mentions.append((claim_id, entity_id))

    for entity_id in sorted(entity_mentions):
        positions = entity_claim_positions[entity_id]
        x = sum(point[0] for point in positions) / len(positions)
        y = sum(point[1] for point in positions) / len(positions)
        jitter = 8 + min(40, math.sqrt(len(positions)) * 3)
        angle = 2 * math.pi * _stable_unit(entity_id, "entity-angle")
        label = entity_labels[entity_id].most_common(1)[0][0]
        entity_type = entity_types[entity_id]
        entity_color = ENTITY_PALETTE[int(_stable_unit(entity_type, "entity-color") * len(ENTITY_PALETTE)) % len(ENTITY_PALETTE)]
        nodes[entity_id] = {
            "id": entity_id, "kind": "entity", "entity_type": entity_type, "label": label,
            "aliases": sorted(entity_labels[entity_id]), "mention_count": entity_mentions[entity_id],
            "supporting_claim_count": len(entity_claim_ids[entity_id]),
            "x": x + jitter * math.cos(angle), "y": y + jitter * math.sin(angle),
            "size": min(16, 3 + 2.2 * math.log2(entity_mentions[entity_id] + 1)),
            "color": entity_color,
        }
    for claim_id, entity_id in pending_mentions:
        add_edge(claim_id, entity_id, "claim_mentions_entity")
    for entity_id in sorted(entity_claim_ids):
        related_claims = sorted(entity_claim_ids[entity_id])
        for left, right in zip(related_claims, related_claims[1:]):
            add_edge(left, right, "claim_related_to_claim")

    ordered_nodes = [nodes[key] for key in sorted(nodes)]
    degrees = Counter()
    for edge in edges:
        degrees[edge["source"]] += 1
        degrees[edge["target"]] += 1
    for node in ordered_nodes:
        node["degree"] = degrees[node["id"]]

    counts = Counter(node["kind"] for node in ordered_nodes)
    if counts["claim"] != claim_count:
        raise RuntimeError("claim preservation check failed")
    node_ids = set(nodes)
    if any(edge["source"] not in node_ids or edge["target"] not in node_ids for edge in edges):
        raise RuntimeError("dangling graph edge")
    return {
        "meta": {
            "schema_version": "1.0", "counts": dict(sorted(counts.items())),
            "node_count": len(ordered_nodes), "edge_count": len(edges),
            "source_count": len(sources), "claim_count": claim_count,
            "invalid_claim_spans": invalid_claim_spans,
        },
        "nodes": ordered_nodes,
        "edges": edges,
    }


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Semantic PDF Knowledge Graph</title><link rel="stylesheet" href="styles.css"></head>
<body><div id="loading"><div class="loader"></div><strong>Loading the complete semantic graph…</strong><span id="loading-status">Fetching graph data</span></div>
<header><div><p class="eyebrow">MP–KG–RAG · PDF CORPUS</p><h1>Semantic Knowledge Atlas</h1></div><div id="stats" class="stats"></div></header>
<main><aside class="controls"><label>Search the corpus<input id="search" placeholder="Entity, claim, PDF, source…" autocomplete="off"></label><div id="results"></div>
<label>Node type<select id="type-filter"><option value="all">All node types</option><option value="document">Documents</option><option value="entity">Entities</option><option value="page">Pages</option><option value="chunk">Chunks</option><option value="claim">Claims</option></select></label>
<label>Source community<select id="source-filter"><option value="all">All source documents</option></select></label>
<label>Semantic type<select id="semantic-filter"><option value="all">All semantic types</option></select></label>
<label>Relationship<select id="relationship-filter"><option value="all">All relationships</option><option value="document_has_page">Document → page</option><option value="page_has_chunk">Page → chunk</option><option value="chunk_asserts_claim">Chunk → claim</option><option value="claim_mentions_entity">Claim → entity</option><option value="claim_related_to_claim">Related claims</option></select></label>
<label>Neighborhood depth<select id="neighborhood-depth"><option value="1">One hop</option><option value="2">Two hops</option></select></label>
<label>Minimum connections <output id="degree-output">0</output><input id="degree" type="range" min="0" max="30" value="0"></label>
<div class="button-grid"><button id="fit">Fit graph</button><button id="neighbors">Selected neighborhood</button><button id="reset">Reset view</button><button id="export">Export PNG</button></div>
<section class="legend"><h2>Node language</h2><p><i class="document"></i>Source documents</p><p><i class="entity"></i>Semantic entities</p><p><i class="page"></i>PDF pages</p><p><i class="chunk"></i>Evidence chunks</p><p><i class="claim"></i>Extracted claims</p></section></aside>
<section class="stage"><div id="sigma-container"></div><div class="hint">Scroll to zoom · drag to pan · click a node to inspect</div></section>
<aside id="details" class="details"><p class="eyebrow">SELECTION</p><h2>Explore the network</h2><p>Click any hub or daughter node to reveal its semantic neighborhood and PDF provenance.</p></aside></main>
<script type="module" src="viewer.js"></script></body></html>"""


CSS = """@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;600;700&display=swap');
:root{--bg:#070a10;--panel:#0d121cdd;--line:#263044;--text:#edf3ff;--muted:#8c9ab1;--accent:#5cf0cf}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 55% 20%,#15213a 0,#070a10 42%);color:var(--text);font:14px Manrope,sans-serif;overflow:hidden}header{height:82px;display:flex;align-items:center;justify-content:space-between;padding:14px 24px;border-bottom:1px solid var(--line);background:#080c13e8}h1{margin:0;font-size:23px;letter-spacing:-.03em}.eyebrow{margin:0 0 3px;color:var(--accent);font:10px DM Mono,monospace;letter-spacing:.16em}.stats{font:11px DM Mono,monospace;color:var(--muted)}main{height:calc(100vh - 82px);display:grid;grid-template-columns:280px 1fr 330px}.controls,.details{background:var(--panel);padding:20px;border-right:1px solid var(--line);overflow:auto}.details{border-right:0;border-left:1px solid var(--line)}label{display:block;color:var(--muted);font-size:11px;margin-bottom:18px;text-transform:uppercase;letter-spacing:.08em}input,select,button{width:100%;margin-top:7px;border:1px solid var(--line);border-radius:8px;background:#111824;color:var(--text);padding:10px;font:12px Manrope}input:focus,select:focus{outline:1px solid var(--accent)}input[type=range]{padding:0;accent-color:var(--accent)}output{float:right;color:var(--accent)}button{cursor:pointer;transition:.2s}button:hover{border-color:var(--accent);background:#172638}.button-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.stage{position:relative;min-width:0}.stage:after{content:'';position:absolute;inset:0;pointer-events:none;box-shadow:inset 0 0 100px #05070c}.hint{position:absolute;bottom:15px;left:50%;transform:translateX(-50%);z-index:2;background:#090d15cc;border:1px solid var(--line);border-radius:20px;padding:7px 14px;color:var(--muted);font-size:11px}#sigma-container{position:absolute;inset:0}.legend{margin-top:24px;padding-top:18px;border-top:1px solid var(--line)}.legend h2{font-size:12px}.legend p{margin:9px 0;color:var(--muted);font-size:11px}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px}.document{background:#4f8cff}.entity{background:#ff5fa2}.page{background:#9b7bff}.chunk{background:#34d6c7}.claim{background:#ffb454}#results{max-height:190px;overflow:auto;margin:-10px 0 16px}.result{padding:8px;border-bottom:1px solid #202838;cursor:pointer;font-size:11px}.result:hover{color:var(--accent)}.details h2{font-size:20px;overflow-wrap:anywhere}.details dl{display:grid;grid-template-columns:90px 1fr;gap:9px;margin-top:20px}.details dt{color:var(--muted);font:10px DM Mono}.details dd{margin:0;overflow-wrap:anywhere}.quote{border-left:2px solid var(--accent);padding:12px;margin:16px 0;background:#101722;color:#dbe6f7;line-height:1.55}.badge{display:inline-block;padding:4px 7px;border:1px solid var(--line);border-radius:20px;color:var(--accent);font:10px DM Mono}#loading{position:fixed;z-index:20;inset:0;background:#070a10;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px}.loader{width:46px;height:46px;border:2px solid #243047;border-top-color:var(--accent);border-radius:50%;animation:spin 1s linear infinite}#loading-status{color:var(--muted);font:11px DM Mono}@keyframes spin{to{transform:rotate(360deg)}}@media(max-width:1000px){main{grid-template-columns:230px 1fr}.details{position:absolute;right:0;top:82px;width:310px;height:calc(100vh - 82px);z-index:5;background:#0d121cf2}}"""


JS = r"""import Graph from 'https://cdn.jsdelivr.net/npm/graphology@0.25.4/+esm';
import Sigma from 'https://cdn.jsdelivr.net/npm/sigma@3.0.0/+esm';
const status=document.querySelector('#loading-status');window.addEventListener('unhandledrejection',e=>{status.textContent=`Unable to load graph: ${e.reason?.message||e.reason}`;status.style.color='#ff7866';document.querySelector('.loader').style.display='none'});
const data=await fetch('graph-data.json').then(r=>{if(!r.ok)throw Error(`Graph data: ${r.status}`);return r.json()});if(!data.meta||data.nodes.length!==data.meta.node_count||data.edges.length!==data.meta.edge_count)throw Error('Graph count/schema validation failed');status.textContent=`Building ${data.meta.node_count.toLocaleString()} nodes`;
const graph=new Graph({multi:true}); for(const n of data.nodes) graph.addNode(n.id,n); for(const e of data.edges){if(!graph.hasNode(e.source)||!graph.hasNode(e.target))throw Error(`Dangling edge: ${e.id}`);graph.addEdgeWithKey(e.id,e.source,e.target,{kind:e.kind,color:e.kind==='claim_mentions_entity'?'#75496a':'#202a3b',size:e.kind==='claim_mentions_entity'?1.1:.45})}
const container=document.querySelector('#sigma-container'); const renderer=new Sigma(graph,container,{renderEdgeLabels:false,labelRenderedSizeThreshold:10,labelDensity:.025,labelColor:{color:'#dce7f7'},defaultEdgeColor:'#202a3b',zIndex:true});
document.querySelector('#stats').textContent=`${data.meta.node_count.toLocaleString()} NODES  ·  ${data.meta.edge_count.toLocaleString()} EDGES  ·  ${data.meta.source_count} SOURCES`;
document.querySelector('#loading').style.display='none'; let selected=null, neighborhood=false; const hidden=new Set();
const details=document.querySelector('#details'); const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function showNode(id){selected=id;const a=graph.getNodeAttributes(id);let provenance=a;if(a.kind==='claim'){const chunk=graph.neighbors(id).find(n=>graph.getNodeAttribute(n,'kind')==='chunk');if(chunk)provenance=graph.getNodeAttributes(chunk)}const rows=[['Type',a.kind],['Entity type',a.entity_type],['Evidence ID',provenance.evidence_id||a.evidence_id],['Source',a.source_id||provenance.source_id],['Page',a.page??provenance.page],['Chunk',a.chunk_id||provenance.chunk_id],['Span valid',a.span_valid],['Characters',a.span_start!==undefined?`${a.span_start}–${a.span_end}`:undefined],['Evidence SHA-256',provenance.evidence_text_sha256||a.evidence_text_sha256],['Connections',a.degree],['Supporting claims',a.supporting_claim_count],['Source path',a.path||provenance.path]].filter(x=>x[1]!==undefined&&x[1]!==null);details.innerHTML=`<p class="eyebrow">${esc(a.kind).toUpperCase()}</p><h2>${esc(a.label)}</h2>${a.claim?`<div class="quote">${esc(a.claim)}</div>`:''}${provenance.text?`<details open><summary>Authenticated supporting passage</summary><div class="quote">${esc(provenance.text)}</div></details>`:''}<dl>${rows.map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl><p><span class="badge">${graph.degree(id)} connected nodes</span></p>`;renderer.refresh();}
renderer.on('clickNode',({node})=>showNode(node)); renderer.on('clickStage',()=>{selected=null;renderer.refresh()});
renderer.setSetting('nodeReducer',(id,a)=>{const out={...a};if(hidden.has(id)){out.hidden=true;return out}if(selected){if(id===selected){out.highlighted=true;out.size=Math.max(out.size,10);out.zIndex=3}else if(graph.areNeighbors(id,selected)){out.color='#eef5ff';out.zIndex=2}else{out.color='#202938';out.label='';out.zIndex=0}}return out});
renderer.setSetting('edgeReducer',(id,a)=>{const out={...a};const relationship=document.querySelector('#relationship-filter').value,[s,t]=graph.extremities(id);if(hidden.has(s)||hidden.has(t)||(relationship!=='all'&&a.kind!==relationship)){out.hidden=true;return out}if(selected&&(s===selected||t===selected)){out.color='#77ffe0';out.size=2;out.zIndex=2}else if(selected&&!neighborhood){out.hidden=true}return out});
const sourceSelect=document.querySelector('#source-filter'),semanticSelect=document.querySelector('#semantic-filter');const sources=new Set(),semanticTypes=new Set();graph.forEachNode((id,a)=>{if(a.source_id)sources.add(a.source_id);if(a.entity_type)semanticTypes.add(a.entity_type)});for(const value of [...sources].sort())sourceSelect.add(new Option(value,value));for(const value of [...semanticTypes].sort())semanticSelect.add(new Option(value,value));
function applyFilters(){hidden.clear();const type=document.querySelector('#type-filter').value,source=sourceSelect.value,semantic=semanticSelect.value,degree=+document.querySelector('#degree').value;let neighborhoodKeep=null;if(neighborhood&&selected){neighborhoodKeep=new Set([selected,...graph.neighbors(selected)]);if(document.querySelector('#neighborhood-depth').value==='2')for(const id of [...neighborhoodKeep])for(const neighbor of graph.neighbors(id))neighborhoodKeep.add(neighbor)}graph.forEachNode((id,a)=>{if((type!=='all'&&a.kind!==type)||(source!=='all'&&a.source_id!==source)||(semantic!=='all'&&a.entity_type!==semantic)||graph.degree(id)<degree||(neighborhoodKeep&&!neighborhoodKeep.has(id)))hidden.add(id)});renderer.refresh()}
for(const id of ['type-filter','source-filter','semantic-filter','relationship-filter'])document.querySelector(`#${id}`).onchange=applyFilters;document.querySelector('#degree').oninput=e=>{document.querySelector('#degree-output').textContent=e.target.value;applyFilters()};
const search=document.querySelector('#search'),results=document.querySelector('#results');search.oninput=()=>{const q=search.value.trim().toLowerCase();if(q.length<2){results.innerHTML='';return}const found=[];graph.forEachNode((id,a)=>{if(found.length<30&&`${a.label} ${a.claim||''} ${a.source_id||''} ${a.path||''} ${a.page??''} ${a.chunk_id||''} ${(a.aliases||[]).join(' ')} ${a.entity_type||''}`.toLowerCase().includes(q))found.push([id,a])});results.innerHTML=found.map(([id,a])=>`<div class="result" data-id="${esc(id)}"><b>${esc(a.label)}</b><br>${esc(a.kind)}${a.source_id?' · '+esc(a.source_id):''}</div>`).join('');results.querySelectorAll('.result').forEach(el=>el.onclick=()=>{const id=el.dataset.id;showNode(id);renderer.getCamera().animate(renderer.getNodeDisplayData(id),{duration:650})})};
document.querySelector('#fit').onclick=()=>renderer.getCamera().animatedReset({duration:700});document.querySelector('#reset').onclick=()=>{selected=null;neighborhood=false;search.value='';results.innerHTML='';for(const id of ['type-filter','source-filter','semantic-filter','relationship-filter'])document.querySelector(`#${id}`).value='all';document.querySelector('#neighborhood-depth').value='1';document.querySelector('#degree').value=0;document.querySelector('#degree-output').textContent='0';hidden.clear();renderer.refresh();renderer.getCamera().animatedReset({duration:700})};
document.querySelector('#neighbors').onclick=()=>{if(!selected)return;neighborhood=!neighborhood;applyFilters()};document.querySelector('#neighborhood-depth').onchange=()=>{if(neighborhood)applyFilters()};
document.querySelector('#export').onclick=()=>{const canvases=[...container.querySelectorAll('canvas')];const out=document.createElement('canvas');out.width=container.clientWidth*2;out.height=container.clientHeight*2;const ctx=out.getContext('2d');ctx.scale(2,2);ctx.fillStyle='#070a10';ctx.fillRect(0,0,container.clientWidth,container.clientHeight);for(const c of canvases)ctx.drawImage(c,0,0,container.clientWidth,container.clientHeight);const a=document.createElement('a');a.download='semantic-knowledge-graph.png';a.href=out.toDataURL('image/png');a.click()};
window.graph=graph;window.renderer=renderer;"""


def write_viewer(output_dir: Path, graph: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_hash = canonical_hash(graph)
    metadata = {**graph["meta"], "graph_sha256": graph_hash}
    (output_dir / "graph-data.json").write_text(json.dumps(graph, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (output_dir / "graph-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "index.html").write_text(HTML, encoding="utf-8")
    (output_dir / "styles.css").write_text(CSS, encoding="utf-8")
    (output_dir / "viewer.js").write_text(JS, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    required = ("semantic_extractions.json", "candidates.json", "chunks.json", "pages.json", "manifest.json", "semantic_gpu_manifest.json", "evidence_ledger.json")
    missing = [name for name in required if not (args.source / name).is_file()]
    if missing:
        raise FileNotFoundError("missing semantic artifacts: " + ", ".join(missing))
    load = lambda name: json.loads((args.source / name).read_text(encoding="utf-8"))
    extractions, candidates, chunks, pages = (load("semantic_extractions.json"), load("candidates.json"), load("chunks.json"), load("pages.json"))
    ledger = load("evidence_ledger.json")
    ledger_rows = ledger.get("records", []) if isinstance(ledger, dict) else ledger
    chunk_ids = {str(row["chunk_id"]) for row in chunks}
    ledger_ids = {str(row.get("chunk_id")) for row in ledger_rows}
    if len(extractions) != len(chunks) or ledger_ids != chunk_ids:
        raise RuntimeError("semantic extraction/chunk/evidence identity mismatch")
    gpu_manifest = load("semantic_gpu_manifest.json")
    semantic_manifest = load("manifest.json")
    artifact_hashes = gpu_manifest.get("artifact_hashes", {})
    observed_hashes = {
        "extractions": file_sha256(args.source / "semantic_extractions.json"),
        "candidates": file_sha256(args.source / "candidates.json"),
        "chunks": file_sha256(args.source / "chunks.json"),
    }
    if gpu_manifest.get("complete") is not True or gpu_manifest.get("rows") != len(extractions):
        raise RuntimeError("semantic GPU manifest is incomplete or has the wrong row count")
    if semantic_manifest.get("complete") is not True or semantic_manifest.get("stage") != "semantic":
        raise RuntimeError("semantic manifest is incomplete or has the wrong stage")
    if semantic_manifest.get("stage_hash") != ledger.get("stage_hash"):
        raise RuntimeError("evidence ledger is not bound to the semantic stage")
    if not isinstance(pages, dict) or pages.get("stage") != "cpu-extraction" or len(pages.get("records", [])) == 0:
        raise RuntimeError("page artifact is not a valid CPU extraction artifact")
    if any(artifact_hashes.get(key) != value for key, value in observed_hashes.items()):
        raise RuntimeError("semantic artifact hash does not match GPU manifest")
    expected_keys = set(gpu_manifest.get("expected_keys") or [])
    observed_keys = {str(row.get("request_id")) for row in extractions}
    if expected_keys != observed_keys:
        raise RuntimeError("semantic request identity set mismatch")
    ledger_by_chunk = {str(row["chunk_id"]): row for row in ledger_rows}
    chunks = [
        {**row, "evidence_id": ledger_by_chunk[str(row["chunk_id"])].get("evidence_id"),
         "evidence_text_sha256": ledger_by_chunk[str(row["chunk_id"])].get("text_sha256")}
        for row in chunks
    ]
    graph = build_graph(extractions, candidates, chunks, pages)
    graph["meta"]["source_artifacts"] = {
        name: {"bytes": (args.source / name).stat().st_size, "sha256": file_sha256(args.source / name)}
        for name in required
    }
    graph["meta"]["validation"] = {
        "accepted_extractions": len(extractions), "evidence_rows": len(ledger_rows),
        "all_claims_preserved": graph["meta"]["claim_count"] == sum(len(row.get("claims") or []) for row in extractions),
        "no_dangling_edges": True, "semantic_stage_bound_to_evidence": True,
        "gpu_manifest_artifact_hashes_matched": True, "request_identity_set_matched": True,
        "page_stage_validated": True, "all_source_artifact_hashes_recorded": True,
    }
    write_viewer(args.output, graph)
    print(json.dumps({**graph["meta"], "output": str(args.output), "graph_sha256": canonical_hash(graph)}, indent=2))


if __name__ == "__main__":
    main()
