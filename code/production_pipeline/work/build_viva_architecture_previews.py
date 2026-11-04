"""Build self-contained HTML chapters for the MP-KG-RAG project viva.

The renderer intentionally has no runtime dependencies.  Chapter dictionaries are
the extension point used by the later architecture chapters; all visual and
interaction behavior lives in this module so the pages stay consistent.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any


NAV_CHAPTERS = (
    ("zero-shot-architecture", "Zero-shot"),
    ("few-shot-architecture", "Few-shot"),
    ("kg-rag-architecture", "KG-RAG"),
    ("mp-kg-rag-architecture", "MP-KG-RAG"),
    ("knowledge-graph-construction", "Knowledge graph"),
)


def _text(value: object) -> str:
    """Escape chapter-authored text for HTML while preserving readable punctuation."""

    return escape(str(value), quote=True)


def _items(values: list[str] | tuple[str, ...]) -> str:
    return "".join(f"<li>{_text(value)}</li>" for value in values)


def finding_cards(findings: list[dict[str, str]]) -> str:
    cards: list[str] = []
    for finding in findings:
        severity = finding.get("severity", "Important")
        cards.append(
            "<article class=\"finding-card\">"
            f"<div class=\"finding-head\"><span class=\"finding-title\">{_text(finding['title'])}</span>"
            f"<span class=\"severity severity-{_text(severity).lower()}\">{_text(severity)}</span></div>"
            f"<p>{_text(finding['summary'])}</p>"
            f"<div class=\"metric\">{_text(finding['metric'])}</div>"
            "</article>"
        )
    return "".join(cards)


def architecture_visual(chapter: dict[str, Any]) -> str:
    branch_flow = chapter.get("branch_flow")
    stages = chapter["stages"]
    movements = chapter.get("movements", [{"kind": "record", "label": "input row"}])
    branches = chapter.get("branch_labels", [])
    stage_nodes: list[str] = []
    for index, stage in enumerate(stages):
        arrow = ""
        if index < len(stages) - 1:
            arrow = f'<span class="flow-stage-arrow" style="--arrow-index:{index + 1}" aria-hidden="true">→</span>'
        stage_nodes.append(
            f'<li><button class="flow-node" data-stage="{index}" type="button" aria-pressed="false" '
            f'aria-label="Stage {index + 1}: {_text(stage["label"])}">'
            f'<span class="flow-stage-number">{index + 1:02d}</span>'
            f'<strong>{_text(stage["label"])}</strong><small>{_text(stage.get("detail", ""))}</small></button>{arrow}</li>'
        )
    movement_tokens = "".join(
        f'<span class="flow-token token-{_text(item["kind"])}" data-movement="{_text(item["kind"])}" '
        f'style="--token-index:{index}">{_text(item["label"])}</span>'
        for index, item in enumerate(movements)
    )
    # Keep a small inline SVG for a familiar visual affordance. The readable,
    # responsive stage cards carry the labels so the diagram also works at 390px.
    branch_key = ""
    if branch_flow:
        branch_key = branch_visual(branch_flow)
    elif branches:
        branch_cards = "".join(
            f'<div class="branch-card"><strong>{_text(branch["label"])}</strong><span>{_text(branch["detail"])}</span></div>'
            for branch in branches
        )
        branch_key = f'<div class="branch-grid" aria-label="Bounded branches">{branch_cards}</div>'
    return (
        '<div class="visual-wrap"><div class="visual-toolbar">'
        '<span class="visual-label">Data movement</span>'
        '<button class="button replay-animation" type="button">Replay animation</button></div>'
        f'<div class="flow-diagram" data-stage-count="{len(stages)}" style="--stage-count:{len(stages)}" role="group" aria-label="{_text(chapter["visual_label"])}">'
        '<svg class="architecture-svg" viewBox="0 0 100 12" preserveAspectRatio="none" aria-hidden="true">'
        '<path class="flow-arrow" d="M2 6 H98" marker-end="url(#flow-arrowhead)" />'
        '<defs><marker id="flow-arrowhead" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">'
        '<path d="M0,0 L0,6 L6,3 z" fill="#79c7dc" /></marker></defs></svg>'
        f'<ol class="flow-stage-list">{"".join(stage_nodes)}</ol>{branch_key}'
        f'<div class="flow-lane" aria-label="Moving data items">{movement_tokens}</div>'
        '<p class="flow-caption">Cards show where information is allowed to travel; moving tokens show the information items carried into generation.</p>'
        '</div></div>'
    )


def graph_visual(config: dict[str, Any]) -> str:
    """Render a replayable, endpoint-addressable source-to-hub graph."""

    nodes = config.get("nodes", [])
    stages: dict[int, list[dict[str, Any]]] = {}
    for node in nodes:
        stages.setdefault(int(node["stage"]), []).append(node)
    stage_labels = config.get(
        "stage_labels",
        {0: "Document record", 1: "Accepted EvidenceChunks", 2: "Qwen extraction records", 3: "Persisted Claim + Entity nodes"},
    )
    stage_groups: list[str] = []
    for stage, stage_nodes in sorted(stages.items()):
        cards = "".join(
            f'<button class="graph-node graph-node-{_text(node["kind"]).lower()}" type="button" '
            f'aria-pressed="false" data-graph-id="{_text(node["id"])}" '
            f'data-graph-node="{_text(node["kind"])}" data-graph-kind="{_text(node["kind"])}" '
            f'data-graph-stage="{stage}" style="--graph-stage:{stage}" '
            f'aria-label="{_text(node["label"])}: {_text(node["detail"])}">'
            f'<span>{_text(node["kind"])}</span><strong>{_text(node["label"])}</strong>'
            f'<small>{_text(node["detail"])}</small></button>'
            for node in stage_nodes
        )
        stage_groups.append(
            f'<section class="graph-stage" data-graph-stage-index="{stage}" aria-labelledby="graph-stage-{stage}">'
            f'<h3 id="graph-stage-{stage}">{_text(stage_labels.get(stage, f"Stage {stage + 1}"))}</h3>'
            f'<div class="graph-stage-nodes">{cards}</div></section>'
        )
    node_labels = {str(node["id"]): str(node["label"]) for node in nodes}
    edge_rows = "".join(
        f'<li class="graph-edge graph-edge-{_text(edge["kind"]).lower()}" '
        f'data-edge-kind="{_text(edge["kind"])}" data-edge-from="{_text(edge["from"])}" '
        f'data-edge-to="{_text(edge["to"])}" data-edge-stage="{_text(edge.get("stage", 0))}" '
        f'style="--graph-edge-stage:{_text(edge.get("stage", 0))}" '
        f'aria-label="{_text(edge["kind"])} edge from {node_labels.get(str(edge["from"]), edge["from"])} to '
        f'{node_labels.get(str(edge["to"]), edge["to"])}">'
        f'<span class="graph-edge-label">{_text(edge["kind"])} · '
        f'{_text(node_labels.get(str(edge["from"]), edge["from"]))} → '
        f'{_text(node_labels.get(str(edge["to"]), edge["to"]))}</span>'
        f'<small>{_text(edge["detail"])}</small></li>'
        for edge in config.get("edges", [])
    )
    return (
        '<div class="graph-visual" data-graph-stage="multi-hub" data-graph-animation="staged-provenance" '
        'data-graph-replay="replayable" data-graph-complete="true" role="group" '
        'aria-label="Document to accepted EvidenceChunk, Qwen extraction record, and persisted Claim and Entity graph">'
        '<div class="graph-visual-head"><span class="visual-label">Worked graph snapshot</span>'
        '<span class="graph-caption">Mention is an extraction record, not a persisted graph node</span>'
        '<button class="button replay-graph" type="button">Replay graph animation</button></div>'
        '<p class="graph-order">Provenance order: Document → accepted EvidenceChunks → Qwen extraction records → persisted Claim + Entity nodes</p>'
        '<div class="graph-token-lane" aria-label="Moving graph provenance records">'
        '<span class="graph-token" style="--graph-token-index:0">accepted chunk</span>'
        '<span class="graph-token" style="--graph-token-index:1">Qwen extraction record</span>'
        '<span class="graph-token" style="--graph-token-index:2">persisted claim → entity</span></div>'
        f'<div class="graph-stage-list" aria-label="Graph nodes in provenance order">{"".join(stage_groups)}</div>'
        f'<ol class="graph-edge-list" aria-label="Graph edges and endpoints">{edge_rows}</ol>'
        '<div class="graph-edge-legend" aria-label="Graph edge legend">'
        '<span class="graph-legend-item graph-legend-endpoints"><b>has_subject / has_object</b> claim endpoints</span>'
        '<span class="graph-legend-item graph-legend-stance"><b>supports / refutes / quotes / contextualizes</b> evidence stance</span>'
        '<span class="graph-legend-item graph-legend-provenance"><b>from_document / evidenced_by</b> provenance</span>'
        '</div><p class="graph-status" aria-live="polite">Graph staged: source evidence is ready.</p></div>'
    )


def branch_visual(config: dict[str, Any]) -> str:
    """Render a semantic, animated fan-out/fan-in visual for retrieval chapters."""

    variant = _text(config["variant"])
    branches = config["branches"]
    branch_rows = "".join(
        '<div class="branch-row" role="listitem" '
        f'data-branch-index="{index}" data-branch-name="{_text(branch["label"])}">'
        f'<button class="branch-node" type="button" aria-pressed="false" data-branch-role="perspective" '
        f'aria-label="{_text(branch["label"])} branch: {_text(branch["detail"])}">'
        f'<span class="branch-node-kicker">Branch {index + 1:02d}</span><strong>{_text(branch["label"])}</strong>'
        f'<small>{_text(branch["detail"])}</small></button>'
        '<span class="branch-connector" aria-hidden="true"><span class="branch-token">'
        f'{_text(branch.get("token", branch["label"]))}</span></span></div>'
        for index, branch in enumerate(branches)
    )
    tail = ""
    if config.get("rejection"):
        rejection = config["rejection"]
        tail = (
            '<div class="branch-tail" data-tail="rerank-filter-evidence">'
            '<div class="branch-tail-node"><button class="branch-node" type="button" aria-pressed="false" '
            'data-branch-role="reranker" aria-label="Cross-encoder reranker filter">'
            '<span class="branch-node-kicker">Precision gate</span><strong>Cross-encoder reranker</strong>'
            '<small>BGE Reranker v2-M3 · threshold 0.55</small></button>'
            '<div class="candidate-filter" aria-label="Reranker candidate decisions">'
            '<span class="candidate candidate-accepted" data-status="accepted">Accepted · 0.82</span>'
            f'<span class="candidate candidate-rejected" data-status="rejected">Rejected · {_text(rejection["score"])} &lt; 0.55</span>'
            '</div></div>'
            '<span class="tail-arrow" aria-hidden="true">→</span>'
            '<div class="branch-tail-node"><button class="branch-node" type="button" aria-pressed="false" '
            'data-branch-role="evidence" aria-label="Bounded evidence ledger">'
            '<span class="branch-node-kicker">Audit boundary</span><strong>Evidence ledger</strong>'
            '<small>maximum five passages · citation validation</small></button>'
            '<dl class="provenance-ledger" aria-label="Concrete evidence provenance">'
            '<div><dt data-provenance-field="evidence ID">evidence ID</dt><dd>EVD-001</dd></div>'
            '<div><dt data-provenance-field="source_id">source_id</dt><dd>WHO-GL-042</dd></div>'
            '<div><dt data-provenance-field="page/locator">page/locator</dt><dd>p. 14 · paragraph 3</dd></div>'
            '<div><dt data-provenance-field="span offsets">span offsets</dt><dd>1184–1327</dd></div>'
            '<div><dt data-provenance-field="source-text key">source-text key</dt><dd>doc42:p14:s03</dd></div>'
            '<div><dt data-provenance-field="SHA-256/hash">SHA-256/hash</dt><dd>sha256:8f2c…d91a</dd></div>'
            '</dl></div>'
            '<span class="tail-arrow" aria-hidden="true">→</span>'
            '<div class="branch-tail-node"><button class="branch-node" type="button" aria-pressed="false" '
            'data-branch-role="generator" aria-label="Qwen grounded final response">'
            '<span class="branch-node-kicker">Bounded generation</span><strong>Qwen final</strong>'
            '<small>citations + schema-validated JSON</small></button></div></div>'
        )
    else:
        tail = (
            '<div class="branch-tail" data-tail="planner-synthesis">'
            '<div class="branch-tail-node"><button class="branch-node" type="button" aria-pressed="false" '
            'data-branch-role="planner" aria-label="Synthesis planner">'
            '<span class="branch-node-kicker">Convergence</span><strong>Planner</strong>'
            '<small>one ordered plan from five analyses</small></button></div>'
            '<span class="tail-arrow" aria-hidden="true">→</span>'
            '<div class="branch-tail-node"><button class="branch-node" type="button" aria-pressed="false" '
            'data-branch-role="generator" aria-label="Final synthesis">'
            '<span class="branch-node-kicker">Bounded generation</span><strong>Final synthesis</strong>'
            '<small>Qwen3.5-4B · validation + checkpoint</small></button></div></div>'
        )
    return (
        f'<div class="branch-flow" data-branch-variant="{variant}" data-branch-count="{len(branches)}" '
        f'data-animation="fan-out-and-converge" role="group" aria-label="{_text(config["label"])}">'
        '<div class="branch-source"><button class="branch-node" type="button" aria-pressed="false" '
        'data-branch-role="source" aria-label="Branch source">'
        f'<span class="branch-node-kicker">Source</span><strong>{_text(config["source"])}</strong>'
        f'<small>{_text(config.get("source_detail", "one bounded input"))}</small></button></div>'
        f'<div class="branch-rows" role="list" aria-label="Parallel branches">{branch_rows}</div>'
        f'<div class="branch-convergence" data-converges-at="{_text(config["convergence_key"])}">'
        '<span class="convergence-line" aria-hidden="true"></span>'
        f'<button class="branch-node" type="button" aria-pressed="false" data-branch-role="convergence" '
        f'aria-label="{_text(config["convergence"])} convergence">'
        '<span class="branch-node-kicker">Fan-in</span>'
        f'<strong>{_text(config["convergence"])}</strong><small>{_text(config["convergence_detail"])}</small></button></div>'
        f'{tail}<p class="branch-status" aria-live="polite">Select a branch to inspect its bounded role.</p></div>'
    )


def comparison_table(comparison: dict[str, Any]) -> str:
    headers = comparison["headers"]
    rows = comparison["rows"]
    head = "".join(f"<th scope=\"col\">{_text(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_text(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def worked_example(chapter: dict[str, Any]) -> str:
    steps = chapter["example_steps"]
    panels: list[str] = []
    for index, step in enumerate(steps):
        panels.append(
            f'<article class="example-step" data-example-step="{index}" aria-hidden="{"false" if index == 0 else "true"}">'
            f'<div class="step-number">{index + 1:02d}</div><div><h3>{_text(step["label"])}</h3>'
            f'<p>{_text(step["body"])}</p><p class="step-boundary"><strong>Boundary:</strong> {_text(step["boundary"])}</p></div></article>'
        )
    return (
        '<section class="section example-section" id="worked-example"><div class="section-kicker">Shared worked example</div>'
        '<h2>One input, one audit trail</h2>'
        '<p class="lede">The intentionally minimal <strong>input hate speech</strong> targets an LGBTQ+ group. '
        'It is shown only to explain the pipeline, not to amplify the wording.</p>'
        '<div class="example-input"><span>Input hate speech</span><p>“That LGBTQ+ group should not be trusted in our community.”</p></div>'
        '<div class="example-controls"><button class="button example-next" type="button">Walk through example</button>'
        '<span class="example-status" aria-live="polite">Stage 1 of ' + str(len(steps)) + '</span></div>'
        '<div class="example-steps">' + "".join(panels) + "</div></section>"
    )


def viva_panels(panels: list[dict[str, str]]) -> str:
    return "".join(
        '<details class="viva-panel"><summary><span>Viva question</span>'
        f'<strong>{_text(panel["question"])}</strong></summary>'
        f'<div class="viva-answer"><p><strong>Short answer:</strong> {_text(panel["short"])}</p>'
        f'<p><strong>Deep follow-up:</strong> {_text(panel["deep"])}</p></div></details>'
        for panel in panels
    )


def challenge_table(challenges: list[dict[str, str]]) -> str:
    if not challenges:
        return ""
    detailed = any(item.get("stage") or item.get("viva") for item in challenges)
    rows = "".join(
        "<tr>"
        f"<th scope=\"row\">{_text(item['symptom'])}</th>"
        f"<td>{_text(item['root'])}</td><td>{_text(item['correction'])}</td>"
        f"<td>{_text(item['lesson'])}</td>"
        + (f"<td>{_text(item.get('stage', 'Pipeline-wide'))}</td>"
           f"<td>{_text(item.get('viva', 'How did you make this boundary observable?'))}</td>" if detailed else "")
        + "</tr>"
        for item in challenges
    )
    extra_headers = (
        '<th scope="col">Pipeline stage</th><th scope="col">Likely viva framing</th>'
        if detailed else ""
    )
    return (
        '<section class="section" id="challenges"><div class="section-kicker">Engineering judgment</div>'
        '<h2>Challenges and corrections</h2><p class="lede">Each fix protects a specific audit boundary; '
        'the trade-off is part of the viva answer.</p><div class="table-scroll"><table><thead><tr>'
        '<th scope="col">Symptom</th><th scope="col">Root cause</th><th scope="col">Correction</th><th scope="col">Lesson</th>'
        + extra_headers +
        '</tr></thead><tbody>' + rows + '</tbody></table></div></section>'
    )


def _section(section: dict[str, Any]) -> str:
    body = [f'<section class="section" id="{_text(section["id"])}">']
    if section.get("kicker"):
        body.append(f'<div class="section-kicker">{_text(section["kicker"])}</div>')
    body.append(f'<h2>{_text(section["heading"])}</h2><p class="lede">{_text(section["intro"])}</p>')
    if section.get("bullets"):
        body.append(f'<ul class="bullet-list">{_items(section["bullets"])}</ul>')
    if section.get("table"):
        body.append(comparison_table(section["table"]))
    body.append("</section>")
    return "".join(body)


def plain_language_glossary() -> str:
    terms = (
        ("Unsloth FastModel + Transformers", "Unsloth FastModel loads Qwen in 4-bit; Transformers model.generate performs the local batched inference path."),
        ("4-bit quantization", "4-bit quantization stores model weights in four-bit values, using less GPU memory with a possible quality trade-off."),
        ("Schema-constrained generation", "Schema-constrained generation means guiding the model toward a required JSON shape, then checking that the result has the required fields."),
        ("Checkpoint", "A checkpoint is a saved progress record for a row, so an interrupted run can resume without losing completed work."),
        ("Embedding", "Embedding means representing text as vectors so dense search can compare semantic similarity."),
        ("BM25", "BM25 is lexical retrieval: it ranks text using token overlap, which helps exact or rare terms."),
        ("Cross-encoder", "A cross-encoder scores a pair by reading the query and passage together for precise reranking."),
        ("RRF", "RRF combines ranked lists by adding reciprocal-rank contributions with configured branch weights."),
        ("Graph traversal", "Graph traversal follows typed edges from accepted seeds for a bounded number of hops."),
        ("Provenance", "Provenance records where evidence came from, including document, chunk, span, and hash."),
        ("Checkpoint identity", "Checkpoint identity is the hashable run contract used to reject incompatible resume state."),
    )
    cards = "".join(f'<div class="glossary-card"><dt>{_text(term)}</dt><dd>{_text(definition)}</dd></div>' for term, definition in terms)
    return (
        '<section class="section" id="plain-language"><div class="section-kicker">Before the diagram</div>'
        '<h2>Plain-language glossary</h2><p class="lede">These terms describe implementation choices in the architecture.</p>'
        f'<dl class="glossary-grid">{cards}</dl></section>'
    )


STYLE = r"""
:root { color-scheme: dark; --ink:#fff; --muted:#c4ccd0; --panel:#0b0d0e; --line:#273238; --accent:#79c7dc; --high:#f17676; --medium:#e8bd70; --low:#83d29c; }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; max-width:100%; overflow-x:clip; }
body { margin:0; max-width:100%; overflow-x:clip; background:#000; color:var(--ink); font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; line-height:1.6; }
a { color:var(--accent); text-decoration:none; }
a:hover { text-decoration:underline; }
:focus-visible { outline:3px solid var(--accent); outline-offset:3px; }
.progress-track { position:fixed; inset:0 0 auto; height:4px; background:#171c1e; z-index:20; }
#progress-bar { display:block; height:100%; width:0; background:var(--accent); transition:width .15s linear; }
.site-nav { position:sticky; top:4px; z-index:10; display:flex; gap:1rem; align-items:center; justify-content:space-between; padding:1rem clamp(1rem,4vw,4rem); background:rgba(0,0,0,.94); border-bottom:1px solid var(--line); }
.brand { color:#fff; font-weight:750; letter-spacing:.04em; }
.chapter-links { display:flex; flex-wrap:wrap; gap:.35rem .8rem; justify-content:flex-end; }
.chapter-links a { padding:.25rem .45rem; border-radius:5px; color:var(--muted); font-size:.9rem; }
.chapter-links a[aria-current="page"] { color:#000; background:var(--accent); font-weight:700; }
.page { width:min(1280px,calc(100% - 2rem)); margin:0 auto; }
.hero { padding:clamp(3rem,8vw,7rem) 0 2.5rem; max-width:920px; }
.eyebrow,.section-kicker { color:var(--accent); text-transform:uppercase; letter-spacing:.14em; font-size:.78rem; font-weight:750; }
h1,h2,h3,p { margin-top:0; }
h1 { font-size:clamp(2.2rem,6vw,5rem); line-height:1.04; letter-spacing:-.045em; margin:.6rem 0 1.25rem; }
h2 { font-size:clamp(1.65rem,3vw,2.6rem); line-height:1.1; letter-spacing:-.02em; margin:.45rem 0 1rem; }
h3 { font-size:1.05rem; line-height:1.25; margin-bottom:.45rem; }
.lede { max-width:780px; color:var(--muted); font-size:1.08rem; }
.findings { padding:1rem 0 3rem; }
.findings h2 { font-size:1.4rem; }
.findings-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:1rem; }
.finding-card,.example-input,.example-step,.visual-wrap,.viva-panel { background:var(--panel); border:1px solid var(--line); border-radius:14px; }
.finding-card { padding:1.15rem; min-height:170px; }
.finding-head { display:flex; gap:.7rem; align-items:start; justify-content:space-between; }
.finding-title { font-weight:750; }
.finding-card p { color:var(--muted); margin:1rem 0 .8rem; }
.metric { color:var(--accent); font-size:.86rem; font-weight:700; }
.severity { border:1px solid currentColor; border-radius:999px; padding:.08rem .45rem; font-size:.7rem; white-space:nowrap; }
.severity-high { color:var(--high); }.severity-medium { color:var(--medium); }.severity-low { color:var(--low); }.severity-important { color:var(--accent); }
.section { padding:3.8rem 0; border-top:1px solid var(--line); }
.bullet-list { display:grid; gap:.75rem; padding-left:1.2rem; color:var(--muted); max-width:920px; }
.bullet-list li::marker { color:var(--accent); }
.visual-wrap { padding:1rem; margin-top:1.4rem; }
.visual-toolbar,.example-controls { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom:.8rem; }
.visual-label { color:var(--muted); font-size:.87rem; }
.button { background:transparent; color:var(--accent); border:1px solid var(--accent); border-radius:6px; padding:.5rem .75rem; font:inherit; cursor:pointer; }
.button:hover { background:#12272e; }
.architecture-svg { display:block; width:100%; min-width:0; height:2.5rem; overflow:visible; }.architecture-svg .flow-arrow { fill:none; stroke:var(--line); stroke-width:1.2; stroke-dasharray:2 2; }
.flow-diagram { --stage-count:5; position:relative; }.flow-stage-list { position:relative; z-index:1; display:grid; grid-template-columns:repeat(var(--stage-count),minmax(0,1fr)); align-items:stretch; gap:1.7rem; list-style:none; margin:-1.3rem 0 0; padding:0; }.flow-stage-list > li { position:relative; min-width:0; }.flow-node { width:100%; min-width:0; min-height:8rem; display:flex; flex-direction:column; justify-content:center; gap:.25rem; padding:.85rem; background:#111719; color:var(--ink); border:1px solid var(--accent); border-radius:12px; cursor:pointer; text-align:left; font:inherit; transition:opacity .3s,transform .3s,background .3s; }.flow-node strong { line-height:1.2; overflow-wrap:anywhere; }.flow-node small { color:var(--muted); font-size:.78rem; line-height:1.25; overflow-wrap:anywhere; }.flow-stage-number { color:var(--accent); font-size:.73rem; font-weight:800; letter-spacing:.08em; }.flow-node.is-active { transform:translateY(-4px); background:#17343c; }.flow-stage-arrow { position:absolute; z-index:2; top:3.5rem; left:calc(100% + .85rem); transform:translateX(-50%); color:var(--accent); font-size:1.4rem; font-weight:800; }
.flow-lane { position:relative; display:grid; gap:.5rem; margin-top:1rem; padding:.7rem 0 2rem; border-top:1px dashed var(--line); overflow:hidden; }.flow-token { position:relative; left:0; z-index:3; width:fit-content; max-width:100%; min-width:0; padding:.2rem .5rem; border:1px solid #83b8c3; border-radius:999px; background:#071114; color:#fff; font-size:.73rem; font-weight:700; opacity:0; white-space:normal; overflow-wrap:anywhere; }.flow-token::before { content:'↗ '; color:#83b8c3; }.flow-diagram.is-running .flow-token { animation:flow-token-horizontal 3.8s cubic-bezier(.35,.05,.65,.95) infinite; animation-delay:calc(var(--token-index) * .35s); }.flow-caption { margin:1rem 0 0; color:var(--muted); font-size:.87rem; }.token-demonstration-card { border-color:#83b8c3; }.token-demonstration-card::before { color:#83b8c3; }.token-learned-pattern { border-color:#a3adb1; }.token-learned-pattern::before { color:#a3adb1; }
.branch-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.55rem; margin-top:1rem; }.branch-card { display:grid; gap:.15rem; min-width:0; padding:.65rem .7rem; border:1px dashed var(--accent); border-radius:9px; background:#0b1417; }.branch-card strong { color:#fff; font-size:.82rem; overflow-wrap:anywhere; }.branch-card span { color:var(--muted); font-size:.74rem; overflow-wrap:anywhere; }
.branch-flow { display:grid; grid-template-columns:minmax(150px,.85fr) minmax(0,2.2fr) minmax(160px,.95fr); gap:1rem; align-items:stretch; margin-top:1.25rem; padding:1rem; border:1px solid #344b52; border-radius:12px; background:#070d0f; overflow:hidden; }.branch-source,.branch-convergence { display:flex; align-items:center; min-width:0; position:relative; }.branch-source::after,.branch-convergence::before { content:''; position:absolute; top:50%; width:1rem; border-top:1px dashed var(--accent); opacity:.8; }.branch-source::after { right:-1rem; }.branch-convergence::before { left:-1rem; }.branch-rows { display:grid; gap:.6rem; min-width:0; padding:.15rem 0; }.branch-row { display:grid; grid-template-columns:minmax(0,1fr) minmax(4rem,.55fr); gap:.65rem; align-items:center; min-width:0; }.branch-row::after { content:''; grid-column:2; height:1px; background:repeating-linear-gradient(90deg,var(--accent) 0 5px,transparent 5px 9px); opacity:.75; }.branch-row > .branch-connector { grid-column:2; grid-row:1; position:relative; display:block; min-width:0; height:2.2rem; align-self:center; }.branch-connector::before { content:'→'; position:absolute; inset:0 0 auto; color:var(--accent); text-align:right; font-weight:800; }.branch-node { width:100%; min-width:0; min-height:4.5rem; display:flex; flex-direction:column; justify-content:center; gap:.18rem; padding:.65rem .72rem; color:var(--ink); background:#101b1e; border:1px solid #79c7dc; border-radius:9px; cursor:pointer; text-align:left; font:inherit; transition:transform .25s,background .25s,border-color .25s; }.branch-node:hover,.branch-node.is-active { background:#17343c; border-color:#b9effc; transform:translateY(-2px); }.branch-node strong { line-height:1.2; overflow-wrap:anywhere; }.branch-node small { color:var(--muted); font-size:.74rem; line-height:1.25; overflow-wrap:anywhere; }.branch-node-kicker { color:var(--accent); font-size:.66rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; }.branch-token { position:absolute; left:0; top:50%; z-index:2; width:max-content; max-width:100%; padding:.18rem .4rem; transform:translateY(-50%); border:1px solid #83b8c3; border-radius:999px; background:#071114; color:#fff; font-size:.68rem; font-weight:700; opacity:0; white-space:normal; overflow-wrap:anywhere; }.branch-flow.is-running .branch-token { animation:branch-token-travel 3.8s cubic-bezier(.35,.05,.65,.95) infinite; animation-delay:calc(var(--branch-index, 0) * .25s); }.branch-row:nth-child(1) .branch-token { --branch-index:0; }.branch-row:nth-child(2) .branch-token { --branch-index:1; }.branch-row:nth-child(3) .branch-token { --branch-index:2; }.branch-row:nth-child(4) .branch-token { --branch-index:3; }.branch-row:nth-child(5) .branch-token { --branch-index:4; }.branch-convergence { justify-content:center; }.branch-convergence .branch-node { border-color:var(--accent); }.convergence-line { position:absolute; inset:10% auto 10% 0; border-left:1px dashed var(--accent); opacity:.65; }.branch-tail { grid-column:1 / -1; display:grid; grid-template-columns:minmax(0,1fr) auto minmax(0,1fr) auto minmax(0,1fr); gap:.7rem; align-items:start; margin-top:.25rem; padding-top:1rem; border-top:1px dashed var(--line); }.branch-tail[data-tail="planner-synthesis"] { grid-template-columns:minmax(0,1fr) auto minmax(0,1fr); }.branch-tail-node { min-width:0; }.tail-arrow { color:var(--accent); align-self:center; font-size:1.3rem; font-weight:800; }.candidate-filter { display:grid; gap:.3rem; margin-top:.45rem; }.candidate { display:block; padding:.25rem .4rem; border-radius:5px; font-size:.71rem; font-weight:700; }.candidate-accepted { color:#b8d0d5; background:#122227; }.candidate-rejected { color:#9aa8ac; background:#1b2427; text-decoration:line-through; }.provenance-ledger { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.35rem .5rem; margin:.55rem 0 0; padding:.55rem; border:1px solid var(--line); border-radius:7px; background:#050809; }.provenance-ledger div { min-width:0; }.provenance-ledger dt { color:var(--accent); font-size:.64rem; font-weight:800; overflow-wrap:anywhere; }.provenance-ledger dd { margin:0; color:var(--muted); font-size:.68rem; overflow-wrap:anywhere; }.branch-status { grid-column:1 / -1; margin:.2rem 0 0; color:var(--muted); font-size:.78rem; }
.graph-visual { position:relative; margin-top:1rem; padding:1rem; border:1px solid #344b52; border-radius:12px; background:#050a0c; overflow:hidden; }.graph-visual-head { display:flex; justify-content:space-between; gap:1rem; align-items:center; flex-wrap:wrap; margin-bottom:.8rem; }.graph-caption,.graph-order { color:var(--muted); font-size:.82rem; }.graph-order { margin:.2rem 0 1rem; }.graph-token-lane { display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; min-height:2rem; margin-bottom:.9rem; border-top:1px dashed var(--line); border-bottom:1px dashed var(--line); padding:.55rem 0; overflow:hidden; }.graph-token { display:block; max-width:100%; padding:.2rem .5rem; border:1px solid #83b8c3; border-radius:999px; background:#071114; color:#fff; font-size:.73rem; font-weight:700; opacity:0; overflow-wrap:anywhere; }.graph-visual.is-running .graph-token { animation:graph-token-travel 3.8s cubic-bezier(.35,.05,.65,.95) both; animation-delay:calc(var(--graph-token-index) * .38s); }.graph-stage-list { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.55rem; align-items:start; }.graph-stage { min-width:0; }.graph-stage h3 { margin:0 0 .45rem; color:var(--accent); font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; }.graph-stage-nodes { display:grid; gap:.5rem; }.graph-node { min-width:0; min-height:5.6rem; padding:.65rem; text-align:left; border:1px solid var(--accent); border-radius:10px; color:var(--ink); background:#0c171a; font:inherit; cursor:pointer; display:flex; flex-direction:column; justify-content:center; gap:.1rem; opacity:0; transform:translateY(.8rem); }.graph-visual.is-running .graph-node { animation:graph-node-reveal 3.2s cubic-bezier(.35,.05,.65,.95) both; animation-delay:calc(var(--graph-stage) * .45s); }.graph-node:hover,.graph-node.is-active { transform:translateY(-3px); background:#17343c; }.graph-node span { color:var(--accent); text-transform:uppercase; font-size:.62rem; letter-spacing:.1em; font-weight:800; }.graph-node strong { overflow-wrap:anywhere; font-size:.85rem; line-height:1.2; }.graph-node small { color:var(--muted); font-size:.7rem; line-height:1.2; overflow-wrap:anywhere; }.graph-node-document { border-color:var(--accent); }.graph-node-claim,.graph-node-entity,.graph-node-evidencechunk,.graph-node-mention { border-color:#9db1b6; }.graph-edge-list { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.55rem; list-style:none; padding:0; margin:1rem 0 0; }.graph-edge { min-width:0; padding:.55rem .65rem; border:1px dashed var(--line); border-radius:9px; opacity:0; transform:translateY(.5rem); }.graph-visual.is-running .graph-edge { animation:graph-edge-reveal 2.4s ease both; animation-delay:calc(1.4s + var(--graph-edge-stage) * .25s); }.graph-edge-label { display:block; color:#fff; font-size:.76rem; font-weight:750; overflow-wrap:anywhere; }.graph-edge small { display:block; margin-top:.2rem; color:var(--muted); font-size:.7rem; }.graph-edge-from_document,.graph-edge-evidence_record,.graph-edge-supports,.graph-edge-contextualizes,.graph-edge-has_subject,.graph-edge-has_object,.graph-edge-evidenced_by { border-color:#9db1b6; }.graph-edge-from_document .graph-edge-label,.graph-edge-evidence_record .graph-edge-label,.graph-edge-supports .graph-edge-label,.graph-edge-contextualizes .graph-edge-label,.graph-edge-has_subject .graph-edge-label,.graph-edge-has_object .graph-edge-label,.graph-edge-evidenced_by .graph-edge-label { color:#b8d0d5; }.graph-edge-legend { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:.8rem; color:var(--muted); font-size:.72rem; }.graph-legend-item { padding:.2rem .45rem; border:1px dashed var(--line); border-radius:999px; }.graph-legend-endpoints b,.graph-legend-stance b,.graph-legend-provenance b { color:#b8d0d5; }.graph-status { margin:.8rem 0 0; color:var(--muted); font-size:.78rem; }
@keyframes flow-token-horizontal { 0% { transform:translateX(0); opacity:0; } 10% { transform:translateX(0); opacity:1; } 80% { transform:translateX(var(--token-distance, 0px)); opacity:1; } 100% { transform:translateX(var(--token-distance, 0px)); opacity:0; } }
.example-input { padding:1rem 1.2rem; border-left:4px solid #83b8c3; margin:1.2rem 0; }.example-input span { color:#83b8c3; text-transform:uppercase; letter-spacing:.12em; font-size:.75rem; font-weight:750; }.example-input p { margin:.45rem 0 0; color:#fff; }
.example-controls { justify-content:flex-start; margin:1.1rem 0; }.example-status { color:var(--muted); font-size:.9rem; }
.example-steps { display:grid; gap:.75rem; }.example-step { display:none; padding:1.1rem; grid-template-columns:3rem 1fr; gap:1rem; }.example-step.is-visible { display:grid; }.example-step[aria-hidden="true"] { visibility:hidden; }.step-number { color:var(--accent); font-size:1.25rem; font-weight:800; }.step-boundary { color:var(--muted); font-size:.9rem; margin-bottom:0; }.step-boundary strong { color:#fff; }
.table-scroll { overflow-x:auto; margin-top:1.3rem; } table { width:100%; border-collapse:collapse; min-width:620px; } th,td { text-align:left; vertical-align:top; padding:.85rem .75rem; border:1px solid var(--line); } th { color:#fff; background:#101416; } td { color:var(--muted); } tbody th { color:var(--accent); }
.viva-list { display:grid; gap:.8rem; }.viva-panel summary { cursor:pointer; list-style:none; padding:1rem 1.1rem; display:flex; flex-wrap:wrap; gap:.55rem .9rem; align-items:baseline; }.viva-panel summary::-webkit-details-marker { display:none; }.viva-panel summary span { color:var(--accent); text-transform:uppercase; letter-spacing:.1em; font-size:.72rem; font-weight:800; }.viva-panel summary strong { font-size:1.02rem; }.viva-answer { border-top:1px solid var(--line); padding:1rem 1.1rem; color:var(--muted); }.viva-answer strong { color:#fff; }
.glossary-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.8rem; margin:1.3rem 0 0; }.glossary-card { padding:1rem; background:var(--panel); border:1px solid var(--line); border-radius:12px; }.glossary-card dt { color:var(--accent); font-weight:800; }.glossary-card dd { margin:.45rem 0 0; color:var(--muted); font-size:.92rem; }
.sr-only { position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); clip-path:inset(50%); border:0; }
.chapter-controls { display:flex; justify-content:space-between; gap:1rem; padding:2.2rem 0 4rem; border-top:1px solid var(--line); }.chapter-controls a { padding:.7rem 1rem; border:1px solid var(--line); border-radius:7px; }.chapter-controls a:hover { border-color:var(--accent); text-decoration:none; }.next-link { text-align:right; }.site-footer { color:var(--muted); border-top:1px solid var(--line); padding:1.5rem 0 3rem; font-size:.85rem; }
@media (max-width:720px) { .site-nav { align-items:flex-start; flex-direction:column; }.chapter-links { justify-content:flex-start; }.page { width:min(100% - 1.2rem,1280px); }.hero { padding-top:3rem; }.finding-card { min-height:0; }.visual-wrap { margin-left:-.2rem; margin-right:-.2rem; }.architecture-svg { display:none; }.flow-stage-list { grid-template-columns:1fr; row-gap:2.5rem; margin:0; padding:0; }.flow-node { position:relative; min-height:0; min-height:5.1rem; }.flow-stage-arrow { position:absolute; top:calc(100% + .5rem); left:50%; display:block; width:2rem; height:1.1rem; text-align:center; transform:translateX(-50%) rotate(90deg); }.flow-stage-arrow:nth-of-type(n) { left:50%; }.flow-lane { position:relative; height:auto; display:grid; gap:.4rem; margin-top:1rem; border-top:1px dashed var(--line); padding-top:.65rem; }.flow-token { position:relative; top:auto; left:auto; max-width:100%; width:fit-content; white-space:normal; }.flow-diagram.is-running .flow-token { animation-name:flow-token-vertical; }.glossary-grid { grid-template-columns:1fr 1fr; }.chapter-controls { align-items:stretch; flex-direction:column; }.next-link { text-align:left; }.graph-stage-list { grid-template-columns:1fr; gap:.9rem; }.graph-stage { min-width:0; }.graph-stage-nodes { grid-template-columns:repeat(2,minmax(0,1fr)); }.graph-edge-list { grid-template-columns:1fr; }.graph-visual-head { align-items:flex-start; flex-direction:column; gap:.2rem; } }
.branch-flow { container-type:inline-size; }
@container (max-width:720px) { .branch-flow { display:flex; flex-direction:column; gap:.75rem; }.branch-source,.branch-convergence { width:100%; }.branch-source::after,.branch-convergence::before { display:none; }.branch-rows { gap:.5rem; }.branch-row { grid-template-columns:minmax(0,1fr); }.branch-row > .branch-connector { grid-column:1; grid-row:2; width:100%; height:1.8rem; }.branch-row::after { display:none; }.branch-connector::before { content:'↓'; text-align:center; }.branch-token { top:0; transform:translateY(0); }.branch-flow.is-running .branch-token { animation-name:branch-token-travel-vertical; }.branch-tail,.branch-tail[data-tail="planner-synthesis"] { grid-template-columns:minmax(0,1fr); }.tail-arrow { transform:rotate(90deg); justify-self:center; }.provenance-ledger { grid-template-columns:1fr 1fr; } }
@media (max-width:720px) { .branch-flow { display:flex; flex-direction:column; gap:.75rem; }.branch-source,.branch-convergence { width:100%; }.branch-source::after,.branch-convergence::before { display:none; }.branch-rows { gap:.5rem; }.branch-row { grid-template-columns:minmax(0,1fr); }.branch-row > .branch-connector { grid-column:1; grid-row:2; width:100%; height:1.8rem; }.branch-row::after { display:none; }.branch-connector::before { content:'↓'; text-align:center; }.branch-token { top:0; transform:translateY(0); }.branch-flow.is-running .branch-token { animation-name:branch-token-travel-vertical; }.branch-tail,.branch-tail[data-tail="planner-synthesis"] { grid-template-columns:minmax(0,1fr); }.tail-arrow { transform:rotate(90deg); justify-self:center; }.provenance-ledger { grid-template-columns:1fr 1fr; } }
@keyframes flow-token-vertical { 0% { opacity:0; transform:translateY(-.8rem); } 12% { opacity:1; transform:translateY(-.8rem); } 80% { opacity:1; transform:translateY(0); } 94%,100% { opacity:0; transform:translateY(.8rem); } }
@keyframes branch-token-travel { 0% { opacity:0; transform:translate(-.2rem,-50%); } 12% { opacity:1; transform:translate(0,-50%); } 80% { opacity:1; transform:translate(3.2rem,-50%); } 94%,100% { opacity:0; transform:translate(3.2rem,-50%); } }
@keyframes branch-token-travel-vertical { 0% { opacity:0; transform:translateY(-.65rem); } 12% { opacity:1; transform:translateY(0); } 80% { opacity:1; transform:translateY(.65rem); } 94%,100% { opacity:0; transform:translateY(1rem); } }
@keyframes graph-node-reveal { 0% { opacity:0; transform:translateY(.8rem); } 18% { opacity:1; transform:translateY(.8rem); } 100% { opacity:1; transform:translateY(0); } }
@keyframes graph-edge-reveal { 0% { opacity:0; transform:translateY(.5rem); } 100% { opacity:1; transform:translateY(0); } }
@keyframes graph-token-travel { 0% { opacity:0; transform:translateX(-1rem); } 12% { opacity:1; transform:translateX(0); } 80% { opacity:1; transform:translateX(1.4rem); } 100% { opacity:0; transform:translateX(2rem); } }
.graph-visual.is-running .graph-node { animation-duration:1.5s; animation-delay:calc(var(--graph-stage) * .25s); }
.graph-visual.is-running .graph-edge { animation-duration:1.1s; animation-delay:calc(.5s + var(--graph-edge-stage) * .15s); }
.graph-visual.is-running .graph-token { animation-duration:2.4s; animation-delay:calc(var(--graph-token-index) * .25s); }
@media (prefers-reduced-motion:reduce) { html { scroll-behavior:auto; } *,*::before,*::after { animation:none !important; transition:none !important; }.flow-lane { display:flex; flex-wrap:wrap; gap:.5rem; }.flow-token { position:relative; left:auto; top:auto; opacity:1; transform:none !important; }.branch-token { opacity:1; transform:none !important; }.graph-node,.graph-edge,.graph-token { opacity:1 !important; transform:none !important; pointer-events:auto; } }
"""


SCRIPT = r"""
(function () {
  const progress = document.getElementById('progress-bar');
  const updateProgress = () => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.width = (max > 0 ? (window.scrollY / max) * 100 : 100) + '%';
  };
  window.addEventListener('scroll', updateProgress, { passive: true });
  window.addEventListener('resize', updateProgress);
  updateProgress();

  const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const syncFlowTokens = (flow) => {
    const lane = flow.querySelector('.flow-lane');
    if (!lane) return;
    const laneWidth = lane.clientWidth;
    lane.querySelectorAll('.flow-token').forEach((token) => {
      const distance = Math.max(0, laneWidth - token.offsetWidth);
      token.style.setProperty('--token-distance', distance + 'px');
    });
  };
  const toggleNode = (node) => {
    const active = node.getAttribute('aria-pressed') !== 'true';
    node.classList.toggle('is-active', active || document.activeElement === node);
    node.setAttribute('aria-pressed', String(active));
    const flow = node.closest('.flow-diagram');
    const status = flow && flow.querySelector('.branch-status');
    if (status && node.textContent) status.textContent = (active ? 'Selected: ' : 'Unselected: ') + node.textContent.replace(/\s+/g, ' ').trim();
    const graph = node.closest('.graph-visual');
    const graphStatus = graph && graph.querySelector('.graph-status');
    if (graphStatus && node.textContent) graphStatus.textContent = (active ? 'Selected: ' : 'Unselected: ') + node.textContent.replace(/\s+/g, ' ').trim();
  };
  const setGraphComplete = (graph, complete) => {
    graph.querySelectorAll('.graph-node, .graph-edge, .graph-token').forEach((item) => item.classList.toggle('is-visible', complete));
    if (complete) {
      const status = graph.querySelector('.graph-status');
      if (status) status.textContent = 'Graph complete: all source-to-hub provenance is visible.';
    }
  };
  const restartGraph = (graph) => {
    graph.classList.remove('is-running');
    setGraphComplete(graph, false);
    graph.querySelectorAll('.graph-node, .graph-edge, .graph-token').forEach((item) => {
      item.addEventListener('animationend', () => item.classList.add('is-visible'));
    });
    if (reducedMotion()) {
      setGraphComplete(graph, true);
      graph.classList.add('is-running');
      return;
    }
    const firstNode = graph.querySelector('.graph-node');
    if (firstNode) firstNode.classList.add('is-visible');
    window.requestAnimationFrame(() => {
      graph.classList.add('is-running');
    });
  };
  document.querySelectorAll('.flow-diagram, .graph-visual').forEach((flow) => {
    syncFlowTokens(flow);
    flow.classList.add('is-running');
    flow.querySelectorAll('.branch-flow').forEach((branch) => branch.classList.add('is-running'));
    if (flow.classList.contains('graph-visual')) {
      flow.querySelectorAll('.graph-node, .graph-edge, .graph-token').forEach((item) => {
        item.addEventListener('animationend', () => item.classList.add('is-visible'));
      });
      if (reducedMotion()) setGraphComplete(flow, true);
    }
    flow.querySelectorAll('.flow-node, .branch-node, .graph-node').forEach((node) => {
      node.addEventListener('focus', () => node.classList.add('is-active'));
      node.addEventListener('blur', () => node.classList.remove('is-active'));
      node.addEventListener('click', () => {
        if (node.dataset.skipClick === 'true') { delete node.dataset.skipClick; return; }
        toggleNode(node);
      });
      node.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); node.dataset.skipClick = 'true'; toggleNode(node); }
      });
    });
    flow.querySelectorAll('.branch-token').forEach((token, index) => token.style.setProperty('--branch-index', String(index)));
  });
  window.addEventListener('resize', () => {
    document.querySelectorAll('.flow-diagram, .graph-visual').forEach(syncFlowTokens);
  });
  document.querySelectorAll('.replay-animation').forEach((button) => {
    button.addEventListener('click', () => {
      const flow = button.closest('.visual-wrap').querySelector('.flow-diagram');
      flow.classList.remove('is-running');
      flow.querySelectorAll('.branch-flow').forEach((branch) => branch.classList.remove('is-running'));
      // A frame boundary restarts the CSS path animation without staged timers.
      if (reducedMotion()) { flow.classList.add('is-running'); flow.querySelectorAll('.branch-flow').forEach((branch) => branch.classList.add('is-running')); return; }
      window.requestAnimationFrame(() => { flow.classList.add('is-running'); flow.querySelectorAll('.branch-flow').forEach((branch) => branch.classList.add('is-running')); });
    });
  });
  document.querySelectorAll('.replay-graph').forEach((button) => {
    button.addEventListener('click', () => restartGraph(button.closest('.graph-visual')));
  });
  document.querySelectorAll('.example-section').forEach((section) => {
    const steps = [...section.querySelectorAll('.example-step')];
    const status = section.querySelector('.example-status');
    const button = section.querySelector('.example-next');
    let current = 0;
    const show = () => {
      steps.forEach((step, index) => { const active = index === current; step.classList.toggle('is-visible', active); step.setAttribute('aria-hidden', String(!active)); });
      status.textContent = 'Stage ' + (current + 1) + ' of ' + steps.length;
      button.textContent = current === steps.length - 1 ? 'Restart example' : 'Walk through example';
    };
    button.addEventListener('click', () => { current = current === steps.length - 1 ? 0 : current + 1; show(); });
    show();
  });
}());
"""


SHARED_EXAMPLE = (
    {
        "label": "Input and normalization",
        "body": "The row is normalized as stored: post text, target, category, dataset identity, and row key remain available for the audit trail.",
        "boundary": "This is the input hate speech, not evidence and not a factual claim to repeat.",
    },
    {
        "label": "Prompt context",
        "body": "The fixed baseline prompt passes Post and Target to a JSON-output contract. It does not inject a language instruction, demonstrations, or external evidence.",
        "boundary": "The input language remains metadata; this baseline does not guarantee same-language output.",
    },
    {
        "label": "Model-emitted reasoning",
        "body": "Qwen may emit a locally captured thinking trace before its answer. This model-emitted reasoning is kept conceptually separate from deterministic audit metadata.",
        "boundary": "A reasoning trace is not proof; request IDs, evidence IDs, hashes, timings, schema status, and checkpoints are deterministic metadata.",
    },
    {
        "label": "Validated response",
        "body": "The final phase asks for schema-constrained JSON, then parsing and validation decide whether the row can be persisted and exported to Excel.",
        "boundary": "The method cannot know whether its tone changed a reader or guarantee that an unsupported statement is true.",
    },
)


ZERO_SHOT: dict[str, Any] = {
    "slug": "zero-shot-architecture",
    "title": "Zero-Shot Architecture",
    "short_title": "Zero-shot",
    "eyebrow": "Chapter 01 · generation baseline",
    "dek": "A transparent baseline that measures the base generator with its fixed system and user prompt before demonstrations or retrieval are introduced.",
    "visual_label": "A normalized input moves through the fixed prompt and two-phase Qwen generation path to a validated output.",
    "findings": [
        {"title": "Scientific baseline", "summary": "No demonstrations or external evidence means changes can be attributed to the base pipeline.", "severity": "High", "metric": "1 input → 1 model path"},
        {"title": "Two-phase answer", "summary": "Thinking is captured first, then a constrained final phase targets parseable output.", "severity": "Medium", "metric": "Reasoning + JSON validation"},
        {"title": "Resumable audit", "summary": "Identity-aware checkpoints preserve progress and make interrupted rows recoverable.", "severity": "Low", "metric": "row key · request ID · hash"},
    ],
    "stages": [
        {"label": "Input row", "detail": "normalize"},
        {"label": "Prompt", "detail": "Post + Target"},
        {"label": "Qwen3.5-4B", "detail": "FastModel · 4-bit"},
        {"label": "Final phase", "detail": "schema JSON"},
        {"label": "Checkpoint", "detail": "Excel trace"},
    ],
    "movements": [
        {"kind": "record", "label": "input row"},
        {"kind": "prompt-context", "label": "fixed prompt context"},
        {"kind": "validated-output", "label": "validated JSON"},
    ],
    "sections": [
        {"id": "why-baseline", "kicker": "The control condition", "heading": "Why zero-shot comes first", "intro": "Zero-shot is the scientific baseline: it measures a fixed system instruction plus user prompt when no demonstrations, retrieval evidence, or weight updates are supplied.", "bullets": ["Dataset record normalization preserves the input language as metadata; language preservation is not an output guarantee in this baseline.", "The fixed system instruction requires one JSON object, while the user prompt supplies Post and Target; it does not inject a language instruction.", "Low prompt-engineering overhead and no demonstration bias make attribution straightforward, even when task consistency is weaker."]},
        {"id": "generation-path", "kicker": "Implemented path", "heading": "One input, two generation phases", "intro": "Unsloth FastModel loads Qwen3.5-4B with 4-bit weights; Transformers model.generate runs the local batched inference path. The first phase enables thinking and captures a locally emitted reasoning trace; a separate final phase requests schema-constrained JSON.", "bullets": ["The generation trace is model-emitted reasoning, not a deterministic explanation produced by the audit layer.", "JSON parsing and validation reject malformed or incomplete answers before persistence.", "Checkpoint persistence records resumable row state; an Excel trace provides a reviewable export while preserving the distinction between generated text and metadata."]},
        {"id": "tradeoffs", "kicker": "Design trade-off", "heading": "Simple attribution, limited grounding", "intro": "This baseline is easy to interpret as a model behavior measurement, but it cannot consult a factual corpus.", "table": {"headers": ["Dimension", "Strength", "Limitation"], "rows": [["Prompting", "Low overhead; behavior is attributable to the base model", "No demonstrations to stabilize tone or structure"], ["Knowledge", "No retrieval errors or corpus setup", "No external factual grounding; hallucination risk"], ["Output", "Two-phase validation creates an explicit contract", "Variable tone and occasional missing final JSON"], ["Operations", "Checkpoint rows can resume", "Reasoning can consume the full token budget"]]}},
    ],
    "example_steps": list(SHARED_EXAMPLE),
    "challenges": [
        {"symptom": "Reasoning consumes the full token budget", "root": "Thinking-enabled generation has no room left for the final answer.", "correction": "Separate thinking-enabled first phase from a shorter schema-constrained final phase.", "lesson": "Budget the observable answer independently from the useful trace."},
        {"symptom": "Final JSON is missing", "root": "A model can stop after its locally emitted reasoning trace.", "correction": "Parse, validate, and checkpoint only after the final phase produces the required fields.", "lesson": "A successful model call is not a successful record."},
        {"symptom": "A run stops mid-dataset", "root": "Long generation jobs can be interrupted or lose a worker.", "correction": "Persist row identity and checkpoint state so recovery resumes append-only.", "lesson": "Operational recovery is part of the scientific method."},
    ],
    "viva": [
        {"question": "Why call this a baseline if the output is already useful?", "short": "It isolates base-model behavior before adding demonstrations, retrieval, or multi-perspective synthesis.", "deep": "The baseline gives later variants a fair reference. Improvements can be discussed as effects of changed information or orchestration rather than assumed model upgrades."},
        {"question": "What is the difference between reasoning and audit metadata?", "short": "Reasoning is locally emitted by Qwen; IDs, hashes, timings, schema status, and checkpoints are deterministic metadata.", "deep": "The trace can be incomplete or model-dependent, while metadata identifies exactly which request and evidence package were processed. Neither category alone proves factual correctness."},
    ],
}


FEW_SHOT: dict[str, Any] = {
    "slug": "few-shot-architecture",
    "title": "Few-Shot Architecture",
    "short_title": "Few-shot",
    "eyebrow": "Chapter 02 · in-context learning",
    "dek": "The zero-shot pipeline with English-only frozen demonstrations that teach response behavior at inference time, without updating model weights.",
    "visual_label": "Frozen demonstration cards condition the same two-phase Qwen generation path.",
    "findings": [
        {"title": "Frozen demonstrations", "summary": "A fixed, reviewable example set teaches tone and structure while keeping evaluation comparable.", "severity": "High", "metric": "weights unchanged"},
        {"title": "More stable behavior", "summary": "English examples show safety, tone, and answer shape directly in the prompt.", "severity": "Medium", "metric": "prompt-context signal"},
        {"title": "Grounding remains absent", "summary": "A polished answer can still contain unsupported claims because examples are not a factual corpus.", "severity": "High", "metric": "same evidence universe: none"},
    ],
    "stages": [
        {"label": "Frozen demos", "detail": "select + format"},
        {"label": "Input row", "detail": "normalize"},
        {"label": "Prompt", "detail": "ICL context"},
        {"label": "Qwen3.5-4B", "detail": "same weights"},
        {"label": "Validated", "detail": "JSON + checkpoint"},
    ],
    "movements": [
        {"kind": "demonstration-card", "label": "English-only demonstration card"},
        {"kind": "record", "label": "new input row"},
        {"kind": "learned-pattern", "label": "learned response pattern"},
        {"kind": "validated-output", "label": "validated JSON"},
    ],
    "sections": [
        {"id": "inheritance", "kicker": "Inherited baseline", "heading": "What changes from zero-shot", "intro": "Few-shot inherits the baseline's fixed JSON system/user prompt, normalization, two-phase Unsloth FastModel + Transformers path, validation, checkpoint persistence, and Excel trace export. It adds English-only frozen demonstrations to prompt context.", "bullets": ["Selection and formatting are deterministic: the English-only examples are fixed before evaluation and shown as input/output pairs.", "Demonstrations teach tone, structure, and safety behavior without changing model weights; they do not provide facts or a language instruction.", "This is in-context learning, not fine-tuning: adaptation happens inside the prompt and disappears when the context is removed."]},
        {"id": "demonstrations", "kicker": "Prompt contract", "heading": "A pattern the model can see", "intro": "Each English-only demonstration pairs a short harmful input with a respectful English counter-narrative and explicit output shape. The new input then follows the same JSON contract.", "bullets": ["Frozen examples make fair evaluation possible: every row sees the same teaching signal and example order.", "The learned response pattern flows into generation, but it is not a claim that demonstration wording is universally appropriate or that output will match the input language.", "The inherited two-phase design still separates model-emitted reasoning from deterministic request IDs, evidence IDs, hashes, timings, schema status, and checkpoints."]},
        {"id": "tradeoffs", "kicker": "Design trade-off", "heading": "Consistency bought with context", "intro": "Few-shot can outperform zero-shot on tone and task consistency while retaining the baseline's lack of external grounding.", "table": {"headers": ["Dimension", "Benefit over zero-shot", "Remaining risk"], "rows": [["Behavior", "Examples make structure and safety concrete", "Demonstration bias can overfit a preferred style"], ["Operations", "Frozen order supports fair evaluation", "Example-order sensitivity and context-window cost"], ["Knowledge", "A relevant example can clarify task framing", "No external factual grounding; leakage risk"], ["Model", "No weight update; quick to change the prompt", "Still subject to reasoning truncation and invalid JSON"]]}},
    ],
    "example_steps": [
        SHARED_EXAMPLE[0],
        {"label": "Frozen demonstrations", "body": "A fixed pair of safe, English-only demonstrations is placed before the new input. Their response patterns teach tone and structure, not facts about the target group or the input language.", "boundary": "Demonstrations are evidence of formatting behavior, not external evidence for the answer."},
        {"label": "In-context generation", "body": "The same Qwen3.5-4B two-phase process sees the examples and emits a more stable, empathetic response pattern without fine-tuning.", "boundary": "The example order can change behavior, and context-window cost grows with every demonstration."},
        {"label": "Validated response", "body": "The final phase is parsed and validated, then checkpointed and exported. The result may sound more consistent than zero-shot while remaining unsupported by a factual corpus.", "boundary": "Few-shot cannot guarantee factual truth merely because its tone is confident or its JSON is valid."},
    ],
    "challenges": [
        {"symptom": "Context-window cost grows", "root": "Every frozen demonstration consumes prompt tokens alongside the input and reasoning budget.", "correction": "Keep a small, fixed set and measure prompt length before generation.", "lesson": "Prompt examples are a compute and evaluation variable."},
        {"symptom": "Example order changes tone", "root": "In-context learning is sensitive to recency and ordering.", "correction": "Freeze selection and order for all evaluation rows.", "lesson": "Fair comparisons require a stable prompt contract."},
        {"symptom": "A demonstration leaks into an answer", "root": "Overly similar examples can expose or copy sensitive wording.", "correction": "Review and minimize demonstrations, then validate generated text and keep the examples versioned.", "lesson": "Teaching behavior is not the same as supplying facts."},
    ],
    "viva": [
        {"question": "Why is few-shot not fine-tuning?", "short": "Weights stay frozen; examples are temporary context supplied at inference time.", "deep": "Fine-tuning changes parameters and persists a learned update. Here, selection and formatting are prompt operations, so removing the demonstrations returns the model to zero-shot behavior."},
        {"question": "Why freeze the demonstrations for evaluation?", "short": "To prevent example selection or order from becoming an uncontrolled source of improvement.", "deep": "A frozen set makes row-to-row and architecture-to-architecture comparisons defensible. It also exposes context-window cost, demonstration bias, and leakage risk instead of hiding them in adaptive retrieval."},
    ],
}


KG_RAG: dict[str, Any] = {
    "slug": "kg-rag-architecture",
    "title": "KG-RAG Architecture",
    "short_title": "KG-RAG",
    "eyebrow": "Chapter 03 · grounded retrieval",
    "dek": "A hybrid retrieval path that combines exact lexical matches, dense semantic search, and bounded knowledge-graph expansion before a cross-encoder reranker selects auditable evidence.",
    "visual_label": "One neutral query fans into BM25, BGE-M3/Qdrant, and two-hop graph branches; RRF fuses candidates before BGE Reranker v2-M3 and Qwen.",
    "branch_labels": [
        {"label": "BM25 lexical", "detail": "token overlap / rare terms"},
        {"label": "BGE-M3 dense", "detail": "semantic vectors in Qdrant"},
        {"label": "KG expansion", "detail": "accepted seeds · two hops"},
    ],
    "branch_flow": {
        "variant": "kg-rag",
        "label": "KG-RAG query fan-out, RRF fan-in, reranker rejection, and bounded evidence",
        "source": "Neutral query",
        "source_detail": "input + task framing",
        "branches": [
            {"label": "BM25", "detail": "lexical rank · rare terms", "token": "BM25 candidate"},
            {"label": "BGE-M3 + Qdrant", "detail": "dense vector rank", "token": "dense candidate"},
            {"label": "Two-hop graph", "detail": "bounded KG expansion", "token": "graph candidate"},
        ],
        "convergence_key": "rrf",
        "convergence": "RRF merge",
        "convergence_detail": "dense 1.1 · BM25 1.0 · graph 0.8 · k=60",
        "rejection": {"score": "0.31"},
    },
    "findings": [
        {"title": "Three retrieval signals", "summary": "Lexical, dense, and graph candidates recover different kinds of relevant context.", "severity": "High", "metric": "BM25 · BGE-M3 · 2 hops"},
        {"title": "Fusion before scoring", "summary": "Reciprocal Rank Fusion combines ranked lists, then a cross-encoder judges input–passage relevance.", "severity": "Medium", "metric": "1.1 / 1.0 / 0.8 · k=60"},
        {"title": "Auditable evidence", "summary": "Only high-scoring passages enter a bounded ledger with immutable IDs and citation checks.", "severity": "Low", "metric": "p ≥ 0.55 · max 5"},
    ],
    "stages": [
        {"label": "Neutral query", "detail": "no reference answer"},
        {"label": "BM25 branch", "detail": "lexical tokens"},
        {"label": "BGE-M3 + Qdrant", "detail": "dense vectors"},
        {"label": "Graph branch", "detail": "up to two hops"},
        {"label": "RRF merge", "detail": "weighted ranks"},
        {"label": "Reranker", "detail": "BGE v2-M3 ≥ 0.55"},
        {"label": "Evidence ledger", "detail": "≤ 5 passages"},
        {"label": "Qwen final", "detail": "citations + JSON"},
    ],
    "movements": [
        {"kind": "neutral-query", "label": "neutral query"},
        {"kind": "bm25-candidate", "label": "BM25 candidate"},
        {"kind": "dense-candidate", "label": "BGE-M3 candidate"},
        {"kind": "graph-candidate", "label": "two-hop graph candidate"},
        {"kind": "rrf-fused", "label": "RRF fused passage"},
        {"kind": "reranked-evidence", "label": "evidence ID · score 0.55+"},
    ],
    "sections": [
        {"id": "retrieval-path", "kicker": "Implemented retrieval path", "heading": "Three branches, one evidence universe", "intro": "The query is constructed from the hate-speech input and task framing without exposing a reference answer. Each retrieval branch returns candidate passages; none is treated as proof until it survives fusion, reranking, and citation validation.", "bullets": ["BM25 is lexical retrieval: it rewards token overlap and is valuable for exact targets, names, or rare terms that semantic similarity may blur.", "BGE-M3 supplies dense embeddings; Qdrant performs the vector search. BGE-M3 is the embedding model, not the generator and not the reranker.", "Accepted dense/BM25 seeds expand through semantic knowledge-graph links for at most two graph hops, adding structural context without allowing an unbounded traversal.", "The graph augments BM25 and dense retrieval; it does not replace either signal or guarantee that a connected passage matches the input." ]},
        {"id": "fusion-reranking", "kicker": "Ranking contract", "heading": "Retrieval is not reranking", "intro": "Retrieval cheaply gathers a broad candidate set using independent ranked lists. Reranking is a second, more expensive relevance judgment over the hate-speech input and each candidate passage together.", "table": {"headers": ["Stage", "Question answered", "Implemented contract"], "rows": [["Retrieval", "Which passages might be useful?", "BM25, BGE-M3/Qdrant, and graph branches produce ranked candidates"], ["RRF", "How do independent lists agree?", "Dense weight 1.1, BM25 weight 1.0, graph weight 0.8, constant 60"], ["Reranking", "Does this passage fit this input?", "BGE Reranker v2-M3 cross-encoder scores the pair"], ["Filter", "What may reach generation?", "Minimum rerank probability 0.55; at most five evidence passages"]]}},
        {"id": "evidence-ledger", "kicker": "Audit boundary", "heading": "From candidate to citation", "intro": "The evidence ledger freezes the selected passages before prompt construction. Each passage receives an immutable evidence ID, and generated citations are checked against those IDs rather than titles or mutable display labels.", "bullets": ["The ledger records evidence IDs and source provenance so a reviewer can recover the exact passage supplied to Qwen.", "Citation validation rejects references that are absent from the frozen ledger; prompt injection is treated as untrusted passage content, not as an instruction.", "The final Qwen3.5-4B stage receives the input plus the bounded evidence package, then produces a schema-validated response and resumable checkpoint. If no candidate clears the gate, the explicit abstention path is preferable to fabricated grounding." ]},
        {"id": "conventional-rag", "kicker": "Comparison", "heading": "Why hybrid KG-RAG over vector-only RAG?", "intro": "A conventional vector-only RAG path can recover semantic neighbors, but it has no lexical branch or explicit structural expansion. KG-RAG adds signals and audit boundaries at a compute cost.", "table": {"headers": ["Dimension", "Vector-only RAG", "KG-RAG here"], "rows": [["Exact terms", "Can miss rare token matches", "BM25 recovers lexical targets"], ["Semantic similarity", "Dense embedding search", "BGE-M3 dense search retained"], ["Structure", "No bounded hops", "Knowledge-graph expansion up to two hops"], ["Precision", "Retriever score only", "RRF followed by BGE Reranker v2-M3"], ["Provenance", "Depends on implementation", "Immutable evidence IDs and ledger"], ["Cost", "Lower retrieval overhead", "Three branches plus cross-encoder scoring"], ["Failure mode", "Embedding mismatch or noisy neighbors", "Also exposed to graph coverage and evidence-to-input mismatch"]]}},
    ],
    "example_steps": [
        SHARED_EXAMPLE[0],
        {"label": "Neutral query", "body": "The input is turned into a neutral retrieval query that asks for relevant factual context without leaking a reference answer into the search.", "boundary": "The query is not an answer and does not assume the target group’s facts."},
        {"label": "Candidate branches", "body": "BM25 finds exact terms, BGE-M3 with Qdrant finds semantic neighbors, and accepted seeds expand through at most two graph hops. Their candidate cards remain separate until fusion.", "boundary": "Candidates are possibilities, not selected evidence."},
        {"label": "RRF and reranking", "body": "RRF combines the ranked lists with dense 1.1, BM25 1.0, graph 0.8, and constant 60. BGE Reranker v2-M3 then scores the input–passage pair and filters below 0.55.", "boundary": "Retrieval rank and reranker probability answer different questions."},
        {"label": "Evidence ledger", "body": "At most five passages become a frozen evidence package with immutable evidence IDs. Citation validation checks those IDs before prompt injection and Qwen generation.", "boundary": "The ledger can make provenance auditable, but it cannot repair missing corpus coverage or an input–evidence mismatch."},
        {"label": "Grounded response", "body": "Qwen receives the input and bounded evidence, emits a counter-narrative under the JSON contract, and leaves a checkpointed audit trail.", "boundary": "Grounding reduces unsupported claims; it does not guarantee factual truth or human persuasion."},
    ],
    "challenges": [
        {"symptom": "Rare exact term is absent from dense neighbors", "root": "Semantic similarity can smooth away a token that matters for the target or a named concept.", "correction": "Keep BM25 as an independent lexical branch and fuse its ranked list.", "lesson": "Hybrid retrieval is a recall hedge, not duplicate infrastructure."},
        {"symptom": "Connected graph passage is irrelevant", "root": "A graph edge expresses a stored relation, not relevance to this input.", "correction": "Bound expansion to two hops, fuse with other branches, and let the cross-encoder filter it.", "lesson": "Structure needs a relevance gate and a hop budget."},
        {"symptom": "Citation names a title rather than evidence", "root": "Human-readable labels are mutable and are not the frozen prompt package.", "correction": "Build a standalone ledger with immutable evidence IDs and validate citations against it.", "lesson": "Audit identity must be explicit at the generation boundary."},
        {"symptom": "Retrieved passage contains an instruction", "root": "Corpus text can include prompt-injection-like language.", "correction": "Treat passages as untrusted data and keep prompt instructions outside the evidence payload.", "lesson": "Grounding does not transfer authority to retrieved text."},
    ],
    "viva": [
        {"question": "Why use both retrieval and reranking?", "short": "Retrieval gathers candidates cheaply; reranking reads each input–passage pair more precisely before generation.", "deep": "BM25, dense search, and graph expansion optimize recall through different signals. BGE Reranker v2-M3 is a cross-encoder precision gate, so its 0.55 threshold and five-passage cap limit what Qwen can cite."},
        {"question": "What do the RRF weights mean?", "short": "They scale each ranked list before reciprocal-rank scores are added: dense 1.1, BM25 1.0, graph 0.8, with constant 60.", "deep": "The weights express an orchestration choice, not calibrated truth probabilities. Fusion happens before cross-encoder reranking, so a high fused rank can still be removed by the 0.55 filter."},
        {"question": "Can a graph hop invent evidence?", "short": "No new fact is created by traversal, but a connected passage can still be irrelevant or unsupported.", "deep": "The graph only expands accepted seeds over two hops. Provenance, reranking, bounded evidence, and citation validation make that risk visible; corpus gaps remain a limitation."},
    ],
}


MP_KG_RAG: dict[str, Any] = {
    "slug": "mp-kg-rag-architecture",
    "title": "MP-KG-RAG Architecture",
    "short_title": "MP-KG-RAG",
    "eyebrow": "Chapter 04 · multi-perspective synthesis",
    "dek": "KG-RAG’s frozen evidence package is reviewed by five bounded analytical perspectives, then a planner and final Qwen synthesis turn those analyses into one auditable response.",
    "visual_label": "One frozen evidence package forks into five bounded perspective calls, converges in a planner, and reaches final synthesis with validation and checkpoints.",
    "branch_labels": [
        {"label": "fact_checking", "detail": "correct supported claims"},
        {"label": "cultural_context", "detail": "respectful context"},
        {"label": "harm_reduction", "detail": "avoid amplification"},
        {"label": "legal_rights", "detail": "source-grounded rights"},
        {"label": "persuasion", "detail": "empathetic guidance"},
    ],
    "branch_flow": {
        "variant": "mp-kg-rag",
        "label": "Frozen evidence fan-out into five perspectives, planner fan-in, and final synthesis",
        "source": "Frozen KG-RAG evidence",
        "source_detail": "same ≤5 passages · immutable IDs",
        "branches": [
            {"label": "fact_checking", "detail": "correct supported claims", "token": "fact_checking analysis"},
            {"label": "cultural_context", "detail": "respectful context", "token": "cultural_context analysis"},
            {"label": "harm_reduction", "detail": "avoid amplification", "token": "harm_reduction analysis"},
            {"label": "legal_rights", "detail": "source-grounded rights", "token": "legal_rights analysis"},
            {"label": "persuasion", "detail": "empathetic guidance", "token": "persuasion analysis"},
        ],
        "convergence_key": "planner",
        "convergence": "Planner",
        "convergence_detail": "five analyses → one synthesis plan",
    },
    "findings": [
        {"title": "Five views, same facts", "summary": "Perspectives vary the analysis lens while sharing one immutable evidence universe.", "severity": "High", "metric": "5 bounded calls / record"},
        {"title": "Plan before prose", "summary": "A synthesis planner turns the analyses into an explicit response strategy before final generation.", "severity": "Medium", "metric": "1 planner + 1 final"},
        {"title": "Observed audit limits", "summary": "Intermediate traces had truncation and recovery mismatches even though every final MP narrative was present.", "severity": "High", "metric": "7,750 perspectives · 1,550 finals"},
    ],
    "stages": [
        {"label": "Frozen KG-RAG evidence", "detail": "same ≤5 passages"},
        {"label": "fact_checking", "detail": "correct supported claims"},
        {"label": "cultural_context", "detail": "respectful context"},
        {"label": "harm_reduction", "detail": "avoid amplification"},
        {"label": "legal_rights", "detail": "source-grounded rights"},
        {"label": "persuasion", "detail": "empathetic guidance"},
        {"label": "Planner", "detail": "synthesis plan"},
        {"label": "Final synthesis", "detail": "Qwen + validation"},
    ],
    "movements": [
        {"kind": "frozen-evidence", "label": "one frozen evidence package"},
        {"kind": "perspective-fact-checking", "label": "fact_checking analysis"},
        {"kind": "perspective-cultural-context", "label": "cultural_context analysis"},
        {"kind": "perspective-harm-reduction", "label": "harm_reduction analysis"},
        {"kind": "perspective-legal-rights", "label": "legal_rights analysis"},
        {"kind": "perspective-persuasion", "label": "persuasion analysis"},
        {"kind": "synthesis-plan", "label": "planner output"},
        {"kind": "final-synthesis", "label": "final MP narrative"},
    ],
    "sections": [
        {"id": "inheritance", "kicker": "Inherited KG-RAG layer", "heading": "The evidence package is frozen once", "intro": "MP-KG-RAG consumes the complete KG-RAG retrieval, RRF, reranking, evidence-ledger, citation, and checkpoint layer. It does not run five different searches: all five analytical calls see the same bounded evidence package.", "bullets": ["The shared package contains up to five reranked passages with immutable evidence IDs, so perspective diversity changes interpretation rather than the source universe.", "Each perspective call emits a structured analysis and a locally emitted Qwen reasoning trace; deterministic request IDs, evidence IDs, hashes, timings, schema status, and checkpoints remain audit metadata.", "The five lenses surface harms, factual rebuttal opportunities, empathy, practical framing, and response strategy before a planner chooses how to combine them."]},
        {"id": "fork-plan-synthesis", "kicker": "Orchestration", "heading": "Fork, plan, synthesize", "intro": "The bounded fan-out holds the five perspective outputs and planner result in memory while the final Qwen3.5-4B synthesis runs. Those MP fields are included in one appended final MP row; they are not separately checkpointed.", "table": {"headers": ["Stage", "Input", "Output and audit boundary"], "rows": [["Perspectives", "One frozen evidence package + input", "Five structured analyses held in memory"], ["Planner", "Five analyses + evidence IDs", "One synthesis plan held in memory"], ["Final synthesis", "Evidence + analyses + plan", "One MP narrative; citation and final schema validation"], ["Checkpoint", "Final parsed output + MP fields", "One appended final MP row with all perspective/plan fields and identity"]]}},
        {"id": "quality-cost", "kicker": "Trade-off", "heading": "Quality breadth has an exact compute cost", "intro": "Multiple perspectives improve issue coverage and make planning explicit, but they add model stages, latency, storage, and recovery paths. The implementation does not claim row-wise monotonic improvement.", "table": {"headers": ["Benefit", "Cost or limitation", "What to say in a viva"], "rows": [["Wider issue coverage", "Six additional model stages per record (five perspectives, one planner; final synthesis is also a model stage)", "Diversity is an orchestration hypothesis measured against baselines, not a guarantee for every row"], ["Explicit plan", "Reasoning and plan traces can truncate", "Inspect intermediates and preserve checkpoint identity"], ["Auditability", "More checkpoint volume and more recovery cases", "A richer trace increases operational surface area"], ["Stable evidence", "All perspectives inherit evidence-to-input mismatch", "Diverse lenses cannot fix a missing or irrelevant passage"]]}},
        {"id": "production-observations", "kicker": "Production disclosure", "heading": "What the strict verifier actually observed", "intro": "The run-level counts describe audit intermediates and recovery behavior, not a claim that the final narratives are perfect. All 1,550 final MP outputs were present even where intermediate traces were imperfect.", "bullets": ["7,750 perspective outputs and 1,550 plans were produced for 1,550 records; there were 1,550 final MP outputs.", "43 perspective traces were truncated (43 truncated perspective traces), four plan traces were truncated (four truncated plan traces), and six perspective-number mismatches were recovered (six recovered perspective-number mismatches).", "One empty intermediate plan was observed by the strict verifier (one empty intermediate plan). These findings affected audit intermediates rather than the presence of all 1,550 final MP narratives.", "Do not claim that every strict-verifier finding was eliminated or that every row improves monotonically; disclose the bounded evidence and the remaining limitations."]},
    ],
    "example_steps": [
        SHARED_EXAMPLE[0],
        {"label": "Frozen evidence package", "body": "KG-RAG supplies the same immutable evidence IDs and at most five reranked passages to every perspective. No branch is allowed to search a different corpus slice.", "boundary": "Five perspectives are not five source sets."},
        {"label": "Five perspectives", "body": "fact_checking, cultural_context, harm_reduction, legal_rights, and persuasion each analyze the input against that same evidence. Their structured outputs expose complementary concerns.", "boundary": "Perspective diversity changes the lens, not the evidence universe."},
        {"label": "Synthesis plan", "body": "A planner reads the five in-memory analyses and drafts an ordered response strategy with evidence IDs, tone, and coverage priorities before prose generation.", "boundary": "The plan can truncate or be empty; it is carried into the final row rather than separately checkpointed."},
        {"label": "Final synthesis", "body": "Qwen receives evidence, analyses, and the plan to produce one MP narrative; citation and final schema checks run before one appended row stores the final plus MP fields.", "boundary": "The final response inherits retrieval gaps and cannot guarantee row-wise improvement or human impact."},
    ],
    "challenges": [
        {"symptom": "Perspective trace is truncated", "root": "A bounded generation budget can be consumed by locally emitted reasoning before structured analysis is complete.", "correction": "Inspect each in-memory perspective output and preserve its trace inside the final appended MP row.", "lesson": "Intermediate diversity is useful only when its failure state is visible."},
        {"symptom": "Perspective number differs after recovery", "root": "A recovered request identity and a human-readable perspective number can diverge.", "correction": "Keep immutable request IDs as the audit key and record the recovered perspective mapping explicitly.", "lesson": "Recovery metadata must not silently rewrite analytical identity."},
        {"symptom": "Planner output is empty", "root": "A model stage can return an empty intermediate despite downstream final output being present.", "correction": "Strictly verify plan content, disclose the one observed empty plan, and retain the plan inside the final appended row.", "lesson": "A present final narrative does not prove every intermediate was healthy."},
        {"symptom": "More perspectives do not help a row", "root": "All branches share the same retrieval package and can inherit its mismatch or omission.", "correction": "Report aggregate adequacy behavior only and avoid a row-wise monotonic-improvement claim.", "lesson": "Diversity broadens analysis; it is not a per-row guarantee."},
    ],
    "viva": [
        {"question": "Why freeze evidence across five perspectives?", "short": "It isolates perspective diversity from retrieval variance and keeps citations auditable.", "deep": "If each branch searched independently, a better answer could be caused by a different evidence universe. One frozen package means the experimental change is analytical orchestration, while immutable evidence IDs keep all branches referentially stable."},
        {"question": "What is the planner doing that final synthesis cannot?", "short": "It turns five structured analyses into an explicit order of claims, tone, and response strategy before prose generation.", "deep": "Planning exposes conflicts and coverage decisions as an intermediate artifact. It adds a model stage and can truncate, so the strict verifier and checkpoints are part of the design."},
        {"question": "Do five perspectives guarantee better outputs?", "short": "No. They provide broader analytical coverage but do not guarantee row-wise monotonic improvement.", "deep": "The evidence package, model budget, and intermediate reliability still constrain each row. The honest result is to compare aggregate adequacy behavior and disclose the 43 truncated traces, four truncated plans, six recovered mismatches, and one empty plan."},
    ],
}


KNOWLEDGE_GRAPH: dict[str, Any] = {
    "slug": "knowledge-graph-construction",
    "title": "Knowledge-Graph Construction",
    "short_title": "Knowledge graph",
    "eyebrow": "Chapter 05 · semantic graph construction",
    "dek": "A provenance-first pipeline turns accepted factual chunks into validated extraction records and a bounded semantic graph. BGE-M3 is loaded after construction for retrieval-time dense search.",
    "visual_label": "Documents and EvidenceChunks are registered, every accepted factual chunk goes through Qwen extraction, and only validated records become persisted graph nodes; BGE-M3 is loaded afterward for dense retrieval.",
    "findings": [
        {"title": "Healthy corpus first", "summary": "Source identities and factual coverage are checked before extraction so placeholders never become graph evidence.", "severity": "High", "metric": "PDF/HTML gate before chunks"},
        {"title": "Qwen sees every accepted chunk", "summary": "Qwen3.5-4B performs mention discovery and semantic extraction for each accepted factual chunk before graph materialization.", "severity": "High", "metric": "chunk → extraction record"},
        {"title": "Four persisted node types", "summary": "Document, EvidenceChunk, Claim, and Entity persist in the graph. Mention remains an extraction record, not a graph node.", "severity": "Medium", "metric": "4 nodes · 5 records"},
    ],
    "stages": [
        {"label": "Register sources", "detail": "identity + hash"},
        {"label": "Accepted EvidenceChunks", "detail": "factual gate + spans"},
        {"label": "Qwen extraction", "detail": "every accepted chunk"},
        {"label": "Validated records", "detail": "Mention + Claim"},
        {"label": "Persisted graph", "detail": "4 node tables"},
        {"label": "BGE-M3 retrieval index", "detail": "loaded afterward"},
        {"label": "Graph retrieval", "detail": "dense/BM25 seeds · ≤2 hops"},
    ],
    "movements": [
        {"kind": "source-document", "label": "source PDF + identity"},
        {"kind": "accepted-chunk", "label": "EvidenceChunk + span/hash"},
        {"kind": "qwen-extraction", "label": "Qwen extraction record"},
        {"kind": "validated-record", "label": "Mention + Claim records"},
        {"kind": "persisted-node", "label": "Claim + Entity nodes"},
        {"kind": "bge-retrieval", "label": "BGE-M3 retrieval vector"},
        {"kind": "graph-retrieval", "label": "seed expansion · two hops"},
    ],
    "graph_visual": {
        "nodes": [
            {"id": "doc-source", "kind": "Document", "stage": 0, "label": "document:doc-9a…", "detail": "source identity · SHA-256"},
            {"id": "evidence-12-a", "kind": "EvidenceChunk", "stage": 1, "label": "evidence:chunk-03", "detail": "accepted text · span/hash"},
            {"id": "evidence-12-b", "kind": "EvidenceChunk", "stage": 1, "label": "evidence:chunk-04", "detail": "accepted text · span/hash"},
            {"id": "mention-01", "kind": "Mention", "stage": 2, "label": "mention:m-01", "detail": "Qwen extraction record · span"},
            {"id": "mention-02", "kind": "Mention", "stage": 2, "label": "mention:m-02", "detail": "Qwen extraction record · span"},
            {"id": "claim-01", "kind": "Claim", "stage": 3, "label": "claim:CLM-01", "detail": "persisted after validation"},
            {"id": "claim-02", "kind": "Claim", "stage": 3, "label": "claim:CLM-02", "detail": "persisted after validation"},
            {"id": "entity-community", "kind": "Entity", "stage": 3, "label": "entity:community", "detail": "canonical persisted entity"},
            {"id": "entity-policy", "kind": "Entity", "stage": 3, "label": "entity:policy", "detail": "canonical persisted entity"},
            {"id": "entity-safety", "kind": "Entity", "stage": 3, "label": "entity:safety", "detail": "canonical persisted entity"},
        ],
        "stage_labels": {0: "Document record", 1: "Accepted EvidenceChunks", 2: "Qwen extraction records", 3: "Persisted Claim + Entity nodes"},
        "edges": [
            {"kind": "from_document", "from": "evidence-12-a", "to": "doc-source", "stage": 1, "detail": "EvidenceChunk records its source document"},
            {"kind": "from_document", "from": "evidence-12-b", "to": "doc-source", "stage": 1, "detail": "EvidenceChunk records its source document"},
            {"kind": "supports", "from": "evidence-12-a", "to": "claim-01", "stage": 3, "detail": "accepted evidence stance"},
            {"kind": "contextualizes", "from": "evidence-12-b", "to": "claim-02", "stage": 3, "detail": "accepted evidence stance"},
            {"kind": "has_subject", "from": "claim-01", "to": "entity-community", "stage": 3, "detail": "claim subject entity"},
            {"kind": "has_object", "from": "claim-01", "to": "entity-policy", "stage": 3, "detail": "claim object entity"},
            {"kind": "has_subject", "from": "claim-02", "to": "entity-community", "stage": 3, "detail": "claim subject entity"},
            {"kind": "has_object", "from": "claim-02", "to": "entity-safety", "stage": 3, "detail": "claim object entity"},
            {"kind": "evidenced_by", "from": "claim-01", "to": "evidence-12-a", "stage": 3, "detail": "claim keeps evidence provenance"},
            {"kind": "evidenced_by", "from": "claim-02", "to": "evidence-12-b", "stage": 3, "detail": "claim keeps evidence provenance"},
        ],
    },
    "sections": [
        {"id": "construction-pipeline", "kicker": "Implemented construction path", "heading": "Build records first, then persist graph nodes", "intro": "Every accepted factual chunk goes through Qwen extraction. Mention and Claim outputs are validated records containing bounded semantic claims; only the Document, EvidenceChunk, Claim, and Entity tables are persisted as graph nodes. BGE-M3 embeddings load after graph construction for retrieval-time dense search, not construction candidate selection.", "bullets": ["Register corpus documents and verify source identities, paths, SHA-256 hashes, and canonical provenance rows before parsing.", "Extract PDF pages and accepted HTML bodies, then apply the healthy factual-corpus gate. Page boundaries remain metadata; there is no Page graph node.", "Split accepted factual text into EvidenceChunks with document identity, character offsets, source-text key, and hash. Every accepted chunk enters Qwen mention discovery and semantic extraction.", "Validate Qwen’s structured Mention and Claim extraction records against the exact chunk text, candidate IDs, spans, and schema. Quarantine malformed or unverifiable records.", "Canonicalize accepted mention labels into Entity rows and build the Document, EvidenceChunk, Claim, and Entity tables. Mention is retained as an extraction record, not a persisted graph node.", "Persist actual typed edges: has_subject, has_object, evidence-stance relations (supports/refutes/quotes/contextualizes), from_document, and evidenced_by.", "Only after graph construction does BGE-M3 load to encode EvidenceChunks for dense retrieval. Retrieval combines those vectors with BM25 and bounded graph traversal over accepted seeds." ]},
        {"id": "graph-walkthrough", "kicker": "Animated worked graph", "heading": "From accepted chunk to persisted claim", "intro": "The visual above shows an accepted EvidenceChunk moving through Qwen extraction records into persisted Claim and Entity nodes. Mention is shown because it is a real extraction table, but it is not a persisted graph node and no Page node is invented.", "bullets": ["Document and EvidenceChunk rows retain source identity and text provenance; page numbers, spans, and hashes are properties on those records rather than Page nodes.", "Qwen Mention records capture surface text, offsets, and candidate/entity identity. Validated Claim rows reference their subject and object entities, which become canonical entity hubs.", "Evidence stance edges and from_document/evidenced_by links preserve support and recovery paths. has_subject and has_object connect claims to persisted entities."]},
        {"id": "graph-schema", "kicker": "Storage contract", "heading": "Actual tables and typed edges", "intro": "The production graph uses five table/record names, with four persisted node tables. Stable namespaced IDs and provenance are derived from the accepted extraction context; the preview does not invent Page or generic containment/mention/related edge types.", "table": {"headers": ["Table / edge", "Stored or endpoint identity", "Stored properties", "Why it exists"], "rows": [["Document", "document_uid", "source_id, source_type, authority, factual gate, document/content hashes", "Registered source identity"], ["EvidenceChunk", "evidence_chunk_id", "document_uid, chunk_id, text, source/page/span/hash metadata", "Accepted retrieval unit and evidence"], ["Mention", "mention_id (extraction record)", "chunk/document, claim occurrence, text, start/end, entity identity/status", "Record Qwen’s extracted surface mentions; not a persisted node"], ["Claim", "claim_id", "subject/object entity IDs, predicate, polarity, modality, stance, confidence, occurrence IDs", "Validated semantic assertion"], ["Entity", "entity_id", "canonical_name, entity_status, accepted status", "Canonical persisted endpoint"], ["has_subject / has_object", "Claim → Entity", "typed subject/object endpoint", "Claim argument structure"], ["supports / refutes / quotes / contextualizes", "EvidenceChunk → Claim", "evidence stance + occurrence", "Evidence relationship"], ["from_document", "EvidenceChunk → Document", "document provenance + occurrence", "Source recovery"], ["evidenced_by", "Claim → EvidenceChunk", "evidence provenance + occurrence", "Claim recovery"]]}},
        {"id": "role-separation", "kicker": "Role separation", "heading": "Each model or index has one job", "intro": "The viva answer should separate retrieval, interpretation, ranking, and generation. BGE-M3 is the embedding model; Qwen is the claim extractor. Calling every component an embedding model hides the actual audit boundary.", "table": {"headers": ["Component", "Role", "What it does here", "What it does not do"], "rows": [["BM25", "Lexical retrieval", "Matches exact tokens and rare terms for retrieval candidates", "Does not create embeddings or claims"], ["BGE-M3", "Dense embedding retrieval", "After graph construction, encodes EvidenceChunks and queries for retrieval-time dense search", "Does not choose construction candidates, generate claims, or create graph relations"], ["Qwen3.5-4B", "Semantic claim extraction + generation", "Returns bounded claims with candidate IDs/spans under validation", "Is not the embedding model or reranker"], ["RRF", "Rank fusion", "Combines dense, BM25, and graph ranked lists before reranking", "Is not a relevance probability"], ["BGE Reranker v2-M3", "Cross-encoder reranker", "Scores the input and candidate passage together for precision", "Is distinct from BGE-M3 retrieval"]]}},
        {"id": "conventional-rag", "kicker": "Comparison", "heading": "Semantic KG augments vector-only RAG", "intro": "The graph does not replace dense or BM25 retrieval. It adds explicit structure and provenance to the candidate universe, with extra extraction and traversal cost.", "table": {"headers": ["Dimension", "Vector-only RAG", "Semantic KG here"], "rows": [["Exact matching", "Dense neighbors can miss rare exact terms", "BM25 remains an independent lexical signal"], ["Semantic similarity", "Embedding search retrieves nearby chunks", "BGE-M3 is loaded after graph construction for retrieval-time dense search"], ["Multi-hop structure", "No explicit bounded relation traversal", "Accepted seeds expand up to two graph hops"], ["Provenance", "Often page metadata only", "Stable node IDs, spans, evidence IDs, and SHA-256"], ["Interpretability", "Similarity score is hard to explain", "Typed stance, endpoint, and provenance edges expose why context connected"], ["Noise control", "Retriever output may be noisy", "Healthy gate, validation, hop limit, RRF, and cross-encoder filter"], ["Compute", "Lower construction cost", "Qwen extraction, canonicalization, graph storage, and retrieval overhead"], ["Failure modes", "Embedding mismatch or noisy neighbors", "Also corpus gaps, bad extraction, stale identity, and evidence mismatch"]]}},
    ],
    "example_steps": [
        {"label": "Register source", "body": "A canonical document record stores source identity, path, catalog provenance, and SHA-256 before any page is trusted. The example PDF is represented as doc:sha256:9a….", "boundary": "A filename alone is not a stable identity and cannot anchor later citations."},
        {"label": "Extract and gate pages", "body": "Page 12 is extracted, hashed, and checked for factual content. Empty placeholder PDFs, cookie pages, and HTML interstitials stop at the healthy factual-corpus gate.", "boundary": "A parseable file is not automatically a healthy factual source."},
        {"label": "Overlap chunks", "body": "The page becomes overlapping chunks with page number, character spans, ordinal, text hash, and embedding key. Overlap protects boundary context while the span preserves recovery.", "boundary": "Chunk overlap improves recall; it must not duplicate or detach provenance."},
        {"label": "Qwen extraction records", "body": "Every accepted factual chunk goes through Qwen mention discovery and semantic extraction. The validated Mention and Claim records retain exact text, candidate IDs, spans, and provenance.", "boundary": "Mention is an extraction record, not a persisted graph node; invalid records are quarantined."},
        {"label": "Persist graph tables", "body": "Validated records materialize Document, EvidenceChunk, Claim, and Entity rows with has_subject, has_object, evidence-stance, from_document, and evidenced_by edges.", "boundary": "No Page graph node or generic containment/mention/related edge is created."},
        {"label": "Load retrieval models", "body": "BGE-M3 loads after graph construction and encodes EvidenceChunks for retrieval-time dense search. BM25 and bounded graph traversal supply the other retrieval signals.", "boundary": "BGE-M3 is not construction candidate selection and does not generate claims or graph relations."},
    ],
    "challenges": [
        {"symptom": "PDF page count read after closing", "root": "The parser asked for document metadata after the PDF handle had already been closed.", "correction": "Read page count and page content inside the open-document scope, then persist extraction status.", "lesson": "Resource lifetime is part of data correctness, not just cleanup.", "stage": "PDF extraction", "viva": "How did you make parser resource ownership observable?"},
        {"symptom": "Empty placeholder PDF or HTML interstitial", "root": "A download had a valid file shape but contained a placeholder, cookie page, or access interstitial instead of factual content.", "correction": "Run the healthy factual-corpus gate on extracted bodies and reject empty placeholders and HTML interstitials before chunking.", "lesson": "A successful download is not successful corpus coverage.", "stage": "Corpus validation", "viva": "What gate stops infrastructure noise becoming evidence?"},
        {"symptom": "PyArrow fails on nested metadata", "root": "Nested dictionaries or lists were placed in scalar table columns that PyArrow could not infer consistently.", "correction": "Normalize nested metadata to stable JSON text or explicitly typed columns before table writes.", "lesson": "Storage schemas must match the physical column type.", "stage": "Artifact export", "viva": "How did you separate rich metadata from scalar export columns?"},
        {"symptom": "Unsloth offline metadata resolution", "root": "Offline startup still tried to resolve model metadata that was not present in the local cache.", "correction": "Pin local model metadata and use an offline-safe loading path with an explicit cache check.", "lesson": "Offline means every dependency and metadata lookup is preflighted.", "stage": "Model runtime", "viva": "What exactly did offline mode guarantee?"},
        {"symptom": "Missing fast linear-attention kernels", "root": "The Blackwell environment lacked the optional optimized kernel required by the fast attention path.", "correction": "Detect kernel availability and use the measured compatible fallback while recording the runtime choice.", "lesson": "A performance optimization cannot be a hidden correctness dependency.", "stage": "GPU runtime", "viva": "How did you preserve reproducibility across GPU capability differences?"},
        {"symptom": "Source compilation targets unrelated GPU architectures", "root": "Build flags included architectures not present or relevant to the active GPU, causing costly or failing compilation.", "correction": "Pin the target architecture set to the deployed hardware and validate it before compilation.", "lesson": "Hardware-specific builds need explicit target boundaries.", "stage": "GPU build", "viva": "Why is narrowing architecture targets an engineering fix rather than a shortcut?"},
        {"symptom": "Qwen spends the full token budget without final JSON", "root": "Thinking consumed the generation budget before a structured answer was emitted.", "correction": "Use two-phase reasoning/final-answer generation with separate budgets and validate only the final phase.", "lesson": "Useful reasoning and observable output need separate budgets.", "stage": "Semantic extraction", "viva": "How did you prevent a valid model call from becoming a missing claim record?"},
        {"symptom": "Two-phase output is conflated", "root": "Model-emitted reasoning was treated as if it were deterministic audit metadata or final structured data.", "correction": "Persist reasoning trace separately from request IDs, hashes, timings, schema status, evidence IDs, and checkpoints.", "lesson": "Generated explanation and deterministic audit identity are different artifacts.", "stage": "Generation contract", "viva": "Which fields can a model change, and which fields must the pipeline own?"},
        {"symptom": "Invalid 41-character reranker revision", "root": "A malformed or non-existent revision identifier was passed to the reranker loader.", "correction": "Pin and validate the real model revision before loading, with a fail-fast configuration check.", "lesson": "Model identity is an input contract, not a display string.", "stage": "Reranking", "viva": "How did you prove the reranker artifact was the intended one?"},
        {"symptom": "Triton JIT needs physical Python source", "root": "The JIT compiler could not resolve source from an in-memory or relocated module, and the wrong program_id namespace was used.", "correction": "Expose physical Python source to Triton and use the compiler’s correct program_id namespace/API.", "lesson": "JIT tooling has runtime and namespace assumptions that tests must exercise.", "stage": "Kernel execution", "viva": "What evidence distinguishes a kernel API bug from a model bug?"},
        {"symptom": "Transformers/LM Format Enforcer import incompatibility", "root": "Installed versions exposed incompatible symbols between Transformers integration and LM Format Enforcer.", "correction": "Pin a compatible pair, import through the supported integration surface, and smoke-test constrained generation.", "lesson": "Dependency compatibility belongs in the reproducibility manifest.", "stage": "Schema-constrained output", "viva": "How did you verify the constraint library rather than only the import?"},
        {"symptom": "Semantic context overflow and invalid JSON quarantines", "root": "Selected chunks exceeded the model context or Qwen returned malformed structured output.", "correction": "Bound candidate context, validate JSON/schema, quarantine invalid records, and preserve the reason and checkpoint identity.", "lesson": "Quarantine is safer than silently creating partial graph facts.", "stage": "Semantic extraction", "viva": "What happens to a claim the model cannot fit or parse?"},
        {"symptom": "Stale checkpoint identity mismatch", "root": "Code, model, prompt, or configuration changed after a checkpoint was written.", "correction": "Hash the stage identity and refuse incompatible resume; start a new checkpoint lineage explicitly.", "lesson": "Resumption is safe only when identity is immutable and checked.", "stage": "Checkpointing", "viva": "Which changes should invalidate a semantic extraction checkpoint?"},
        {"symptom": "Citations use titles rather than frozen evidence IDs", "root": "Human-readable titles were used as references even though they can be duplicated or changed.", "correction": "Attach immutable evidence IDs and validate every citation against the standalone frozen ledger.", "lesson": "Readable labels are not audit keys.", "stage": "Evidence provenance", "viva": "How can a reviewer recover exactly what Qwen saw?"},
        {"symptom": "Missing standalone evidence ledger", "root": "Evidence was recoverable only indirectly from generation identity, so citation and recovery boundaries were ambiguous.", "correction": "Persist a standalone evidence ledger and retain generation identity as a recovery link, not as the ledger itself.", "lesson": "Evidence deserves its own durable artifact.", "stage": "Evidence provenance", "viva": "Why is a generation record insufficient as an evidence ledger?"},
        {"symptom": "Excel-illegal control characters", "root": "Generated or extracted text contained characters Excel rejects even though JSON accepted them.", "correction": "Sanitize forbidden control characters at export while retaining the canonical machine-readable record.", "lesson": "Presentation exports need a separate compatibility boundary.", "stage": "Excel export", "viva": "How did you avoid corrupting the canonical text while making Excel readable?"},
        {"symptom": "MP perspective-number mismatches after recovery", "root": "Human-readable perspective numbers differed from immutable request IDs during restart recovery.", "correction": "Use request IDs as audit keys and record the recovered number-to-request mapping explicitly.", "lesson": "Recovery must preserve analytical identity rather than renumbering silently.", "stage": "MP recovery", "viva": "Which identifier wins when a display number and request identity disagree?"},
        {"symptom": "Excel floating-point round-trip differences", "root": "Latency metadata changed representation when written to and read back from Excel.", "correction": "Keep exact canonical latency in JSONL/Parquet and treat rounded Excel values as presentation metadata.", "lesson": "Do not use a floating-point presentation export as the audit source.", "stage": "Trace export", "viva": "Which artifact is authoritative for numeric audit fields?"},
        {"symptom": "Interruptible-instance restarts", "root": "Preemptible workers can stop between requests or while artifacts are being written.", "correction": "Use append-only row checkpoints, fsync/atomic summaries, identity-aware resume, and never truncate accepted JSONL.", "lesson": "Recovery design is part of the pipeline, not an afterthought.", "stage": "Operations", "viva": "How does a restart avoid duplicating or deleting accepted graph records?"},
    ],
    "viva": [
        {"question": "Why not build the graph directly from embeddings?", "short": "Embeddings find semantic candidates; Qwen extracts bounded claims and relations from those candidates under validation.", "deep": "BGE-M3 is a retrieval representation, not a truth or relation generator. Keeping selection, interpretation, validation, and canonicalization separate makes provenance and failure handling explicit."},
        {"question": "Which graph records are persisted?", "short": "Document, EvidenceChunk, Claim, and Entity are persisted node tables; Mention is a Qwen extraction record, not a graph node.", "deep": "EvidenceChunk rows retain page/span metadata without inventing a Page node. Claims connect to entities with has_subject and has_object, while stance, from_document, and evidenced_by edges preserve provenance."},
        {"question": "How does graph retrieval remain safe?", "short": "It expands accepted lexical/dense seeds by at most two hops, fuses ranks, reranks passages, and preserves evidence IDs.", "deep": "Traversal augments BM25 and dense retrieval; it does not replace them or prove relevance. Hop limits, RRF, the cross-encoder gate, evidence ledger, and citation validation make structural context auditable."},
    ],
}


CHAPTERS: list[dict[str, Any]] = [ZERO_SHOT, FEW_SHOT, KG_RAG, MP_KG_RAG, KNOWLEDGE_GRAPH]


def render_page(chapter: dict[str, object]) -> str:
    """Render one chapter as a complete, directly-openable HTML document."""

    current = str(chapter["slug"])
    # Keep the stylesheet aligned with the production edge vocabulary. The
    # legacy selector is not emitted because extraction records are not edges.
    style = STYLE.replace(".graph-edge-evidence_record,", "").replace(".graph-edge-evidence_record .graph-edge-label,", "")
    nav = "".join(
        f'<a href="{slug}.html" aria-current="{"page" if slug == current else "false"}">{_text(label)}</a>'
        for slug, label in NAV_CHAPTERS
    )
    index = next((i for i, (slug, _) in enumerate(NAV_CHAPTERS) if slug == current), 0)
    previous = NAV_CHAPTERS[index - 1] if index else None
    following = NAV_CHAPTERS[index + 1] if index + 1 < len(NAV_CHAPTERS) else None
    previous_link = f'<a href="{previous[0]}.html">← {previous[1]}</a>' if previous else '<span></span>'
    next_link = f'<a class="next-link" href="{following[0]}.html">{following[1]} →</a>' if following else '<span></span>'
    section_html = "".join(_section(section) for section in chapter["sections"])  # type: ignore[index]
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>{_text(chapter['title'])} · Viva architecture</title>"
        f"<style>{style}</style></head><body><div id=\"reading-progress\" class=\"progress-track\" aria-label=\"Reading progress\"><span id=\"progress-bar\"></span></div>"
        f'<nav class="site-nav" aria-label="Chapter navigation"><a class="brand" href="zero-shot-architecture.html">MP-KG-RAG / VIVA</a><div class="chapter-links">{nav}</div></nav>'
        '<main class="page">'
        f'<p class="sr-only" id="chapter-context">Chapter context: {_text(chapter["title"])}. {_text(chapter["eyebrow"])}. {_text(chapter["dek"])}</p>'
        '<section class="findings" aria-labelledby="findings-heading"><div class="section-kicker">At a glance</div><h2 id="findings-heading">Top Findings</h2><div class="findings-grid">'
        f'{finding_cards(chapter["findings"])}' '</div></section>'
        f'<header class="hero" aria-describedby="chapter-context"><div class="eyebrow">{_text(chapter["eyebrow"])}</div><h1>{_text(chapter["title"])}</h1><p class="lede">{_text(chapter["dek"])}</p></header>'
        f'{plain_language_glossary()}'
        f'<section class="section" id="architecture"><div class="section-kicker">Architecture visual</div><h2>Follow the information boundary</h2><p class="lede">{_text(chapter["visual_label"])}</p>{architecture_visual(chapter)}{graph_visual(chapter["graph_visual"]) if chapter.get("graph_visual") else ""}</section>'
        f'{section_html}{worked_example(chapter)}{challenge_table(chapter.get("challenges", []))}'
        '<section class="section" id="viva"><div class="section-kicker">Interviewer prompts</div><h2>Viva questions</h2><p class="lede">Use the short answer first; expand into the follow-up when asked about implementation judgment.</p><div class="viva-list">'
        f'{viva_panels(chapter["viva"])}' '</div></section>'
        f'<div class="chapter-controls">{previous_link}{next_link}</div><footer class="site-footer">Standalone local preview · no server, fonts, scripts, images, or network dependencies.</footer></main>'
        f'<script>{SCRIPT}</script></body></html>'
    )


def build_all(output_dir: Path) -> list[Path]:
    """Write all registered chapter pages into ``output_dir`` and return their paths."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for chapter in CHAPTERS:
        path = output_dir / f"{chapter['slug']}.html"
        path.write_text(render_page(chapter), encoding="utf-8")
        paths.append(path)
    return paths


if __name__ == "__main__":
    build_all(Path(__file__).resolve().parents[1] / ".preview")
