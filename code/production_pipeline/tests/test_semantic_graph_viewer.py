import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work"))

import build_semantic_graph_viewer as builder


def fixture_data():
    chunks = [
        {"chunk_id": "c1", "source_id": "SRC1", "page": 1, "relative_path": "documents/report.pdf", "text": "Alpha rights evidence."},
        {"chunk_id": "c2", "source_id": "SRC2", "page": 4, "relative_path": "documents/study.pdf", "text": "Beta inclusion evidence."},
    ]
    extractions = [
        {"chunk_id": "c1", "source_id": "SRC1", "page": 1, "claims": [{"claim": "Equal treatment protects people.", "candidate_ids": ["m1"], "span_start": 0, "span_end": 5}]},
        {"chunk_id": "c2", "source_id": "SRC2", "page": 4, "claims": [
            {"claim": "Equal treatment improves inclusion.", "candidate_ids": ["m2"], "span_start": 2, "span_end": 8},
            {"claim": "An unlinked but preserved claim.", "candidate_ids": [], "span_start": 9, "span_end": 15},
        ]},
    ]
    candidates = [
        {"candidate_id": "m1", "entity_type": "concept", "normalized_text": "equal treatment", "label": "Equal treatment"},
        {"candidate_id": "m2", "entity_type": "concept", "normalized_text": "equal treatment", "label": "equal treatment"},
    ]
    pages = {"records": [
        {"source_id": "SRC1", "page": 1, "relative_path": "documents/report.pdf"},
        {"source_id": "SRC2", "page": 4, "relative_path": "documents/study.pdf"},
    ]}
    return extractions, candidates, chunks, pages


def test_build_graph_preserves_every_claim_and_canonicalizes_entity_hubs():
    graph = builder.build_graph(*fixture_data())
    nodes = graph["nodes"]
    edges = graph["edges"]

    assert graph["meta"]["counts"]["claim"] == 3
    assert sum(node["kind"] == "claim" for node in nodes) == 3
    entities = [node for node in nodes if node["kind"] == "entity"]
    assert len(entities) == 1
    assert entities[0]["label"].casefold() == "equal treatment"
    assert entities[0]["mention_count"] == 2
    assert sum(edge["kind"] == "claim_mentions_entity" for edge in edges) == 2


def test_build_graph_has_no_dangling_edges_and_preserves_lineage():
    graph = builder.build_graph(*fixture_data())
    node_ids = {node["id"] for node in graph["nodes"]}

    assert all(edge["source"] in node_ids and edge["target"] in node_ids for edge in graph["edges"])
    assert {edge["kind"] for edge in graph["edges"]} >= {
        "document_has_page", "page_has_chunk", "chunk_asserts_claim", "claim_mentions_entity"
    }
    unlinked = next(node for node in graph["nodes"] if node.get("claim") == "An unlinked but preserved claim.")
    assert unlinked["source_id"] == "SRC2"
    assert unlinked["page"] == 4
    assert unlinked["chunk_id"] == "c2"


def test_build_is_deterministic_and_writes_valid_browser_assets(tmp_path):
    data = fixture_data()
    first = builder.build_graph(*data)
    second = builder.build_graph(*data)

    assert builder.canonical_hash(first) == builder.canonical_hash(second)
    builder.write_viewer(tmp_path, first)
    assert json.loads((tmp_path / "graph-data.json").read_text())["meta"]["counts"]["claim"] == 3
    for name in ("index.html", "styles.css", "viewer.js", "graph-data.json", "graph-metadata.json"):
        assert (tmp_path / name).is_file()
    assert "sigma" in (tmp_path / "viewer.js").read_text().casefold()


def test_graph_does_not_shadow_sigmas_reserved_node_type_attribute():
    graph = builder.build_graph(*fixture_data())

    assert all("type" not in node and node["kind"] for node in graph["nodes"])
    assert all("type" not in edge and edge["kind"] for edge in graph["edges"])


def test_all_node_colors_are_webgl_safe_hex_values():
    graph = builder.build_graph(*fixture_data())

    assert all(node["color"].startswith("#") and len(node["color"]) == 7 for node in graph["nodes"])


def test_shared_entities_create_bounded_claim_relationships_and_distinct_counts():
    extractions, candidates, chunks, pages = fixture_data()
    extractions[0]["claims"][0]["candidate_ids"] = ["m1", "m1"]
    graph = builder.build_graph(extractions, candidates, chunks, pages)
    entity = next(node for node in graph["nodes"] if node["kind"] == "entity")

    assert entity["mention_count"] == 2
    assert entity["supporting_claim_count"] == 2
    assert sum(edge["kind"] == "claim_mentions_entity" for edge in graph["edges"]) == 2
    related = [edge for edge in graph["edges"] if edge["kind"] == "claim_related_to_claim"]
    assert len(related) == 1


def test_claim_span_validation_is_explicit():
    extractions, candidates, chunks, pages = fixture_data()
    extractions[0]["claims"][0].update(span_start=8, span_end=2)
    graph = builder.build_graph(extractions, candidates, chunks, pages)
    claim = next(node for node in graph["nodes"] if node["kind"] == "claim" and node["chunk_id"] == "c1")

    assert claim["span_valid"] is False
    assert graph["meta"]["invalid_claim_spans"] == 1
