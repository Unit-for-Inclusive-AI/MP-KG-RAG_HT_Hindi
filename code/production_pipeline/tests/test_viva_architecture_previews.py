"""Contract tests for the self-contained viva architecture previews."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from work.build_viva_architecture_previews import build_all, CHAPTERS, NAV_CHAPTERS


EXPECTED_TASK_1_FILES = {
    "zero-shot-architecture.html",
    "few-shot-architecture.html",
}

EXPECTED_TASK_2_FILES = {
    "zero-shot-architecture.html",
    "few-shot-architecture.html",
    "kg-rag-architecture.html",
    "mp-kg-rag-architecture.html",
}

EXPECTED_TASK_3_FILES = EXPECTED_TASK_2_FILES | {"knowledge-graph-construction.html"}


@pytest.fixture()
def rendered_pages(tmp_path: Path) -> dict[str, str]:
    paths = build_all(tmp_path)
    return {path.name: path.read_text(encoding="utf-8") for path in paths}


def test_task_one_registry_has_the_two_generation_chapters() -> None:
    assert EXPECTED_TASK_1_FILES <= {
        f"{chapter['slug']}.html" for chapter in CHAPTERS
    }


def test_build_all_writes_exact_task_three_filenames(
    rendered_pages: dict[str, str],
) -> None:
    assert EXPECTED_TASK_3_FILES == rendered_pages.keys()
    for filename in EXPECTED_TASK_1_FILES:
        assert rendered_pages[filename].startswith("<!doctype html>")


def test_pages_are_self_contained_and_do_not_use_code_blocks(
    rendered_pages: dict[str, str],
) -> None:
    external_url = re.compile(
        r"(?:src|href)\s*=\s*['\"](?:https?:|//|data:|javascript:)",
        re.IGNORECASE,
    )
    for html in rendered_pages.values():
        assert not external_url.search(html)
        assert "<pre" not in html.lower()
        assert "<code" not in html.lower()
        assert "<style" in html and "<script" in html and "<svg" in html


def test_shared_shell_contains_required_visual_and_accessibility_contract(
    rendered_pages: dict[str, str],
) -> None:
    for html in rendered_pages.values():
        assert "Top Findings" in html
        assert 'class="findings-grid"' in html
        assert 'aria-label="Chapter navigation"' in html
        assert "zero-shot-architecture.html" in html
        assert "few-shot-architecture.html" in html
        assert "reading-progress" in html
        assert 'id="progress-bar"' in html
        assert "Walk through example" in html
        assert "Replay animation" in html
        assert "Viva question" in html
        assert "prefers-reduced-motion" in html
        assert "#000" in html or "black" in html.lower()


def test_architecture_flow_is_responsive_and_not_clipped(
    rendered_pages: dict[str, str],
) -> None:
    for html in rendered_pages.values():
        assert "min-width:680px" not in html.replace(" ", "")
        assert "min-width:0" in html.replace(" ", "")
        assert 'class="flow-stage-list"' in html
        assert 'class="flow-token' in html
        assert ".flow-lane{position:relative" in html.replace(" ", "")


def test_top_findings_are_first_visible_page_content_with_accessible_context(
    rendered_pages: dict[str, str],
) -> None:
    for html in rendered_pages.values():
        assert html.index('id="findings-heading"') < html.index('<header class="hero"')
        assert 'class="sr-only"' in html
        assert "Chapter context" in html


def test_data_movement_names_demonstration_and_learned_pattern(
    rendered_pages: dict[str, str],
) -> None:
    few_shot = rendered_pages["few-shot-architecture.html"]
    assert 'data-movement="demonstration-card"' in few_shot
    assert 'data-movement="learned-pattern"' in few_shot
    assert "flow-token" in few_shot
    assert "is-running" in few_shot


def test_flow_stage_list_has_only_li_children_and_keeps_arrows_accessible(
    rendered_pages: dict[str, str],
) -> None:
    """The ordered list must retain valid list semantics while arrows stay decorative."""

    class _DirectChildParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.stack: list[str] = []
            self.direct_children: list[str] = []
            self.arrow_attrs: list[dict[str, str]] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            attributes = dict(attrs)
            if self.stack and self.stack[-1] == "ol":
                self.direct_children.append(tag)
            if tag == "span" and attributes.get("class") == "flow-stage-arrow":
                self.arrow_attrs.append(attributes)
            self.stack.append(tag)

        def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            self.handle_starttag(tag, attrs)
            self.handle_endtag(tag)

        def handle_endtag(self, tag: str) -> None:
            assert self.stack and self.stack[-1] == tag
            self.stack.pop()

    for html in rendered_pages.values():
        parser = _DirectChildParser()
        list_start = html.index('<ol class="flow-stage-list">')
        list_end = html.index("</ol>", list_start) + len("</ol>")
        parser.feed(html[list_start:list_end])
        assert parser.direct_children
        assert set(parser.direct_children) == {"li"}
        assert parser.arrow_attrs
        assert all(attrs.get("aria-hidden") == "true" for attrs in parser.arrow_attrs)


def test_example_input_uses_muted_non_severity_colors(
    rendered_pages: dict[str, str],
) -> None:
    """The quoted input is context, not a finding, so it cannot borrow severity colors."""

    for html in rendered_pages.values():
        style = html[html.index(".example-input {") : html.index(".example-steps {")]
        assert "var(--high)" not in style
        assert "var(--medium)" not in style
        assert "var(--low)" not in style
        assert "#f17676" not in style
        assert "#e8bd70" not in style
        assert "#83d29c" not in style
        assert "#83b8c3" in style


def test_animation_has_no_timer_staging_and_reduced_motion_is_immediate(
    rendered_pages: dict[str, str],
) -> None:
    for html in rendered_pages.values():
        assert "setTimeout" not in html
        assert "prefers-reduced-motion:reduce" in html.replace(" ", "")
        assert "animation:none" in html.replace(" ", "")


def test_plain_language_definitions_precede_architecture_terms(
    rendered_pages: dict[str, str],
) -> None:
    for html in rendered_pages.values():
        glossary = html.index('id="plain-language"')
        architecture = html.index('id="architecture"')
        assert glossary < architecture
        for definition in (
            "Unsloth FastModel loads Qwen",
            "4-bit quantization stores model weights",
            "Schema-constrained generation means",
            "A checkpoint is a saved progress record",
            "Embedding means representing text as vectors",
            "BM25 is lexical retrieval",
            "A cross-encoder scores a pair",
            "RRF combines ranked lists",
            "Graph traversal follows typed edges",
            "Provenance records where evidence came from",
            "Checkpoint identity is the hashable run contract",
        ):
            assert definition in html


def test_source_fidelity_contract_matches_production_implementation(
    rendered_pages: dict[str, str],
) -> None:
    """Architecture claims must be anchored to production source, not a generic design."""

    production = Path(__file__).resolve().parents[1] / "work" / "build_remote_vm_qwen35_mpkg_rag.py"
    source = production.read_text(encoding="utf-8")
    assert "from unsloth import FastModel" in source
    assert "FastModel.from_pretrained" in source
    assert "model.generate(**encoded" in source
    assert "SentenceTransformer(CONFIG[\\\"embedding_model\\\"]" in source
    assert source.index("graph_tables = build_semantic_graph") < source.index("from sentence_transformers import SentenceTransformer")
    assert "canonical_extraction_chunks(chunks[chunks.factual_index_allowed]" in source
    assert "mention_prompts = [build_mention_prompt" in source
    assert "prompts = [build_extraction_prompt" in source
    assert "semantic_kg_nodes = pd.DataFrame" in source
    assert 'node_type\\": \\"Mention\\"' not in source
    perspective_names = ["fact_checking", "cultural_context", "harm_reduction", "legal_rights", "persuasion"]
    for name in perspective_names:
        assert f'\\"{name}\\"' in source
    for html in rendered_pages.values():
        assert "vLLM" not in html and "vllm" not in html
        assert "FastModel" in html and "model.generate" in html
    mp = rendered_pages["mp-kg-rag-architecture.html"]
    for name in perspective_names:
        assert name in mp
    assert "one appended final MP row" in mp
    branch_names = set(re.findall(r'data-branch-name="([^"]+)"', mp))
    assert branch_names == set(perspective_names)


def test_source_fidelity_contract_matches_core_graph_and_checkpoint_boundaries(
    rendered_pages: dict[str, str],
) -> None:
    root = Path(__file__).resolve().parents[1]
    core = (root / "work" / "mpkg_rag_core.py").read_text(encoding="utf-8")
    production = (root / "work" / "build_remote_vm_qwen35_mpkg_rag.py").read_text(encoding="utf-8")
    graph_source = core[core.index("def build_semantic_graph") : core.index("_RETRIEVAL_WEIGHT_KEYS")]
    for table in ('"Document": []', '"EvidenceChunk": []', '"Mention": []', '"Entity": []', '"Claim": []'):
        assert table in graph_source
    for edge in ('"has_subject"', '"has_object"', '"evidenced_by"', '"from_document"'):
        assert edge in graph_source
    for stance in ("supports", "refutes", "quotes", "contextualizes"):
        assert stance in core
    assert 'graph["Mention"] = list(mentions.values())' in graph_source
    assert 'graph["edges"] = edges' in graph_source
    assert production.index("graph_tables = build_semantic_graph") < production.index("from sentence_transformers import SentenceTransformer")
    assert "mp_perspective_outputs" in production and "mp_response_plan" in production
    assert "append_checkpoint_row(checkpoint, row)" in production
    assert "checkpoint_raw_envelope" in production
    assert "not separately checkpointed" in rendered_pages["mp-kg-rag-architecture.html"]


def test_knowledge_graph_contract_uses_actual_tables_edges_and_topology(
    rendered_pages: dict[str, str],
) -> None:
    html = rendered_pages["knowledge-graph-construction.html"]
    for term in ("Document", "EvidenceChunk", "Mention", "Entity", "Claim"):
        assert term in html
    assert "Mention is an extraction record, not a persisted graph node" in html
    schema = html[html.index('id="graph-schema"') : html.index('id="role-separation"')]
    assert "Page graph node" in html
    assert '<td>Page</td>' not in schema
    for edge in ("has_subject", "has_object", "supports", "refutes", "quotes", "contextualizes", "from_document", "evidenced_by"):
        assert edge in html
    assert "BGE-M3 loads after graph construction" in html
    assert "Every accepted factual chunk goes through Qwen extraction" in html


def test_generated_preview_policy_is_explicit_and_ignored() -> None:
    ignore = Path(__file__).resolve().parents[1] / ".gitignore"
    assert ".preview/" in ignore.read_text(encoding="utf-8")


def test_language_claims_match_prompt_and_frozen_examples(
    rendered_pages: dict[str, str],
) -> None:
    zero_shot = rendered_pages["zero-shot-architecture.html"]
    few_shot = rendered_pages["few-shot-architecture.html"]
    assert "fixed system instruction" in zero_shot
    assert "does not guarantee same-language output" in zero_shot
    assert "English-only" in few_shot
    assert "language-matched" not in few_shot
    assert "does not inject a language instruction" in zero_shot


def test_preview_files_open_directly_at_mobile_and_desktop_without_network(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    paths = build_all(tmp_path)
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - depends on local browser install
            pytest.skip(f"BROWSER ACCEPTANCE NOT RUN — Chromium unavailable: {exc}")
        try:
            for path in paths:
                for viewport in ({"width": 1440, "height": 1000}, {"width": 390, "height": 844}):
                    context = browser.new_context(viewport=viewport)
                    page = context.new_page()
                    external_requests: list[str] = []
                    page.on("request", lambda request: external_requests.append(request.url) if not request.url.startswith("file:") else None)
                    errors: list[str] = []
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(path.resolve().as_uri(), wait_until="load")
                    assert not external_requests
                    assert not errors
                    flow = page.locator(".flow-diagram").first
                    assert flow.bounding_box()["width"] <= viewport["width"]
                    assert page.locator(".example-next").count() == 1
                    page.get_by_role("button", name="Replay animation").click()
                    context.close()
        finally:
            browser.close()


def test_playwright_regression_contract_covers_interactions_layout_and_motion(tmp_path: Path) -> None:
    """Keep the direct-file preview behavior covered at both required viewports."""

    playwright = pytest.importorskip("playwright.sync_api")
    paths = build_all(tmp_path)
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - depends on local browser install
            pytest.skip(f"BROWSER ACCEPTANCE NOT RUN — Chromium unavailable: {exc}")
        try:
            for path in paths:
                for viewport in ({"width": 1440, "height": 1000}, {"width": 390, "height": 844}):
                    context = browser.new_context(viewport=viewport, reduced_motion="no-preference")
                    page = context.new_page()
                    console_errors: list[str] = []
                    page_errors: list[str] = []
                    page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
                    page.on("pageerror", lambda error: page_errors.append(str(error)))
                    page.goto(path.resolve().as_uri(), wait_until="load")

                    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                    assert not console_errors
                    assert not page_errors

                    example = page.locator(".example-section")
                    before = example.locator(".example-step.is-visible").inner_text()
                    example.get_by_role("button", name="Walk through example").click()
                    after = example.locator(".example-step.is-visible").inner_text()
                    assert after != before

                    disclosure = page.locator(".viva-panel").first
                    disclosure.locator("summary").click()
                    assert disclosure.get_attribute("open") is not None
                    assert disclosure.locator(".viva-answer").is_visible()

                    node = page.locator(".flow-node").first
                    page.keyboard.press("Tab")
                    node.focus()
                    assert node.evaluate("element => element.matches(':focus-visible')")
                    node.press("Enter")
                    node.press("Enter")
                    assert "is-active" in (node.get_attribute("class") or "")

                    caption_box = page.locator(".flow-caption").bounding_box()
                    token_boxes = [page.locator(".flow-token").nth(i).bounding_box() for i in range(page.locator(".flow-token").count())]
                    assert caption_box is not None and all(box is not None for box in token_boxes)
                    assert caption_box["y"] >= max(box["y"] + box["height"] for box in token_boxes if box)

                    first_token = page.locator(".flow-token").first
                    first_position = first_token.bounding_box()
                    moved = False
                    for _ in range(30):
                        page.wait_for_timeout(100)
                        moved_position = first_token.bounding_box()
                        if first_position and moved_position and (
                            (first_position["x"], first_position["y"])
                            != (moved_position["x"], moved_position["y"])
                        ):
                            moved = True
                            break
                    assert moved, "flow token did not transition during the observation window"
                    context.close()

                context = browser.new_context(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
                page = context.new_page()
                page.goto(path.resolve().as_uri(), wait_until="load")
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                assert page.locator(".flow-token").count() >= 2
                assert all(page.locator(".flow-token").nth(i).evaluate("element => getComputedStyle(element).opacity") == "1" for i in range(page.locator(".flow-token").count()))
                assert all(page.locator(".flow-token").nth(i).evaluate("element => getComputedStyle(element).animationName") == "none" for i in range(page.locator(".flow-token").count()))
                caption_box = page.locator(".flow-caption").bounding_box()
                token_boxes = [page.locator(".flow-token").nth(i).bounding_box() for i in range(page.locator(".flow-token").count())]
                assert caption_box is not None and caption_box["y"] >= max(box["y"] + box["height"] for box in token_boxes if box)
                context.close()
        finally:
            browser.close()


def test_desktop_few_shot_long_token_stays_inside_its_lane(tmp_path: Path) -> None:
    """A long demonstration label must remain visually contained at the animation endpoint."""

    playwright = pytest.importorskip("playwright.sync_api")
    paths = build_all(tmp_path)
    few_shot = next(path for path in paths if path.name == "few-shot-architecture.html")
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - depends on local browser install
            pytest.skip(f"BROWSER ACCEPTANCE NOT RUN — Chromium unavailable: {exc}")
        try:
            context = browser.new_context(viewport={"width": 1440, "height": 1000})
            page = context.new_page()
            page.goto(few_shot.resolve().as_uri(), wait_until="load")
            page.locator(".flow-token").first.evaluate(
                """(token) => {
                    token.textContent = 'English-only demonstration card with a deliberately long label that must wrap inside the lane';
                    token.style.animationDelay = '0s';
                    token.style.animationDuration = '1s';
                    window.dispatchEvent(new Event('resize'));
                }"""
            )
            metrics = None
            for _ in range(30):
                page.wait_for_timeout(100)
                metrics = page.locator(".flow-token").first.evaluate(
                    """(token) => {
                        const lane = token.closest('.flow-lane');
                        const tokenBox = token.getBoundingClientRect();
                        const laneBox = lane.getBoundingClientRect();
                        return {
                            left: tokenBox.left,
                            right: tokenBox.right,
                            laneLeft: laneBox.left,
                            laneRight: laneBox.right,
                            clientWidth: token.clientWidth,
                            scrollWidth: token.scrollWidth,
                        };
                    }"""
                )
                if (
                    metrics["left"] >= metrics["laneLeft"] - 1
                    and metrics["right"] <= metrics["laneRight"] + 1
                    and metrics["scrollWidth"] <= metrics["clientWidth"] + 1
                ):
                    break
            assert metrics is not None
            assert metrics["left"] >= metrics["laneLeft"] - 1
            assert metrics["right"] <= metrics["laneRight"] + 1
            assert metrics["scrollWidth"] <= metrics["clientWidth"] + 1
            context.close()
        finally:
            browser.close()


def test_generation_chapters_include_required_architecture_and_example_content(
    rendered_pages: dict[str, str],
) -> None:
    zero_shot = rendered_pages["zero-shot-architecture.html"]
    few_shot = rendered_pages["few-shot-architecture.html"]

    for term in (
        "scientific baseline",
        "dataset record normalization",
        "language preservation",
        "Qwen3.5-4B",
        "FastModel",
        "4-bit",
        "thinking-enabled",
        "schema-constrained",
        "JSON",
        "checkpoint",
        "Excel",
        "full token budget",
        "final JSON",
        "resumable",
        "hallucination",
    ):
        assert term.lower() in zero_shot.lower(), term

    for term in (
        "frozen",
        "demonstration",
        "in-context learning",
        "fine-tuning",
        "context-window",
        "example-order",
        "demonstration bias",
        "leakage",
        "fair evaluation",
    ):
        assert term.lower() in few_shot.lower(), term

    for html in (zero_shot, few_shot):
        assert "LGBTQ" in html
        assert "input hate speech" in html.lower()
        assert "cannot know" in html.lower() or "cannot guarantee" in html.lower()
        assert html.count("<details") >= 2
        assert html.count('type="button"') >= 2


def test_render_page_is_available_for_later_chapter_extensions() -> None:
    from work.build_viva_architecture_previews import render_page

    html = render_page(CHAPTERS[0])
    assert isinstance(html, str)
    assert "Zero-Shot" in html


def test_task_two_registry_adds_kg_rag_and_mp_kg_rag_pages() -> None:
    registered = {f"{chapter['slug']}.html" for chapter in CHAPTERS}
    assert EXPECTED_TASK_2_FILES <= registered
    assert "knowledge-graph-construction.html" in registered


def test_task_three_registry_is_the_final_five_page_set() -> None:
    assert EXPECTED_TASK_3_FILES == {f"{chapter['slug']}.html" for chapter in CHAPTERS}


def test_task_three_registry_and_construction_contract(
    rendered_pages: dict[str, str],
) -> None:
    html = rendered_pages["knowledge-graph-construction.html"]
    for term in (
        "Register corpus documents",
        "source identity",
        "Extract PDF pages",
        "healthy factual-corpus gate",
        "empty placeholders",
        "HTML interstitials",
        "overlapping chunks",
        "BGE-M3 embeddings",
        "retrieval candidates",
        "Qwen",
        "bounded semantic claims",
        "canonical entity hubs",
        "stable namespaced IDs",
        "evidence IDs",
        "SHA-256",
        "graph retrieval",
        "two graph hops",
        "BM25",
        "RRF",
        "BGE Reranker v2-M3",
    ):
        assert term.lower() in html.lower(), term
    for node_type in ("Document", "EvidenceChunk", "Mention", "Claim", "Entity"):
        assert node_type in html
    for edge_type in ("has_subject", "has_object", "supports", "from_document", "evidenced_by"):
        assert edge_type in html
    assert "Role separation" in html
    assert "BGE-M3 is the embedding model" in html
    assert "Qwen is the claim extractor" in html
    assert 'data-graph-stage="multi-hub"' in html


def test_task_three_challenge_catalogue_covers_every_specified_issue(
    rendered_pages: dict[str, str],
) -> None:
    html = rendered_pages["knowledge-graph-construction.html"].lower()
    issues = (
        "page count read after closing",
        "empty placeholder pdf",
        "html interstitial",
        "pyarrow",
        "unsloth offline metadata",
        "fast linear-attention kernels",
        "unrelated gpu architectures",
        "full token budget",
        "two-phase",
        "41-character reranker revision",
        "physical python source",
        "program_id namespace",
        "lm format enforcer",
        "context overflow",
        "invalid json quarantines",
        "checkpoint identity mismatch",
        "titles rather than frozen evidence ids",
        "standalone evidence ledger",
        "excel-illegal control characters",
        "perspective-number mismatches",
        "floating-point round-trip",
        "interruptible-instance restarts",
        "symptom",
        "root cause",
        "correction",
        "lesson",
        "pipeline stage",
        "viva framing",
    )
    for issue in issues:
        assert issue in html, issue


def test_task_three_graph_has_ordered_multihub_nodes_and_explicit_edges(
    rendered_pages: dict[str, str],
) -> None:
    """The graph contract is endpoint-addressable rather than decorative SVG art."""

    html = rendered_pages["knowledge-graph-construction.html"]
    graph_start = html.index('<div class="graph-visual"')
    graph_end = html.index('<div class="graph-edge-legend"', graph_start)
    graph = html[graph_start:graph_end]
    node_matches = re.findall(
        r'<button[^>]+class="graph-node[^>]+data-graph-id="([^"]+)"[^>]+data-graph-kind="([^"]+)"[^>]+data-graph-stage="(\d+)"',
        graph,
    )
    assert node_matches
    kinds = [kind for _, kind, _ in node_matches]
    assert set(kinds) == {"Document", "EvidenceChunk", "Mention", "Claim", "Entity"}
    assert kinds.count("Claim") >= 2
    assert kinds.count("Entity") >= 3
    assert [int(stage) for _, _, stage in node_matches] == sorted(int(stage) for _, _, stage in node_matches)
    assert all('aria-label="' in button and "<small>" in button for button in re.findall(r'<button[^>]*class="graph-node[^>]*>.*?</button>', graph))
    ids = {node_id for node_id, _, _ in node_matches}
    edge_matches = re.findall(
        r'<(?:li|div)[^>]+class="graph-edge[^>]+data-edge-kind="([^"]+)"[^>]+data-edge-from="([^"]+)"[^>]+data-edge-to="([^"]+)"',
        graph,
    )
    assert edge_matches
    assert {kind for kind, _, _ in edge_matches} >= {"from_document", "supports", "contextualizes", "has_subject", "has_object", "evidenced_by"}
    assert {kind for kind, _, _ in edge_matches} <= {"from_document", "supports", "refutes", "quotes", "contextualizes", "has_subject", "has_object", "evidenced_by"}
    assert all(source in ids and target in ids and source != target for _, source, target in edge_matches)
    assert any(kind == "from_document" and source.startswith("evidence-") and target.startswith("doc-") for kind, source, target in edge_matches)
    assert any(kind == "has_subject" and source.startswith("claim-") and target.startswith("entity-") for kind, source, target in edge_matches)
    assert any(kind == "has_object" and source.startswith("claim-") and target.startswith("entity-") for kind, source, target in edge_matches)
    assert any(kind == "evidenced_by" and source.startswith("claim-") and target.startswith("evidence-") for kind, source, target in edge_matches)


def test_task_three_challenges_render_exactly_nineteen_rows_with_six_fields(
    rendered_pages: dict[str, str],
) -> None:
    html = rendered_pages["knowledge-graph-construction.html"]
    start = html.index('<section class="section" id="challenges">')
    end = html.index('</section>', start)
    rows = re.findall(r'<tr>(.*?)</tr>', html[start:end], flags=re.DOTALL)
    assert len(rows) == 20  # header plus 19 engineering issues
    body_rows = rows[1:]
    assert len(body_rows) == 19
    assert all(len(re.findall(r'<(?:th|td)\b', row)) == 6 for row in body_rows)


def test_task_three_graph_staging_contract_is_present(
    rendered_pages: dict[str, str],
) -> None:
    html = rendered_pages["knowledge-graph-construction.html"]
    assert 'data-graph-animation="staged-provenance"' in html
    assert 'data-graph-replay="replayable"' in html
    assert 'data-graph-complete="true"' in html
    assert "graph-stage" in html
    assert "graph-token" in html
    assert "graph-edge-label" in html
    assert "prefers-reduced-motion:reduce" in html.replace(" ", "")


def test_task_three_graph_playwright_replays_stages_and_fits_mobile(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    path = next(p for p in build_all(tmp_path) if p.name == "knowledge-graph-construction.html")
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"BROWSER ACCEPTANCE NOT RUN — Chromium unavailable: {exc}")
        try:
            context = browser.new_context(viewport={"width": 390, "height": 844}, reduced_motion="no-preference")
            page = context.new_page()
            page.goto(path.resolve().as_uri(), wait_until="load")
            graph = page.locator(".graph-visual")
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            assert graph.locator(".graph-node").count() >= 9
            before = graph.locator(".graph-node").first.bounding_box()
            moved = False
            for _ in range(30):
                page.wait_for_timeout(100)
                after = graph.locator(".graph-node").first.bounding_box()
                if before and after and (before["x"], before["y"]) != (after["x"], after["y"]):
                    moved = True
                    break
            assert moved, "graph node did not transition during the observation window"
            node = graph.locator(".graph-node").first
            node.focus()
            node.press("Enter")
            assert node.get_attribute("aria-pressed") == "true"
            assert "Selected:" in graph.locator(".graph-status").inner_text()
            graph.get_by_role("button", name="Replay graph animation").click()
            assert graph.locator(".graph-node.is-visible").count() >= 1
            context.close()

            context = browser.new_context(viewport={"width": 390, "height": 844}, reduced_motion="reduce")
            page = context.new_page()
            page.goto(path.resolve().as_uri(), wait_until="load")
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            graph = page.locator(".graph-visual")
            assert graph.locator(".graph-node").count() >= 9
            assert graph.locator(".graph-node:not(.is-visible)").count() == 0
            assert all(graph.locator(".graph-node").nth(i).evaluate("el => getComputedStyle(el).opacity") == "1" for i in range(graph.locator(".graph-node").count()))
            context.close()
        finally:
            browser.close()


def test_kg_rag_factual_contract_is_explicit(rendered_pages: dict[str, str]) -> None:
    html = rendered_pages["kg-rag-architecture.html"]
    for term in (
        "BM25",
        "BGE-M3",
        "Qdrant",
        "two graph hops",
        "dense weight 1.1",
        "BM25 weight 1.0",
        "graph weight 0.8",
        "constant 60",
        "BGE Reranker v2-M3",
        "0.55",
        "five evidence passages",
        "evidence ID",
        "retrieval",
        "reranking",
    ):
        assert term.lower() in html.lower(), term
    assert "retrieval" in html.lower() and "reranking" in html.lower()


def test_mp_kg_rag_factual_contract_discloses_bounded_costs_and_limits(
    rendered_pages: dict[str, str],
) -> None:
    html = rendered_pages["mp-kg-rag-architecture.html"]
    for term in (
        "five perspectives",
        "planner",
        "final synthesis",
        "7,750",
        "1,550 plans",
        "1,550 final",
        "43 perspective traces were truncated",
        "four plan traces were truncated",
        "six perspective-number mismatches were recovered",
        "one empty intermediate plan was observed",
        "audit",
        "limitations",
    ):
        assert term.lower() in html.lower(), term
    assert "monotonic" in html.lower()


def test_kg_rag_has_true_fan_in_branch_flow_with_rejected_candidate_and_provenance(
    rendered_pages: dict[str, str],
) -> None:
    html = rendered_pages["kg-rag-architecture.html"]
    assert 'data-branch-variant="kg-rag"' in html
    assert 'data-branch-count="3"' in html
    assert html.count('class="branch-row"') == 3
    assert 'class="branch-convergence"' in html
    assert 'data-converges-at="rrf"' in html
    assert 'data-status="rejected"' in html
    assert "Rejected" in html and "0.31" in html and "0.55" in html
    assert 'class="branch-token"' in html
    assert 'data-provenance-field="source_id"' in html
    for field in ("page/locator", "span offsets", "source-text key", "SHA-256/hash", "evidence ID"):
        assert field in html


def test_mp_kg_rag_has_five_parallel_perspective_branches_and_planner_convergence(
    rendered_pages: dict[str, str],
) -> None:
    html = rendered_pages["mp-kg-rag-architecture.html"]
    assert 'data-branch-variant="mp-kg-rag"' in html
    assert 'data-branch-count="5"' in html
    assert html.count('class="branch-row"') == 5
    assert 'data-converges-at="planner"' in html
    assert 'class="branch-convergence"' in html
    assert "Final synthesis" in html
    assert html.count('class="branch-token"') >= 5


def test_branch_flow_nodes_are_real_buttons_with_state_and_keyboard_metadata(
    rendered_pages: dict[str, str],
) -> None:
    for slug in ("kg-rag-architecture.html", "mp-kg-rag-architecture.html"):
        html = rendered_pages[slug]
        assert 'class="branch-node"' in html
        assert 'type="button"' in html
        assert 'aria-pressed="false"' in html
        assert 'aria-label=' in html
        assert 'data-branch-role=' in html


def test_kg_rag_comparison_and_branch_browser_behavior(tmp_path: Path) -> None:
    """The standalone branch diagrams must move, converge, and remain operable."""

    playwright = pytest.importorskip("playwright.sync_api")
    paths = build_all(tmp_path)
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - depends on local browser install
            pytest.skip(f"BROWSER ACCEPTANCE NOT RUN — Chromium unavailable: {exc}")
        try:
            for filename, count, convergence in (
                ("kg-rag-architecture.html", 3, "rrf"),
                ("mp-kg-rag-architecture.html", 5, "planner"),
            ):
                path = next(item for item in paths if item.name == filename)
                for viewport in ({"width": 1440, "height": 1000}, {"width": 390, "height": 844}):
                    context = browser.new_context(viewport=viewport, reduced_motion="no-preference")
                    page = context.new_page()
                    page.goto(path.resolve().as_uri(), wait_until="load")
                    flow = page.locator(".branch-flow")
                    assert flow.get_attribute("data-branch-count") == str(count)
                    assert flow.locator(".branch-row").count() == count
                    assert flow.locator(f'[data-converges-at="{convergence}"]').count() == 1
                    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                    token = flow.locator(".branch-token").first
                    before = token.bounding_box()
                    moved = False
                    for _ in range(30):
                        page.wait_for_timeout(100)
                        after = token.bounding_box()
                        if before and after and (before["x"], before["y"]) != (after["x"], after["y"]):
                            moved = True
                            break
                    assert moved, "branch token did not transition during the observation window"
                    branch = flow.locator(".branch-node").first
                    branch.focus()
                    branch.press("Enter")
                    assert branch.get_attribute("aria-pressed") == "true"
                    context.close()

                context = browser.new_context(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
                page = context.new_page()
                page.goto(path.resolve().as_uri(), wait_until="load")
                assert all(
                    page.locator(".branch-token").nth(i).evaluate("element => getComputedStyle(element).animationName") == "none"
                    for i in range(page.locator(".branch-token").count())
                )
                context.close()
        finally:
            browser.close()


def test_reduced_motion_branch_tokens_are_visible_and_laid_out(tmp_path: Path) -> None:
    """Reduced motion keeps both retrieval branch token sets readable and placed."""

    playwright = pytest.importorskip("playwright.sync_api")
    paths = build_all(tmp_path)
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - depends on local browser install
            pytest.skip(f"BROWSER ACCEPTANCE NOT RUN — Chromium unavailable: {exc}")
        try:
            for filename in ("kg-rag-architecture.html", "mp-kg-rag-architecture.html"):
                path = next(item for item in paths if item.name == filename)
                for viewport in ({"width": 1440, "height": 1000}, {"width": 390, "height": 844}):
                    context = browser.new_context(viewport=viewport, reduced_motion="reduce")
                    page = context.new_page()
                    page.goto(path.resolve().as_uri(), wait_until="load")
                    tokens = page.locator(".branch-flow .branch-token")
                    assert tokens.count() > 0
                    for index in range(tokens.count()):
                        token = tokens.nth(index)
                        assert token.evaluate("element => getComputedStyle(element).animationName") == "none"
                        assert float(token.evaluate("element => getComputedStyle(element).opacity")) == 1
                        assert token.inner_text().strip()
                        box = token.bounding_box()
                        assert box is not None and box["width"] > 0 and box["height"] > 0
                        assert box["x"] >= 0 and box["x"] + box["width"] <= viewport["width"]
                    context.close()
        finally:
            browser.close()


def test_kg_rag_conventional_rag_comparison_is_present(rendered_pages: dict[str, str]) -> None:
    html = rendered_pages["kg-rag-architecture.html"]
    comparison = html[html.index('id="conventional-rag"') :]
    assert "Vector-only RAG" in comparison
    assert "KG-RAG here" in comparison
    assert "lexical" in comparison.lower() and "two hops" in comparison.lower()


def test_task_four_direct_file_acceptance_covers_all_pages_and_navigation(tmp_path: Path) -> None:
    """Exercise every chapter as a standalone page at desktop, mobile, and reduced motion."""

    playwright = pytest.importorskip("playwright.sync_api")
    paths = build_all(tmp_path)
    expected_filenames = [f"{slug}.html" for slug, _ in NAV_CHAPTERS]
    assert [path.name for path in paths] == expected_filenames

    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - depends on local browser install
            pytest.skip(f"BROWSER ACCEPTANCE NOT RUN — Chromium unavailable: {exc}")
        try:
            for viewport, reduced_motion in (
                ({"width": 1440, "height": 1000}, "no-preference"),
                ({"width": 390, "height": 844}, "no-preference"),
                ({"width": 390, "height": 844}, "reduce"),
            ):
                for index, path in enumerate(paths):
                    context = browser.new_context(
                        viewport=viewport,
                        reduced_motion=reduced_motion,
                    )
                    page = context.new_page()
                    external_requests: list[str] = []
                    console_errors: list[str] = []
                    page_errors: list[str] = []
                    page.on(
                        "request",
                        lambda request: external_requests.append(request.url)
                        if not request.url.startswith("file:")
                        else None,
                    )
                    page.on(
                        "console",
                        lambda message: console_errors.append(message.text)
                        if message.type == "error"
                        else None,
                    )
                    page.on("pageerror", lambda error: page_errors.append(str(error)))

                    page.goto(path.resolve().as_uri(), wait_until="load")
                    assert page.title() == f"{CHAPTERS[index]['title']} · Viva architecture"
                    assert page.locator('.chapter-links a[aria-current="page"]').count() == 1
                    assert page.locator(
                        f'.chapter-links a[aria-current="page"][href="{path.name}"]'
                    ).count() == 1
                    assert page.locator(".findings-grid .finding-card").count() >= 3
                    assert page.locator(".flow-diagram").count() == 1
                    assert page.locator(".flow-node").count() >= 4
                    visual_box = page.locator(".visual-wrap").bounding_box()
                    assert visual_box is not None
                    assert visual_box["x"] >= 0
                    assert visual_box["x"] + visual_box["width"] <= viewport["width"]
                    assert page.locator(".example-section").count() == 1
                    assert page.locator(".example-step").count() >= 3
                    assert page.locator(".viva-panel").count() >= 1
                    tables = page.locator(".table-scroll")
                    assert tables.count() >= 1
                    for table_index in range(tables.count()):
                        table = tables.nth(table_index)
                        table_box = table.bounding_box()
                        assert table_box is not None
                        assert table_box["width"] <= viewport["width"]
                        assert table.evaluate("element => getComputedStyle(element).overflowX") == "auto"
                    assert page.evaluate(
                        "document.documentElement.scrollWidth <= window.innerWidth"
                    )
                    assert not external_requests
                    assert not console_errors
                    assert not page_errors

                    previous = page.locator('.chapter-controls a:not(.next-link)').first
                    next_link = page.locator(".chapter-controls a.next-link")
                    if index:
                        assert previous.get_attribute("href") == expected_filenames[index - 1]
                    else:
                        assert page.locator(".chapter-controls > span").count() == 1
                    if index + 1 < len(expected_filenames):
                        assert next_link.get_attribute("href") == expected_filenames[index + 1]
                    else:
                        assert next_link.count() == 0

                    example = page.locator(".example-section")
                    initial_status = example.locator(".example-status").inner_text()
                    # Start from the document and use the browser's actual tab order.  The
                    # architecture node is a button, the example control is a button, and the
                    # viva summary is a native disclosure control; each must receive a visible
                    # focus ring before it is activated with Enter.
                    page.evaluate("document.activeElement && document.activeElement.blur()")

                    def tab_to(selector: str) -> None:
                        for _ in range(120):
                            page.keyboard.press("Tab")
                            if page.locator(f"{selector}:focus").count() == 1:
                                assert page.evaluate(
                                    "document.activeElement.matches(':focus-visible')"
                                )
                                return
                        raise AssertionError(f"Tab traversal never reached {selector}")

                    tab_to(".flow-node")
                    page.keyboard.press("Enter")
                    assert page.locator(".flow-node").first.get_attribute("aria-pressed") == "true"

                    tab_to(".example-next")
                    page.keyboard.press("Enter")
                    assert example.locator(".example-status").inner_text() != initial_status

                    disclosure = page.locator(".viva-panel").first
                    tab_to(".viva-panel summary")
                    page.keyboard.press("Enter")
                    assert disclosure.get_attribute("open") is not None
                    assert disclosure.locator(".viva-answer").is_visible()

                    if reduced_motion == "reduce":
                        animated = page.locator(".flow-token, .branch-token, .graph-token")
                        assert animated.count() >= 1
                        for animated_index in range(animated.count()):
                            item = animated.nth(animated_index)
                            assert item.evaluate(
                                "element => getComputedStyle(element).animationName"
                            ) == "none"
                            assert float(item.evaluate("element => getComputedStyle(element).opacity")) == 1

                    context.close()

            # Follow the next links from the first chapter through the full cycle,
            # then use the brand link to return to the first direct-file page.
            context = browser.new_context(viewport={"width": 1440, "height": 1000})
            page = context.new_page()
            page.goto(paths[0].resolve().as_uri(), wait_until="load")
            for expected in expected_filenames[1:]:
                page.locator(".chapter-controls a.next-link").click()
                assert page.url.endswith(f"/{expected}")
            assert page.locator(".chapter-controls a.next-link").count() == 0
            page.locator(".brand").click()
            assert page.url.endswith(f"/{expected_filenames[0]}")
            context.close()

            # Walk backwards as well: every non-first chapter's previous control must be
            # keyboard/click navigable and land on the immediately preceding file.
            context = browser.new_context(viewport={"width": 1440, "height": 1000})
            page = context.new_page()
            page.goto(paths[-1].resolve().as_uri(), wait_until="load")
            for expected in reversed(expected_filenames[:-1]):
                previous_link = page.locator(".chapter-controls a:not(.next-link)")
                assert previous_link.count() == 1
                assert previous_link.get_attribute("href") == expected
                previous_link.click()
                assert page.url.endswith(f"/{expected}")
            assert page.locator(".chapter-controls a:not(.next-link)").count() == 0
            context.close()
        finally:
            browser.close()


def test_task_four_mobile_reflow_and_scrolled_region_inspection(tmp_path: Path) -> None:
    """Inspect mobile geometry and every below-fold learning region on every page."""

    playwright = pytest.importorskip("playwright.sync_api")
    paths = build_all(tmp_path)
    viewport = {"width": 390, "height": 844}

    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - depends on local browser install
            pytest.skip(f"BROWSER ACCEPTANCE NOT RUN — Chromium unavailable: {exc}")
        try:
            for path in paths:
                context = browser.new_context(viewport=viewport, reduced_motion="reduce")
                page = context.new_page()
                page.goto(path.resolve().as_uri(), wait_until="load")

                def assert_in_viewport(locator, label: str) -> dict[str, float]:
                    box = locator.bounding_box()
                    assert box is not None, f"{label} has no rendered geometry"
                    assert box["width"] > 0 and box["height"] > 0, f"{label} is not usable"
                    assert box["x"] >= -1, f"{label} clips past the left viewport edge: {box}"
                    assert box["x"] + box["width"] <= viewport["width"] + 1, (
                        f"{label} clips past the right viewport edge: {box}"
                    )
                    return box

                # Cards must genuinely reflow into multiple rows, keep both title and summary
                # readable, and never create an internal horizontal clip at 390px.
                findings = page.locator(".findings-grid")
                assert_in_viewport(findings, "Top Findings grid")
                card_boxes = []
                for card_index in range(page.locator(".finding-card").count()):
                    card = page.locator(".finding-card").nth(card_index)
                    card_box = assert_in_viewport(card, f"Top Findings card {card_index + 1}")
                    card_boxes.append(card_box)
                    assert card.locator(".finding-title").is_visible()
                    assert card.locator("p").is_visible()
                    assert card.inner_text().strip()
                    assert card.evaluate("element => element.scrollWidth <= element.clientWidth")
                    for child_selector in (".finding-title", "p", ".metric"):
                        child = card.locator(child_selector)
                        if child.count():
                            child_box = child.first.bounding_box()
                            assert child_box is not None
                            assert child_box["x"] >= card_box["x"] - 1
                            assert child_box["x"] + child_box["width"] <= (
                                card_box["x"] + card_box["width"] + 1
                            )
                assert len({round(box["y"]) for box in card_boxes}) >= 2

                # Sticky navigation and chapter controls must wrap inside the viewport too.
                for selector in (".site-nav", ".chapter-links", ".chapter-links a"):
                    for item_index in range(page.locator(selector).count()):
                        assert_in_viewport(
                            page.locator(selector).nth(item_index),
                            f"{selector} {item_index + 1}",
                        )
                controls = page.locator(".chapter-controls")
                controls.scroll_into_view_if_needed()
                assert_in_viewport(controls, "chapter controls")
                for item_index in range(controls.locator("a").count()):
                    assert_in_viewport(
                        controls.locator("a").nth(item_index),
                        f"chapter control link {item_index + 1}",
                    )

                # Every table is intentionally wider than its mobile viewport, but its wrapper
                # must expose that width through real horizontal scrolling.
                tables = page.locator(".table-scroll")
                assert tables.count() >= 1
                for table_index in range(tables.count()):
                    table_scroll = tables.nth(table_index)
                    table_scroll.scroll_into_view_if_needed()
                    assert_in_viewport(table_scroll, f"table container {table_index + 1}")
                    dimensions = table_scroll.evaluate(
                        """element => ({
                            clientWidth: element.clientWidth,
                            scrollWidth: element.scrollWidth,
                            tableWidth: element.querySelector('table').getBoundingClientRect().width,
                        })"""
                    )
                    assert dimensions["clientWidth"] > 0
                    assert dimensions["scrollWidth"] > dimensions["clientWidth"]
                    assert dimensions["tableWidth"] > dimensions["clientWidth"]
                    assert table_scroll.locator("th").first.is_visible()
                    right_scroll = table_scroll.evaluate(
                        "element => { element.scrollLeft = element.scrollWidth; return element.scrollLeft; }"
                    )
                    assert right_scroll > 0
                    table_scroll.evaluate("element => { element.scrollLeft = 0; }")

                # Scroll each learning region into view and verify its geometry, descendants,
                # and direct siblings do not clip or overlap.  This covers content below the
                # first 844px viewport rather than only testing the page header.
                regions = {
                    "#architecture": (".visual-wrap", ".flow-diagram"),
                    "#worked-example": (".example-input", ".example-step.is-visible"),
                    "#challenges": (".table-scroll",),
                    "#viva": (".viva-panel",),
                }
                for region_selector, child_selectors in regions.items():
                    region = page.locator(region_selector)
                    assert region.count() == 1
                    region.scroll_into_view_if_needed()
                    region_box = assert_in_viewport(region, region_selector)
                    geometry = region.evaluate(
                        """element => {
                            const rect = element.getBoundingClientRect();
                            const children = [...element.children]
                                .map(child => child.getBoundingClientRect())
                                .filter(child => child.width > 0 && child.height > 0);
                            const overlaps = [];
                            for (let i = 0; i < children.length; i += 1) {
                                for (let j = i + 1; j < children.length; j += 1) {
                                    const horizontal = Math.min(children[i].right, children[j].right)
                                        - Math.max(children[i].left, children[j].left);
                                    const vertical = Math.min(children[i].bottom, children[j].bottom)
                                        - Math.max(children[i].top, children[j].top);
                                    if (horizontal > 1 && vertical > 1) overlaps.push([i, j]);
                                }
                            }
                            const clipped = [...element.querySelectorAll('*')]
                                .filter(child => !child.closest('.table-scroll'))
                                .map(child => child.getBoundingClientRect())
                                .filter(child => child.width > 0 && child.height > 0)
                                .some(child => child.left < -1 || child.right > window.innerWidth + 1);
                            return { scrollWidth: element.scrollWidth, clientWidth: element.clientWidth, overlaps, clipped };
                        }"""
                    )
                    # A mobile visual may intentionally use a tiny negative margin inside the
                    # page column; it is still valid as long as it remains inside the viewport.
                    assert geometry["scrollWidth"] <= viewport["width"] + 1
                    assert not geometry["overlaps"], f"overlap in {region_selector}: {geometry}"
                    assert not geometry["clipped"], f"clipping in {region_selector}: {geometry}"
                    for child_selector in child_selectors:
                        children = region.locator(child_selector)
                        assert children.count() >= 1
                        for child_index in range(children.count()):
                            child = children.nth(child_index)
                            child.scroll_into_view_if_needed()
                            child_box = assert_in_viewport(
                                child,
                                f"{region_selector} {child_selector} {child_index + 1}",
                            )
                            assert child_box["width"] <= viewport["width"] + 1

                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                context.close()
        finally:
            browser.close()
