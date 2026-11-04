import json
import ast
import copy
import hashlib
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work"))


class RemoteVmNotebookBuilderTests(unittest.TestCase):
    def _build_notebook_json(self):
        from build_remote_vm_qwen35_mpkg_rag import build_notebook

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "remote_vm_qwen35_mpkg_rag.ipynb"
            build_notebook(output)
            return json.loads(output.read_text(encoding="utf-8"))

    def build_text(self):
        from build_remote_vm_qwen35_mpkg_rag import build_notebook

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "remote_vm_qwen35_mpkg_rag.ipynb"
            build_notebook(output)
            notebook = json.loads(output.read_text(encoding="utf-8"))
        return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    def test_notebook_contains_batched_agents_and_real_evaluation(self):
        text = self.build_text()
        for expected in [
            "Qwen3.5-4B", "generate_batch", "perspective_rationale",
            "human ratings are required", "paired_permutation",
            "build_human_annotation_workbook", "HUMAN_METRICS",
        ]:
            self.assertIn(expected, text)

    def test_notebook_has_separate_mp_checkpoint_and_output_column(self):
        text = self.build_text()
        for expected in [
            "mp_kg_rag_rows.jsonl",
            "mp-kg-rag-generated-source-grounded-counter-narrative",
            "perspective_batch_size",
        ]:
            self.assertIn(expected, text)

    def test_notebook_extracts_html_and_records_reproducible_manifest_fields(self):
        text = self.build_text()
        for expected in [
            "BeautifulSoup",
            "extract_html_document",
            "extraction_method",
            "relative_path",
            "content_sha256",
            "document_type",
            "source_manifest.json",
            "hidden_metadata_file",
        ]:
            self.assertIn(expected, text)

    def test_notebook_has_schema_validation_repair_and_parse_rate_gate(self):
        text = self.build_text()
        for expected in [
            "SEMANTIC_EXTRACTION_SCHEMA",
            "validate_extraction",
            "repair_json_output",
            "parse_rate",
            "minimum_parse_rate",
            "invalid_json",
            "enable_thinking=False",
        ]:
            self.assertIn(expected, text)

    def test_notebook_uses_one_canonical_citation_ledger_for_validation_and_metrics(self):
        text = self.build_text()
        for expected in [
            "build_evidence_ledger",
            "evidence_ledger",
            "parsed_counter_narrative",
            "claim_level_citations",
            "citation_entailment",
            "resolve_citation_tokens",
            "citation_metrics",
            "citation_precision",
            "citation_recall",
            "citation_necessity",
        ]:
            self.assertIn(expected, text)
        self.assertIn("mp_kg_rag", text)
        self.assertNotIn("re.findall(r\\\"\\\\[(SRC", text)

    def test_indicxnli_uses_pinned_data_only_parquet_and_rejects_legacy_script(self):
        text = self.build_text()
        self.assertIn("mteb/IndicXnliPairClassification", text)
        self.assertIn("027e97b9afe84ea3447b57b7705b8864bb2b3a83", text)
        self.assertIn('"source_format": "parquet"', text)
        self.assertIn('"sentence1": "premise"', text)
        self.assertIn('"sentence2": "hypothesis"', text)
        self.assertIn('"labels": "label"', text)
        self.assertNotIn("Divyanshu/indicxnli", text)
        self.assertIn("load_nli_dataset_rows", text)

    def test_dependency_install_is_hashed_transitive_and_non_destructive(self):
        text = self.build_text()
        self.assertNotIn("--force-reinstall", text)
        self.assertIn("hashed, transitive application lock", text)
        self.assertIn("--hash=sha256:", text)
        self.assertIn("--require-hashes", text)
        self.assertIn("--no-deps", text)
        self.assertIn("managed_torch_cuda_mismatch", text)
        self.assertIn("MANAGED_ACCELERATOR_CONTRACT_HASH", text)
        self.assertIn("bitsandbytes_kernel_unavailable", text)
        self.assertIn("managed_accelerator_smoke_failed", text)
        for contract_value in ["2.11.0+cu128", "0.26.0+cu128", "3.6.0", "0.0.35", "0.17.0+cu128", "12.8.90", "12.8"]:
            self.assertIn(contract_value, text)
        self.assertNotIn("not a transitive lock", text)

    def test_lock_counts_distinguish_emitted_application_and_managed_packages(self):
        root = Path(__file__).resolve().parents[1]
        lock = (root / "requirements-remote-vm.lock").read_text(encoding="utf-8")
        for expected in [
            "# emitted_application_package_count=115",
            "# managed_package_count=23",
            "# resolved_package_count=138",
        ]:
            self.assertIn(expected, lock)
        text = self.build_text()
        for expected in [
            "emitted_application_package_count",
            "managed_package_count",
            "resolved_package_count",
            "prompt_budget_irreducible",
        ]:
            self.assertIn(expected, text)

    def test_qwen35_transformers_compatibility_is_pinned_and_fail_closed(self):
        root = Path(__file__).resolve().parents[1]
        direct = (root / "requirements-remote-vm.in").read_text(encoding="utf-8")
        lock = (root / "requirements-remote-vm.lock").read_text(encoding="utf-8")
        text = self.build_text()
        self.assertIn("transformers==5.2.0", direct)
        self.assertRegex(lock, r"(?m)^transformers==5\.2\.0 \\")
        self.assertRegex(lock, r"(?m)^huggingface-hub==1\.3\.2 \\")
        self.assertRegex(lock, r"(?m)^sentence-transformers==5\.2\.2 \\")
        self.assertIn("QWEN35_MIN_TRANSFORMERS_VERSION", text)
        self.assertIn("qwen35_transformers_incompatible", text)
        self.assertIn("validate_qwen35_transformers_compatibility", text)
        self.assertIn('CONFIG["qwen35_transformers_compatibility"]', text)
        self.assertIn('"qwen35_transformers_compatibility": QWEN35_TRANSFORMERS_COMPATIBILITY', text)

    def test_qwen35_transformers_compatibility_function_rejects_transformers_4(self):
        from packaging.version import Version

        notebook = self._build_notebook_json()
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_qwen35_transformers_compatibility"]
        namespace = {
            "Version": Version,
            "QWEN35_MIN_TRANSFORMERS_VERSION": "5.2.0",
            "importlib": types.SimpleNamespace(metadata=types.SimpleNamespace(version=lambda name: "4.57.1")),
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "qwen35-compatibility-probe", "exec"), namespace)
        with self.assertRaisesRegex(RuntimeError, "qwen35_transformers_incompatible"):
            namespace["validate_qwen35_transformers_compatibility"]()

    def test_accelerator_contract_runs_xformers_and_triton_kernel_smokes(self):
        text = self.build_text()
        for expected in [
            '"nvidia-nvjitlink-cu12": "12.8.93"',
            '"nvidia-nvtx-cu12": "12.8.90"',
            "xformers.ops.memory_efficient_attention",
            "@triton.jit",
            "triton_kernel_smoke_failed",
            "xformers_smoke_failed",
        ]:
            self.assertIn(expected, text)

    def test_dependency_lock_has_transitive_resolution_and_excludes_managed_stack(self):
        root = Path(__file__).resolve().parents[1]
        direct = (root / "requirements-remote-vm.in").read_text(encoding="utf-8")
        lock = (root / "requirements-remote-vm.lock").read_text(encoding="utf-8")
        direct_names = {line.split("==", 1)[0].strip().lower().replace("_", "-") for line in direct.splitlines() if line.strip() and not line.lstrip().startswith("#")}
        lock_names = {line.split("==", 1)[0].strip().lower().replace("_", "-") for line in lock.splitlines() if "==" in line and not line.lstrip().startswith(("#", "--"))}
        self.assertTrue(direct_names <= lock_names)
        self.assertGreater(len(lock_names), len(direct_names))
        self.assertIn("datasets==4.3.0", lock)
        self.assertIn("aiohttp==", lock)
        self.assertIn("--hash=sha256:", lock)
        self.assertIn("managed PyTorch/CUDA stack", lock)
        self.assertIn("cut-cross-entropy==25.1.1", lock)
        self.assertNotRegex(lock, r"(?m)^nvidia-(nvjitlink|nvtx)-cu12==")
        self.assertNotRegex(lock, r"(?m)^(torch|torchvision|torchaudio|triton|xformers|torchao)==")
        self.assertNotIn("# cut-cross-entropy", lock)

    def test_all_runtime_model_loads_require_immutable_revisions(self):
        text = self.build_text()
        for expected in [
            "require_model_revision",
            "generator_model_revision",
            "embedding_model_revision",
            "reranker_model_revision",
            "revision=CONFIG[\"generator_model_revision\"]",
            "revision=CONFIG[\"embedding_model_revision\"]",
            "revision=CONFIG[\"reranker_model_revision\"]",
            "3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636",
            "5617a9f61b028005a4858fdac845db406aefb181",
            "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        ]:
            self.assertIn(expected, text)
        self.assertIn("immutable_model_revision_unavailable", text)

    def test_triton_smoke_uses_language_intrinsics(self):
        text = self.build_text()
        self.assertIn("pid = tl.program_id(0)", text)
        self.assertIn("tl.arange(0, BLOCK_SIZE)", text)
        self.assertNotIn("pid = triton.program_id(0)", text)
        self.assertNotIn("triton.arange(0, BLOCK_SIZE)", text)

    def test_parquet_artifacts_canonicalize_structured_object_columns(self):
        text = self.build_text()
        self.assertIn("def parquet_safe_frame(frame):", text)
        self.assertIn("parquet_safe_frame(source_registry).to_parquet", text)
        self.assertIn("parquet_safe_frame(dataset).to_parquet", text)
        self.assertIn("parquet_safe_frame(pages).to_parquet", text)
        self.assertIn("parquet_safe_frame(chunks).to_parquet", text)

    def test_pdf_page_count_is_captured_before_document_close(self):
        text = self.build_text()
        self.assertIn("pdf_candidate_count = len(pdf)", text)
        self.assertIn('"pdf_candidate_count": pdf_candidate_count', text)
        self.assertNotIn('"pdf_candidate_count": len(pdf)', text)

    def test_missing_sources_are_excluded_behind_a_factual_coverage_gate(self):
        text = self.build_text()
        self.assertIn('"minimum_healthy_factual_documents": 40', text)
        self.assertIn('"minimum_factual_document_coverage": 0.80', text)
        self.assertIn('"factual_availability_gate": factual_availability_gate', text)
        self.assertIn('if not factual_availability_gate["pass"]:', text)

    def test_transformers_five_lmformatenforcer_shim_follows_unsloth(self):
        text = self.build_text()
        shim = 'transformers_tokenization_utils.PreTrainedTokenizerBase = transformers.PreTrainedTokenizerBase'
        self.assertIn(shim, text)
        self.assertLess(text.index("from unsloth import FastModel"), text.index(shim))
        self.assertLess(text.index(shim), text.index("from lmformatenforcer import JsonSchemaParser"))

    def test_nli_is_one_shared_pipeline_with_explicit_release_and_no_fitz_leak(self):
        text = self.build_text()
        self.assertIn("load_shared_nli_pipeline", text)
        self.assertIn("release_shared_nli_pipeline", text)
        self.assertIn("nli_qwen_gpu_coexistence", text)
        self.assertIn("with fitz.open", text)
        self.assertIn("fallback_sha256", text)
        self.assertIn("fallback_source_identity", text)

    def test_generation_releases_qwen_before_metrics_and_enforces_memory_gate(self):
        text = self.build_text()
        for expected in [
            "generation_frame = generate_all_variants(dataset)",
            "unload_generator()",
            "qwen_generation_unloaded_before_metrics",
            "qwen_metrics_memory_gate",
            "BERTSCORE_DEVICE",
            "detoxify_model_unloaded_before_metrics",
            "device=BERTSCORE_DEVICE",
        ]:
            self.assertIn(expected, text)
        self.assertLess(text.index("generation_frame = generate_all_variants(dataset)"), text.index("# 13 - Language-aware automatic metrics"))

    def test_generation_prompt_budget_is_adaptive_and_quarantines_only_irreducible_rows(self):
        text = self.build_text()
        for expected in [
            "fit_adaptive_prompt_with_evidence",
            "PromptBudgetQuarantine",
            "prompt_budget_quarantine",
            "prompt_budget_irreducible",
            "dropped_evidence_ids",
            "payload_candidates",
            "whole ledger spans",
        ]:
            self.assertIn(expected, text)
        self.assertNotIn("evidence[i]['text'][:limit]", text)

    def test_perspective_parse_rate_gate_is_typed_and_audits_actual_count(self):
        text = self.build_text()
        for expected in [
            "enforce_perspective_parse_rate",
            "ParseRateQuarantine",
            "minimum_parse_rate",
            "perspective_parse_rate",
            "parsed_count",
            "total_count",
            "parse_status_counts",
        ]:
            self.assertIn(expected, text)
        notebook = self._build_notebook_json()
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "enforce_perspective_parse_rate")
        base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GenerationRowQuarantine")
        quarantine = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ParseRateQuarantine")
        namespace = {"CONFIG": {"minimum_parse_rate": 0.98}, "json": json, "ACCEPTED_PARSE_STATUSES": {"initial", "repair"}}
        exec(compile(ast.Module(body=[base, quarantine, definition], type_ignores=[]), "parse-rate-probe", "exec"), namespace)
        with self.assertRaises(namespace["ParseRateQuarantine"]) as failure:
            namespace["enforce_perspective_parse_rate"]([{"structured_output": {"parse_status": "initial"}}] + [{"structured_output": {"parse_status": "schema_invalid"}}] * 4)
        self.assertEqual(failure.exception.audit["parsed_count"], 1)
        self.assertEqual(failure.exception.audit["total_count"], 5)
        self.assertEqual(failure.exception.audit["parse_rate"], 0.2)

    def test_repair_prompt_has_adaptive_budget_gate_and_immutable_schema_tail(self):
        text = self.build_text()
        for expected in [
            "_repair_payload_candidates",
            "_adaptive_repair_messages",
            "fit_adaptive_prompt_with_evidence",
            "repair_prompt_budget",
            "schema_tail",
            "PromptBudgetQuarantine",
        ]:
            self.assertIn(expected, text)
        self.assertNotIn("raw[:", text)

    def test_compact_repair_candidate_preserves_invalid_output(self):
        notebook = self._build_notebook_json()
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_repair_payload_candidates")
        namespace = {"json": json, "split_visible_thinking": lambda value: str(value or "").strip()}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), "repair-candidate-probe", "exec"), namespace)
        full, compact = namespace["_repair_payload_candidates"]("not valid JSON with source claim", "repair it")
        self.assertIn("not valid JSON with source claim", full)
        self.assertIn("not valid JSON with source claim", compact)
        self.assertNotIn("prior_output_status", compact)

    def test_typed_generation_failures_are_row_quarantined_but_unexpected_errors_propagate(self):
        text = self.build_text()
        for expected in [
            "GenerationRowQuarantine",
            "SchemaValidationQuarantine",
            "except GenerationRowQuarantine",
            "generation_quarantine.json",
            "counts_by_quarantine_type",
            "counts_by_reason",
            "generation_quarantine",
            "append_checkpoint_row(checkpoint, row)",
        ]:
            self.assertIn(expected, text)
        self.assertNotIn("except Exception as failure", text[text.index("def generate_all_variants"):text.index("def load_qwen_for_generation")])

    def test_detoxify_is_explicit_cpu_and_audited_in_identity_manifest(self):
        text = self.build_text()
        for expected in [
            "Detoxify('original', device='cpu')",
            "DETOXIFY_DEVICE",
            "DETOXIFY_STATUS",
            "detoxify_device",
            "detoxify_status",
        ]:
            self.assertIn(expected, text)

    def test_irreducible_prompt_is_a_non_checkpointed_per_row_quarantine(self):
        text = self.build_text()
        notebook = self._build_notebook_json()
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prompt_budget_quarantine_row")
        row_definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "row_quarantine_row")
        namespace = {"CONFIG": {"split_name": "test"}}
        exec(compile(ast.Module(body=[row_definition, definition], type_ignores=[]), "prompt-quarantine-row", "exec"), namespace)
        row = namespace["prompt_budget_quarantine_row"]({"ID": "r1", "Text": "post", "Category": "cat", "Target": "target", "Counter Narrative": "ref"}, "mp_kg_rag", {"reason": "prompt_budget_irreducible", "dropped_evidence_ids": ["E1"]})
        self.assertTrue(row["prompt_quarantine"])
        self.assertEqual(row["prompt_quarantine_reason"], "prompt_budget_irreducible")
        self.assertIsNone(row["raw_output"])
        self.assertNotIn("checkpoint_identity", row)
        # Prompt-budget failures share the typed row-quarantine base with
        # parse-rate and schema failures, so none of them are checkpointed.
        self.assertIn("except GenerationRowQuarantine", text)
        self.assertIn("append_checkpoint_row(checkpoint, row)", text)

    def test_notebook_uses_exact_sentence_spans_and_fail_closed_citation_nli(self):
        text = self.build_text()
        for expected in [
            "sentence_aligned_windows",
            "span_start",
            "span_end",
            "displayed_text",
            "build_claim_citation_records",
            "aggregate_citation_support",
            "require_citation_nli",
            "citation_entailment_unavailable",
            "format_and_entailment",
        ]:
            self.assertIn(expected, text)
        self.assertNotIn("evidence[i]['text'][:limit]", text)

    def test_tokenizer_preflight_rejects_over_budget_without_truncation(self):
        notebook = self._build_notebook_json()
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "preflight_prompt_token_budget"]
        class FakeBatch:
            def __init__(self, rows): self.rows = rows
            def __getitem__(self, key): return self.rows[key]
        class FakeTokenizer:
            def __init__(self): self.calls = []
            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return FakeBatch({"attention_mask": [[1] * len(text) for text in kwargs["text"]]})
        fake = FakeTokenizer()
        namespace = {"tokenizer": fake, "CONFIG": {"max_seq_length": 8}, "json": json}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "budget-probe", "exec"), namespace)
        with self.assertRaisesRegex(RuntimeError, "prompt_token_budget_exceeded"):
            namespace["preflight_prompt_token_budget"](["1234567"], 2)
        self.assertFalse(fake.calls[-1]["truncation"])

    def test_notebook_has_exact_multilingual_abstention_templates(self):
        text = self.build_text()
        for expected in [
            "I cannot verify this from the available evidence.",
            "இந்தக் கூற்றை கிடைக்கும் ஆதாரங்களிலிருந்து சரிபார்க்க முடியவில்லை.",
            "उपलब्ध साक्ष्यों से इस दावे की पुष्टि नहीं कर सकता।",
            "safe_abstention_validator",
            "prompt_token_budget_exceeded",
            "truncation=False",
        ]:
            self.assertIn(expected, text)

    def test_missing_citation_helpers_fail_closed_without_abstention_fallback(self):
        notebook = self._build_notebook_json()
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        wanted = {"build_evidence_ledger", "validate_evidence_ids", "resolve_citation_tokens", "validate_response"}
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        namespace = {"re": re, "math": __import__("math"), "json": json, "INTERNAL_RESPONSE_METADATA_KEYS": set(), "ACCEPTED_PARSE_STATUSES": set(), "canonical_page": lambda value: value}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "missing-citation-helper-probe", "exec"), namespace)
        evidence = [{"evidence_chunk_id": "EV1", "text": "Evidence text."}]
        result = namespace["validate_response"]({"counter_narrative": "safe abstention", "cited_evidence_ids": [], "factual_claims": []}, evidence)
        assert result["pass"] is False
        assert result["citation_entailment_status"] == "helper_unavailable"

    def test_tracked_notebook_is_byte_fresh_from_builder(self):
        from build_remote_vm_qwen35_mpkg_rag import build_notebook

        tracked = Path(__file__).resolve().parents[1] / "outputs" / "remote_vm_qwen35_mpkg_rag.ipynb"
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / tracked.name
            build_notebook(generated)
            self.assertEqual(tracked.read_bytes(), generated.read_bytes())

    def test_notebook_records_precision_memory_and_unloads_retrieval_models(self):
        text = self.build_text()
        for expected in [
            "gpu_snapshot",
            "bf16_supported",
            "is_loaded_in_4bit",
            "quantization_config",
            "first_parameter_dtype",
            "unload_retrieval_models",
            "torch.cuda.empty_cache()",
            "retrieval_models_unloaded",
        ]:
            self.assertIn(expected, text)

    def test_notebook_freezes_stratified_controls_and_run_identity(self):
        text = self.build_text()
        for expected in [
            "evaluation_categories",
            "make_frozen_split",
            "stratify_key",
            "frozen_split.json",
            "split_name",
            "input_text_sha256",
            "config_hash",
            "prompt_template_hash",
            "resume_identity_mismatch",
        ]:
            self.assertIn(expected, text)

    def test_notebook_builds_typed_source_backed_semantic_kg_and_uses_graph_scores(self):
        text = self.build_text()
        for expected in [
            "Document",
            "EvidenceChunk",
            "Mention",
            "Entity",
            "Claim",
            "has_subject",
            "has_object",
            "supports",
            "refutes",
            "provenance",
            "review_status",
            "semantic_kg_nodes.parquet",
            "semantic_kg_edges.parquet",
            "semantic_kg_manifest.json",
            "graph_score",
            "graph_ablation",
            "polarity",
            "modality",
            "attribution",
        ]:
            self.assertIn(expected, text)

    def test_notebook_hardens_semantic_kg_reuse_multilingual_ids_and_graph_runtime(self):
        text = self.build_text()
        for expected in [
            "index_manifest.json",
            "extraction_manifest.json",
            "extraction_identity",
            "INDEX_IDENTITY",
            "canonical_page",
            "unicodedata.normalize",
            "no_indexable_chunks",
            "def graph_search",
        ]:
            self.assertIn(expected, text)

    def test_notebook_injects_the_exact_tested_core_source(self):
        from build_remote_vm_qwen35_mpkg_rag import CORE_SOURCE

        text = self.build_text()
        self.assertIn(CORE_SOURCE, text)
        self.assertIn("CORE_SOURCE_SHA256", text)
        self.assertIn("core_source_sha256", text)

    def test_remote_environment_uses_checked_in_exact_lock_without_floating_installs(self):
        from build_remote_vm_qwen35_mpkg_rag import LOCK_PATH, LOCK_SOURCE

        self.assertTrue(LOCK_PATH.exists())
        self.assertEqual(LOCK_SOURCE, LOCK_PATH.read_text(encoding="utf-8"))
        self.assertNotIn("%pip install -U", LOCK_SOURCE)
        requirements = [line.strip().rstrip("\\").rstrip() for line in LOCK_SOURCE.splitlines()
                        if line.strip() and not line.lstrip().startswith(("#", "--"))]
        self.assertGreater(len(requirements), 10)
        for requirement in requirements:
            self.assertRegex(requirement, r"^[A-Za-z0-9_.-]+==[^=\s]+$")
        self.assertIn("--hash=sha256:", LOCK_SOURCE)
        text = self.build_text()
        self.assertNotIn("%pip install -U", text)
        self.assertIn("requirements-remote-vm.lock", text)
        self.assertIn("--requirement", text)
        self.assertIn("verify_locked_environment", text)

    def test_environment_fingerprint_is_complete_and_part_of_identity(self):
        from build_remote_vm_qwen35_mpkg_rag import LOCK_SOURCE

        text = self.build_text()
        for expected in [
            "LOCKFILE_SHA256",
            "locked_requirements",
            "installed_package_versions",
            "verify_locked_environment",
            "environment_fingerprint",
            "torch_cuda_version",
            "cuda_driver_version",
            "cudnn_version",
            "gpu_name",
            "model_revisions",
            "LOCKFILE_SHA256",
        ]:
            self.assertIn(expected, text)
        self.assertIn("environment_fingerprint", text[text.index("IDENTITY ="):])
        self.assertIn("LOCKFILE_SHA256", text[text.index("IDENTITY ="):])
        self.assertNotEqual(LOCK_SOURCE.strip(), "")

    def test_seed_controls_cover_hashing_and_plotting_bootstrap(self):
        text = self.build_text()
        for expected in [
            "PYTHONHASHSEED",
            "random.seed(SEED)",
            "np.random.seed(SEED)",
            "torch.manual_seed(SEED)",
            "np.random.default_rng(seed)",
            "BOOTSTRAP_SEED",
            "errorbar=None",
        ]:
            self.assertIn(expected, text)

    def test_notebook_removes_the_old_alias_ppr_and_duplicate_extractor(self):
        text = self.build_text()
        for forbidden in [
            "personalized_graph_scores",
            "nx.pagerank",
            "import networkx as nx",
            "deterministic_relation_patterns",
            "extract_semantic_claims(chunk)",
        ]:
            self.assertNotIn(forbidden, text)
        for required in [
            "load_source_registry",
            "build_semantic_graph",
            "expand_graph_from_seeds",
            "reciprocal_rank_fusion",
            "select_evidence",
        ]:
            self.assertIn(required, text)

    def test_notebook_uses_collision_safe_document_uids_end_to_end(self):
        text = self.build_text()
        for expected in [
            "document_uid",
            "legacy_source_id",
            "document_uid_manifest_hash",
            "document_uid",
            "chunk_document_uid",
            "document_uid",
        ]:
            self.assertIn(expected, text)
        self.assertNotIn('assert not pd.Series([row["source_id"] for row in rows]).duplicated().any()', text)
        self.assertNotIn('stable_id(page["source_id"], page_value', text)

    def test_notebook_has_sequential_single_model_lifecycle(self):
        text = self.build_text()
        ordered = [
            "load_qwen_for_extraction",
            "unload_generator",
            "load_retrieval_models",
            "unload_retrieval_models",
            "load_qwen_for_generation",
        ]
        positions = [text.index(value) for value in ordered]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(text.count('os.environ.get("MODEL_ID", "unsloth/Qwen3.5-4B")'), 1)
        self.assertIn("same Qwen model is loaded sequentially", text)

    def test_notebook_precomputes_full_query_signatures_and_graph_retrieval(self):
        text = self.build_text()
        for expected in [
            "build_query_signature",
            "query_signature",
            "predicate",
            "polarity",
            "modality",
            "stance",
            "expand_graph_from_seeds",
            "query_signature=query_signature",
        ]:
            self.assertIn(expected, text)
        self.assertNotIn("seed_entities = aliases", text)

    def test_notebook_uses_canonical_evidence_ids_for_all_model_and_metric_interfaces(self):
        text = self.build_text()
        for expected in [
            "validate_evidence_ids",
            "evidence_id",
            'f"E{i + 1}"',
            "supported_evidence_ids",
            "selected_evidence_ids",
            "cited_evidence_ids",
            "evidence_ledger",
            "ledger_id",
        ]:
            self.assertIn(expected, text)
        self.assertNotIn("cited_chunk_ids", text)

    def test_notebook_checkpoint_identity_and_paired_ablation_are_frozen(self):
        text = self.build_text()
        for expected in [
            "input_text_sha256",
            "corpus_manifest_hash",
            "chunk_manifest_hash",
            "graph_manifest_hash",
            "extraction_model",
            "extraction_prompt_revision",
            "retrieval_thresholds",
            "graph_config",
            "checkpoint_identity",
            "resume_identity_mismatch",
            "frozen_evidence_ids",
            "graph_on",
            "graph_off",
            "paired_retrieval_evaluation",
        ]:
            self.assertIn(expected, text)

    def test_generate_batch_uses_named_tokenizer_text_argument(self):
        text = self.build_text()
        self.assertIn("tokenizer(text=prompts", text)
        self.assertNotIn("tokenizer(prompts, return_tensors", text)

    def test_strict_payload_quarantines_malformed_array_elements(self):
        notebook = self._build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definitions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_strict_payload"
        ]
        namespace = {"re": re, "math": __import__("math")}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "strict-payload-probes", "exec"), namespace)
        schema = {
            "required": ["cited_evidence_ids"],
            "properties": {"cited_evidence_ids": {"type": "array", "items": {"type": "string"}}},
        }
        for value in ([7, "E1"], [["E1"]], ["E1", None]):
            result = namespace["_strict_payload"]({"cited_evidence_ids": value}, schema, "final_response", {"E1"})
            self.assertFalse(result["valid"], value)
            self.assertTrue(result["quarantine"], value)

    def test_parse_rates_accept_only_initial_or_repair(self):
        text = self.build_text()
        self.assertIn("ACCEPTED_PARSE_STATUSES", text)
        self.assertIn("get(\"parse_status\") in ACCEPTED_PARSE_STATUSES", text)
        self.assertNotIn('get("parse_status") != "schema_invalid"', text)

    def test_perspective_name_is_bound_to_expected_agent(self):
        text = self.build_text()
        self.assertIn("expected_perspective", text)
        self.assertIn("perspective_name_mismatch", text)
        self.assertIn("expected_perspective=name", text)
        notebook = self._build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definitions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in {"_strict_payload", "validate_perspective"}
        ]
        namespace = {"re": re, "math": __import__("math"), "PERSPECTIVE_SCHEMA": {
            "required": ["perspective", "rationale", "claims_to_address", "supported_evidence_ids", "response_guidance", "risk_flags", "confidence"],
            "properties": {
                "perspective": {"type": "string"}, "rationale": {"type": "string"},
                "claims_to_address": {"type": "array", "items": {"type": "string"}},
                "supported_evidence_ids": {"type": "array", "items": {"type": "string"}},
                "response_guidance": {"type": "array", "items": {"type": "string"}},
                "risk_flags": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
        }}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "perspective-probes", "exec"), namespace)
        payload = {
            "perspective": "wrong_agent", "rationale": "r", "claims_to_address": [],
            "supported_evidence_ids": [], "response_guidance": [], "risk_flags": [], "confidence": 0.5,
        }
        result = namespace["validate_perspective"](payload, set(), "fact_checking")
        self.assertFalse(result["valid"])
        self.assertIn("perspective_name_mismatch", result["reasons"])

    def test_chat_template_receives_thinking_control(self):
        text = self.build_text()
        self.assertIn(
            "tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)",
            text,
        )

    def test_shared_universe_is_used_as_metric_domain(self):
        text = self.build_text()
        self.assertIn("selected evidence outside shared frozen universe", text)
        self.assertIn("set(universe)", text)
        notebook = self._build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definitions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "paired_retrieval_metrics"
        ]
        class FakeNumpy:
            @staticmethod
            def mean(values):
                values = list(values)
                return sum(values) / len(values) if values else 0.0

        namespace = {"np": FakeNumpy, "CONFIG": {"minimum_authority": 0.5}}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "paired-metrics-probes", "exec"), namespace)
        on = {"evidence": [{"evidence_chunk_id": "EV1", "authority_score": 1.0, "status": "accepted", "rerank_probability": 0.8}]}
        off = {"evidence": [{"evidence_chunk_id": "EV2", "authority_score": 0.0, "status": "quarantined", "rerank_probability": 0.2}]}
        metrics = namespace["paired_retrieval_metrics"](on, off, ["EV1", "EV2", "EV3"])
        self.assertEqual(metrics["shared_frozen_universe"], ["EV1", "EV2", "EV3"])
        self.assertEqual(metrics["overlap"], 0.0)
        self.assertEqual(metrics["graph_only_gain"], 0.0)
        with self.assertRaises(ValueError):
            namespace["paired_retrieval_metrics"](on, off, ["EV1"])

    def test_all_generation_schemas_are_strict_and_unknown_eids_quarantine(self):
        text = self.build_text()
        for expected in [
            "QUERY_SIGNATURE_SCHEMA",
            "PERSPECTIVE_SCHEMA",
            "PLAN_SCHEMA",
            "FINAL_RESPONSE_SCHEMA",
            "validate_query_signature",
            "validate_perspective",
            "validate_plan",
            "validate_final_response",
            "additionalProperties",
            "schema_invalid",
            "quarantine",
            "confidence",
            '"minimum": 0',
            '"maximum": 1',
        ]:
            self.assertIn(expected, text)
        self.assertIn("unknown_evidence_ids", text)
        self.assertIn("raise ValueError", text)

    def test_stale_checkpoint_identity_is_rejected_by_executable_helper(self):
        notebook = self._build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definitions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "validate_checkpoint_identity"
        ]
        self.assertEqual(len(definitions), 1, "checkpoint validation must be executable and testable")
        namespace = {}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "checkpoint-probe", "exec"), namespace)
        expected = {"input_text_sha256": "input-a", "graph_manifest_hash": "graph-a"}
        self.assertTrue(namespace["validate_checkpoint_identity"](expected, expected))
        with self.assertRaisesRegex(RuntimeError, "resume_identity_mismatch"):
            namespace["validate_checkpoint_identity"](
                {**expected, "graph_manifest_hash": "graph-stale"}, expected
            )

    def test_checkpoint_revalidation_reconstructs_all_materialized_fields_and_verification(self):
        notebook = self._build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        wanted = {"authenticated_checkpoint_evidence", "checkpoint_materialization", "checkpoint_raw_envelope", "canonical_checkpoint_row", "revalidate_checkpoint_row"}
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        namespace = {
            "json": json,
            "VARIANT_COLUMNS": {"kg_rag": "kg-rag-generated-source-grounded-counter-narrative"},
            "EVIDENCE_CACHE": {"1": {"evidence": [{"evidence_chunk_id": "EV1", "chunk_id": "C1", "document_uid": "D1", "source_id": "SRC1", "text": "Evidence."}]}},
            "ACCEPTED_PARSE_STATUSES": {"initial", "repair"},
            "parse_json_object": json.loads,
            "build_evidence_ledger": lambda evidence: [{"evidence_id": "E1", "evidence_chunk_id": evidence[0]["evidence_chunk_id"]}],
            "validate_response": lambda payload, evidence: {"pass": True, "schema_valid": True, "citation_count": 1, "detail": payload["counter_narrative"]},
            "checkpoint_identity": lambda record, variant: {"record_id": str(record["ID"]), "variant": variant},
            "validate_checkpoint_identity": lambda saved, expected: True if saved == expected else (_ for _ in ()).throw(RuntimeError("resume_identity_mismatch")),
            "CONFIG": {"split_name": "test"},
            "CONFIG_HASH": "CONFIG",
            "PROMPT_TEMPLATE_HASH": "PROMPT",
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "checkpoint-revalidation-probe", "exec"), namespace)
        record = {"ID": "1", "Text": "post", "Category": "Homophobic", "Target": "WHO", "Counter Narrative": "reference", "input_text_sha256": "input"}
        payload = {"counter_narrative": "Claim [E1].", "cited_evidence_ids": ["E1"], "factual_claims": [], "safety_notes": [], "parse_status": "initial"}
        evidence = namespace["EVIDENCE_CACHE"]["1"]["evidence"]
        row = namespace["canonical_checkpoint_row"](record, "kg_rag", json.dumps({key: value for key, value in payload.items() if key != "parse_status"}), payload, evidence, {})
        self.assertEqual(namespace["revalidate_checkpoint_row"](row, record, "kg_rag"), row)
        for field, replacement in [
            ("response", "tampered"),
            ("parsed_counter_narrative", "tampered"),
            ("kg-rag-generated-source-grounded-counter-narrative", "tampered"),
            ("evidence_ledger", []),
            ("verification", {"pass": True, "schema_valid": True, "citation_count": 999, "detail": "Claim [E1]."}),
            ("evidence", [{"evidence_chunk_id": "EV-TAMPERED", "text": "Evidence."}]),
        ]:
            tampered = dict(row, **{field: replacement})
            with self.assertRaisesRegex(RuntimeError, "resume_identity_mismatch"):
                namespace["revalidate_checkpoint_row"](tampered, record, "kg_rag")

    def test_checkpoint_revalidation_requires_exact_canonical_row_and_all_mp_fields(self):
        notebook = self._build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        wanted = {"authenticated_checkpoint_evidence", "checkpoint_materialization", "checkpoint_raw_envelope", "canonical_checkpoint_row", "revalidate_checkpoint_row"}
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        namespace = {
            "json": json,
            "VARIANT_COLUMNS": {"mp_kg_rag": "mp-kg-rag-generated-source-grounded-counter-narrative"},
            "EVIDENCE_CACHE": {"1": {"evidence": [{"evidence_chunk_id": "EV1", "chunk_id": "C1", "document_uid": "D1", "source_id": "SRC1", "text": "Evidence."}]}},
            "ACCEPTED_PARSE_STATUSES": {"initial", "repair"},
            "parse_json_object": json.loads,
            "build_evidence_ledger": lambda evidence: [{"evidence_id": "E1", "evidence_chunk_id": evidence[0]["evidence_chunk_id"]}],
            "validate_response": lambda payload, evidence: {"pass": True, "schema_valid": True, "citation_count": 1, "detail": payload["counter_narrative"]},
            "checkpoint_identity": lambda record, variant: {"record_id": str(record["ID"]), "variant": variant},
            "validate_checkpoint_identity": lambda saved, expected: True if saved == expected else (_ for _ in ()).throw(RuntimeError("resume_identity_mismatch")),
            "CONFIG": {"split_name": "test"},
            "CONFIG_HASH": "CONFIG",
            "PROMPT_TEMPLATE_HASH": "PROMPT",
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "canonical-checkpoint-probe", "exec"), namespace)
        record = {"ID": "1", "Text": "post", "Category": "Homophobic", "Target": "WHO", "Counter Narrative": "reference", "input_text_sha256": "input"}
        payload = {"counter_narrative": "Claim [E1].", "cited_evidence_ids": ["E1"], "factual_claims": [], "safety_notes": {}, "parse_status": "initial"}
        evidence = namespace["EVIDENCE_CACHE"]["1"]["evidence"]
        mp_fields = {"perspective_rationale": {"fact_checking": "grounded"}, "perspective_parse_rate": 1.0, "mp_perspective_outputs": [{"perspective": "fact_checking", "perspective_rationale": "grounded", "structured_output": {"parse_status": "initial"}, "supported_evidence_ids": [], "raw_output": "{}"}], "mp_response_plan": {"claim_focus": "focus"}, "mp_plan_raw_output": "{}"}
        row = namespace["canonical_checkpoint_row"](record, "mp_kg_rag", json.dumps(payload), payload, evidence, mp_fields)
        self.assertEqual(namespace["revalidate_checkpoint_row"](row, record, "mp_kg_rag"), row)
        fields = ["Text", "Category", "Target", "Counter Narrative", "variant", "split_name", "input_text_sha256", "config_hash", "prompt_template_hash", "checkpoint_identity", "response", "parsed_counter_narrative", "mp-kg-rag-generated-source-grounded-counter-narrative", "evidence", "evidence_ledger", "verification", "parse_status", "perspective_rationale", "perspective_parse_rate", "mp_perspective_outputs", "mp_response_plan", "mp_plan_raw_output", "raw_output"]
        for field in fields:
            replacement = [] if isinstance(row[field], list) else ({} if isinstance(row[field], dict) else "tampered")
            with self.assertRaisesRegex(RuntimeError, "resume_identity_mismatch", msg=field):
                namespace["revalidate_checkpoint_row"]({**row, field: replacement}, record, "mp_kg_rag")
        with self.assertRaisesRegex(RuntimeError, "resume_identity_mismatch"):
            namespace["revalidate_checkpoint_row"]({key: value for key, value in row.items() if key != "Text"}, record, "mp_kg_rag")
        with self.assertRaisesRegex(RuntimeError, "resume_identity_mismatch"):
            namespace["revalidate_checkpoint_row"]({**row, "unexpected": True}, record, "mp_kg_rag")

    def test_acceptance_gate_forbids_legacy_graph_and_raw_chunk_validation_paths(self):
        text = self.build_text()
        forbidden = (
            "nx.pagerank",
            "personalized_graph_scores",
            "seed_entities = aliases",
            "cited_chunk_ids",
            "keyword_relation_classifier",
        )
        for token in forbidden:
            self.assertNotIn(token, text)
        self.assertIn("expand_graph_from_seeds", text)
        self.assertIn('"graph_score"', text)
        self.assertIn("validate_evidence_ids", text)
        self.assertIn('r\"E[1-9][0-9]*\"', text)

    def test_validate_response_rejects_malformed_and_inconsistent_outputs(self):
        text = self.build_text()
        notebook = self._build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        wanted = {"build_evidence_ledger", "validate_evidence_ids", "resolve_citation_tokens", "claim_level_citations", "validate_response"}
        tree = ast.parse(source)
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        namespace = {
            "re": re,
            "math": __import__("math"),
            "ACCEPTED_PARSE_STATUSES": {"initial", "repair"},
            "INTERNAL_RESPONSE_METADATA_KEYS": {"parse_status", "quarantine", "schema_errors"},
            "canonical_page": lambda value: value,
            "FINAL_RESPONSE_SCHEMA": {
                "properties": {
                    "counter_narrative": {},
                    "cited_evidence_ids": {},
                    "factual_claims": {},
                    "safety_notes": {},
                },
                "required": [
                    "counter_narrative",
                    "cited_evidence_ids",
                    "factual_claims",
                    "safety_notes",
                ],
            },
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "notebook-validation-probes", "exec"), namespace)
        evidence = [{"evidence_chunk_id": "EV1", "chunk_id": "C1", "document_uid": "D1", "source_id": "SRC1", "section": "s", "text": "Evidence text."}]
        self.assertFalse(namespace["validate_response"]({}, evidence)["pass"])
        self.assertFalse(namespace["validate_response"]({"counter_narrative": "", "cited_evidence_ids": []}, evidence)["pass"])
        self.assertFalse(namespace["validate_response"]({"counter_narrative": "Claim [E9].", "cited_evidence_ids": ["E9"]}, evidence)["pass"])
        self.assertFalse(namespace["validate_response"]({"counter_narrative": "Claim [E1].", "cited_evidence_ids": []}, evidence)["pass"])
        self.assertFalse(namespace["validate_response"]({"counter_narrative": "Claim [E1].", "cited_evidence_ids": ["E2"]}, evidence)["pass"])

    def test_validate_response_quarantines_non_list_evidence_ids_without_throwing(self):
        notebook = self._build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        wanted = {"_strict_payload", "validate_final_response", "build_evidence_ledger", "validate_evidence_ids", "resolve_citation_tokens", "claim_level_citations", "validate_response", "split_visible_thinking", "parse_json_object", "repair_json_output", "parse_with_one_repair", "baseline_response"}
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        namespace = {
            "re": re,
            "math": __import__("math"),
            "json": json,
            "ACCEPTED_PARSE_STATUSES": {"initial", "repair"},
            "INTERNAL_RESPONSE_METADATA_KEYS": {"parse_status", "quarantine", "schema_errors"},
            "CONFIG": {"answer_max_new_tokens": 32},
            "generate_batch": lambda *args, **kwargs: ['{"counter_narrative":"ok","cited_evidence_ids":[],"factual_claims":[],"safety_notes":[],"extra_wrapper":true}'],
            "canonical_page": lambda value: value,
            "FINAL_RESPONSE_SCHEMA": {
                "properties": {
                    "counter_narrative": {"type": "string"},
                    "cited_evidence_ids": {"type": "array"},
                    "factual_claims": {"type": "array"},
                    "safety_notes": {"type": "array"},
                },
                "required": ["counter_narrative", "cited_evidence_ids", "factual_claims", "safety_notes"],
            },
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "evidence-id-probes", "exec"), namespace)
        evidence = [{"evidence_chunk_id": "EV1", "chunk_id": "C1", "document_uid": "D1", "source_id": "SRC1", "section": "s", "text": "Evidence text."}]
        for bad_ids in (7, None, ["E1", 7], ["E1", None]):
            try:
                result = namespace["validate_response"]({
                    "counter_narrative": "Claim [E1].",
                    "cited_evidence_ids": bad_ids,
                    "factual_claims": [],
                    "safety_notes": [],
                    "parse_status": "initial",
                }, evidence)
            except Exception as exc:
                self.fail(f"validate_response raised for {bad_ids!r}: {exc}")
            self.assertFalse(result["pass"], bad_ids)

    def test_query_signature_cache_rejects_duplicate_swapped_tampered_and_invalid_rows(self):
        notebook = self._build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        wanted = {"_strict_payload", "validate_query_signature", "query_signature_entity_ids_usable", "verify_query_signature_cache"}
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        namespace = {
            "re": re,
            "math": __import__("math"),
            "ACCEPTED_PARSE_STATUSES": {"initial", "repair"},
            "QUERY_SIGNATURE_SCHEMA": {
                "required": ["entity_ids", "predicates", "polarities", "modalities", "desired_stances"],
                "properties": {key: {"type": "array"} for key in ["entity_ids", "predicates", "polarities", "modalities", "desired_stances"]},
            },
            "record_identities": [
                {"record_id": "1", "input_text_sha256": "sha1", "target_sha256": "t1", "category_sha256": "c1"},
                {"record_id": "2", "input_text_sha256": "sha2", "target_sha256": "t2", "category_sha256": "c2"},
            ],
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "query-cache-probes", "exec"), namespace)
        signature = {key: [] for key in ["entity_ids", "predicates", "polarities", "modalities", "desired_stances"]}
        rows = [
            {"ID": "1", "record_identity": namespace["record_identities"][0], "input_text_sha256": "sha1", "query_signature": signature, "parse_status": "initial"},
            {"ID": "2", "record_identity": namespace["record_identities"][1], "input_text_sha256": "sha2", "query_signature": signature, "parse_status": "repair"},
        ]
        self.assertTrue(namespace["verify_query_signature_cache"](rows))
        duplicate = [rows[0], copy.deepcopy(rows[0])]
        self.assertFalse(namespace["verify_query_signature_cache"](duplicate))
        swapped = copy.deepcopy(rows)
        swapped[0]["ID"], swapped[1]["ID"] = swapped[1]["ID"], swapped[0]["ID"]
        self.assertFalse(namespace["verify_query_signature_cache"](swapped))
        tampered = copy.deepcopy(rows)
        tampered[0]["record_identity"]["input_text_sha256"] = "tampered"
        self.assertFalse(namespace["verify_query_signature_cache"](tampered))
        invalid_signature = copy.deepcopy(rows)
        invalid_signature[0]["query_signature"] = {**signature, "extra_wrapper": True}
        self.assertFalse(namespace["verify_query_signature_cache"](invalid_signature))

        catalog = {
            "entities": [
                {"entity_id": "ENT:linked", "namespace": "corpus", "retrieval_allowed": True, "factual_identity_allowed": True, "link_status": "linked"},
                {"entity_id": "TGT:anchor", "namespace": "target", "retrieval_allowed": True, "factual_identity_allowed": False, "link_status": "linked"},
            ]
        }
        linked_rows = copy.deepcopy(rows)
        linked_rows[0]["query_signature"] = {**signature, "entity_ids": ["ENT:linked"]}
        self.assertTrue(namespace["verify_query_signature_cache"](linked_rows, catalog=catalog))
        target_only = copy.deepcopy(rows)
        target_only[0]["query_signature"] = {**signature, "entity_ids": ["TGT:anchor"]}
        self.assertTrue(namespace["verify_query_signature_cache"](target_only, catalog=catalog))
        unknown = copy.deepcopy(rows)
        unknown[0]["query_signature"] = {**signature, "entity_ids": ["ENT:tampered"]}
        self.assertFalse(namespace["verify_query_signature_cache"](unknown, catalog=catalog))

    def test_quality_gates_require_meaningful_graph_and_linked_query_coverage(self):
        text = self.build_text()
        for expected in [
            "semantic_linkage_quality_gate",
            "minimum_graph_linked_claim_rate",
            "minimum_query_linked_entity_rate",
            "target_anchor_only_signatures",
            "uncalibrated_diagnostic_only",
            "citation_entailment_unavailable",
        ]:
            self.assertIn(expected, text)
        notebook = self._build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definitions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"semantic_linkage_quality_gate", "query_signature_entity_ids_usable"}
        ]
        namespace = {
            "SCORING_CALIBRATION_STATUS": "uncalibrated_diagnostic_only",
            "SELF_CONFIDENCE_STATUS": "uncalibrated_diagnostic_only",
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "quality-gate-probes", "exec"), namespace)
        catalog = {"entities": [
            {"entity_id": "ENT:linked", "namespace": "corpus", "retrieval_allowed": True, "factual_identity_allowed": True, "link_status": "linked"},
            {"entity_id": "TGT:only", "namespace": "target", "retrieval_allowed": True, "factual_identity_allowed": False, "link_status": "linked"},
        ]}
        thresholds = {"minimum_graph_linked_claim_rate": 0.02, "minimum_query_linked_entity_rate": 0.02, "minimum_linked_claims": 1, "minimum_linked_queries": 1}
        graph = {"Claim": [{"review_status": "accepted", "subject_entity_id": "ENT:linked"}] + [{"review_status": "accepted", "subject_entity_id": "ENT:unlinked"}] * 99}
        result = namespace["semantic_linkage_quality_gate"](graph, catalog, [{"entity_ids": ["ENT:linked"]}], thresholds)
        self.assertFalse(result["pass"], "one linked claim out of corpus must fail the meaningful coverage gate")
        zero_queries = namespace["semantic_linkage_quality_gate"](
            {"Claim": [{"review_status": "accepted", "subject_entity_id": "ENT:linked"}]}, catalog,
            [{"entity_ids": ["TGT:only"]}], thresholds,
        )
        self.assertFalse(zero_queries["pass"])
        self.assertEqual(zero_queries["target_anchor_only_signatures"], 1)

    def test_paired_metrics_align_vectors_to_shared_universe_domain(self):
        notebook = self._build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "paired_retrieval_metrics"]
        class FakeNumpy:
            @staticmethod
            def mean(values):
                values = list(values)
                return sum(values) / len(values) if values else 0.0

        namespace = {"np": FakeNumpy, "CONFIG": {"minimum_authority": 0.5}}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "aligned-metrics-probes", "exec"), namespace)
        on = {"evidence": [{"evidence_chunk_id": "EV1", "authority_score": 1.0, "status": "accepted", "rerank_probability": 0.8}]}
        off = {"evidence": [{"evidence_chunk_id": "EV2", "authority_score": 0.0, "status": "quarantined", "rerank_probability": 0.2}]}
        short = namespace["paired_retrieval_metrics"](on, off, ["EV1", "EV2"])
        long = namespace["paired_retrieval_metrics"](on, off, ["EV1", "EV2", "EV3"])
        self.assertIn("graph_on_selection_vector", long)
        self.assertEqual(len(long["graph_on_selection_vector"]), 3)
        self.assertEqual(long["universe_traces"][-1]["evidence_id"], "EV3")
        self.assertEqual(long["universe_traces"][-1]["graph_on_selected"], 0)
        self.assertLess(long["authority_rate"], short["authority_rate"])
        self.assertLess(long["selected_score_mean"], short["selected_score_mean"])
        self.assertIn("graph_on_score_vector", source)
        self.assertIn("graph_off_score_vector", source)

    def test_response_schema_does_not_drop_model_wrappers_and_baseline_is_strict(self):
        text = self.build_text()
        self.assertIn("INTERNAL_RESPONSE_METADATA_KEYS", text)
        self.assertIn("baseline_response", text)
        self.assertIn("validate_final_response", text)
        self.assertIn("parse_status", text)
        self.assertIn("output_schema=FINAL_RESPONSE_SCHEMA", text)
        notebook = self._build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        wanted = {"_strict_payload", "validate_final_response", "build_evidence_ledger", "validate_evidence_ids", "resolve_citation_tokens", "claim_level_citations", "validate_response", "split_visible_thinking", "parse_json_object", "repair_json_output", "parse_with_one_repair", "baseline_response"}
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        namespace = {
            "re": re,
            "math": __import__("math"),
            "json": json,
            "ACCEPTED_PARSE_STATUSES": {"initial", "repair"},
            "INTERNAL_RESPONSE_METADATA_KEYS": {"parse_status", "quarantine", "schema_errors"},
            "CONFIG": {"answer_max_new_tokens": 32},
            "generate_batch": lambda *args, **kwargs: ['{"counter_narrative":"ok","cited_evidence_ids":[],"factual_claims":[],"safety_notes":[],"extra_wrapper":true}'],
            "canonical_page": lambda value: value,
            "FINAL_RESPONSE_SCHEMA": {
                "properties": {
                    "counter_narrative": {"type": "string"},
                    "cited_evidence_ids": {"type": "array"},
                    "factual_claims": {"type": "array"},
                    "safety_notes": {"type": "array"},
                },
                "required": ["counter_narrative", "cited_evidence_ids", "factual_claims", "safety_notes"],
            },
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "wrapper-probes", "exec"), namespace)
        evidence = [{"evidence_chunk_id": "EV1", "chunk_id": "C1", "document_uid": "D1", "source_id": "SRC1", "section": "s", "text": "Evidence text."}]
        wrapped = {"counter_narrative": "Claim [E1].", "cited_evidence_ids": ["E1"], "factual_claims": ["Claim"], "safety_notes": [], "parse_status": "initial", "extra_wrapper": {"payload": "hidden"}}
        self.assertFalse(namespace["validate_response"](wrapped, evidence)["pass"])
        baseline, _ = namespace["baseline_response"]("post", "target")
        self.assertEqual(baseline["parse_status"], "schema_invalid")
        self.assertTrue(baseline["quarantine"])

    def test_semantic_extraction_quarantine_is_not_parse_accepted(self):
        text = self.build_text()
        self.assertIn("semantic_invalid", text)
        self.assertIn("extraction_parse_status", text)
        self.assertIn("validation.get(\"quarantined\")", text)
        self.assertIn("output_schema=SEMANTIC_EXTRACTION_SCHEMA", text)
        notebook = self._build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"extraction_parse_status", "extraction_parse_accepted"}]
        def trusted_validation_fingerprint(validation):
            material = {key: value for key, value in validation.items() if key != "validation_fingerprint"}
            return hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        namespace = {
            "ACCEPTED_PARSE_STATUSES": {"initial", "repair"},
            "hashlib": hashlib,
            "json": json,
            "_validation_fingerprint": trusted_validation_fingerprint,
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "extraction-status-probes", "exec"), namespace)
        validation = {"status": "quarantined", "quarantined": [{"reasons": ["mention_span_mismatch"]}]}
        self.assertEqual(namespace["extraction_parse_status"]("initial", validation), "semantic_invalid")
        self.assertFalse(namespace["extraction_parse_accepted"]("initial", validation))

    def test_extraction_cache_is_complete_identity_bound_and_fail_closed(self):
        text = self.build_text()
        for expected in [
            "EXPECTED_EXTRACTION_UNIVERSE",
            "EXPECTED_EXTRACTION_UNIVERSE_HASH",
            "verify_extraction_cache",
            "max(1, len(EXPECTED_EXTRACTION_UNIVERSE))",
            "raw_output",
        ]:
            self.assertIn(expected, text)
        self.assertLess(text.index("verify_extraction_cache(extraction_rows"), text.index("need_extraction ="))
        notebook = self._build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "verify_extraction_cache"]
        def trusted_validation_fingerprint(validation):
            material = {key: value for key, value in validation.items() if key != "validation_fingerprint"}
            return hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        namespace = {
            "ACCEPTED_PARSE_STATUSES": {"initial", "repair"},
            "hashlib": hashlib,
            "json": json,
            "_validation_fingerprint": trusted_validation_fingerprint,
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "extraction-cache-probes", "exec"), namespace)
        def source_text(identity):
            return f"source text for {identity['chunk_id']}"
        expected = [
            {"document_uid": "D1", "chunk_id": "C1", "text_sha256": hashlib.sha256(source_text({"chunk_id": "C1"}).encode()).hexdigest()},
            {"document_uid": "D2", "chunk_id": "C2", "text_sha256": hashlib.sha256(source_text({"chunk_id": "C2"}).encode()).hexdigest()},
        ]
        def row(identity):
            text_value = source_text(identity)
            validation = {
                "status": "accepted",
                "quarantined": [],
                "source_context": dict(identity),
                "text": text_value,
                "validation_marker": "mpkg-rag.validated-extraction.v1",
                "validation_fingerprint": "",
            }
            validation["validation_fingerprint"] = namespace["_validation_fingerprint"](validation)
            return {
                **identity,
                "validation": validation,
                "parse_status": "initial",
                "raw_output": "{}",
            }
        valid = [row(item) for item in expected]
        self.assertTrue(namespace["verify_extraction_cache"](valid, expected))
        for attack in [
            valid[:1],
            valid + [row({"document_uid": "D3", "chunk_id": "C3", "text_sha256": "T3"})],
            [valid[0], copy.deepcopy(valid[0])],
            [row(expected[1]), row(expected[0])],
            [{**valid[0], "text_sha256": "tampered"}, valid[1]],
            [{**valid[0], "validation": []}, valid[1]],
            [{key: value for key, value in valid[0].items() if key != "raw_output"}, valid[1]],
            [{**valid[0], "validation": {**valid[0]["validation"], "source_context": None}}, valid[1]],
            [{**valid[0], "validation": {**valid[0]["validation"], "source_context": {**expected[0], "chunk_id": "C2"}}}, valid[1]],
            [{**valid[0], "validation": {**valid[0]["validation"], "source_context": valid[1]["validation"]["source_context"]}}, valid[1]],
            [{**valid[0], "validation": {**valid[0]["validation"], "text": valid[1]["validation"]["text"]}}, valid[1]],
            [{**valid[0], "validation": {key: value for key, value in valid[0]["validation"].items() if key != "text"}}, valid[1]],
            [{**valid[0], "validation": {**valid[0]["validation"], "validation_marker": ""}}, valid[1]],
            [{**valid[0], "validation": {key: value for key, value in valid[0]["validation"].items() if key != "validation_fingerprint"}}, valid[1]],
            [{**valid[0], "validation": {**valid[0]["validation"], "validation_fingerprint": "tampered-nonempty"}}, valid[1]],
        ]:
            self.assertFalse(namespace["verify_extraction_cache"](attack, expected), attack)

    def test_extraction_cache_state_persists_before_gate_and_reuses_complete_invalid_cache(self):
        text = self.build_text()
        for expected in [
            "write_json_atomic",
            "cache_state",
            "complete_validation_failed",
            "complete_ready",
            "expected_extraction_universe",
            "parse_status_counts",
            "parse_rate_gate_failed",
            "cache_reuse_state",
        ]:
            self.assertIn(expected, text)
        self.assertLess(text.rindex("persist_extraction_identity_state(cache_state, parse_rate"), text.index("if cache_state !="))
        notebook = self._build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definitions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"cache_reuse_state", "extraction_identity_payload"}
        ]
        identity = {"corpus_manifest_hash": "C1", "expected_extraction_universe_hash": "U1"}
        query_identity = {"record_identities": [{"record_id": "R1"}], "schema_revision": "Q1"}
        expected_universe = [{"document_uid": "D1", "chunk_id": "C1", "text_sha256": "H1"}]
        namespace = {
            "identity": identity,
            "query_identity": query_identity,
            "EXPECTED_EXTRACTION_UNIVERSE": expected_universe,
            "EXPECTED_EXTRACTION_UNIVERSE_HASH": "U1",
            "mention_identity": {"stage": "mention-discovery"},
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "extraction-state-probes", "exec"), namespace)
        saved_failed = {
            "identity": identity,
            "query_identity": query_identity,
            "expected_extraction_universe": expected_universe,
            "expected_extraction_universe_hash": "U1",
            "cache_state": "complete_validation_failed",
        }
        self.assertEqual(
            namespace["cache_reuse_state"](saved_failed, identity, query_identity, expected_universe, True, True),
            (False, False),
        )
        self.assertEqual(
            namespace["cache_reuse_state"](saved_failed, identity, query_identity, expected_universe, False, True),
            (True, False),
        )
        tampered = {**saved_failed, "identity": {**identity, "corpus_manifest_hash": "tampered"}}
        self.assertEqual(
            namespace["cache_reuse_state"](tampered, identity, query_identity, expected_universe, True, True),
            (True, True),
        )
        payload = namespace["extraction_identity_payload"]("complete_validation_failed", 0.0, {"schema_invalid": 1})
        self.assertEqual(payload["cache_state"], "complete_validation_failed")
        self.assertEqual(payload["expected_extraction_universe"], expected_universe)
        self.assertEqual(payload["parse_rate"], 0.0)
        self.assertEqual(payload["parse_status_counts"], {"schema_invalid": 1})

    def test_extraction_writer_uses_canonical_hash_order(self):
        text = self.build_text()
        self.assertIn("canonical_extraction_chunks", text)
        self.assertIn("extraction_chunks = canonical_extraction_chunks", text)
        notebook = self._build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definitions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"canonical_extraction_chunks", "verify_extraction_cache"}
        ]
        def trusted_validation_fingerprint(validation):
            material = {key: value for key, value in validation.items() if key != "validation_fingerprint"}
            return hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        namespace = {"hashlib": hashlib, "json": json, "_validation_fingerprint": trusted_validation_fingerprint}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "canonical-extraction-order-probes", "exec"), namespace)
        chunks = [
            {"document_uid": "D2", "chunk_id": "C2", "text_sha256": hashlib.sha256(b"second").hexdigest(), "text": "second"},
            {"document_uid": "D1", "chunk_id": "C1", "text_sha256": hashlib.sha256(b"first").hexdigest(), "text": "first"},
        ]
        expected = sorted(
            [{key: row[key] for key in ("document_uid", "chunk_id", "text_sha256")} for row in chunks],
            key=lambda row: (row["document_uid"], row["chunk_id"], row["text_sha256"]),
        )
        ordered = namespace["canonical_extraction_chunks"](chunks, expected)
        self.assertEqual(
            [(row["document_uid"], row["chunk_id"], row["text_sha256"]) for row in ordered],
            [(row["document_uid"], row["chunk_id"], row["text_sha256"]) for row in expected],
        )
        self.assertEqual([row["text"] for row in ordered], ["first", "second"])
        fresh_rows = []
        for row in ordered:
            validation = {
                "status": "accepted",
                "quarantined": [],
                "source_context": {key: row[key] for key in ("document_uid", "chunk_id", "text_sha256")},
                "text": row["text"],
                "validation_marker": "mpkg-rag.validated-extraction.v1",
                "validation_fingerprint": "",
            }
            validation["validation_fingerprint"] = namespace["_validation_fingerprint"](validation)
            fresh_rows.append({
                **{key: row[key] for key in ("document_uid", "chunk_id", "text_sha256")},
                "validation": validation,
                "parse_status": "initial",
                "raw_output": "{}",
            })
        self.assertTrue(namespace["verify_extraction_cache"](fresh_rows, expected))

    def test_extraction_semantic_validation_consumes_at_most_one_repair(self):
        text = self.build_text()
        for expected in [
            "parse_extraction_with_one_repair",
            "semantic_repair",
            "validation reasons",
            "Exact source text",
            "Source context",
            "output_schema=SEMANTIC_EXTRACTION_SCHEMA",
        ]:
            self.assertIn(expected, text)
        notebook = self._build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_extraction_with_one_repair"]
        namespace = {
            "json": json,
            "SEMANTIC_EXTRACTION_SCHEMA": {"type": "object"},
            "ACCEPTED_PARSE_STATUSES": {"initial", "repair"},
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "extraction-repair-probes", "exec"), namespace)

        def run(initial_raw, repair_raw=None, initial_status="accepted", repair_status="accepted"):
            calls = []
            queue = [repair_raw]
            def fake_parse(raw):
                if raw == "STRUCTURAL_INVALID":
                    raise ValueError("invalid_json")
                return json.loads(raw)
            def fake_validate(payload, source_text, context):
                status = initial_status if payload.get("phase") == "bad" and not calls else repair_status if payload.get("phase") == "bad" else "accepted"
                return {"status": status, "quarantined": [{"reasons": ["bad_span"]}] if status == "quarantined" else [], "accepted": []}
            def fake_repair(raw, prompt, output_schema):
                calls.append(prompt)
                return queue.pop(0)
            namespace.update({"parse_json_object": fake_parse, "validate_extraction": fake_validate, "repair_json_output": fake_repair})
            return namespace["parse_extraction_with_one_repair"](initial_raw, "base prompt", "EXACT SOURCE", {"document_uid": "D1", "chunk_id": "C1"}), calls

        result, calls = run('{"phase":"good"}', '{"phase":"good"}')
        self.assertEqual(result[2], "initial")
        self.assertEqual(len(calls), 0)
        result, calls = run('{"phase":"bad"}', '{"phase":"good"}', initial_status="quarantined")
        self.assertEqual(result[2], "repair")
        self.assertEqual(len(calls), 1)
        self.assertIn("EXACT SOURCE", calls[0])
        self.assertIn("bad_span", calls[0])
        result, calls = run("STRUCTURAL_INVALID", '{"phase":"bad"}', repair_status="quarantined")
        self.assertEqual(result[2], "semantic_invalid")
        self.assertEqual(len(calls), 1)
        result, calls = run('{"phase":"bad"}', '{"phase":"bad"}', initial_status="quarantined", repair_status="quarantined")
        self.assertEqual(result[2], "semantic_invalid")
        self.assertEqual(len(calls), 1)
        self.assertTrue(result[3]["quarantined"])

    def test_extraction_invalid_fallback_retains_validation_binding(self):
        text = self.build_text()
        self.assertIn("validate_extraction(None, source_text, context)", text)
        notebook = self._build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_extraction_with_one_repair"]
        namespace = {"SEMANTIC_EXTRACTION_SCHEMA": {"type": "object"}}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "extraction-fallback-probes", "exec"), namespace)
        calls = []
        context = {"document_uid": "D1", "chunk_id": "C1", "text_sha256": "H1"}
        bound_validation = {
            "status": "quarantined",
            "quarantined": [{"reasons": ["invalid_json"]}],
            "source_context": context,
            "text": "exact source",
            "validation_marker": "mpkg-rag.validated-extraction.v1",
            "validation_fingerprint": "fp",
        }
        def fake_parse(raw):
            raise ValueError("invalid_json")
        def fake_validate(payload, source_text, source_context):
            calls.append((payload, source_text, source_context))
            return bound_validation
        def fake_repair(raw, prompt, output_schema):
            return "still invalid"
        namespace.update({
            "parse_json_object": fake_parse,
            "validate_extraction": fake_validate,
            "repair_json_output": fake_repair,
        })
        result = namespace["parse_extraction_with_one_repair"]("invalid", "prompt", "exact source", context)
        self.assertEqual(result[2], "schema_invalid")
        self.assertEqual(calls, [(None, "exact source", context)])
        self.assertEqual(result[3], bound_validation)

    def test_structured_generation_uses_lmfe_and_chat_template_compatibility(self):
        text = self.build_text()
        for expected in [
            "lm-format-enforcer",
            "JsonSchemaParser",
            "build_transformers_prefix_allowed_tokens_fn",
            "prefix_allowed_tokens_fn",
            'getattr(tokenizer, "tokenizer", tokenizer)',
            "base_tokenizer.decode",
            "except TypeError",
            "chat_template_compatibility_fallback",
            "output_schema",
            "output_schema=SEMANTIC_EXTRACTION_SCHEMA",
            "output_schema=QUERY_SIGNATURE_SCHEMA",
            "output_schema=PERSPECTIVE_SCHEMA",
            "output_schema=PLAN_SCHEMA",
            "output_schema=FINAL_RESPONSE_SCHEMA",
        ]:
            self.assertIn(expected, text)

    def test_query_signature_cache_identity_is_per_record_and_core_bound(self):
        text = self.build_text()
        for expected in [
            "record_identities",
            "record_id",
            "input_text_sha256",
            "target_sha256",
            "category_sha256",
            "core_source_sha256",
            "schema_revision",
            "verify_query_signature_cache",
            "cache_identity_mismatch",
        ]:
            self.assertIn(expected, text)

    def test_corpus_manifest_hash_covers_all_retrieval_affecting_reviewed_metadata(self):
        text = self.build_text()
        for expected in [
            "canonical_manifest_rows",
            "authority_score",
            "source_type",
            "factual_index_allowed",
            "status",
            "status_reason",
            "quarantine_reasons",
            "review_status",
            "document_uid",
            "relative_path",
            "content_sha256",
        ]:
            self.assertIn(expected, text)

    def test_paired_retrieval_evaluation_persists_shared_universe_and_metrics(self):
        text = self.build_text()
        for expected in [
            "shared_frozen_universe",
            "paired_retrieval_metrics",
            "overlap",
            "graph_only_gain",
            "authority_rate",
            "accepted_evidence_rate",
            "selected_score_mean",
            "paired_permutation",
            "paired_statistical_comparison",
            "frozen_evidence_ids",
        ]:
            self.assertIn(expected, text)
        self.assertGreaterEqual(text.count("paired_retrieval_metrics("), 2)

    def test_builder_regeneration_is_byte_identical(self):
        from build_remote_vm_qwen35_mpkg_rag import build_notebook

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.ipynb"
            second = Path(directory) / "second.ipynb"
            build_notebook(first)
            build_notebook(second)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_final_fix_has_closed_catalog_two_pass_linking_and_quality_gates(self):
        text = self.build_text()
        for expected in [
            "build_entity_catalog",
            "build_entity_candidates",
            "build_mention_prompt",
            "validate_mentions",
            "candidate_index",
            "resolve_query_signature_entities",
            "candidate_set_authentication_failed",
            "catalog_manifest_hash",
            "namespace_filter",
            "namespace_preference",
            "factual_only",
            "allow_target_fallback",
            "entity-candidates.v3",
            "graph_yield",
            "no_linked_entities",
            "accepted_linked_claims",
        ]:
            self.assertIn(expected, text)
        self.assertNotIn("Use canonical external entity IDs when supplied by context", text)

    def test_final_fix_has_per_source_html_audit_recovery_and_coverage_gate(self):
        text = self.build_text()
        for expected in [
            "extraction_audit",
            "extractor_revision",
            "role=main",
            "cookie_interstitial",
            "blocked",
            "missing_factual_documents",
            "SRC029",
            "SRC059",
            "insufficient_factual_corpus",
            "factual_availability_gate",
            "parent_document_uid",
        ]:
            self.assertIn(expected, text)

    def test_final_fix_hardens_split_checkpoint_quantization_and_real_few_shot_identity(self):
        text = self.build_text()
        for expected in [
            "target_sha256",
            "category_sha256",
            "reference_answer_sha256",
            "split_membership_hash",
            "duplicate_checkpoint_ids",
            "unknown_checkpoint_id",
            "revalidate_checkpoint_row",
            "checkpoint_materialization",
            "authenticated_checkpoint_evidence",
            "parse_status",
            "verify_effective_4bit",
            "effective_4bit_verification_failed",
            "FEW_SHOT_EXAMPLES",
            "few_shot_prompt_revision",
        ]:
            self.assertIn(expected, text)

    def test_effective_4bit_rejects_metadata_only_float32_fallback_and_allows_small_fp32_modules(self):
        notebook = self._build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "verify_effective_4bit")
        namespace = {"json": json, "CONFIG": {"load_in_4bit": True}}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), "quantization-probe", "exec"), namespace)

        class Parameter:
            dtype = "torch.float32"
            def __init__(self, size): self.size = size
            def numel(self): return self.size

        class Fallback:
            is_loaded_in_4bit = True
            config = type("Config", (), {"quantization_config": "4bit"})()
            def modules(self): return []
            def parameters(self): return [Parameter(100)]

        with self.assertRaisesRegex(RuntimeError, "effective_4bit_verification_failed"):
            namespace["verify_effective_4bit"](Fallback())

        class Params4bit(Parameter):
            pass
        class Linear4bit:
            pass
        class Quantized:
            is_loaded_in_4bit = True
            config = type("Config", (), {"quantization_config": "metadata-only"})()
            def modules(self): return [Linear4bit()]
            def parameters(self): return [Params4bit(90), Parameter(10)]

        self.assertTrue(namespace["verify_effective_4bit"](Quantized())["effective"])

    def test_mention_discovery_cache_is_ordered_text_bound_and_invalidates_downstream(self):
        text = self.build_text()
        for expected in [
            "MENTION_DISCOVERY_CACHE",
            "MENTION_DISCOVERY_IDENTITY",
            "verify_mention_cache",
            "expected_mention_universe_hash",
            "mention_cache_identity_mismatch",
            "mention_records",
        ]:
            self.assertIn(expected, text)
        notebook = self._build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definitions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"mention_discovery_identity", "verify_mention_cache", "cache_reuse_state"}
        ]
        namespace = {
            "hashlib": hashlib,
            "json": json,
            "MENTION_DISCOVERY_SCHEMA": {"version": "mention-discovery.v1"},
            "MENTION_DISCOVERY_PROMPT_REVISION": "mention-discovery.v1-exact-spans",
            "EXTRACTION_MODEL": "unsloth/Qwen3.5-4B",
            "CORPUS_MANIFEST_HASH": "C1",
            "CORE_SOURCE_SHA256": "CORE",
            "EXPECTED_EXTRACTION_UNIVERSE_HASH": "U1",
            "stable_id": lambda *parts: hashlib.sha256(json.dumps([str(part) for part in parts], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()[:20],
            "validate_mentions": lambda payload, value, context: {
                "status": "accepted",
                "quarantined": [],
            },
            "_validation_fingerprint": lambda value: "fingerprint",
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "mention-cache-probes", "exec"), namespace)
        expected = [
            {"document_uid": "D1", "chunk_id": "C1", "text_sha256": hashlib.sha256(b"first text").hexdigest(), "text": "first text"},
            {"document_uid": "D2", "chunk_id": "C2", "text_sha256": hashlib.sha256(b"second text").hexdigest(), "text": "second text"},
        ]
        identity = namespace["mention_discovery_identity"](expected)
        def row(item):
            return {
                **item,
                "mentions": [],
                "parse_status": "initial",
                "raw_output": "{}",
            }
        valid = [row(item) for item in expected]
        self.assertTrue(namespace["verify_mention_cache"](valid, expected, identity))
        self.assertFalse(namespace["verify_mention_cache"](valid[:1], expected, identity))
        self.assertFalse(namespace["verify_mention_cache"]([valid[1], valid[0]], expected, identity))
        self.assertFalse(namespace["verify_mention_cache"]([{**valid[0], "text": "tampered"}, valid[1]], expected, identity))
        self.assertEqual(
            namespace["cache_reuse_state"](
                {"identity": {"catalog_manifest_hash": "CAT"}, "query_identity": {"catalog_manifest_hash": "CAT"}, "expected_extraction_universe": expected, "expected_extraction_universe_hash": "U1"},
                {"catalog_manifest_hash": "CAT"},
                {"catalog_manifest_hash": "CAT"},
                expected,
                True,
                True,
                mention_cache_complete=True,
                mention_identity_matches=True,
            ),
            (False, False),
        )
        self.assertEqual(
            namespace["cache_reuse_state"](
                {"identity": {"catalog_manifest_hash": "CAT"}, "query_identity": {"catalog_manifest_hash": "CAT"}, "expected_extraction_universe": expected, "expected_extraction_universe_hash": "U1"},
                {"catalog_manifest_hash": "CAT"},
                {"catalog_manifest_hash": "CAT"},
                expected,
                True,
                True,
                mention_cache_complete=False,
                mention_identity_matches=False,
            ),
            (True, True),
        )

    def test_catalog_hash_is_frozen_before_downstream_identities_and_second_run_skips_qwen(self):
        text = self.build_text()
        self.assertIn("final_catalog_frozen", text)
        self.assertIn("mention_identity", text)
        self.assertIn("model_loaded", text)
        self.assertIn("need_mentions", text)
        self.assertNotIn("CATALOG_MANIFEST_HASH = ENTITY_CATALOG[\"catalog_hash\"]", text[text.index("identity = {\"corpus_manifest_hash\""):])

    def test_html_deduplicates_contained_parent_child_text_and_allows_hashed_fallbacks(self):
        text = self.build_text()
        for expected in [
            "deduplicate_contained_text_rows",
            "contained_text",
            "fallback_path",
            "fallback_sha256",
            "content_based_interstitial",
        ]:
            self.assertIn(expected, text)
        self.assertNotIn("KNOWN_BLOCKED_FACTUAL_SOURCES", text)
        notebook = self._build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definitions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"deduplicate_contained_text_rows", "_validated_fallback_path"}
        ]
        namespace = {"hashlib": hashlib, "Path": Path, "normalize_text": lambda value: " ".join(str(value or "").split())}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "html-probes", "exec"), namespace)
        parent = "A long factual paragraph with enough content to exercise containment-aware deduplication."
        child = "enough content to exercise containment-aware deduplication."
        self.assertEqual([row["text"] for row in namespace["deduplicate_contained_text_rows"]([{"text": parent}, {"text": child}])], [parent])
        self.assertEqual([row["text"] for row in namespace["deduplicate_contained_text_rows"]([{"text": child}, {"text": parent}])], [parent])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); primary = root / "blocked.html"; fallback = root / "fallback.html"; fallback.write_text("validated fallback", encoding="utf-8"); primary.write_text("interstitial", encoding="utf-8")
            resolved = namespace["_validated_fallback_path"]({"path": str(primary), "fallback_path": str(fallback), "fallback_sha256": hashlib.sha256(fallback.read_bytes()).hexdigest()})
            self.assertEqual(resolved[0], fallback)
            self.assertIsNone(namespace["_validated_fallback_path"]({"path": str(primary), "fallback_path": str(fallback), "fallback_sha256": "stale"}))

    def test_metrics_use_named_reference_and_exclude_missing_reference_rows(self):
        text = self.build_text()
        self.assertIn('row["Counter Narrative"]', text)
        self.assertIn("reference_available", text)
        self.assertIn("reference_metrics_excluded", text)
        self.assertNotIn("row._4", text)
        self.assertNotIn("str(row._4)", text)

    def test_prompt_preflight_passes_real_schema_tail_and_repair_uses_same_gate(self):
        notebook = self._build_notebook_json()
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        preflight = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "preflight_prompt_token_budget")
        calls = [node for node in ast.walk(preflight) if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "fit_prompt_to_budget"]
        self.assertTrue(calls)
        self.assertTrue(any(keyword.arg == "schema_tail" and isinstance(keyword.value, ast.Name) and keyword.value.id == "schema_tail" for call in calls for keyword in call.keywords))
        self.assertIn("schema_tail=schema_tail", source)
        self.assertIn("repair_json_output", source)

    def test_prompt_preflight_spy_sees_schema_tail_text(self):
        notebook = self._build_notebook_json()
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "preflight_prompt_token_budget"]
        calls = []
        class FakeBatch:
            def __getitem__(self, key): return {"attention_mask": [[1, 1]]}[key]
        class FakeTokenizer:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return FakeBatch()
        namespace = {"tokenizer": FakeTokenizer(), "CONFIG": {"max_seq_length": 16}, "fit_prompt_to_budget": lambda prompt, tokenizer, budget, **kwargs: calls.append({"prompt": prompt, **kwargs}) or prompt}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "preflight-spy", "exec"), namespace)
        namespace["preflight_prompt_token_budget"](["body"], 2, schema_tail="FINAL_SCHEMA_TAIL")
        assert any(item.get("schema_tail") == "FINAL_SCHEMA_TAIL" for item in calls)

    def test_cache_snapshots_and_filter_manifests_are_persisted_with_source_audit_events(self):
        text = self.build_text()
        for expected in [
            "cache_snapshot",
            "evictions",
            "cache_capacity",
            "dataset_filter_manifest.json",
            "corpus_filter_manifest.json",
            "html_short_element",
            "page_short_text",
            "audit_events",
            "fcntl.flock",
            "LOCK_EX",
        ]:
            self.assertIn(expected, text)

    def test_checkpoint_append_is_idempotent_for_identical_rows_and_fails_on_conflict(self):
        notebook = self._build_notebook_json()
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        wanted = {"load_jsonl", "checkpoint_lock", "append_checkpoint_row"}
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        namespace = {"Path": Path, "json": json, "fcntl": __import__("fcntl"), "os": __import__("os")}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "checkpoint-lock-probe", "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            row = {"ID": "1", "value": "same"}
            self.assertTrue(namespace["append_checkpoint_row"](path, row))
            self.assertFalse(namespace["append_checkpoint_row"](path, dict(row)))
            with self.assertRaisesRegex(RuntimeError, "duplicate_checkpoint_conflict"):
                namespace["append_checkpoint_row"](path, {"ID": "1", "value": "different"})

    def test_reference_metric_path_is_named_and_quarantines_missing_values(self):
        notebook = self._build_notebook_json()
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "reference_metric_row"]
        class FakeRouge:
            def score(self, reference, response):
                assert reference == "named reference"
                assert response == "generated"
                return {"rougeL": type("Score", (), {"fmeasure": 0.5})()}
        namespace = {"normalize_optional_text": lambda value: None if value is None or (isinstance(value, float) and value != value) else str(value).strip() or None, "ROUGE": FakeRouge()}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "metric-probe", "exec"), namespace)
        missing = namespace["reference_metric_row"]({"Counter Narrative": float("nan"), "parsed_counter_narrative": "generated", "reference_available": False})
        scored = namespace["reference_metric_row"]({"Counter Narrative": "named reference", "parsed_counter_narrative": "generated", "reference_available": True})
        self.assertIsNone(missing["rouge_l"])
        self.assertTrue(missing["reference_metrics_excluded"])
        self.assertEqual(scored["rouge_l"], 0.5)

    def test_cache_capacity_is_hashed_before_config_and_not_mutated_after_dataset_scope(self):
        text = self.build_text()
        self.assertIn("derive_effective_cache_capacity", text)
        self.assertIn("EFFECTIVE_CACHE_CAPACITY", text)
        self.assertLess(text.index("EFFECTIVE_CACHE_CAPACITY"), text.index("CONFIG ="))
        self.assertNotIn('CONFIG["cache_max_records"] = max(', text)

    def test_corpus_filter_fixture_reports_every_drop_with_source_identity_and_locator(self):
        notebook = self._build_notebook_json()
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        wanted = {"deduplicate_contained_text_rows"}
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        namespace = {"hashlib": hashlib, "normalize_text": lambda value: " ".join(str(value or "").split())}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "corpus-filter-probe", "exec"), namespace)
        rows = [
            {"document_uid": "D1", "relative_path": "doc.html", "paragraph": 1, "text": "A long factual paragraph with enough content to retain."},
            {"document_uid": "D1", "relative_path": "doc.html", "paragraph": 2, "text": "A long factual paragraph with enough content to retain."},
            {"document_uid": "D1", "relative_path": "doc.html", "paragraph": 3, "text": "A long factual paragraph with enough content to"},
        ]
        kept, audit = namespace["deduplicate_contained_text_rows"](rows, return_audit=True)
        assert len(kept) == 1
        assert {item["reason"] for item in audit} == {"duplicate_text", "contained_text"}
        assert all(item["document_uid"] == "D1" and item["relative_path"] == "doc.html" and item["locator"] for item in audit)

    def test_html_extraction_duplicate_candidate_is_counted_and_audited(self):
        notebook = self._build_notebook_json()
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        tree = ast.parse(source)
        wanted = {"normalize_text", "content_based_interstitial", "_validated_fallback_path", "_html_root", "deduplicate_contained_text_rows", "extract_pdf_document", "extract_html_document"}
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        from bs4 import BeautifulSoup
        namespace = {"Path": Path, "BeautifulSoup": BeautifulSoup, "hashlib": hashlib, "re": re, "fitz": __import__("fitz"), "COOKIE_MARKERS": ("enable cookies", "please enable cookies", "javascript is disabled", "checking your browser", "access denied", "captcha")}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "html-audit-probe", "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory) / "doc.html"
            paragraph = "A sufficiently long factual paragraph that remains in the extracted source."
            html.write_text(f"<html><body><p>{paragraph}</p><p>{paragraph}</p><p>short</p></body></html>", encoding="utf-8")
            rows, outcome = namespace["extract_html_document"]({"path": str(html), "relative_path": "doc.html", "document_uid": "D1", "content_sha256": "hash", "source_type": "official"})
        duplicate_events = [event for event in outcome["html_filter_events"] if event["reason"] == "duplicate_text"]
        assert duplicate_events and duplicate_events[0]["document_uid"] == "D1"
        assert outcome["html_candidate_count"] == outcome["html_filter_kept_count"] + len(outcome["html_filter_events"])
        assert outcome["html_candidate_count"] == 3

    def test_run_lock_and_checkpoint_reads_are_nonblocking_and_lock_protected(self):
        text = self.build_text()
        for expected in ["acquire_run_lock", "run_already_active", "release_run_lock", "RUN_LOCK_HANDLE", "LOCK_NB", "read_checkpoint_rows_locked", "checkpoint_lock"]:
            self.assertIn(expected, text)
        self.assertIn("source_registry_validation_errors", text)


if __name__ == "__main__":
    unittest.main()
