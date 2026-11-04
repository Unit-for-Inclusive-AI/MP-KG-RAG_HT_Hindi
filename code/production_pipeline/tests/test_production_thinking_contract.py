import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProductionThinkingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys

        sys.path.insert(0, str(ROOT / "work"))
        import build_remote_vm_qwen35_mpkg_rag as builder

        cls.output = Path(tempfile.mkdtemp()) / "production.ipynb"
        builder.build_notebook(cls.output)
        notebook = json.loads(cls.output.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

    def test_all_four_variants_are_generated_in_progressive_order(self):
        self.assertIn(
            '"generation_variants": ["qwen_zero_shot", "qwen_few_shot", "kg_rag", "mp_kg_rag"]',
            self.source,
        )
        self.assertIn('"MAX_EXPERIMENT_ROWS", "1550"', self.source)
        self.assertIn('"extraction_batch_size": 24', self.source)
        self.assertIn('CONFIG["extraction_batch_size"]', self.source)

    def test_user_facing_generation_enables_thinking_and_records_traces(self):
        self.assertIn('"thinking_enabled": True', self.source)
        self.assertIn('enable_thinking=CONFIG["thinking_enabled"]', self.source)
        self.assertIn('"reasoning_trace": qwen_generation_trace(final_raw)', self.source)
        self.assertIn('"reasoning_trace": reasoning_trace', self.source)
        self.assertIn('"reasoning_token_count"', self.source)
        self.assertIn('"reasoning_truncated"', self.source)

    def test_thinking_generation_does_not_apply_json_prefix_constraint(self):
        self.assertIn('"reasoning_max_new_tokens": 192', self.source)
        self.assertIn("def _generate_prompt_batch(", self.source)
        self.assertIn("reasoning_outputs = _generate_prompt_batch(", self.source)
        self.assertIn("enable_thinking=False,", self.source)
        self.assertIn("prefix_allowed_tokens_fn", self.source)
        self.assertIn('f"<think>\\n{reasoning.strip()}\\n</think>\\n\\n{final.strip()}"', self.source)

    def test_production_workbook_has_outputs_traces_evidence_and_manifest(self):
        for sheet in (
            "Outputs",
            "Zero-Shot Trace",
            "Few-Shot Trace",
            "KG-RAG Trace",
            "MP-KG-RAG Trace",
            "Evidence Ledger",
            "Run Manifest",
            "Quality Summary",
        ):
            self.assertIn(f'\"{sheet}\"', self.source)
        self.assertIn("dataset_with_all_rag_counter_narratives.xlsx", self.source)


if __name__ == "__main__":
    unittest.main()
