import ast
import errno
import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
import unicodedata
import unittest
from pathlib import Path


def canonical_test_hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_notebook_functions(notebook, wanted, namespace_overrides=None):
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    source = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))
    )
    tree = ast.parse(source)
    wanted = set(wanted)
    dependencies = {
        "validate_generated_row": {
            "normalize_for_dedup", "_language_matches", "_harmful_text_checks",
            "_counter_narrative_checks"
        },
        "_harmful_text_checks": {"_harmful_text_matches"},
        "_language_matches": {"_detected_language_code"},
        "is_near_duplicate": {"find_near_duplicate_pairs"},
        "batch_near_duplicate_reasons": {"find_near_duplicate_pairs"},
        "run_generation": {
            "batch_near_duplicate_reasons", "validate_checkpoint_manifest_structure"
        },
        "reconcile_accepted_rows": {
            "validate_generated_row", "find_near_duplicate_pairs"
        },
        "atomic_write_json": {"fsync_directory"},
        "append_jsonl": {"fsync_directory"},
        "_quarantine_jsonl_tail": {"fsync_directory"},
        "load_jsonl": {"_quarantine_jsonl_tail"},
        "publish_data_artifacts": {
            "atomic_write_json", "fsync_file", "fsync_directory"
        },
    }
    while True:
        expanded = wanted | {
            dependency
            for name in wanted
            for dependency in dependencies.get(name, ())
        }
        if expanded == wanted:
            break
        wanted = expanded
    extracted_constant_names = {
        "GENERATED_RECORD_KEYS",
        "TARGETS_BY_CATEGORY",
        "REFUSAL_MARKERS",
        "PLACEHOLDER_PROMPT_LEAKAGE_MARKERS",
        "OPERATIONAL_ATTACK_MARKERS",
        "CONTACT_PATTERN",
        "ADDRESS_PATTERN",
        "PROPER_NAME_PATTERN",
        "HINGLISH_MARKERS",
        "ENGLISH_MARKERS",
        "HINDI_MARKERS",
        "COUNTER_ENDORSEMENT_MARKERS",
        "COUNTER_OPPOSITION_PATTERNS",
        "COUNTER_SUPPORT_PATTERNS",
        "COUNTER_ENDORSEMENT_PATTERNS",
        "HARMFUL_TEXT_PATTERNS",
        "ANTI_HATE_TEXT_PATTERNS",
        "SEVERE_HOSTILITY_PATTERNS",
        "CATEGORY_TEXT_PATTERNS",
        "ABUSE_TYPE_TEXT_PATTERNS",
        "IDENTITY_PACKAGE_NAMES",
    }

    def is_extracted_constant(node):
        return (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in extracted_constant_names
        )

    definitions = [
        node
        for node in tree.body
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in wanted
        )
        or is_extracted_constant(node)
    ]
    def test_detect(value):
        normalized = str(value).casefold()
        if "मराठी" in normalized:
            return "mr"
        if re.search(r"[\u0900-\u097f]", normalized):
            return "hi"
        if any(marker in normalized for marker in ("bonjour", "monde")):
            return "fr"
        if any(marker in normalized for marker in ("odio", "personas", "voy", "tolerar", "esto")):
            return "es"
        return "en"

    namespace = {
        "hashlib": hashlib,
        "errno": errno,
        "json": json,
        "math": math,
        "pd": __import__("pandas"),
        "os": os,
        "Path": Path,
        "tempfile": tempfile,
        "time": time,
        "random": random,
        "re": re,
        "unicodedata": unicodedata,
        "detect": test_detect,
        "EXPORT_COLUMNS": ["ID", "Text", "Category", "Target", "Counter Narrative"],
        "PRIMARY_MODEL_ID": "lukey03/Qwen3.5-9B-abliterated",
        "FALLBACK_MODEL_ID": "wangzhang/Qwen3.5-4B-abliterated",
        "CATEGORIES": (
            "Gay Men",
            "Lesbian Women",
            "Bisexual People",
            "Transgender People",
            "Non-binary/Gender-nonconforming People",
        ),
        "SUPPORTED_TOTALS": {1500, 2000},
        "SMOKE_TOTAL": 15,
        "LANGUAGES": ("English", "Hindi", "Hinglish"),
        "PLATFORM_STYLES": (
            "X/Twitter reply",
            "Instagram comment",
            "Instagram meme caption",
            "YouTube reply",
            "Public chat/forum message",
        ),
        "ABUSE_TYPES": (
            "coded mockery",
            "identity shaming",
            "stereotype/inferiority",
            "misgendering",
            "profanity/slur",
            "sexualized degradation",
            "dehumanization",
            "exclusion/rights denial",
            "threat/intimidation",
        ),
        "SEVERITIES": ("coded", "explicit", "severe", "extreme"),
        "SYSTEM_PROMPT": "synthetic research dataset system prompt",
        "validate_generated_row": lambda row, scheduled: [],
        "normalize_for_dedup": lambda value: re.sub(r"\s+", " ", str(value).casefold()).strip(),
    }
    namespace.update(namespace_overrides or {})
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "notebook-functions", "exec"), namespace)
    namespace.update(namespace_overrides or {})
    return namespace


class KaggleHateSpeechNotebookTests(unittest.TestCase):
    @staticmethod
    def scheduled_row(request_id="REQ000001"):
        return {
            "request_id": request_id,
            "category": "Transgender People",
            "target": "a fictional transgender creator",
            "language": "English",
            "platform_style": "Instagram comment",
            "abuse_type": "identity shaming",
            "severity": "severe",
        }

    @staticmethod
    def accepted_row(request_id="REQ000001"):
        return {
            "ID": request_id,
            "Text": (
                "I hate these fictional transgender people; they are disgusting "
                "freaks and a disgrace who should go away."
            ),
            "Category": "Transgender People",
            "Target": "a fictional transgender creator",
            "Counter Narrative": "Identity-based abuse is wrong and every person deserves equal respect.",
            "language": "English",
            "platform_style": "Instagram comment",
            "abuse_type": "identity shaming",
            "severity": "severe",
            "request_id": request_id,
        }

    def build_notebook_json(self):
        from work.build_kaggle_qwen35_lgbtq_hatespeech_dataset import build_notebook

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "notebook.ipynb"
            build_notebook(output)
            return json.loads(output.read_text(encoding="utf-8"))

    def build_text(self):
        notebook = self.build_notebook_json()
        return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    def test_notebook_has_kaggle_metadata_and_model_choices(self):
        notebook = self.build_notebook_json()
        text = self.build_text()
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["metadata"]["kaggle"]["accelerator"], "gpu")
        self.assertIn("lukey03/Qwen3.5-9B-abliterated", text)
        self.assertIn("wangzhang/Qwen3.5-4B-abliterated", text)

    def test_notebook_freezes_categories_totals_and_export_schema(self):
        text = self.build_text()
        for value in [
            "Gay Men",
            "Lesbian Women",
            "Bisexual People",
            "Transgender People",
            "Non-binary/Gender-nonconforming People",
            "SUPPORTED_TOTALS = {1500, 2000}",
            "SMOKE_TOTAL = 15",
            "EXPORT_COLUMNS = [\"ID\", \"Text\", \"Category\", \"Target\", \"Counter Narrative\"]",
        ]:
            self.assertIn(value, text)

    def test_quota_schedule_is_balanced_and_deterministic(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook, {"_language_counts", "build_quota_schedule"}
        )
        first = functions["build_quota_schedule"](2000, 3407)
        second = functions["build_quota_schedule"](2000, 3407)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2000)
        for category in functions["CATEGORIES"]:
            rows = [row for row in first if row["category"] == category]
            self.assertEqual(len(rows), 400)
            language_counts = {
                language: sum(row["language"] == language for row in rows)
                for language in functions["LANGUAGES"]
            }
            self.assertEqual(language_counts, {"English": 134, "Hindi": 133, "Hinglish": 133})
        self.assertEqual(len({row["request_id"] for row in first}), 2000)

    def test_quota_schedule_supports_1500_and_smoke_but_rejects_other_totals(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook, {"_language_counts", "build_quota_schedule"}
        )
        schedule = functions["build_quota_schedule"](1500, 9)
        self.assertEqual(len(schedule), 1500)
        for category in functions["CATEGORIES"]:
            rows = [row for row in schedule if row["category"] == category]
            self.assertEqual(len(rows), 300)
            self.assertEqual(
                {
                    language: sum(row["language"] == language for row in rows)
                    for language in functions["LANGUAGES"]
                },
                {"English": 100, "Hindi": 100, "Hinglish": 100},
            )
        self.assertEqual(len(functions["build_quota_schedule"](15, 9)), 15)
        with self.assertRaisesRegex(ValueError, "unsupported_total_rows"):
            functions["build_quota_schedule"](1700, 9)

    def test_row_validator_rejects_identity_mismatch_refusal_and_private_markers(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {"normalize_for_dedup", "_language_matches", "validate_generated_row"},
        )
        scheduled = {
            "request_id": "REQ000001",
            "category": "Transgender People",
            "target": "a fictional creator",
            "language": "Hinglish",
            "platform_style": "Instagram comment",
            "abuse_type": "identity shaming",
            "severity": "severe",
        }
        row = {
            "ID": "REQ000001",
            "Text": "Main yeh generate nahi kar sakta, contact @realperson at test@example.com",
            "Category": "Gay Men",
            "Target": "a fictional creator",
            "Counter Narrative": "I cannot assist with that request.",
            "language": "Hinglish",
            "platform_style": "Instagram comment",
            "abuse_type": "identity shaming",
            "severity": "severe",
            "request_id": "REQ000001",
        }
        reasons = functions["validate_generated_row"](row, scheduled)
        self.assertIn("category_mismatch", reasons)
        self.assertIn("refusal_or_disclaimer", reasons)
        self.assertIn("private_or_contact_marker", reasons)

    def test_row_validator_rejects_private_marker_in_target(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {"normalize_for_dedup", "_language_matches", "validate_generated_row"},
        )
        scheduled = {
            "request_id": "REQ000001",
            "category": "Transgender People",
            "target": "contact @realperson for details",
            "language": "English",
            "platform_style": "Instagram comment",
            "abuse_type": "identity shaming",
            "severity": "severe",
        }
        row = {
            "ID": "REQ000001",
            "Text": "This is a fictional harmful comment for validation.",
            "Category": "Transgender People",
            "Target": "contact @realperson for details",
            "Counter Narrative": "Respectful responses should reject identity-based abuse.",
            "language": "English",
            "platform_style": "Instagram comment",
            "abuse_type": "identity shaming",
            "severity": "severe",
            "request_id": "REQ000001",
        }
        reasons = functions["validate_generated_row"](row, scheduled)
        self.assertIn("private_or_contact_marker", reasons)

    def test_row_validator_rejects_parenthesized_phone_number(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {"normalize_for_dedup", "_language_matches", "validate_generated_row"},
        )
        row = dict(
            self.accepted_row(),
            Text="Call (987) 654-3210 to harass this fictional target.",
        )
        self.assertIn(
            "private_or_contact_marker",
            functions["validate_generated_row"](row, self.scheduled_row()),
        )

    def test_row_validator_rejects_case_insensitive_urls(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "normalize_for_dedup",
                "_language_matches",
                "_counter_narrative_checks",
                "validate_generated_row",
            },
        )
        row = dict(
            self.accepted_row(),
            Text="Harass this fictional target at HTTPS://EXAMPLE.COM/path now.",
        )
        self.assertIn(
            "private_or_contact_marker",
            functions["validate_generated_row"](row, self.scheduled_row()),
        )

    def test_row_validator_rejects_addresses_and_unscheduled_proper_names(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "normalize_for_dedup",
                "_language_matches",
                "_harmful_text_checks",
                "_counter_narrative_checks",
                "validate_generated_row",
            },
        )
        cases = [
            (
                dict(
                    self.accepted_row(),
                    Text="I will hurt them at 12 Main Street tonight.",
                ),
                "private_or_contact_marker",
            ),
            (
                dict(
                    self.accepted_row(),
                    Text="They are disgusting freaks, and Taylor Swift should go away.",
                ),
                "real_or_unscheduled_name_marker",
            ),
            (
                dict(
                    self.accepted_row(),
                    **{
                        "Counter Narrative": (
                            "Taylor Swift says this abuse is wrong and every person "
                            "deserves equal respect."
                        )
                    },
                ),
                "real_or_unscheduled_name_marker",
            ),
        ]
        for row, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason, row=row):
                self.assertIn(
                    expected_reason,
                    functions["validate_generated_row"](
                        row, self.scheduled_row()
                    ),
                )

    def test_row_validator_requires_exact_record_keys_and_projection(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "normalize_for_dedup",
                "_language_matches",
                "_counter_narrative_checks",
                "validate_generated_row",
                "project_generated_record",
            },
        )
        row = dict(self.accepted_row(), reasoning="hidden analysis")
        reasons = functions["validate_generated_row"](row, self.scheduled_row())
        self.assertIn("record_keys_mismatch", reasons)
        projected = functions["project_generated_record"](row)
        self.assertEqual(tuple(projected), functions["GENERATED_RECORD_KEYS"])
        self.assertNotIn("reasoning", projected)

    def test_quota_schedule_assigns_deterministic_varied_targets_from_closed_allowlist(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook, {"_language_counts", "build_quota_schedule"}
        )
        first = functions["build_quota_schedule"](1500, 3407)
        second = functions["build_quota_schedule"](1500, 3407)
        self.assertEqual(first, second)
        for category in functions["CATEGORIES"]:
            targets = {row["target"] for row in first if row["category"] == category}
            self.assertGreater(len(targets), 1)
            self.assertLessEqual(targets, set(functions["TARGETS_BY_CATEGORY"][category]))

    def test_row_validator_requires_candidate_target_to_match_schedule(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "normalize_for_dedup",
                "_language_matches",
                "_counter_narrative_checks",
                "validate_generated_row",
            },
        )
        row = dict(self.accepted_row(), Target="a fictional transgender student")
        reasons = functions["validate_generated_row"](row, self.scheduled_row())
        self.assertIn("target_mismatch", reasons)

    def test_row_validator_rejects_neutral_text_for_an_abuse_quota(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "normalize_for_dedup",
                "_language_matches",
                "_harmful_text_matches",
                "_counter_narrative_checks",
                "validate_generated_row",
            },
        )
        for text in (
            "This community member deserves equal respect in public.",
            "Hate is wrong and every person deserves equal respect.",
            "This harmful post targets them.",
        ):
            with self.subTest(text=text):
                row = dict(self.accepted_row(), Text=text)
                self.assertIn(
                    "text_not_obviously_harmful",
                    functions["validate_generated_row"](row, self.scheduled_row()),
                )

    def test_harmful_text_heuristic_accepts_examples_in_each_language(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook, {"normalize_for_dedup", "_harmful_text_matches"}
        )
        cases = {
            "English": "I hate this fictional group and want them gone.",
            "Hindi": "मुझे इन काल्पनिक लोगों से नफरत है और इन्हें निकालो।",
            "Hinglish": "Yeh log ghatiya hain, inko yahan se nikalo.",
        }
        for language, text in cases.items():
            with self.subTest(language=language):
                self.assertTrue(functions["_harmful_text_matches"](text, language))
        self.assertFalse(
            functions["_harmful_text_matches"](
                "Hate is wrong and everyone deserves equal respect.", "English"
            )
        )
        for non_harmful in (
            "We hate prejudice and support equal rights.",
            "I hate broccoli.",
            "They should not be hurt or attacked.",
            "An urban community deserves equal respect.",
        ):
            with self.subTest(non_harmful=non_harmful):
                self.assertFalse(
                    functions["_harmful_text_matches"](non_harmful, "English")
                )
        multilingual_non_harmful = {
            "Hindi": "मैं नफरत का विरोध करता हूँ और सभी का सम्मान करता हूँ।",
            "Hinglish": "Main nafrat ka virodh karta hoon aur sabko izzat deta hoon.",
        }
        for language, text in multilingual_non_harmful.items():
            with self.subTest(language=language, text=text):
                self.assertFalse(functions["_harmful_text_matches"](text, language))

    def test_severe_and_extreme_rows_require_strong_hostility_evidence(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "normalize_for_dedup",
                "_language_matches",
                "_harmful_text_matches",
                "_harmful_text_checks",
                "_counter_narrative_checks",
                "validate_generated_row",
            },
        )
        mild = dict(
            self.accepted_row(),
            Text="I hate these fictional people and want them to leave.",
        )
        severe_reasons = functions["validate_generated_row"](
            mild, self.scheduled_row()
        )
        self.assertNotIn("text_not_obviously_harmful", severe_reasons)
        self.assertIn("text_severity_evidence_missing", severe_reasons)
        coded = dict(self.scheduled_row(), severity="coded")
        coded_row = dict(mild, severity="coded")
        self.assertNotIn(
            "text_severity_evidence_missing",
            functions["validate_generated_row"](coded_row, coded),
        )
        boundary_cases = {
            "English": "I hate these fictional people for their skills.",
            "Hindi": "मैं इन काल्पनिक लोगों से नफरत करता हूँ; हमारा समाज इसे देखता है।",
            "Hinglish": "Main in fictional logon se nafrat karta hoon; humaara samaj dekhta hai.",
        }
        for language, text in boundary_cases.items():
            with self.subTest(language=language, text=text):
                reasons = functions["_harmful_text_checks"](
                    text,
                    language,
                    "extreme",
                    "Transgender People",
                    "identity shaming",
                )
                self.assertNotIn("text_not_obviously_harmful", reasons)
                self.assertIn("text_severity_evidence_missing", reasons)

    def test_harmful_text_checks_ground_category_and_abuse_type(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "normalize_for_dedup",
                "_harmful_text_matches",
                "_harmful_text_checks",
            },
        )
        checks = functions["_harmful_text_checks"]
        generic = checks(
            "I hate those people; they are disgusting freaks and a disgrace.",
            "English",
            "severe",
            "Transgender People",
            "identity shaming",
        )
        self.assertIn("text_category_evidence_missing", generic)
        self.assertNotIn("text_abuse_type_evidence_missing", generic)

        wrong_type = checks(
            "I will hurt these transgender people and attack them tonight.",
            "English",
            "severe",
            "Transgender People",
            "misgendering",
        )
        self.assertNotIn("text_category_evidence_missing", wrong_type)
        self.assertIn("text_abuse_type_evidence_missing", wrong_type)

        grounded = checks(
            "These transgender people are disgusting; that trans woman is not a real woman.",
            "English",
            "severe",
            "Transgender People",
            "misgendering",
        )
        self.assertNotIn("text_category_evidence_missing", grounded)
        self.assertNotIn("text_abuse_type_evidence_missing", grounded)

    def test_language_heuristic_handles_short_english_hindi_and_hinglish_posts(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(notebook, {"_language_matches"})
        matches = functions["_language_matches"]
        self.assertTrue(matches("This hate is wrong.", "English"))
        self.assertTrue(matches("Go away now.", "English"))
        self.assertTrue(matches("यह गलत है।", "Hindi"))
        self.assertTrue(matches("गलत है।", "Hindi"))
        self.assertTrue(matches("Yeh galat hai.", "Hinglish"))
        self.assertFalse(matches("Yeh galat hai.", "English"))
        self.assertFalse(matches("This hate is wrong.", "Hinglish"))
        self.assertFalse(matches("bonjour tout monde", "Hindi"))
        self.assertFalse(matches("bonjour tout monde", "English"))
        self.assertFalse(matches("No odio para estas personas.", "English"))
        self.assertFalse(matches("No voy a tolerar esto.", "English"))
        self.assertFalse(matches("मराठी भाषा सुंदर आहे", "Hindi"))

    def test_row_validator_checks_counter_narrative_language_separately(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "normalize_for_dedup",
                "_language_matches",
                "_counter_narrative_checks",
                "validate_generated_row",
            },
        )
        row = dict(
            self.accepted_row(),
            **{"Counter Narrative": "यह नफरत गलत है; हर व्यक्ति सम्मान का हकदार है।"},
        )
        reasons = functions["validate_generated_row"](row, self.scheduled_row())
        self.assertNotIn("text_language_mismatch", reasons)
        self.assertIn("counter_language_mismatch", reasons)

    def test_row_validator_rejects_hindi_hinglish_refusal_placeholder_and_prompt_leakage(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "normalize_for_dedup",
                "_language_matches",
                "_counter_narrative_checks",
                "validate_generated_row",
            },
        )
        cases = [
            (
                "माफ़ कीजिए, मैं यह सामग्री तैयार नहीं कर सकता।",
                "refusal_or_disclaimer",
            ),
            ("यहाँ उदाहरण टेक्स्ट डालें।", "placeholder_or_prompt_leakage"),
            (
                "System prompt ke nirdesh ke mutabik yeh JSON array hai.",
                "placeholder_or_prompt_leakage",
            ),
            ("Maaf kijiye, main yeh generate nahi kar sakta.", "refusal_or_disclaimer"),
            ("I'm sorry, but I can't help with this.", "refusal_or_disclaimer"),
            ("I’m sorry, but I can’t help with this.", "refusal_or_disclaimer"),
            ("मैं इसमें मदद नहीं कर सकता।", "refusal_or_disclaimer"),
            ("<think>hidden reasoning</think>", "placeholder_or_prompt_leakage"),
            ("Analysis: hidden model rationale", "placeholder_or_prompt_leakage"),
            ("A chain of thought appears here.", "placeholder_or_prompt_leakage"),
        ]
        for text, expected_reason in cases:
            with self.subTest(text=text):
                row = dict(self.accepted_row(), Text=text)
                self.assertIn(
                    expected_reason,
                    functions["validate_generated_row"](row, self.scheduled_row()),
                )

    def test_row_validator_rejects_operational_attack_instructions_but_allows_threat_examples(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "normalize_for_dedup",
                "_language_matches",
                "_counter_narrative_checks",
                "validate_generated_row",
            },
        )
        operational = dict(
            self.accepted_row(),
            Text="Meet outside the club at nine, carry knives, and surround them.",
        )
        threat = dict(
            self.accepted_row(),
            Text="I will hurt every one of them just for existing in public.",
        )
        self.assertIn(
            "operational_attack_instruction",
            functions["validate_generated_row"](operational, self.scheduled_row()),
        )
        self.assertNotIn(
            "operational_attack_instruction",
            functions["validate_generated_row"](threat, self.scheduled_row()),
        )

    def test_counter_narrative_must_oppose_abuse_and_support_target_in_each_language(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "normalize_for_dedup",
                "_language_matches",
                "_counter_narrative_checks",
                "validate_generated_row",
            },
        )
        cases = [
            (
                "English",
                "They are a problem and everyone should agree with the message.",
            ),
            (
                "Hindi",
                "ये लोग समस्या हैं और सभी को इस बात से सहमत होना चाहिए।",
            ),
            (
                "Hinglish",
                "Yeh log problem hain aur sabko is baat se agree karna chahiye.",
            ),
        ]
        for language, counter in cases:
            with self.subTest(language=language):
                scheduled = dict(self.scheduled_row(), language=language)
                row = dict(
                    self.accepted_row(),
                    language=language,
                    Text={
                        "English": "This hateful post attacks a protected group.",
                        "Hindi": "यह नफरत भरी पोस्ट एक संरक्षित समुदाय पर हमला करती है।",
                        "Hinglish": "Yeh hateful post protected logon ko target karti hai.",
                    }[language],
                    **{"Counter Narrative": counter},
                )
                reasons = functions["validate_generated_row"](row, scheduled)
                self.assertIn("counter_missing_opposition", reasons)
                self.assertIn("counter_missing_support", reasons)

    def test_counter_narrative_rejects_polarity_reversed_marker_stuffing(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {"normalize_for_dedup", "_counter_narrative_checks"},
        )
        cases = [
            ("English", "I hate these people; they deserve abuse and do not belong."),
            ("English", "Reject their equal rights."),
            ("English", "Stop supporting these people."),
            ("English", "Hate is wrong; disrespecting them is good."),
            ("Hindi", "मैं इन लोगों से नफरत करता हूँ; वे दुर्व्यवहार के हकदार हैं और सम्मान नहीं चाहिए।"),
            ("Hinglish", "Main nafrat karta hoon; yeh log abuse deserve karte hain aur respect nahi chahiye."),
        ]
        for language, counter in cases:
            with self.subTest(language=language, counter=counter):
                reasons = functions["_counter_narrative_checks"](counter, language)
                self.assertIn("counter_endorses_abuse", reasons)

    def test_counter_narrative_accepts_clear_opposition_and_support_in_each_language(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {"normalize_for_dedup", "_counter_narrative_checks"},
        )
        cases = {
            "English": "This hate is wrong; every person deserves equal dignity and respect.",
            "Hindi": "यह नफरत गलत है; हर व्यक्ति समान सम्मान और अधिकार का हकदार है।",
            "Hinglish": "Yeh nafrat galat hai; sabko barabar izzat aur haq milna chahiye.",
        }
        for language, counter in cases.items():
            with self.subTest(language=language):
                self.assertEqual(
                    functions["_counter_narrative_checks"](counter, language), []
                )

    def test_normalization_and_near_duplicate_detection(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook, {"normalize_for_dedup", "is_near_duplicate"}
        )
        normalized = functions["normalize_for_dedup"]("  SAME\u00a0Text!! ")
        self.assertEqual(normalized, "same text!!")
        self.assertTrue(
            functions["is_near_duplicate"](
                "This social media comment repeats almost exactly",
                ["This social-media comment repeats almost exactly"],
                0.80,
            )
        )

    def test_batched_near_duplicate_scan_preserves_threshold_and_scales_modestly(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook, {"normalize_for_dedup", "find_near_duplicate_pairs"}
        )
        pairs = functions["find_near_duplicate_pairs"](
            [
                "This social media comment repeats almost exactly",
                "This social-media comment repeats almost exactly",
                "A wholly unrelated sentence with different vocabulary",
            ],
            0.80,
        )
        self.assertTrue(any(first == 0 and second == 1 for first, second, _ in pairs))

        values = [
            f"synthetic-row-{index}-{hashlib.sha256(str(index).encode()).hexdigest()}"
            for index in range(1000)
        ]
        started = time.perf_counter()
        functions["find_near_duplicate_pairs"](values, 0.999999, chunk_size=128)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 8.0)

    def test_batch_near_duplicate_reasons_checks_existing_and_same_batch_rows(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "normalize_for_dedup",
                "find_near_duplicate_pairs",
                "batch_near_duplicate_reasons",
            },
        )
        candidates = [
            dict(
                self.accepted_row("REQ000002"),
                Text=self.accepted_row()["Text"] + " Again.",
                **{"Counter Narrative": "A unique counter says hate is wrong and supports equal dignity."},
            ),
            dict(
                self.accepted_row("REQ000003"),
                Text="A distinct harmful post with entirely separate wording for this sample.",
                **{"Counter Narrative": "Another response rejects hate and gives equal respect to people."},
            ),
            dict(
                self.accepted_row("REQ000004"),
                Text="A distinct harmful post with entirely separate wording for this sample. Again.",
                **{"Counter Narrative": "A third response rejects abuse and supports dignity and rights."},
            ),
        ]
        reasons = functions["batch_near_duplicate_reasons"](
            candidates,
            [self.accepted_row()["Text"]],
            [self.accepted_row()["Counter Narrative"]],
            0.80,
        )
        self.assertIn("near_duplicate_text", reasons[0])
        self.assertIn("near_duplicate_text", reasons[2])

    def test_batch_duplicate_reasons_rejects_normalized_exact_matches_independently(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "normalize_for_dedup",
                "find_near_duplicate_pairs",
                "batch_near_duplicate_reasons",
            },
        )
        first = dict(
            self.accepted_row("REQ000002"),
            Text="  " + self.accepted_row()["Text"].upper() + "  ",
        )
        second = dict(
            self.accepted_row("REQ000003"),
            Text="An unrelated harmful sample with distinct wording for exact checks.",
            **{"Counter Narrative": "A UNIQUE response says hate is wrong and supports dignity."},
        )
        third = dict(
            self.accepted_row("REQ000004"),
            Text="A third unrelated harmful sample with other wording for exact checks.",
            **{"Counter Narrative": "a unique   response says hate is wrong and supports dignity."},
        )
        reasons = functions["batch_near_duplicate_reasons"](
            [first, second, third],
            [self.accepted_row()["Text"]],
            [self.accepted_row()["Counter Narrative"]],
            1.0,
        )
        self.assertIn("exact_duplicate_text", reasons[0])
        self.assertIn("exact_duplicate_counter", reasons[2])

    def test_batch_duplicate_scan_starts_at_candidate_suffix(self):
        notebook = self.build_notebook_json()
        scan_starts = []

        def recording_scan(values, threshold, chunk_size=256, start_index=1):
            scan_starts.append((len(values), start_index))
            return []

        functions = extract_notebook_functions(
            notebook,
            {"normalize_for_dedup", "batch_near_duplicate_reasons"},
            {"find_near_duplicate_pairs": recording_scan},
        )
        candidates = [self.accepted_row("REQ000011"), self.accepted_row("REQ000012")]
        functions["batch_near_duplicate_reasons"](
            candidates,
            [f"accepted text {index}" for index in range(10)],
            [f"accepted counter {index}" for index in range(10)],
            0.88,
        )
        self.assertEqual(scan_starts, [(12, 10), (12, 10)])

    def test_rejected_batch_candidate_does_not_block_later_candidate(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "normalize_for_dedup",
                "find_near_duplicate_pairs",
                "batch_near_duplicate_reasons",
            },
        )
        first = dict(
            self.accepted_row("REQ000002"),
            Text="A distinctive harmful post about a fictional target appears here.",
        )
        second = dict(
            self.accepted_row("REQ000003"),
            Text="A distinctive harmful post about a fictional target appears here. Again.",
            **{
                "Counter Narrative": (
                    "This abuse is wrong and every person deserves equal dignity."
                )
            },
        )
        reasons = functions["batch_near_duplicate_reasons"](
            [first, second],
            ["An unrelated accepted harmful text."],
            [self.accepted_row()["Counter Narrative"]],
            0.80,
        )
        self.assertIn("exact_duplicate_counter", reasons[0])
        self.assertNotIn("near_duplicate_text", reasons[1])

    def test_reconciliation_and_generation_use_batched_near_duplicate_scans(self):
        text = self.build_text()
        reconcile_source = text[
            text.index("def reconcile_accepted_rows("):text.index("def reconstruct_retry_counts(")
        ]
        generation_source = text[
            text.index("def run_generation("):text.index("ACCEPTED_ROWS = run_generation")
        ]
        self.assertIn("find_near_duplicate_pairs(", reconcile_source)
        self.assertNotIn("is_near_duplicate(", reconcile_source)
        self.assertIn("batch_near_duplicate_reasons(", generation_source)
        self.assertNotIn("is_near_duplicate(", generation_source)

    def test_json_parser_extracts_array_and_rejects_non_array(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(notebook, {"extract_first_json_array"})
        parsed = functions["extract_first_json_array"]('prefix [{"request_id":"REQ000001"}] suffix')
        self.assertEqual(parsed, [{"request_id": "REQ000001"}])
        with self.assertRaisesRegex(ValueError, "json_array_not_found"):
            functions["extract_first_json_array"]('{"request_id":"REQ000001"}')
        with self.assertRaisesRegex(ValueError, "json_array_contains_non_object"):
            functions["extract_first_json_array"]('[{"request_id":"REQ000001"}, 7]')

    def test_json_parser_skips_empty_unsuitable_and_wrong_assignment_arrays(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(notebook, {"extract_first_json_array"})
        valid = self.accepted_row()
        text = (
            'noise [] then [1, 2] then [{"request_id":"REQ999999"}] then '
            + json.dumps([valid])
        )
        self.assertEqual(
            functions["extract_first_json_array"](
                text, expected_request_ids=["REQ000001"], expected_count=1
            ),
            [valid],
        )
        with self.assertRaisesRegex(ValueError, "json_array_empty"):
            functions["extract_first_json_array"]("prefix [] suffix")

    def test_json_parser_skips_prompt_echo_before_complete_generated_array(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(notebook, {"extract_first_json_array"})
        echo = [self.scheduled_row()]
        generated = [self.accepted_row()]
        text = f"echo {json.dumps(echo)} generated {json.dumps(generated)}"
        self.assertEqual(
            functions["extract_first_json_array"](
                text,
                expected_request_ids=["REQ000001"],
                expected_count=1,
            ),
            generated,
        )

    def test_candidate_batch_validation_rejects_non_string_duplicate_and_unknown_ids(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(notebook, {"validate_candidate_batch"})
        cases = [
            (
                [{"request_id": ["REQ000001"]}],
                "candidate_request_id_not_string:index=0",
            ),
            (
                [{"request_id": "REQ000001"}, {"request_id": "REQ000001"}],
                "duplicate_candidate_request_id:REQ000001",
            ),
            (
                [{"request_id": "REQ999999"}],
                "unknown_candidate_request_id:REQ999999",
            ),
        ]
        for candidates, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                mapping, reasons = functions["validate_candidate_batch"](
                    candidates,
                    ["REQ000001", "REQ000002"],
                    ["REQ000001", "REQ000002", "REQ000003"],
                )
                self.assertEqual(mapping, {})
                self.assertIn(expected_reason, reasons)

    def test_run_generation_recovers_after_malformed_batch_and_persists_retry(self):
        notebook = self.build_notebook_json()
        calls = []

        def fake_generate_batch(assignments):
            calls.append(assignments)
            if len(calls) == 1:
                return [{"request_id": [assignments[0]["request_id"]]}]
            return [self.accepted_row(assignments[0]["request_id"])]

        class TorchStub:
            class cuda:
                class OutOfMemoryError(Exception):
                    pass

                @staticmethod
                def empty_cache():
                    return None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            functions = extract_notebook_functions(
                notebook,
                {
                    "canonical_hash",
                    "atomic_write_json",
                    "append_jsonl",
                    "load_jsonl",
                    "validate_checkpoint_presence",
                    "read_checkpoint_manifest",
                    "validate_resume_manifest",
                    "manifest_requires_accepted_count_recovery",
                    "reconcile_accepted_rows",
                    "reconstruct_retry_counts",
                    "consume_retry_budget",
                    "ensure_retry_budgets",
                    "validate_candidate_batch",
                    "project_generated_record",
                    "is_near_duplicate",
                    "_manifest",
                    "run_generation",
                },
                {
                    "CONFIG": {
                        "model_id": "fake/model",
                        "seed": 7,
                        "generation_batch_size": 1,
                        "max_request_retries": 2,
                        "near_duplicate_threshold": 0.95,
                    },
                    "MODEL_REVISION": "a" * 40,
                    "PIPELINE_IDENTITY": {
                        "model": {"id": "fake/model", "revision": "a" * 40},
                        "generation_config": {"seed": 7},
                    },
                    "PIPELINE_IDENTITY_HASH": canonical_test_hash({
                        "model": {"id": "fake/model", "revision": "a" * 40},
                        "generation_config": {"seed": 7},
                    }),
                    "ACCEPTED_PATH": root / "accepted_rows.jsonl",
                    "REJECTED_PATH": root / "rejected_events.jsonl",
                    "MANIFEST_PATH": root / "manifest.json",
                    "generate_batch": fake_generate_batch,
                    "seed_generation": lambda *args, **kwargs: 1,
                    "torch": TorchStub,
                    "tqdm": type("Progress", (), {"write": staticmethod(lambda value: None)}),
                },
            )
            rows = functions["run_generation"]([self.scheduled_row()])
            self.assertEqual(rows, [self.accepted_row()])
            self.assertEqual(len(calls), 2)
            events = functions["load_jsonl"](root / "rejected_events.jsonl")
            self.assertEqual(events[0]["request_ids"], ["REQ000001"])
            self.assertEqual(events[0]["reason"], "malformed_candidate_batch")
            self.assertIn("candidate_request_id_not_string:index=0", events[0]["reasons"])
            self.assertEqual(
                functions["load_jsonl"](root / "accepted_rows.jsonl"),
                [self.accepted_row()],
            )
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["accepted_count"], 1)

    def test_resume_fails_closed_when_accepted_rows_exist_without_manifest(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(notebook, {"validate_checkpoint_presence"})
        with self.assertRaisesRegex(RuntimeError, "checkpoint_manifest_missing"):
            functions["validate_checkpoint_presence"](True, False)

    def test_resume_rejects_exact_duplicate_and_repeated_request_ids(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "canonical_hash",
                "normalize_for_dedup",
                "_language_matches",
                "validate_generated_row",
                "reconcile_accepted_rows",
            },
        )
        scheduled = self.scheduled_row()
        accepted = self.accepted_row()
        manifest = {"accepted_count": 2}
        with self.assertRaisesRegex(RuntimeError, "checkpoint_duplicate_accepted_row"):
            functions["reconcile_accepted_rows"](
                [accepted, dict(accepted)], [scheduled], manifest
            )

        changed = dict(accepted, Text=accepted["Text"] + " Changed.")
        with self.assertRaisesRegex(RuntimeError, "checkpoint_duplicate_request_id"):
            functions["reconcile_accepted_rows"](
                [accepted, changed], [scheduled], manifest
            )

    def test_resume_rejects_unknown_and_stale_accepted_rows(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "canonical_hash",
                "normalize_for_dedup",
                "_language_matches",
                "validate_generated_row",
                "reconcile_accepted_rows",
            },
        )
        manifest = {"accepted_count": 1}
        with self.assertRaisesRegex(RuntimeError, "checkpoint_unknown_request_id"):
            functions["reconcile_accepted_rows"](
                [self.accepted_row("REQ999999")], [self.scheduled_row()], manifest
            )

        stale = dict(self.accepted_row(), Category="Gay Men")
        with self.assertRaisesRegex(RuntimeError, "checkpoint_invalid_accepted_row"):
            functions["reconcile_accepted_rows"](
                [stale], [self.scheduled_row()], manifest
            )

    def test_resume_rejects_non_string_accepted_request_id_without_type_error(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "canonical_hash",
                "normalize_for_dedup",
                "_language_matches",
                "validate_generated_row",
                "reconcile_accepted_rows",
            },
        )
        malformed = dict(self.accepted_row(), request_id=[])
        with self.assertRaisesRegex(
            RuntimeError, "checkpoint_invalid_accepted_request_id:index=0"
        ):
            functions["reconcile_accepted_rows"](
                [malformed], [self.scheduled_row()], {"accepted_count": 1}
            )

    def test_resume_recovers_validated_append_only_suffix(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "canonical_hash",
                "normalize_for_dedup",
                "_language_matches",
                "validate_generated_row",
                "reconcile_accepted_rows",
            },
        )
        second = dict(
            self.accepted_row("REQ000002"),
            Text=(
                "I despise those fictional transgender people; they are worthless "
                "freaks and a disgrace who should get out."
            ),
            **{
                "Counter Narrative": (
                    "A separate respectful response rejects prejudice and supports equal dignity."
                )
            },
        )
        reconciled = functions["reconcile_accepted_rows"](
            [self.accepted_row(), second],
            [self.scheduled_row(), self.scheduled_row("REQ000002")],
            {"accepted_count": 1},
        )
        self.assertEqual(set(reconciled), {"REQ000001", "REQ000002"})

    def test_resume_rejects_manifest_count_greater_than_durable_rows(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {"canonical_hash", "reconcile_accepted_rows"},
        )
        with self.assertRaisesRegex(RuntimeError, "checkpoint_accepted_rows_missing"):
            functions["reconcile_accepted_rows"](
                [], [self.scheduled_row()], {"accepted_count": 1}
            )

    def test_recovered_suffix_requires_manifest_count_rewrite(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook, {"manifest_requires_accepted_count_recovery"}
        )
        self.assertTrue(
            functions["manifest_requires_accepted_count_recovery"](
                {"accepted_count": 1}, 2
            )
        )
        self.assertFalse(
            functions["manifest_requires_accepted_count_recovery"](
                {"accepted_count": 2}, 2
            )
        )

    def test_resume_rejects_unknown_or_duplicate_rows_in_recovery_suffix(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "canonical_hash",
                "normalize_for_dedup",
                "_language_matches",
                "validate_generated_row",
                "reconcile_accepted_rows",
            },
        )
        prefix = self.accepted_row()
        with self.assertRaisesRegex(RuntimeError, "checkpoint_unknown_request_id"):
            functions["reconcile_accepted_rows"](
                [prefix, self.accepted_row("REQ999999")],
                [self.scheduled_row(), self.scheduled_row("REQ000002")],
                {"accepted_count": 1},
            )
        duplicate_id = dict(
            prefix,
            Text="A different valid English message for the same durable request ID.",
        )
        with self.assertRaisesRegex(RuntimeError, "checkpoint_duplicate_request_id"):
            functions["reconcile_accepted_rows"](
                [prefix, duplicate_id],
                [self.scheduled_row(), self.scheduled_row("REQ000002")],
                {"accepted_count": 1},
            )

    def test_resume_rejects_cross_request_text_and_counter_duplicates(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "canonical_hash",
                "normalize_for_dedup",
                "is_near_duplicate",
                "_language_matches",
                "validate_generated_row",
                "reconcile_accepted_rows",
            },
        )
        first = self.accepted_row()
        second_base = dict(
            self.accepted_row("REQ000002"),
            Text=(
                "I loathe those fictional transgender people; they are diseased "
                "perverts and a shame who should go away."
            ),
            **{
                "Counter Narrative": (
                    "This different response firmly supports equality and rejects targeted abuse."
                )
            },
        )
        schedule = [self.scheduled_row(), self.scheduled_row("REQ000002")]
        manifest = {"accepted_count": 2}
        cases = [
            (
                "checkpoint_duplicate_accepted_text",
                dict(second_base, Text=first["Text"]),
            ),
            (
                "checkpoint_near_duplicate_accepted_text",
                dict(second_base, Text=first["Text"] + " Again."),
            ),
            (
                "checkpoint_duplicate_accepted_counter",
                dict(second_base, **{"Counter Narrative": first["Counter Narrative"]}),
            ),
            (
                "checkpoint_near_duplicate_accepted_counter",
                dict(
                    second_base,
                    **{"Counter Narrative": first["Counter Narrative"] + " Always."},
                ),
            ),
        ]
        for diagnostic, second in cases:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(RuntimeError, diagnostic):
                    functions["reconcile_accepted_rows"](
                        [first, second], schedule, manifest, 0.80
                    )

    def test_resume_manifest_fails_closed_on_identity_mismatch(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook, {"canonical_hash", "validate_resume_manifest"}
        )
        identity = {"model": {"id": "fake/model", "revision": "a" * 40}}
        identity_hash = functions["canonical_hash"](identity)
        manifest = {
            "pipeline_identity": identity,
            "identity_hash": identity_hash,
            "model_id": "fake/model",
            "model_revision": "a" * 40,
        }
        self.assertTrue(
            functions["validate_resume_manifest"](
                manifest, identity, identity_hash
            )
        )
        with self.assertRaisesRegex(RuntimeError, "checkpoint_identity_mismatch"):
            functions["validate_resume_manifest"](
                manifest, {"model": {"id": "other"}}, identity_hash
            )
        with self.assertRaisesRegex(RuntimeError, "checkpoint_identity_mismatch"):
            functions["validate_resume_manifest"](
                dict(manifest, pipeline_identity={"tampered": True}),
                identity,
                identity_hash,
            )

    def test_checkpoint_manifest_structure_is_validated_before_model_loading(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook, {"validate_checkpoint_manifest_structure"}
        )
        valid = {
            "identity_hash": "b" * 64,
            "pipeline_identity": {"identity_version": 1},
            "model_id": "fake/model",
            "model_revision": "a" * 40,
            "seed": 7,
            "accepted_count": 0,
            "updated_at": "2026-08-24T00:00:00Z",
        }
        self.assertTrue(
            functions["validate_checkpoint_manifest_structure"](valid, 15)
        )
        for accepted_count in (None, "1", -1, 16):
            with self.subTest(accepted_count=accepted_count):
                with self.assertRaisesRegex(
                    RuntimeError, "checkpoint_manifest_invalid:accepted_count"
                ):
                    functions["validate_checkpoint_manifest_structure"](
                        dict(valid, accepted_count=accepted_count), 15
                    )
        source = self.build_text()
        self.assertLess(
            source.index(
                "validate_checkpoint_manifest_structure(CHECKPOINT_MANIFEST"
            ),
            source.index("tokenizer, model = load_generator"),
        )

    def test_model_revision_must_be_resolved_to_an_immutable_commit(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(notebook, {"require_model_revision"})
        self.assertEqual(functions["require_model_revision"]("a" * 40), "a" * 40)
        for unresolved in (None, "", "main", "refs/heads/main"):
            with self.subTest(unresolved=unresolved):
                with self.assertRaisesRegex(RuntimeError, "immutable_model_revision_unavailable"):
                    functions["require_model_revision"](unresolved)

    def test_builder_embeds_sha256_of_exact_emitted_pipeline_cells(self):
        from work import build_kaggle_qwen35_lgbtq_hatespeech_dataset as builder

        notebook = self.build_notebook_json()
        emitted_pipeline_sources = [
            "".join(notebook["cells"][index]["source"])
            for index in (3, 4, 5, 6, 7)
        ]
        expected = builder.compute_pipeline_code_sha256(emitted_pipeline_sources)
        config_source = "".join(notebook["cells"][2]["source"])
        match = re.search(r'PIPELINE_CODE_SHA256 = "([0-9a-f]{64})"', config_source)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), expected)

    def test_pipeline_identity_hash_is_sensitive_to_every_effective_input(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "canonical_hash",
                "require_model_revision",
                "_json_safe_config",
                "build_pipeline_identity",
            },
        )
        config = {
            "model_id": "fake/model",
            "total_rows": 15,
            "smoke_test": True,
            "seed": 7,
            "generation_batch_size": 2,
            "temperature": 0.9,
            "top_p": 0.91,
            "repetition_penalty": 1.05,
            "max_new_tokens": 100,
            "max_request_retries": 3,
            "near_duplicate_threshold": 0.88,
            "run_root": Path("/tmp/fake-run"),
        }
        schedule = [{"request_id": "REQ000001", "target": "synthetic"}]
        package_versions = {"transformers": "5.3.0", "torch": "2.8.0"}
        runtime_identity = {
            "python": "3.11.0", "platform": "test", "cuda": "12.6", "gpu": "Fake GPU"
        }

        def identity_hash(**overrides):
            identity = functions["build_pipeline_identity"](
                overrides.get("config", config),
                overrides.get("schedule", schedule),
                overrides.get("model_revision", "a" * 40),
                overrides.get("effective_dtype", "float16"),
                overrides.get("package_versions", package_versions),
                overrides.get("runtime_identity", runtime_identity),
                overrides.get("pipeline_code_sha256", "b" * 64),
                overrides.get("system_prompt", "exact prompt"),
                overrides.get("generated_record_keys", functions["GENERATED_RECORD_KEYS"]),
            )
            return identity, functions["canonical_hash"](identity)

        identity, baseline = identity_hash()
        self.assertEqual(
            identity["quantization"],
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": "float16",
            },
        )
        variants = [
            {"config": dict(config, seed=8)},
            {"schedule": schedule + [{"request_id": "REQ000002", "target": "synthetic"}]},
            {"model_revision": "c" * 40},
            {"effective_dtype": "bfloat16"},
            {"package_versions": dict(package_versions, torch="2.9.0")},
            {"runtime_identity": dict(runtime_identity, gpu="Other GPU")},
            {"pipeline_code_sha256": "d" * 64},
            {"system_prompt": "changed prompt"},
            {"generated_record_keys": tuple(functions["GENERATED_RECORD_KEYS"]) + ("extra",)},
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(identity_hash(**variant)[1], baseline)

    def test_identity_records_versions_for_generation_export_and_audit_packages(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(notebook, set())
        self.assertLessEqual(
            {
                "transformers", "accelerate", "bitsandbytes", "sentencepiece",
                "torch", "numpy", "pandas", "openpyxl", "tqdm", "langdetect",
                "scikit-learn", "matplotlib", "seaborn", "tokenizers",
                "huggingface-hub", "safetensors",
            },
            set(functions["IDENTITY_PACKAGE_NAMES"]),
        )

    def test_resume_selects_recorded_revision_without_resolving_current_head(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook, {"require_model_revision", "select_model_revision"}
        )
        calls = []

        def resolver(model_id):
            calls.append(model_id)
            return "b" * 40

        manifest = {"model_id": "fake/model", "model_revision": "a" * 40}
        self.assertEqual(
            functions["select_model_revision"]("fake/model", manifest, resolver),
            "a" * 40,
        )
        self.assertEqual(calls, [])
        with self.assertRaisesRegex(RuntimeError, "checkpoint_model_id_mismatch"):
            functions["select_model_revision"]("other/model", manifest, resolver)
        self.assertEqual(calls, [])
        self.assertEqual(
            functions["select_model_revision"]("fake/model", None, resolver),
            "b" * 40,
        )
        self.assertEqual(calls, ["fake/model"])

    def test_validate_config_rejects_invalid_values_before_model_access(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(notebook, {"validate_config"})
        valid = {
            "model_id": functions["PRIMARY_MODEL_ID"],
            "total_rows": 2000,
            "smoke_test": False,
            "seed": 3407,
            "generation_batch_size": 3,
            "temperature": 1.0,
            "top_p": 0.92,
            "repetition_penalty": 1.08,
            "max_new_tokens": 1100,
            "max_request_retries": 6,
            "near_duplicate_threshold": 0.88,
            "run_root": Path("/tmp/fake-run"),
        }
        self.assertTrue(functions["validate_config"](valid))
        invalid_cases = [
            ("total_rows", 1700),
            ("generation_batch_size", 0),
            ("generation_batch_size", True),
            ("generation_batch_size", 65),
            ("max_request_retries", -1),
            ("max_request_retries", 101),
            ("max_new_tokens", 0),
            ("max_new_tokens", 32769),
            ("temperature", 0.0),
            ("temperature", float("nan")),
            ("top_p", 1.1),
            ("repetition_penalty", 0.0),
            ("near_duplicate_threshold", 1.1),
            ("seed", True),
            ("seed", -1),
            ("smoke_test", "yes"),
            ("model_id", "unapproved/model"),
        ]
        for key, value in invalid_cases:
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(ValueError, f"invalid_config:{key}"):
                    functions["validate_config"](dict(valid, **{key: value}))

    def test_pre_model_order_gpu_diagnostics_identity_and_oom_guidance(self):
        text = self.build_text()
        weight_load = text.index("AutoModelForCausalLM.from_pretrained")
        for marker in [
            "validate_config(CONFIG)",
            "SCHEDULE = build_quota_schedule",
            "CHECKPOINT_MANIFEST = read_checkpoint_manifest",
            "validate_resume_manifest(",
            "GPU before model download",
            "Validated pre-model plan",
        ]:
            self.assertLess(text.index(marker), weight_load)
        self.assertLess(
            text.index("GPU before model download"), text.index("AutoConfig.from_pretrained")
        )
        self.assertIn("torch.cuda.get_device_properties(0).total_memory", text)
        self.assertIn('"schedule_hash": canonical_hash(SCHEDULE)', text)
        self.assertIn("restart_kernel_and_select_4b_fallback", text)
        self.assertIn("PIPELINE_IDENTITY_HASH", text)
        self.assertIn('"identity_hash": PIPELINE_IDENTITY_HASH', text)
        self.assertNotIn('"config_hash"', text)

    def test_model_loading_cuda_oom_raises_restart_and_4b_fallback_diagnostic(self):
        notebook = self.build_notebook_json()

        class TorchStub:
            class cuda:
                class OutOfMemoryError(Exception):
                    pass

        class QuantizationStub:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class TokenizerStub:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return object()

        class ModelStub:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise TorchStub.cuda.OutOfMemoryError("fake oom")

        functions = extract_notebook_functions(
            notebook,
            {"load_generator"},
            {
                "torch": TorchStub,
                "BitsAndBytesConfig": QuantizationStub,
                "AutoTokenizer": TokenizerStub,
                "AutoModelForCausalLM": ModelStub,
            },
        )
        with self.assertRaisesRegex(
            RuntimeError, "restart_kernel_and_select_4b_fallback"
        ):
            functions["load_generator"](
                "fake/model", object(), "a" * 40, "float16"
            )

    def test_retry_counts_are_reconstructed_from_single_and_batch_events(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(notebook, {"reconstruct_retry_counts"})
        counts = functions["reconstruct_retry_counts"](
            [
                {"request_id": "REQ000001", "reasons": ["missing_candidate"]},
                {"request_ids": ["REQ000001", "REQ000002"], "reason": "cuda_out_of_memory"},
            ],
            ["REQ000001", "REQ000002", "REQ000003"],
        )
        self.assertEqual(
            counts,
            {"REQ000001": 2, "REQ000002": 1, "REQ000003": 0},
        )

    def test_retry_history_rejects_non_string_ids_without_type_error(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(notebook, {"reconstruct_retry_counts"})
        cases = [
            {"request_id": [], "reason": "malformed"},
            {"request_ids": [[]], "reason": "malformed"},
        ]
        for event in cases:
            with self.subTest(event=event):
                with self.assertRaisesRegex(
                    RuntimeError, "checkpoint_invalid_retry_request_id"
                ):
                    functions["reconstruct_retry_counts"](
                        [event], ["REQ000001"]
                    )

    def test_persistent_batch_size_one_oom_exhausts_retry_budget(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook, {"consume_retry_budget", "reduce_batch_size_after_oom"}
        )
        counts = {"REQ000001": 0}
        self.assertEqual(
            functions["reduce_batch_size_after_oom"](
                1, counts, ["REQ000001"], 2
            ),
            1,
        )
        self.assertEqual(
            functions["reduce_batch_size_after_oom"](
                1, counts, ["REQ000001"], 2
            ),
            1,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"retry_budget_exhausted:REQ000001:cuda_out_of_memory:retry_count=3:max=2",
        ):
            functions["reduce_batch_size_after_oom"](
                1, counts, ["REQ000001"], 2
            )

    def test_sampled_generation_seed_includes_durable_retry_state(self):
        notebook = self.build_notebook_json()

        class SeedRecorder:
            def __init__(self):
                self.values = []

            def seed(self, value):
                self.values.append(value)

        class TorchStub:
            def __init__(self):
                self.values = []
                self.cuda = SeedRecorder()
                self.cuda.manual_seed_all = self.cuda.seed

            def manual_seed(self, value):
                self.values.append(value)

        python_random = SeedRecorder()
        numpy_random = SeedRecorder()
        numpy_stub = type("NumpyStub", (), {"random": numpy_random})()
        torch_stub = TorchStub()
        functions = extract_notebook_functions(
            notebook,
            {"canonical_hash", "derive_generation_seed", "seed_generation"},
            {"random": python_random, "np": numpy_stub, "torch": torch_stub},
        )
        request_ids = ["REQ000001", "REQ000002"]
        retry_counts = {"REQ000001": 2, "REQ000002": 1}
        derived = functions["seed_generation"](3407, request_ids, retry_counts)
        self.assertEqual(
            derived,
            functions["derive_generation_seed"](3407, request_ids, retry_counts),
        )
        self.assertNotEqual(
            derived,
            functions["derive_generation_seed"](
                3407, request_ids, {"REQ000001": 3, "REQ000002": 1}
            ),
        )
        self.assertEqual(python_random.values, [derived])
        self.assertEqual(numpy_random.values, [derived % (2**32)])
        self.assertEqual(torch_stub.values, [derived])
        self.assertEqual(torch_stub.cuda.values, [derived])

    def test_generation_allows_only_one_json_repair_attempt(self):
        notebook = self.build_notebook_json()
        calls = []

        def model_text(messages):
            calls.append(messages)
            return "not json" if len(calls) == 1 else "[1]"

        functions = extract_notebook_functions(
            notebook,
            {
                "extract_first_json_array",
                "build_messages",
                "generate_batch_with_model_text",
            },
        )
        with self.assertRaisesRegex(ValueError, "json_array_contains_non_object"):
            functions["generate_batch_with_model_text"](
                [self.scheduled_row()], model_text
            )
        self.assertEqual(len(calls), 2)

    def test_generation_cell_has_quantization_prompt_and_refill_controls(self):
        text = self.build_text()
        for value in [
            "BitsAndBytesConfig",
            "AutoConfig",
            "load_in_4bit=True",
            'bnb_4bit_quant_type="nf4"',
            "torch.cuda.is_bf16_supported()",
            "AutoModelForCausalLM",
            "apply_chat_template",
            "enable_thinking=False",
            "research dataset for automated hate-speech detection and reporting",
            "Do not soften, euphemize, or sanitize the harmful Text field",
            "repair_json_once",
            "max_request_retries",
            "pending_request_ids",
            "accepted_rows.jsonl",
            "rejected_events.jsonl",
            "checkpoint_identity_mismatch",
            "torch.cuda.OutOfMemoryError",
            "revision=model_revision",
            "PIPELINE_IDENTITY = build_pipeline_identity",
            "PIPELINE_IDENTITY_HASH = canonical_hash(PIPELINE_IDENTITY)",
            '"identity_hash": identity_hash',
            "read_checkpoint_manifest",
            'seed_generation(CONFIG["seed"], request_ids, retry_counts)',
            "reconstruct_retry_counts",
            "reconcile_accepted_rows",
            "manifest_requires_accepted_count_recovery",
        ]:
            self.assertIn(value, text)
        recovery_start = text.index("if manifest_requires_accepted_count_recovery")
        generation_start = text.index('batch_size = CONFIG["generation_batch_size"]')
        self.assertLess(recovery_start, generation_start)
        self.assertIn(
            "atomic_write_json(", text[recovery_start:generation_start]
        )
        self.assertNotRegex(text, r"(?<!\.)\beval\(")

    def test_finalize_dataset_enforces_exact_schema_ids_and_quotas(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(notebook, {"finalize_dataset"})
        schedule = []
        rows = []
        for index, category in enumerate(functions["CATEGORIES"], 1):
            request_id = f"REQ{index:06d}"
            scheduled = {
                "request_id": request_id,
                "category": category,
                "language": "English",
                "platform_style": "X/Twitter reply",
                "abuse_type": "coded mockery",
                "severity": "coded",
            }
            schedule.append(scheduled)
            rows.append({
                "ID": request_id,
                "Text": f"Synthetic harmful sample {index}",
                "Category": category,
                "Target": "a fictional community member",
                "Counter Narrative": f"Synthetic supportive response {index}",
                "language": "English",
                "platform_style": "X/Twitter reply",
                "abuse_type": "coded mockery",
                "severity": "coded",
                "request_id": request_id,
            })
        frame = functions["finalize_dataset"](rows, schedule)
        self.assertEqual(list(frame.columns), ["ID", "Text", "Category", "Target", "Counter Narrative"])
        self.assertEqual(frame["ID"].tolist(), [f"HS{index:06d}" for index in range(1, 6)])
        with self.assertRaisesRegex(RuntimeError, "incomplete_generation"):
            functions["finalize_dataset"](rows[:-1], schedule)

    def test_finalize_dataset_rejects_invalid_rows(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {"finalize_dataset"},
            {"validate_generated_row": lambda row, scheduled: ["invalid_row"]},
        )
        with self.assertRaisesRegex(RuntimeError, "final_validation_failed:REQ000001:invalid_row"):
            functions["finalize_dataset"]([self.accepted_row()], [self.scheduled_row()])

    def test_finalize_dataset_rejects_normalized_text_duplicates(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(notebook, {"finalize_dataset"})
        schedule = [self.scheduled_row("REQ000001"), self.scheduled_row("REQ000002")]
        rows = [self.accepted_row("REQ000001"), self.accepted_row("REQ000002")]
        rows[0]["Text"] = "A synthetic   harmful sample"
        rows[1]["Text"] = "a SYNTHETIC\tharmful sample"
        with self.assertRaisesRegex(RuntimeError, "exact_text_duplicate"):
            functions["finalize_dataset"](rows, schedule)
        rows[1]["Text"] = "A different synthetic harmful sample"
        rows[0]["Counter Narrative"] = "A supportive   answer"
        rows[1]["Counter Narrative"] = "a SUPPORTIVE\tanswer"
        with self.assertRaisesRegex(RuntimeError, "exact_counter_duplicate"):
            functions["finalize_dataset"](rows, schedule)

    def test_audit_quotas_rejects_category_and_language_mismatches(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(notebook, {"audit_quotas"})
        pd = functions["pd"]
        schedule = []
        rows = []
        index = 0
        for category in functions["CATEGORIES"]:
            for language in functions["LANGUAGES"]:
                index += 1
                schedule.append({"request_id": f"REQ{index:06d}", "category": category, "language": language})
                rows.append({"Category": category, "language": language})
        final = pd.DataFrame(rows)
        audit = pd.DataFrame(rows)
        functions["audit_quotas"](final, audit, schedule)
        final.loc[0, "Category"] = functions["CATEGORIES"][1]
        with self.assertRaisesRegex(RuntimeError, "category_quota_mismatch"):
            functions["audit_quotas"](final, audit, schedule)
        final.loc[0, "Category"] = functions["CATEGORIES"][0]
        audit.loc[0, "language"] = "Hindi"
        with self.assertRaisesRegex(RuntimeError, "language_quota_mismatch"):
            functions["audit_quotas"](final, audit, schedule)

    def test_audit_helpers_compute_lengths_duplicates_and_inferred_language_consistency(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "normalize_for_dedup",
                "_language_matches",
                "find_near_duplicate_pairs",
                "compute_length_summaries",
                "compute_duplicate_statistics",
                "compute_language_consistency",
            },
        )
        rows = [
            self.accepted_row("REQ000001"),
            dict(
                self.accepted_row("REQ000002"),
                Text=self.accepted_row()["Text"] + " Again.",
                **{"Counter Narrative": "Hate is wrong; every person deserves equal dignity and respect."},
            ),
        ]
        lengths = functions["compute_length_summaries"](rows)
        self.assertEqual(set(lengths), {"Text", "Counter Narrative"})
        self.assertEqual(lengths["Text"]["count"], 2)
        duplicates = functions["compute_duplicate_statistics"](rows, 0.80)
        self.assertGreaterEqual(duplicates["near_text_pair_count"], 1)
        self.assertEqual(duplicates["exact_text_duplicate_count"], 0)
        consistency = functions["compute_language_consistency"](rows)
        self.assertEqual(consistency["both_match_count"], 2)
        rows[1]["Text"] = "यह नफरत भरी पोस्ट गलत है।"
        mismatch = functions["compute_language_consistency"](rows)
        self.assertEqual(mismatch["text_mismatch_count"], 1)
        self.assertEqual(mismatch["both_match_count"], 1)

    def test_publish_data_artifacts_uses_exact_names_and_leaves_no_temps(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook, {"atomic_write_json", "publish_data_artifacts"}
        )
        pd = functions["pd"]
        final = pd.DataFrame([{
            "ID": "HS000001", "Text": "sample", "Category": "Gay Men",
            "Target": "fictional", "Counter Narrative": "supportive",
        }])
        audit = final.assign(language="English")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            functions["publish_data_artifacts"](final, audit, root, {"total_rows": 1})
            self.assertEqual(
                {path.name for path in root.iterdir()},
                {
                    "lgbtq_hatespeech_counter_narratives.csv",
                    "lgbtq_hatespeech_counter_narratives.xlsx",
                    "generation_audit.csv",
                    "run_manifest.json",
                },
            )

    def test_publish_data_artifacts_cleans_temps_and_withholds_manifest_on_failure(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook, {"atomic_write_json", "publish_data_artifacts"}
        )
        pd = functions["pd"]
        final = pd.DataFrame([{
            "ID": "HS000001", "Text": "sample", "Category": "Gay Men",
            "Target": "fictional", "Counter Narrative": "supportive",
        }])

        class FailingAudit:
            def to_csv(self, *args, **kwargs):
                Path(args[0]).write_text("partial", encoding="utf-8")
                raise OSError("serialization_failed")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(OSError, "serialization_failed"):
                functions["publish_data_artifacts"](final, FailingAudit(), root, {"total_rows": 1})
            self.assertEqual(list(root.iterdir()), [])

    def test_publish_data_artifacts_removes_old_manifest_before_replacement_failure(self):
        notebook = self.build_notebook_json()

        class FailSecondArtifactReplace:
            def __init__(self):
                self.artifact_replacements = 0

            def replace(self, source, destination):
                if Path(destination).name != "run_manifest.json":
                    self.artifact_replacements += 1
                    if self.artifact_replacements == 2:
                        raise OSError("second_artifact_replace_failed")
                return os.replace(source, destination)

            def __getattr__(self, name):
                return getattr(os, name)

        functions = extract_notebook_functions(
            notebook,
            {"atomic_write_json", "publish_data_artifacts"},
            {"os": FailSecondArtifactReplace()},
        )
        pd = functions["pd"]
        final = pd.DataFrame([{
            "ID": "HS000001", "Text": "sample", "Category": "Gay Men",
            "Target": "fictional", "Counter Narrative": "supportive",
        }])
        audit = final.assign(language="English")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_manifest.json").write_text('{"old": true}', encoding="utf-8")
            with self.assertRaisesRegex(OSError, "second_artifact_replace_failed"):
                functions["publish_data_artifacts"](final, audit, root, {"total_rows": 1})
            self.assertFalse((root / "run_manifest.json").exists())
            self.assertFalse(any(path.name.endswith(".tmp") for path in root.iterdir()))

    def test_publish_withholds_manifest_when_directory_fsync_fails(self):
        notebook = self.build_notebook_json()

        def fail_directory_fsync(path):
            raise OSError(errno.EIO, "directory_fsync_failed", str(path))

        functions = extract_notebook_functions(
            notebook,
            {"publish_data_artifacts"},
            {"fsync_directory": fail_directory_fsync},
        )
        pd = functions["pd"]
        final = pd.DataFrame([{
            "ID": "HS000001", "Text": "sample", "Category": "Gay Men",
            "Target": "fictional", "Counter Narrative": "supportive",
        }])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(OSError, "directory_fsync_failed"):
                functions["publish_data_artifacts"](
                    final, final.assign(language="English"), root, {"total_rows": 1}
                )
            self.assertFalse((root / "run_manifest.json").exists())

    def test_publish_removes_manifest_when_post_replace_fsync_fails(self):
        notebook = self.build_notebook_json()
        directory_sync_calls = []

        def fail_manifest_directory_fsync(path):
            directory_sync_calls.append(Path(path))
            if len(directory_sync_calls) == 2:
                raise OSError(errno.EIO, "manifest_post_replace_fsync_failed")
            return True

        functions = extract_notebook_functions(
            notebook,
            {"atomic_write_json", "publish_data_artifacts"},
            {"fsync_directory": fail_manifest_directory_fsync},
        )
        pd = functions["pd"]
        final = pd.DataFrame([{
            "ID": "HS000001", "Text": "sample", "Category": "Gay Men",
            "Target": "fictional", "Counter Narrative": "supportive",
        }])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                OSError, "manifest_post_replace_fsync_failed"
            ):
                functions["publish_data_artifacts"](
                    final, final.assign(language="English"), root, {"total_rows": 1}
                )
            self.assertGreaterEqual(len(directory_sync_calls), 2)
            self.assertFalse((root / "run_manifest.json").exists())

    def test_checkpoint_manifest_stays_resumable_after_post_replace_fsync_error(self):
        notebook = self.build_notebook_json()
        directory_sync_calls = []

        def fail_first_directory_fsync(path):
            directory_sync_calls.append(Path(path))
            if len(directory_sync_calls) == 1:
                raise OSError(errno.EIO, "checkpoint_post_replace_fsync_failed")
            return True

        functions = extract_notebook_functions(
            notebook,
            {
                "atomic_write_json",
                "validate_checkpoint_presence",
                "read_checkpoint_manifest",
            },
            {"fsync_directory": fail_first_directory_fsync},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "manifest.json"
            path.write_text('{"accepted_count": 0}', encoding="utf-8")
            with self.assertRaisesRegex(
                OSError, "checkpoint_post_replace_fsync_failed"
            ):
                functions["atomic_write_json"](
                    path, {"accepted_count": 1, "identity_hash": "a" * 64}
                )
            self.assertTrue(path.exists())
            self.assertTrue(
                functions["validate_checkpoint_presence"](True, path.exists())
            )
            self.assertEqual(
                functions["read_checkpoint_manifest"](path)["accepted_count"], 1
            )

    def test_append_jsonl_flushes_and_fsyncs_record(self):
        notebook = self.build_notebook_json()
        fsync_calls = []
        directory_fsyncs = []

        class RecordingOS:
            def fsync(self, descriptor):
                fsync_calls.append(descriptor)
                return os.fsync(descriptor)

            def __getattr__(self, name):
                return getattr(os, name)

        functions = extract_notebook_functions(
            notebook,
            {"append_jsonl"},
            {
                "os": RecordingOS(),
                "fsync_directory": lambda path: directory_fsyncs.append(Path(path)),
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            functions["append_jsonl"](path, {"request_id": "REQ000001"})
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                '{"request_id": "REQ000001"}\n',
            )
            self.assertEqual(len(fsync_calls), 1)
            self.assertEqual(directory_fsyncs, [path.parent])

    def test_fsync_directory_propagates_real_io_errors(self):
        notebook = self.build_notebook_json()

        class FailingOS:
            O_RDONLY = os.O_RDONLY
            O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)

            def __init__(self):
                self.closed = []

            def open(self, path, flags):
                return 91

            def fsync(self, descriptor):
                raise OSError(errno.EIO, "simulated_eio")

            def close(self, descriptor):
                self.closed.append(descriptor)

        failing_os = FailingOS()
        functions = extract_notebook_functions(
            notebook, {"fsync_directory"}, {"os": failing_os}
        )
        with self.assertRaisesRegex(OSError, "simulated_eio"):
            functions["fsync_directory"](Path("/tmp/fake-run"))
        self.assertEqual(failing_os.closed, [91])

    def test_load_jsonl_quarantines_and_truncates_only_malformed_final_line(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook, {"fsync_directory", "_quarantine_jsonl_tail", "load_jsonl"}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted.jsonl"
            good_line = '{"request_id": "REQ000001"}\n'
            torn_tail = '{"request_id": "REQ000002"'
            path.write_text(good_line + torn_tail, encoding="utf-8")
            self.assertEqual(
                functions["load_jsonl"](path),
                [{"request_id": "REQ000001"}],
            )
            self.assertEqual(path.read_text(encoding="utf-8"), good_line)
            quarantines = list(path.parent.glob("accepted.jsonl.corrupt-tail-*"))
            self.assertEqual(len(quarantines), 1)
            self.assertEqual(quarantines[0].read_text(encoding="utf-8"), torn_tail)

    def test_load_jsonl_durably_terminates_valid_unterminated_tail_before_append(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook,
            {
                "append_jsonl",
                "fsync_directory",
                "_quarantine_jsonl_tail",
                "load_jsonl",
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted.jsonl"
            path.write_bytes(b'{"request_id": "REQ000001"}')

            first = functions["load_jsonl"](path)
            self.assertEqual(first, [{"request_id": "REQ000001"}])
            self.assertTrue(path.read_bytes().endswith(b"\n"))

            functions["append_jsonl"](path, {"request_id": "REQ000002"})
            self.assertEqual(
                functions["load_jsonl"](path),
                [
                    {"request_id": "REQ000001"},
                    {"request_id": "REQ000002"},
                ],
            )
            self.assertEqual(
                list(path.parent.glob("accepted.jsonl.corrupt-tail-*")), []
            )

    def test_load_jsonl_fails_on_interior_corruption_without_modifying_file(self):
        notebook = self.build_notebook_json()
        functions = extract_notebook_functions(
            notebook, {"fsync_directory", "_quarantine_jsonl_tail", "load_jsonl"}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted.jsonl"
            original = b'{"ok": 1}\n{"broken":\n{"ok": 2}\n'
            path.write_bytes(original)
            with self.assertRaisesRegex(RuntimeError, "jsonl_interior_corruption:line=2"):
                functions["load_jsonl"](path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob("accepted.jsonl.corrupt-tail-*")), [])

    def test_export_publication_fsyncs_temps_and_directory_before_manifest(self):
        notebook = self.build_notebook_json()
        events = []

        class RecordingOS:
            def replace(self, source, destination):
                events.append(("replace", Path(destination).name))
                return os.replace(source, destination)

            def __getattr__(self, name):
                return getattr(os, name)

        def atomic_manifest(path, value, remove_on_post_replace_failure=False):
            events.append(("manifest", Path(path).name))
            self.assertTrue(remove_on_post_replace_failure)
            Path(path).write_text(json.dumps(value), encoding="utf-8")

        functions = extract_notebook_functions(
            notebook,
            {"publish_data_artifacts"},
            {
                "os": RecordingOS(),
                "fsync_file": lambda path: events.append(("file", Path(path).name)),
                "fsync_directory": lambda path: events.append(("directory", Path(path).name)),
                "atomic_write_json": atomic_manifest,
            },
        )
        pd = functions["pd"]
        final = pd.DataFrame([{
            "ID": "HS000001", "Text": "sample", "Category": "Gay Men",
            "Target": "fictional", "Counter Narrative": "supportive",
        }])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            functions["publish_data_artifacts"](
                final, final.assign(language="English"), root, {"total_rows": 1}
            )
        file_events = [event for event in events if event[0] == "file"]
        replace_events = [event for event in events if event[0] == "replace"]
        self.assertEqual(len(file_events), 3)
        self.assertEqual(len(replace_events), 3)
        manifest_index = events.index(("manifest", "run_manifest.json"))
        self.assertTrue(
            any(
                event[0] == "directory"
                for event in events[max(index for index, event in enumerate(events) if event[0] == "replace") + 1:manifest_index]
            )
        )
        for _, filename in replace_events:
            temp_name = f".{filename}.tmp"
            self.assertLess(events.index(("file", temp_name)), events.index(("replace", filename)))

    def test_export_cell_contains_audit_outputs_and_exact_names(self):
        text = self.build_text()
        for value in [
            "lgbtq_hatespeech_counter_narratives.csv",
            "lgbtq_hatespeech_counter_narratives.xlsx",
            "generation_audit.csv",
            "run_manifest.json",
            "category_counts",
            "language_counts",
            "rejection_reason_counts",
            "near_duplicate_threshold",
            "seaborn",
            "platform_counts.rename",
            "severity_counts.rename",
            "length_summaries",
            "duplicate_statistics",
            "language_consistency_counts",
            'AUDIT_DATASET["text_language_consistent"]',
            'AUDIT_DATASET["counter_language_consistent"]',
        ]:
            self.assertIn(value, text)

    def test_all_ordinary_python_cells_compile(self):
        notebook = self.build_notebook_json()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))
        )
        compile(source, "kaggle-notebook", "exec")

    def test_tracked_notebook_is_byte_fresh_from_builder(self):
        from work.build_kaggle_qwen35_lgbtq_hatespeech_dataset import build_notebook

        tracked = (
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "kaggle_qwen35_lgbtq_hatespeech_dataset.ipynb"
        )
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / tracked.name
            build_notebook(generated)
            self.assertEqual(tracked.read_bytes(), generated.read_bytes())


if __name__ == "__main__":
    unittest.main()
