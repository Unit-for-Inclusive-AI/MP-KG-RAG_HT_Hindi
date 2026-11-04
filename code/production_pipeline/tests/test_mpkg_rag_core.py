import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


WORK = Path(__file__).resolve().parents[1] / "work"
sys.path.insert(0, str(WORK))

try:
    import mpkg_rag_core as core
except ModuleNotFoundError:
    core = None


DEFAULT_CORPUS = Path(
    "/Users/pika/Downloads/Hate_Speech_Counter_Narrative_RAG_Project/"
    "kg_sources/extracted/lgbt_hate_speech_kg_sources"
)


def resolve_real_corpus():
    configured = os.environ.get("MPKG_RAG_CORPUS_ROOT")
    return Path(configured).expanduser() if configured else DEFAULT_CORPUS


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class ManifestDrivenRegistryTests(unittest.TestCase):
    def require_core(self):
        self.assertIsNotNone(core, "work/mpkg_rag_core.py is not implemented")
        return core

    def make_corpus(self, files, source_manifest=None, openalex=None, inventory=None):
        root = Path(tempfile.mkdtemp())
        documents = root / "documents"
        documents.mkdir()
        inventory_rows = []
        for filename, content in files.items():
            path = documents / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            inventory_rows.append(
                {
                    "filename": filename,
                    "bytes": len(content),
                    "sha256": sha256(content),
                    "kind": "pdf" if path.suffix == ".pdf" else "html_or_other",
                }
            )
        write_json(root / "source_manifest.json", source_manifest or [])
        write_json(root / "added_openalex_sources.json", openalex or [])
        write_json(root / "local_file_inventory.json", inventory or inventory_rows)
        return root

    def test_stable_id_is_deterministic_and_part_safe(self):
        registry = self.require_core()

        self.assertEqual(registry.stable_id("SRC061", "abc"), registry.stable_id("SRC061", "abc"))
        self.assertNotEqual(registry.stable_id("a||b"), registry.stable_id("a", "b"))
        self.assertRegex(registry.stable_id("SRC061", "abc"), r"^[0-9a-f]{20,64}$")

    def test_qwen_thinking_is_split_from_final_content(self):
        registry = self.require_core()

        trace = registry.split_qwen_thinking(
            '<think>\nCheck the evidence, then answer.\n</think>\n\n'
            '{"counter_narrative":"Respect people."}'
        )

        self.assertEqual(trace["reasoning_content"], "Check the evidence, then answer.")
        self.assertEqual(
            trace["final_content"],
            '{"counter_narrative":"Respect people."}',
        )
        self.assertEqual(trace["thinking_status"], "complete")
        self.assertFalse(trace["reasoning_truncated"])

    def test_qwen_thinking_supports_closing_tag_only_output(self):
        registry = self.require_core()

        trace = registry.split_qwen_thinking(
            'First inspect the claim.\n</think>\n\n{"counter_narrative":"No."}'
        )

        self.assertEqual(trace["reasoning_content"], "First inspect the claim.")
        self.assertEqual(trace["final_content"], '{"counter_narrative":"No."}')
        self.assertEqual(trace["thinking_status"], "complete_closing_tag_only")

    def test_unclosed_qwen_thinking_is_marked_truncated(self):
        registry = self.require_core()

        trace = registry.split_qwen_thinking("<think>unfinished reasoning")

        self.assertEqual(trace["reasoning_content"], "unfinished reasoning")
        self.assertEqual(trace["final_content"], "")
        self.assertEqual(trace["thinking_status"], "truncated")
        self.assertTrue(trace["reasoning_truncated"])

    def test_repeated_legacy_id_stays_distinct_and_duplicate_content_merges_provenance(self):
        registry = self.require_core()
        first = b"first document"
        second = b"second document"
        files = {
            "SRC061_first.pdf": first,
            "SRC061_second.pdf": second,
            "copy-of-first.pdf": first,
        }
        root = self.make_corpus(
            files,
            source_manifest=[
                {
                    "source_id": "SRC061",
                    "title": "First reviewed source",
                    "organisation": "Example Journal",
                    "type": "Peer-reviewed research",
                    "local_file": "documents/SRC061_first.pdf",
                    "sha256": sha256(first),
                }
            ],
            openalex=[
                {
                    "source_id": "SRC061",
                    "title": "Second reviewed source",
                    "doi": "https://doi.org/10.1234/example",
                    "openalex": "https://openalex.org/W1",
                    "path": "documents/SRC061_second.pdf",
                    "sha256": sha256(second),
                }
            ],
        )

        rows = registry.load_source_registry(root)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows.file_records_before_deduplication, 3)
        collision_rows = [row for row in rows if row["legacy_source_id"] == "SRC061"]
        self.assertEqual(len(collision_rows), 2)
        self.assertNotEqual(collision_rows[0]["document_uid"], collision_rows[1]["document_uid"])
        merged = next(row for row in rows if row["content_sha256"] == sha256(first))
        self.assertEqual(
            set(merged["relative_paths"]),
            {"documents/SRC061_first.pdf", "documents/copy-of-first.pdf"},
        )
        self.assertEqual(len(merged["provenance"]), 2)

    def test_authority_uses_reviewed_metadata_and_keeps_unknown_metadata_unknown(self):
        registry = self.require_core()
        root = self.make_corpus(
            {
                "WHO_review.pdf": b"unknown despite filename",
                "plain.pdf": b"reviewed government report",
            },
            source_manifest=[
                {
                    "source_id": "SRC001",
                    "title": "Unclassified notes",
                    "organisation": "Unknown publisher",
                    "type": "Community notes",
                    "local_file": "documents/WHO_review.pdf",
                    "sha256": sha256(b"unknown despite filename"),
                },
                {
                    "source_id": "SRC002",
                    "title": "Reviewed report",
                    "organisation": "UNESCO",
                    "type": "UN report",
                    "local_file": "documents/plain.pdf",
                    "sha256": sha256(b"reviewed government report"),
                },
            ],
        )

        rows = registry.load_source_registry(root)
        unknown = next(row for row in rows if row["relative_path"].endswith("WHO_review.pdf"))
        reviewed = next(row for row in rows if row["relative_path"].endswith("plain.pdf"))

        self.assertEqual(unknown["source_type"], "unknown")
        self.assertIsNone(unknown["authority_score"])
        self.assertFalse(unknown["factual_index_allowed"])
        self.assertEqual(reviewed["source_type"], "un_agency")
        self.assertEqual(reviewed["authority_score"], 0.92)
        self.assertTrue(reviewed["factual_index_allowed"])

    def test_hash_mismatch_is_explicit_but_does_not_drop_the_file(self):
        registry = self.require_core()
        content = b"actual bytes"
        root = self.make_corpus(
            {"SRC010_source.pdf": content},
            source_manifest=[
                {
                    "source_id": "SRC010",
                    "type": "UN report",
                    "organisation": "UNESCO",
                    "local_file": "documents/SRC010_source.pdf",
                    "sha256": "0" * 64,
                }
            ],
            inventory=[
                {
                    "filename": "SRC010_source.pdf",
                    "bytes": len(content),
                    "sha256": "1" * 64,
                    "kind": "pdf",
                }
            ],
        )

        rows = registry.load_source_registry(root)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content_sha256"], sha256(content))
        self.assertTrue(rows.validation_errors)
        self.assertTrue(all(error["event"] == "hash_mismatch" for error in rows.validation_errors))
        self.assertTrue(rows[0]["validation_errors"])
        self.assertFalse(rows[0]["factual_index_allowed"])
        self.assertEqual(rows[0]["status"], "quarantined")
        self.assertEqual(rows[0]["status_reason"], "manifest_hash_mismatch")

    def test_strict_loading_rejects_hash_mismatch(self):
        registry = self.require_core()
        content = b"actual bytes"
        root = self.make_corpus(
            {"SRC010_source.pdf": content},
            source_manifest=[
                {
                    "source_id": "SRC010",
                    "type": "UN report",
                    "local_file": "documents/SRC010_source.pdf",
                    "sha256": "0" * 64,
                }
            ],
        )

        with self.assertRaises(registry.CorpusValidationError):
            registry.load_source_registry(root, strict=True)

    def test_manifest_cannot_overwrite_derived_quarantine_identity_or_authority(self):
        registry = self.require_core()
        content = b"actual bytes"
        root = self.make_corpus(
            {"SRC010_source.pdf": content},
            source_manifest=[
                {
                    "source_id": "SRC010",
                    "type": "UN report",
                    "organisation": "UNESCO",
                    "local_file": "documents/SRC010_source.pdf",
                    "sha256": "0" * 64,
                    "factual_index_allowed": True,
                    "status": "accepted",
                    "status_reason": "validated",
                    "document_uid": "attacker-document-uid",
                    "authority_score": 1.0,
                }
            ],
        )

        row = registry.load_source_registry(root)[0]
        content_sha256 = sha256(content)

        self.assertFalse(row["factual_index_allowed"])
        self.assertEqual(row["status"], "quarantined")
        self.assertEqual(row["status_reason"], "manifest_hash_mismatch")
        self.assertEqual(row["authority_score"], 0.92)
        self.assertEqual(row["document_uid"], registry.stable_id("document", content_sha256))
        self.assertEqual(row["content_sha256"], content_sha256)
        self.assertEqual(row["manifest_metadata"]["factual_index_allowed"], True)
        self.assertEqual(row["manifest_metadata"]["status"], "accepted")
        self.assertEqual(row["manifest_metadata"]["document_uid"], "attacker-document-uid")
        self.assertEqual(row["manifest_metadata"]["authority_score"], 1.0)

    def test_ambiguous_basename_overlay_does_not_merge_metadata(self):
        registry = self.require_core()
        root = self.make_corpus(
            {
                "left/shared.pdf": b"left bytes",
                "right/shared.pdf": b"right bytes",
            },
            source_manifest=[
                {"source_id": "SRC001", "filename": "shared.pdf", "type": "UN report"},
                {"source_id": "SRC002", "filename": "shared.pdf", "type": "official guidance"},
            ],
        )

        rows = registry.load_source_registry(root)

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["source_type"] == "unknown" for row in rows))
        self.assertTrue(all(not row["factual_index_allowed"] for row in rows))
        self.assertTrue(
            all(
                any(error["event"] == "ambiguous_basename" for error in row["validation_errors"])
                for row in rows
            )
        )
        self.assertEqual(
            sum(event["event"] == "ambiguous_basename" for event in rows.audit_events),
            2,
        )

    def test_unrecognized_type_label_does_not_become_known_source_type(self):
        registry = self.require_core()
        root = self.make_corpus(
            {"ordinary.pdf": b"community activism"},
            source_manifest=[
                {
                    "source_id": "SRC003",
                    "type": "Community activism",
                    "organisation": "Unknown publisher",
                    "local_file": "documents/ordinary.pdf",
                    "sha256": sha256(b"community activism"),
                }
            ],
        )

        row = registry.load_source_registry(root)[0]

        self.assertEqual(row["source_type"], "unknown")
        self.assertIsNone(row["authority_score"])
        self.assertFalse(row["factual_index_allowed"])

    def test_unsupported_and_hidden_files_are_audit_events_not_source_rows(self):
        registry = self.require_core()
        root = self.make_corpus(
            {
                "SRC011_supported.pdf": b"supported",
                "notes.txt": b"unsupported",
                "._hidden.pdf": b"metadata",
                ".DS_Store": b"metadata",
            }
        )

        rows = registry.load_source_registry(root)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["relative_path"], "documents/SRC011_supported.pdf")
        self.assertEqual(
            {event["event"] for event in rows.audit_events},
            {"unsupported_file", "hidden_metadata_file"},
        )
        self.assertTrue(all("relative_path" not in row or row["relative_path"].endswith("supported.pdf") for row in rows))

    @unittest.skipUnless(
        resolve_real_corpus().is_dir() and (resolve_real_corpus() / "documents").is_dir(),
        "real corpus is not available",
    )
    def test_real_corpus_accounts_for_all_files_and_preserves_both_src061_documents(self):
        registry = self.require_core()

        rows = registry.load_source_registry(resolve_real_corpus())

        self.assertEqual(rows.file_records_before_deduplication, 90)
        self.assertEqual(len(rows), 90)
        src061 = [row for row in rows if row["legacy_source_id"] == "SRC061"]
        self.assertEqual(len(src061), 2)
        self.assertEqual(len({row["document_uid"] for row in src061}), 2)
        self.assertEqual(
            {row["relative_path"] for row in src061},
            {
                "documents/SRC061_Lafayette_Queer_Archives_Project.html",
                "documents/SRC061_OpenAlex_From_prejudice_to_marginalization_Tracing_the_forms_of_online_hate_speech_targeting_LGBTQ_and_Muslim_communiti.pdf",
            },
        )
        self.assertTrue(all(row["content_sha256"] for row in rows))


class SemanticClaimGraphTests(unittest.TestCase):
    def require_core(self):
        self.assertIsNotNone(core, "work/mpkg_rag_core.py is not implemented")
        return core

    def make_payload(self, text, *, subject_id="wikidata:Q1", object_id="wikidata:Q2",
                     predicate="is", polarity="affirmed", stance="supports",
                     claim_id="claim-1", subject_text="queer people",
                     object_text="dangerous", object_status=None):
        subject_start = text.index(subject_text)
        object_start = text.index(object_text)
        claim = {
            "schema_version": "semantic-claims.v1",
            "claims": [
                {
                    "claim_id": claim_id,
                    "mentions": [
                        {
                            "mention_id": "m-subject",
                            "text": subject_text,
                            "start": subject_start,
                            "end": subject_start + len(subject_text),
                            "entity_id": subject_id,
                            "entity_status": "canonical",
                            "canonical_name": "Queer people",
                        },
                        {
                            "mention_id": "m-object",
                            "text": object_text,
                            "start": object_start,
                            "end": object_start + len(object_text),
                            "entity_id": object_id,
                            "entity_status": object_status or ("canonical" if object_id else "nil"),
                            "canonical_name": object_text,
                        },
                    ],
                    "subject": {"mention_id": "m-subject"},
                    "predicate": predicate,
                    "object": {"mention_id": "m-object"},
                    "polarity": polarity,
                    "modality": "asserted",
                    "attribution": {"type": "reported", "source": "the report"},
                    "evidence_stance": stance,
                    "model_confidence": 0.91,
                }
            ],
        }
        if polarity != "affirmed":
            scope_start = text.index("not dangerous") if "not dangerous" in text else object_start
            scope_end = scope_start + len("not dangerous") if "not dangerous" in text else object_start + len(object_text)
            claim["claims"][0]["polarity_scope"] = {
                "text": text[scope_start:scope_end], "start": scope_start, "end": scope_end
            }
        return claim

    def test_prompt_exposes_versioned_json_schema_without_loading_a_model(self):
        registry = self.require_core()

        prompt = registry.build_extraction_prompt(
            "queer people are dangerous", {"document_uid": "doc-1", "chunk_id": "chunk-1"}, "strict"
        )

        self.assertEqual(registry.SEMANTIC_EXTRACTION_SCHEMA["version"], "semantic-claims.v1")
        self.assertIsInstance(prompt, list)
        self.assertEqual([message["role"] for message in prompt], ["system", "user"])
        self.assertIn("claims", prompt[0]["content"])
        self.assertIn("document_uid", prompt[1]["content"])
        self.assertNotIn("\u0008", json.dumps(prompt))

    def test_validation_preserves_scoped_negation_subject_object_and_distinct_evidence_stance(self):
        registry = self.require_core()
        text = "The report says queer people are not dangerous."
        context = {"document_uid": "doc-1", "chunk_id": "chunk-1", "source_id": "SRC1"}

        result = registry.validate_extraction(self.make_payload(text, polarity="negated", stance="refutes"), text, context)

        self.assertEqual(result["status"], "accepted")
        claim = result["accepted"][0]
        self.assertEqual(claim["subject"]["mention_id"], "m-subject")
        self.assertEqual(claim["object"]["mention_id"], "m-object")
        self.assertEqual(claim["polarity"], "negated")
        self.assertEqual(claim["evidence_stance"], "refutes")
        self.assertEqual(
            claim["polarity_scope"],
            {"text": "not dangerous", "start": text.index("not dangerous"), "end": len(text) - 1},
        )
        self.assertEqual(result["quarantined"], [])

    def test_invalid_span_is_quarantined_with_a_reason(self):
        registry = self.require_core()
        text = "queer people are dangerous"
        payload = self.make_payload(text)
        payload["claims"][0]["mentions"][0]["text"] = "queer  people"

        result = registry.validate_extraction(
            payload, text, {"document_uid": "doc-1", "chunk_id": "chunk-1", "source_id": "SRC1"}
        )

        self.assertEqual(result["accepted"], [])
        self.assertEqual(result["status"], "quarantined")
        self.assertIn("mention_span_mismatch", result["quarantined"][0]["reasons"])

    def test_unknown_predicate_is_quarantined_instead_of_inferred(self):
        registry = self.require_core()
        text = "queer people are dangerous"

        result = registry.validate_extraction(
            self.make_payload(text, predicate="invented_relation"),
            text,
            {"document_uid": "doc-1", "chunk_id": "chunk-1", "source_id": "SRC1"},
        )

        self.assertEqual(result["accepted"], [])
        self.assertIn("unknown_predicate", result["quarantined"][0]["reasons"])

    def test_canonical_entities_are_preserved(self):
        registry = self.require_core()
        text = "queer people are dangerous"
        context = {"document_uid": "doc-1", "chunk_id": "chunk-1", "source_id": "SRC1"}

        first = registry.validate_extraction(self.make_payload(text), text, context)

        first_mentions = {mention["mention_id"]: mention for mention in first["accepted"][0]["mentions"]}
        self.assertEqual(first_mentions["m-subject"]["entity_id"], "wikidata:Q1")
        self.assertEqual(first_mentions["m-object"]["entity_id"], "wikidata:Q2")
        self.assertEqual(first_mentions["m-object"]["entity_status"], "canonical")

    def test_overlapping_chunks_deduplicate_claim_but_preserve_each_evidence_occurrence(self):
        registry = self.require_core()
        contexts = [
            {"document_uid": "doc-1", "chunk_id": "chunk-1", "source_id": "SRC1"},
            {"document_uid": "doc-1", "chunk_id": "chunk-2", "source_id": "SRC1"},
        ]
        chunks = [
            {**contexts[0], "text": "Intro: queer people are dangerous.", "status": "accepted", "factual_index_allowed": True},
            {**contexts[1], "text": "queer people are dangerous. Tail.", "status": "accepted", "factual_index_allowed": True},
        ]
        extractions = [
            registry.validate_extraction(self.make_payload(chunks[0]["text"]), chunks[0]["text"], contexts[0]),
            registry.validate_extraction(self.make_payload(chunks[1]["text"]), chunks[1]["text"], contexts[1]),
        ]

        graph = registry.build_semantic_graph(chunks, extractions)

        self.assertEqual(len(graph["Claim"]), 1)
        self.assertEqual(len(graph["EvidenceChunk"]), 2)
        stance_edges = [edge for edge in graph["edges"] if edge["type"] == "supports"]
        self.assertEqual(len(stance_edges), 2)
        self.assertEqual({edge["stance"] for edge in stance_edges}, {"supports"})
        self.assertEqual(len({edge["occurrence_id"] for edge in stance_edges}), 2)

    def test_quarantined_extractions_do_not_enter_accepted_graph_tables(self):
        registry = self.require_core()
        text = "queer people are dangerous"
        context = {"document_uid": "doc-2", "chunk_id": "chunk-q", "source_id": "SRC2"}
        invalid = self.make_payload(text, predicate="invented_relation")

        result = registry.validate_extraction(invalid, text, context)
        graph = registry.build_semantic_graph(
            [{**context, "text": text, "status": "accepted", "factual_index_allowed": True}], [result]
        )

        self.assertEqual(graph["Document"], [])
        self.assertEqual(graph["EvidenceChunk"], [])
        self.assertEqual(graph["Claim"], [])
        self.assertEqual(graph["quarantined"], result["quarantined"])

    def test_graph_quarantines_raw_status_accepted_and_raw_claim_bypasses(self):
        registry = self.require_core()
        text = "queer people are dangerous"
        context = {"document_uid": "doc-bypass", "chunk_id": "chunk-1", "source_id": "SRC1"}
        validated = registry.validate_extraction(self.make_payload(text), text, context)
        raw_result = {
            "status": "accepted",
            "source_context": context,
            "accepted": validated["accepted"],
        }
        quarantined_result = dict(validated)
        quarantined_result["status"] = "quarantined"

        graph = registry.build_semantic_graph(
            [{**context, "text": text, "status": "accepted", "factual_index_allowed": True}],
            [raw_result, validated["accepted"][0], quarantined_result],
        )

        self.assertEqual(graph["Claim"], [])
        self.assertEqual(graph["EvidenceChunk"], [])
        self.assertTrue(any("unvalidated_extraction_result" in row["reasons"] for row in graph["quarantined"]))
        self.assertTrue(any("validation_result_stale" in row["reasons"] for row in graph["quarantined"]))

    def test_graph_quarantines_mutated_or_forged_validation_results(self):
        registry = self.require_core()
        text = "queer people are dangerous"
        context = {"document_uid": "doc-digest", "chunk_id": "chunk-1", "source_id": "SRC1"}
        validated = registry.validate_extraction(self.make_payload(text), text, context)
        chunk = {**context, "text": text, "status": "accepted", "factual_index_allowed": True}

        stale_accepted = dict(validated)
        stale_accepted["accepted"] = []
        stale_reviewed = dict(validated)
        stale_reviewed["reviewed"] = [{"status": "reviewed", "record": "bogus"}]
        unknown_status = dict(validated)
        unknown_status["status"] = "bogus"
        mutated_status = dict(validated)
        mutated_status["status"] = "reviewed"
        forged_marker = dict(validated)
        forged_marker["validation_marker"] = "forged"

        for candidate in (stale_accepted, stale_reviewed, unknown_status, mutated_status, forged_marker):
            graph = registry.build_semantic_graph([chunk], [candidate])
            self.assertEqual(graph["Claim"], [])
            self.assertEqual(graph["EvidenceChunk"], [])
            self.assertTrue(graph["quarantined"])

        self.assertTrue(any("validation_result_stale" in row["reasons"] for row in registry.build_semantic_graph([chunk], [stale_accepted])["quarantined"]))
        self.assertTrue(any("validation_result_status_invalid" in row["reasons"] for row in registry.build_semantic_graph([chunk], [unknown_status])["quarantined"]))

    def test_graph_denies_missing_none_false_and_harmful_source_policy(self):
        registry = self.require_core()
        text = "queer people are dangerous"
        context = {"document_uid": "doc-policy-default", "chunk_id": "chunk-1", "source_id": "SRC1"}
        validated = registry.validate_extraction(self.make_payload(text), text, context)
        chunks = [
            {**context, "text": text, "status": "accepted"},
            {**context, "text": text, "status": "accepted", "factual_index_allowed": None},
            {**context, "text": text, "status": "accepted", "factual_index_allowed": False},
            {**context, "text": text, "status": "accepted", "factual_index_allowed": True, "source_type": "harmful_examples"},
            {**context, "text": text, "status": None, "factual_index_allowed": True},
        ]

        for chunk in chunks:
            graph = registry.build_semantic_graph([chunk], [validated])
            self.assertEqual(graph["Document"], [])
            self.assertEqual(graph["EvidenceChunk"], [])
            self.assertEqual(graph["Claim"], [])
            self.assertTrue(graph["quarantined"])

    def test_graph_separates_document_hash_from_chunk_text_hash(self):
        registry = self.require_core()
        text = "queer people are dangerous"
        document_hash = hashlib.sha256((text + " plus surrounding source text").encode()).hexdigest()
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        context = {
            "document_uid": "doc-hashes",
            "chunk_id": "chunk-1",
            "source_id": "SRC1",
            "source_type": "official",
            "authority_score": 0.92,
            "factual_index_allowed": True,
            "status": "accepted",
            "content_sha256": document_hash,
            "document_sha256": document_hash,
            "text_sha256": text_hash,
        }
        validated = registry.validate_extraction(self.make_payload(text), text, context)
        graph = registry.build_semantic_graph([{**context, "text": text}], [validated])

        self.assertEqual(len(graph["Claim"]), 1)
        self.assertEqual(graph["Document"][0]["content_sha256"], document_hash)
        self.assertEqual(graph["EvidenceChunk"][0]["text_sha256"], text_hash)
        self.assertEqual(graph["quarantined"], [])

    def test_ambiguous_and_nil_validated_records_remain_review_audit_only(self):
        registry = self.require_core()
        text = "queer people are dangerous"
        context = {"document_uid": "doc-unresolved", "chunk_id": "chunk-1", "source_id": "SRC1"}
        chunks = [{**context, "text": text, "status": "accepted", "factual_index_allowed": True}]
        ambiguous = registry.validate_extraction(
            self.make_payload(text, object_id=None, object_status="ambiguous"), text, context
        )
        nil = registry.validate_extraction(
            self.make_payload(text, object_id="NIL-explicit", object_status="nil"), text, context
        )

        for result in (ambiguous, nil):
            graph = registry.build_semantic_graph(chunks, [result])
            self.assertEqual(graph["Document"], [])
            self.assertEqual(graph["EvidenceChunk"], [])
            self.assertEqual(graph["Claim"], [])
            self.assertTrue(graph["reviewed"])

    def test_schema_rejects_missing_ids_status_extra_fields_and_string_subjects(self):
        registry = self.require_core()
        text = "queer people are dangerous"
        payload = self.make_payload(text)
        mention = payload["claims"][0]["mentions"][0]
        del mention["mention_id"]
        del mention["entity_status"]
        payload["claims"][0]["subject"] = "m-subject"
        payload["claims"][0]["unexpected"] = True

        result = registry.validate_extraction(
            payload, text, {"document_uid": "doc-schema", "chunk_id": "chunk-1", "source_id": "SRC1"}
        )

        self.assertEqual(result["accepted"], [])
        reasons = set(result["quarantined"][0]["reasons"])
        self.assertTrue({"missing_mention_id", "missing_entity_status", "subject_not_object", "extra_claim_fields"} <= reasons)

        payload["extra"] = True
        top_level_result = registry.validate_extraction(
            payload, text, {"document_uid": "doc-schema", "chunk_id": "chunk-1", "source_id": "SRC1"}
        )
        self.assertIn("extra_payload_fields", top_level_result["quarantined"][0]["reasons"])

    def test_invalid_external_id_is_quarantined_and_ambiguous_or_nil_is_review_only(self):
        registry = self.require_core()
        text = "queer people are dangerous"
        context = {"document_uid": "doc-entity", "chunk_id": "chunk-1", "source_id": "SRC1"}

        invalid = registry.validate_extraction(self.make_payload(text, object_id="bad id"), text, context)
        ambiguous = registry.validate_extraction(
            self.make_payload(text, object_id=None, object_status="ambiguous"), text, context
        )
        nil = registry.validate_extraction(
            self.make_payload(text, object_id="NIL-explicit", object_status="nil"), text, context
        )

        self.assertIn("invalid_entity_id", invalid["quarantined"][0]["reasons"])
        self.assertEqual(ambiguous["accepted"], [])
        self.assertEqual(nil["accepted"], [])
        self.assertIn("unresolved_entity", ambiguous["reviewed"][0]["reasons"])
        self.assertIn("unresolved_entity", nil["reviewed"][0]["reasons"])

    def test_quoted_only_and_reviewed_claims_are_not_retrievable(self):
        registry = self.require_core()
        text = "queer people are dangerous"
        context = {"document_uid": "doc-review", "chunk_id": "chunk-1", "source_id": "SRC1"}
        quoted = registry.validate_extraction(self.make_payload(text, stance="quotes"), text, context)

        graph = registry.build_semantic_graph(
            [{**context, "text": text, "status": "accepted", "factual_index_allowed": True}], [quoted]
        )

        self.assertEqual(graph["Claim"], [])
        self.assertEqual(graph["Document"], [])
        self.assertIn("quoted_only", quoted["reviewed"][0]["reasons"])
        self.assertTrue(graph["reviewed"])

    def test_quarantined_or_nonfactual_chunks_are_excluded_from_graph_tables(self):
        registry = self.require_core()
        text = "queer people are dangerous"
        context = {"document_uid": "doc-policy", "chunk_id": "chunk-1", "source_id": "SRC1"}
        validated = registry.validate_extraction(self.make_payload(text), text, context)

        for chunk in (
            {**context, "text": text, "status": "quarantined", "factual_index_allowed": True},
            {**context, "text": text, "status": "accepted", "factual_index_allowed": False},
        ):
            graph = registry.build_semantic_graph([chunk], [validated])
            self.assertEqual(graph["Claim"], [])
            self.assertEqual(graph["EvidenceChunk"], [])
            self.assertTrue(graph["quarantined"])

        quarantined_context = {**context, "status": "quarantined"}
        context_result = registry.validate_extraction(self.make_payload(text), text, quarantined_context)
        context_graph = registry.build_semantic_graph(
            [{"document_uid": context["document_uid"], "chunk_id": context["chunk_id"], "text": text}],
            [context_result],
        )
        self.assertEqual(context_graph["Claim"], [])
        self.assertIn("chunk_status_not_accepted", context_graph["quarantined"][0]["reasons"])

    def test_missing_or_mismatched_chunk_provenance_is_quarantined(self):
        registry = self.require_core()
        text = "queer people are dangerous"
        context = {"document_uid": "doc-provenance", "chunk_id": "chunk-1", "source_id": "SRC1"}
        validated = registry.validate_extraction(self.make_payload(text), text, context)

        missing = registry.build_semantic_graph([], [validated])
        mismatched = registry.build_semantic_graph(
            [{**context, "text": "different text", "source_id": "SRC2"}], [validated]
        )
        context_text = {**context, "text": "different text"}
        context_text_result = registry.validate_extraction(self.make_payload(text), text, context_text)
        context_text_graph = registry.build_semantic_graph(
            [{**context, "text": text}], [context_text_result]
        )

        self.assertEqual(missing["EvidenceChunk"], [])
        self.assertEqual(mismatched["EvidenceChunk"], [])
        self.assertEqual(context_text_graph["EvidenceChunk"], [])
        self.assertIn("missing_chunk_provenance", missing["quarantined"][0]["reasons"])
        self.assertTrue({"chunk_text_mismatch", "source_context_mismatch"} <= set(mismatched["quarantined"][0]["reasons"]))
        self.assertIn("source_context_mismatch", context_text_graph["quarantined"][0]["reasons"])

    def test_graph_retains_audit_fields_and_never_emits_empty_evidence_text(self):
        registry = self.require_core()
        text = "queer people are dangerous"
        document_hash = hashlib.sha256((text + " plus surrounding source text").encode()).hexdigest()
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        context = {
            "document_uid": "doc-audit",
            "chunk_id": "chunk-1",
            "source_id": "SRC1",
            "source_type": "official",
            "authority_score": 0.92,
            "factual_index_allowed": True,
            "status": "accepted",
            "content_sha256": document_hash,
            "document_sha256": document_hash,
            "text_sha256": text_hash,
        }
        validated = registry.validate_extraction(self.make_payload(text), text, context)
        graph = registry.build_semantic_graph([{**context, "text": text}], [validated])

        document = graph["Document"][0]
        evidence = graph["EvidenceChunk"][0]
        claim = graph["Claim"][0]
        for row in (document, evidence):
            self.assertEqual(row["source_type"], "official")
            self.assertEqual(row["authority_score"], 0.92)
            self.assertTrue(row["factual_index_allowed"])
            self.assertEqual(row["status"], "accepted")
            self.assertEqual(row["content_sha256"], document_hash)
            self.assertEqual(row["document_sha256"], document_hash)
            self.assertEqual(row["text_sha256"], text_hash)
            self.assertIsNotNone(row["text"] if row is evidence else row["document_uid"])
        self.assertEqual(claim["model_confidence"], 0.91)
        self.assertEqual(claim["polarity"], "affirmed")
        self.assertEqual(claim["modality"], "asserted")
        self.assertEqual(claim["attribution"], {"type": "reported", "source": "the report"})
        self.assertEqual(claim["stance"], "supports")
        self.assertEqual(claim["review_status"], "accepted")

    def test_negated_claim_requires_and_validates_actual_polarity_scope_offsets(self):
        registry = self.require_core()
        text = "queer people are not dangerous"
        context = {"document_uid": "doc-scope", "chunk_id": "chunk-1", "source_id": "SRC1"}
        payload = self.make_payload(text, polarity="negated", stance="refutes")
        payload["claims"][0]["polarity_scope"]["text"] = "dangerous"

        result = registry.validate_extraction(payload, text, context)

        self.assertEqual(result["accepted"], [])
        self.assertIn("polarity_scope_mismatch", result["quarantined"][0]["reasons"])

    def test_schema_rejects_wrong_types_in_declared_fields(self):
        registry = self.require_core()
        text = "queer people are dangerous"
        payload = self.make_payload(text)
        payload["claims"][0]["claim_id"] = 7
        payload["claims"][0]["attribution"] = 7

        result = registry.validate_extraction(
            payload, text, {"document_uid": "doc-types", "chunk_id": "chunk-1", "source_id": "SRC1"}
        )

        self.assertEqual(result["accepted"], [])
        reasons = set(result["quarantined"][0]["reasons"])
        self.assertTrue({"invalid_claim_id", "invalid_attribution"} <= reasons)


class SemanticRetrievalTests(unittest.TestCase):
    def require_core(self):
        self.assertIsNotNone(core, "work/mpkg_rag_core.py is not implemented")
        return core

    def require_function(self, name):
        function = getattr(self.require_core(), name, None)
        self.assertTrue(callable(function), f"missing retrieval interface: {name}")
        return function

    def expansion_config(self):
        return {
            "max_hops": 2,
            "minimum_dense_score": 0.5,
            "minimum_graph_score": 0.0,
            "hop_decay": 0.5,
            "minimum_rerank_probability": 0.75,
            "max_evidence": 3,
            "weights": {
                "query_entity": 1.0,
                "predicate": 2.0,
                "polarity": 1.5,
                "modality": 1.25,
                "stance": 1.75,
                "review_state": 0.5,
                "authority": 0.75,
                "extraction_confidence": 0.75,
                "seed_score": 1.0,
                "hop_decay": 1.0,
            },
            "review_state_scores": {"accepted": 1.0, "reviewed": 0.25, "unknown": 0.0},
        }

    def graph_fixture(self):
        return {
            "Document": [
                {"document_uid": "d1", "authority_score": 0.9, "status": "accepted"},
                {"document_uid": "d2", "authority_score": 0.7, "status": "accepted"},
                {"document_uid": "d3", "authority_score": 0.8, "status": "accepted"},
            ],
            "EvidenceChunk": [
                {
                    "evidence_chunk_id": "e1",
                    "document_uid": "d1",
                    "chunk_id": "c1",
                    "authority_score": 0.9,
                    "status": "accepted",
                },
                {
                    "evidence_chunk_id": "e2",
                    "document_uid": "d2",
                    "chunk_id": "c2",
                    "authority_score": 0.7,
                    "status": "accepted",
                },
                {
                    "evidence_chunk_id": "e3",
                    "document_uid": "d3",
                    "chunk_id": "c3",
                    "authority_score": 0.8,
                    "status": "accepted",
                },
                {
                    "evidence_chunk_id": "e4",
                    "document_uid": "d1",
                    "chunk_id": "c4",
                    "authority_score": 0.9,
                    "status": "accepted",
                },
            ],
            "Claim": [
                {
                    "claim_id": "cl1",
                    "subject_entity_id": "Q1",
                    "object_entity_id": "Q2",
                    "predicate": "supports",
                    "polarity": "affirmed",
                    "modality": "asserted",
                    "stance": "supports",
                    "model_confidence": 0.9,
                    "review_status": "accepted",
                },
                {
                    "claim_id": "cl2",
                    "subject_entity_id": "Q1",
                    "object_entity_id": "Q2",
                    "predicate": "denies",
                    "polarity": "negated",
                    "modality": "possible",
                    "stance": "refutes",
                    "model_confidence": 0.8,
                    "review_status": "accepted",
                },
                {
                    "claim_id": "cl3",
                    "subject_entity_id": "Q9",
                    "object_entity_id": "Q8",
                    "predicate": "contains",
                    "polarity": "affirmed",
                    "modality": "asserted",
                    "stance": "contextualizes",
                    "model_confidence": 0.95,
                    "review_status": "accepted",
                },
            ],
            "Entity": [
                {"entity_id": "Q1", "canonical_name": "same alias"},
                {"entity_id": "Q2", "canonical_name": "same alias"},
                {"entity_id": "Q8", "canonical_name": "other"},
                {"entity_id": "Q9", "canonical_name": "other"},
            ],
            "Mention": [],
            "edges": [
                {"source_id": "e1", "type": "supports", "target_id": "cl1", "stance": "supports"},
                {"source_id": "cl1", "type": "has_subject", "target_id": "Q1", "target_type": "entity"},
                {"source_id": "cl1", "type": "has_object", "target_id": "Q2", "target_type": "entity"},
                {"source_id": "cl1", "type": "evidenced_by", "target_id": "e1"},
                {"source_id": "cl1", "type": "evidenced_by", "target_id": "e2"},
                {"source_id": "e2", "type": "refutes", "target_id": "cl2", "stance": "refutes"},
                {"source_id": "cl2", "type": "has_subject", "target_id": "Q1", "target_type": "entity"},
                {"source_id": "cl2", "type": "has_object", "target_id": "Q2", "target_type": "entity"},
                {"source_id": "cl2", "type": "evidenced_by", "target_id": "e2"},
                {"source_id": "e3", "type": "supports", "target_id": "cl3", "stance": "contextualizes"},
                {"source_id": "cl3", "type": "has_subject", "target_id": "Q9", "target_type": "entity"},
                {"source_id": "cl3", "type": "has_object", "target_id": "Q8", "target_type": "entity"},
                {"source_id": "cl3", "type": "evidenced_by", "target_id": "e3"},
                {"source_id": "e1", "type": "from_document", "target_id": "d1"},
                {"source_id": "e4", "type": "from_document", "target_id": "d1"},
            ],
        }

    def test_seed_filters_and_disconnected_nodes_do_not_enter_expansion(self):
        expand_graph_from_seeds = self.require_function("expand_graph_from_seeds")
        hits = [
            {"branch": "dense", "rank": 1, "raw_score": 0.9, "chunk_id": "c1", "document_uid": "d1"},
            {"branch": "dense", "rank": 2, "raw_score": 0.1, "chunk_id": "c3", "document_uid": "d3"},
            {"branch": "bm25", "rank": 1, "raw_score": 0.0, "chunk_id": "c3", "document_uid": "d3"},
        ]

        results = expand_graph_from_seeds(
            hits,
            {"entity_ids": ["Q1"], "predicates": ["supports"]},
            self.graph_fixture(),
            self.expansion_config(),
        )

        self.assertEqual({row["chunk_id"] for row in results}, {"c1", "c2"})
        self.assertNotIn("c3", {row["chunk_id"] for row in results})
        self.assertTrue(all(row["graph_score"] > 0 for row in results))

    def test_claimless_same_document_chunk_is_not_reached_through_provenance_edges(self):
        expand_graph_from_seeds = self.require_function("expand_graph_from_seeds")

        results = expand_graph_from_seeds(
            [{"branch": "dense", "rank": 1, "raw_score": 0.9, "chunk_id": "c1", "document_uid": "d1"}],
            {"entity_ids": ["Q1"]},
            self.graph_fixture(),
            self.expansion_config(),
        )

        self.assertNotIn("c4", {row["chunk_id"] for row in results})

    def test_seed_branch_and_rank_contract_rejects_aliases_unknowns_and_invalid_ranks(self):
        expand_graph_from_seeds = self.require_function("expand_graph_from_seeds")
        base = {"rank": 1, "raw_score": 0.9, "chunk_id": "c1", "document_uid": "d1"}

        for hit in (
            {**base, "branch": "sparse"},
            {**base, "branch": "vector"},
            {**base, "branch": "graph"},
            {**base, "branch": None},
            {**base, "branch": "dense", "rank": 0},
            {**base, "branch": "dense", "rank": -1},
            {**base, "branch": "dense", "rank": 1.5},
            {**base, "branch": "dense", "rank": True},
        ):
            with self.assertRaises(ValueError):
                expand_graph_from_seeds(
                    [hit], {"entity_ids": ["Q1"]}, self.graph_fixture(), self.expansion_config()
                )

    def test_seed_scores_are_normalized_per_branch_and_trace_raw_scores(self):
        expand_graph_from_seeds = self.require_function("expand_graph_from_seeds")
        config = self.expansion_config()
        config["max_hops"] = 0

        results = expand_graph_from_seeds(
            [
                {"branch": "dense", "rank": 1, "raw_score": 0.8, "chunk_id": "c1", "document_uid": "d1"},
                {"branch": "dense", "rank": 2, "raw_score": 0.8, "chunk_id": "c2", "document_uid": "d2"},
                {"branch": "bm25", "rank": 1, "raw_score": 100.0, "chunk_id": "c1", "document_uid": "d1"},
            ],
            {"entity_ids": ["Q1"]},
            self.graph_fixture(),
            config,
        )

        e1 = next(row for row in results if row["chunk_id"] == "c1")
        seed_trace = e1["trace"]["seed_hits"]
        self.assertEqual({hit["branch"] for hit in seed_trace}, {"dense", "bm25"})
        self.assertEqual({hit["raw_score"] for hit in seed_trace}, {0.8, 100.0})
        self.assertEqual({hit["normalized_score"] for hit in seed_trace}, {1.0})
        self.assertEqual(e1["trace"]["components"]["seed_score"]["value"], 1.0)

    def test_oov_bm25_zero_score_is_excluded_before_graph_expansion(self):
        expand_graph_from_seeds = self.require_function("expand_graph_from_seeds")

        results = expand_graph_from_seeds(
            [{"branch": "bm25", "rank": 1, "raw_score": 0.0, "chunk_id": "c3", "document_uid": "d3"}],
            {"entity_ids": ["Q9"]},
            self.graph_fixture(),
            self.expansion_config(),
        )

        self.assertEqual(results, [])

    def test_positive_score_unknown_evidence_id_is_excluded_from_expansion(self):
        expand_graph_from_seeds = self.require_function("expand_graph_from_seeds")

        results = expand_graph_from_seeds(
            [{
                "branch": "dense",
                "rank": 1,
                "raw_score": 0.99,
                "evidence_chunk_id": "not-in-graph",
                "chunk_id": "c1",
                "document_uid": "d1",
            }],
            {"entity_ids": ["Q1"]},
            self.graph_fixture(),
            self.expansion_config(),
        )

        self.assertEqual(results, [])

    def test_none_or_empty_explicit_evidence_id_is_rejected_without_coordinate_fallback(self):
        expand_graph_from_seeds = self.require_function("expand_graph_from_seeds")

        for explicit_id in (None, ""):
            results = expand_graph_from_seeds(
                [{
                    "branch": "dense",
                    "rank": 1,
                    "raw_score": 0.99,
                    "evidence_chunk_id": explicit_id,
                    "chunk_id": "c1",
                    "document_uid": "d1",
                }],
                {"entity_ids": ["Q1"]},
                self.graph_fixture(),
                self.expansion_config(),
            )
            self.assertEqual(results, [], explicit_id)

    def test_absent_evidence_id_uses_valid_coordinate_fallback(self):
        expand_graph_from_seeds = self.require_function("expand_graph_from_seeds")

        results = expand_graph_from_seeds(
            [{
                "branch": "dense",
                "rank": 1,
                "raw_score": 0.99,
                "chunk_id": "c1",
                "document_uid": "d1",
            }],
            {"entity_ids": ["Q1"]},
            self.graph_fixture(),
            self.expansion_config(),
        )

        self.assertTrue(results)
        self.assertEqual(results[0]["chunk_id"], "c1")

    def test_two_hop_claim_evidence_traversal_preserves_path(self):
        expand_graph_from_seeds = self.require_function("expand_graph_from_seeds")

        results = expand_graph_from_seeds(
            [{"branch": "dense", "rank": 1, "raw_score": 0.9, "chunk_id": "c1", "document_uid": "d1"}],
            {"entity_ids": ["Q1"], "predicates": ["supports"]},
            self.graph_fixture(),
            self.expansion_config(),
        )
        related = next(row for row in results if row["chunk_id"] == "c2")

        self.assertEqual(related["hop"], 2)
        self.assertTrue(any(path["nodes"] == ["e1", "cl1", "e2"] for path in related["graph_paths"]))
        self.assertIn("evidenced_by", related["trace"]["relations"])

    def test_canonical_entity_and_predicate_change_rankings_without_alias_matching(self):
        expand_graph_from_seeds = self.require_function("expand_graph_from_seeds")
        hits = [
            {"branch": "dense", "rank": 1, "raw_score": 0.9, "chunk_id": "c1", "document_uid": "d1"},
            {"branch": "dense", "rank": 2, "raw_score": 0.9, "chunk_id": "c2", "document_uid": "d2"},
        ]
        graph = self.graph_fixture()

        supports = expand_graph_from_seeds(
            hits,
            {"entity_ids": ["Q1"], "predicates": ["supports"]},
            graph,
            self.expansion_config(),
        )
        denies = expand_graph_from_seeds(
            hits,
            {"entity_ids": ["Q1"], "predicates": ["denies"]},
            graph,
            self.expansion_config(),
        )

        self.assertEqual(supports[0]["chunk_id"], "c1")
        self.assertEqual(denies[0]["chunk_id"], "c2")
        self.assertNotEqual(supports[0]["trace"]["components"]["predicate"]["value"], 0)

    def test_polarity_modality_and_desired_stance_are_scored_components(self):
        expand_graph_from_seeds = self.require_function("expand_graph_from_seeds")
        results = expand_graph_from_seeds(
            [
                {"branch": "dense", "rank": 1, "raw_score": 0.9, "chunk_id": "c1", "document_uid": "d1"},
                {"branch": "dense", "rank": 2, "raw_score": 0.9, "chunk_id": "c2", "document_uid": "d2"},
            ],
            {
                "entity_ids": ["Q1"],
                "predicates": [],
                "polarities": ["negated"],
                "modalities": ["possible"],
                "desired_stances": ["refutes"],
            },
            self.graph_fixture(),
            self.expansion_config(),
        )

        self.assertEqual(results[0]["chunk_id"], "c2")
        components = results[0]["trace"]["components"]
        self.assertEqual(components["polarity"]["value"], 1.0)
        self.assertEqual(components["modality"]["value"], 1.0)
        self.assertEqual(components["stance"]["value"], 1.0)

    def test_rrf_preserves_all_branch_traces_paths_relations_and_stances(self):
        reciprocal_rank_fusion = self.require_function("reciprocal_rank_fusion")
        fused = reciprocal_rank_fusion(
            {
                "dense": [{
                    "evidence_chunk_id": "e1", "rank": 1, "raw_score": 0.9,
                    "graph_paths": [{"nodes": ["e1", "cl1"]}], "relation": "supports", "stance": "supports",
                    "source_trace": {"origin": "dense-path"},
                }],
                "bm25": [{
                    "evidence_chunk_id": "e1", "rank": 2, "raw_score": 0.4,
                    "graph_paths": [{"nodes": ["e1", "cl2"]}], "relation": "refutes", "stance": "refutes",
                    "source_trace": {"origin": "bm25-path"},
                }],
            },
            {"dense": 1.0, "bm25": 0.5},
            60,
        )

        row = fused[0]
        self.assertEqual(len(row["branch_traces"]), 2)
        self.assertEqual({item["branch"] for item in row["branch_traces"]}, {"dense", "bm25"})
        self.assertEqual({item["rank"] for item in row["branch_traces"]}, {1, 2})
        self.assertEqual({item["raw_score"] for item in row["branch_traces"]}, {0.9, 0.4})
        self.assertEqual(len(row["graph_paths"]), 2)
        self.assertEqual(set(row["stances"]), {"supports", "refutes"})
        dense_trace = next(item for item in row["branch_traces"] if item["branch"] == "dense")
        bm25_trace = next(item for item in row["branch_traces"] if item["branch"] == "bm25")
        self.assertEqual(dense_trace["graph_paths"], [{"nodes": ["e1", "cl1"]}])
        self.assertEqual(dense_trace["relations"], ["supports"])
        self.assertEqual(dense_trace["stances"], ["supports"])
        self.assertEqual(dense_trace["source_trace"], {"origin": "dense-path"})
        self.assertEqual(bm25_trace["graph_paths"], [{"nodes": ["e1", "cl2"]}])
        self.assertEqual(bm25_trace["source_trace"], {"origin": "bm25-path"})

    def test_rrf_deduplicates_duplicate_candidate_within_branch_before_contribution(self):
        reciprocal_rank_fusion = self.require_function("reciprocal_rank_fusion")

        fused = reciprocal_rank_fusion(
            {
                "dense": [
                    {"evidence_chunk_id": "e1", "rank": 3, "raw_score": 0.9},
                    {"evidence_chunk_id": "e1", "rank": 1, "raw_score": 0.4},
                    {"evidence_chunk_id": "e1", "rank": 1, "raw_score": 0.8},
                ]
            },
            {"dense": 1.0},
            60,
        )

        self.assertEqual(len(fused), 1)
        self.assertEqual(len(fused[0]["branch_traces"]), 1)
        self.assertEqual(fused[0]["branch_traces"][0]["rank"], 1)
        self.assertEqual(fused[0]["branch_traces"][0]["raw_score"], 0.8)
        self.assertEqual(fused[0]["rrf_score"], 1 / 61)

    def test_rrf_graph_branch_requires_finite_positive_numeric_graph_score(self):
        reciprocal_rank_fusion = self.require_function("reciprocal_rank_fusion")

        invalid_scores = [None, "1.0", True, float("nan"), float("inf"), 0.0, -1.0]
        for index, graph_score in enumerate(invalid_scores):
            fused = reciprocal_rank_fusion(
                {"graph": [{
                    "evidence_chunk_id": f"invalid-{index}",
                    "rank": 1,
                    "raw_score": 100.0,
                    "graph_score": graph_score,
                }]},
                {"graph": 1.0},
                60,
            )
            self.assertEqual(fused, [], graph_score)

        valid = reciprocal_rank_fusion(
            {"graph": [{
                "evidence_chunk_id": "valid",
                "rank": 1,
                "raw_score": 100.0,
                "graph_score": 0.01,
            }]},
            {"graph": 1.0},
            60,
        )
        self.assertEqual([row["candidate_id"] for row in valid], ["valid"])

    def test_rrf_accepts_only_exact_semantic_branch_names_and_matching_weights(self):
        reciprocal_rank_fusion = self.require_function("reciprocal_rank_fusion")
        hit = {"evidence_chunk_id": "e1", "rank": 1, "raw_score": 1.0}

        for branch in ("sparse", "vector", "retrieval", "unknown"):
            with self.assertRaises(ValueError):
                reciprocal_rank_fusion({branch: [hit]}, {branch: 1.0}, 60)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion({"dense": [hit]}, {"dense": 1.0, "other": 1.0}, 60)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion({"dense": [hit]}, {"bm25": 1.0}, 60)

    def test_expansion_rows_are_graph_branch_hits_consumable_by_rrf(self):
        expand_graph_from_seeds = self.require_function("expand_graph_from_seeds")
        reciprocal_rank_fusion = self.require_function("reciprocal_rank_fusion")

        expanded = expand_graph_from_seeds(
            [{"branch": "dense", "rank": 1, "raw_score": 0.9, "chunk_id": "c1", "document_uid": "d1"}],
            {"entity_ids": ["Q1"]},
            self.graph_fixture(),
            self.expansion_config(),
        )
        fused = reciprocal_rank_fusion({"graph": expanded}, {"graph": 1.0}, 60)

        self.assertTrue(expanded)
        self.assertEqual([row["rank"] for row in expanded], list(range(1, len(expanded) + 1)))
        self.assertTrue(all(row["branch"] == "graph" for row in expanded))
        self.assertTrue(all(row["raw_score"] == row["graph_score"] > 0 for row in expanded))
        self.assertEqual(len(fused), len(expanded))

    def test_reranker_uses_stable_sigmoid_and_abstains_below_threshold(self):
        select_evidence = self.require_function("select_evidence")
        config = self.expansion_config()
        selected = select_evidence(
            [{"evidence_chunk_id": "e1"}, {"evidence_chunk_id": "e2"}],
            {"e1": 1000.0, "e2": -1000.0},
            config,
        )
        abstained = select_evidence(
            [{"evidence_chunk_id": "e2"}], {"e2": -1000.0}, config
        )

        self.assertEqual([row["evidence_chunk_id"] for row in selected["selected"]], ["e1"])
        self.assertGreater(selected["selected"][0]["rerank_probability"], 0.99)
        self.assertTrue(abstained["abstained"])
        self.assertEqual(abstained["reason"], "all_candidates_below_minimum_rerank_probability")

    def test_empty_reranker_input_abstains_with_explicit_reason(self):
        select_evidence = self.require_function("select_evidence")

        result = select_evidence([], {}, self.expansion_config())

        self.assertEqual(result["selected"], [])
        self.assertTrue(result["abstained"])
        self.assertEqual(result["reason"], "no_candidates")


class Task5AcceptanceGateTests(unittest.TestCase):
    def require_core(self):
        self.assertIsNotNone(core, "work/mpkg_rag_core.py is not implemented")
        return core

    @unittest.skipUnless(
        resolve_real_corpus().is_dir() and (resolve_real_corpus() / "documents").is_dir(),
        "real corpus is not available",
    )
    def test_real_corpus_manifest_coverage_and_duplicate_src061_are_accounted_for(self):
        registry = self.require_core()
        rows = registry.load_source_registry(resolve_real_corpus())

        self.assertEqual(rows.file_records_before_deduplication, 90)
        self.assertEqual(len(rows), 90)
        self.assertTrue(all(row["document_uid"] for row in rows))
        self.assertTrue(all(row["content_sha256"] for row in rows))
        self.assertTrue(all(row["manifest_sources"] for row in rows))
        self.assertEqual(
            {source: sum(source in row["manifest_sources"] for row in rows)
             for source in ("source_manifest", "added_openalex_sources", "local_file_inventory")},
            {"source_manifest": 42, "added_openalex_sources": 40, "local_file_inventory": 50},
        )
        src061 = [row for row in rows if row["legacy_source_id"] == "SRC061"]
        self.assertEqual(len(src061), 2)
        self.assertEqual(len({row["document_uid"] for row in src061}), 2)
        self.assertEqual(len({row["content_sha256"] for row in src061}), 2)

    def test_structured_extraction_to_semantic_graph_to_rrf_to_abstention_is_end_to_end(self):
        registry = self.require_core()
        text = "The report says queer people are not dangerous."
        document_uid = "doc-acceptance-1"
        chunk_id = "chunk-acceptance-1"
        text_sha256 = hashlib.sha256(text.encode()).hexdigest()
        context = {
            "document_uid": document_uid,
            "chunk_id": chunk_id,
            "source_id": "SRC-ACCEPTANCE",
            "source_type": "official",
            "authority_score": 1.0,
            "factual_index_allowed": True,
            "status": "accepted",
            "text_sha256": text_sha256,
        }
        payload = SemanticClaimGraphTests().make_payload(
            text, polarity="negated", stance="refutes"
        )
        validated = registry.validate_extraction(payload, text, context)
        self.assertEqual(validated["status"], "accepted")
        self.assertEqual(len(validated["accepted"]), 1)

        chunk = {**context, "text": text}
        graph = registry.build_semantic_graph([chunk], [validated])
        self.assertEqual(len(graph["Document"]), 1)
        self.assertEqual(len(graph["EvidenceChunk"]), 1)
        self.assertEqual(len(graph["Claim"]), 1)
        self.assertTrue(any(edge["type"] == "has_subject" for edge in graph["edges"]))
        self.assertTrue(any(edge["type"] == "refutes" for edge in graph["edges"]))
        self.assertFalse(graph["quarantined"])

        retrieval_config = {
            "minimum_dense_score": 0.2,
            "minimum_graph_score": 0.0,
            "max_hops": 2,
            "hop_decay": 0.75,
            "minimum_rerank_probability": 0.8,
            "max_evidence": 2,
            "weights": {
                "query_entity": 2.0, "predicate": 1.0, "polarity": 0.5,
                "modality": 0.4, "stance": 0.8, "review_state": 0.8,
                "authority": 0.8, "extraction_confidence": 0.8,
                "seed_score": 1.0, "hop_decay": 0.6,
            },
            "review_state_scores": {"accepted": 1.0, "reviewed": 0.25, "unknown": 0.0},
        }
        evidence_id = graph["EvidenceChunk"][0]["evidence_chunk_id"]
        query = {
            "entity_ids": ["wikidata:Q1"],
            "predicates": ["is"],
            "polarities": ["negated"],
            "modalities": ["asserted"],
            "desired_stances": ["refutes"],
        }
        graph_hits = registry.expand_graph_from_seeds(
            [
                {"branch": "dense", "rank": 1, "raw_score": 0.9, "evidence_chunk_id": evidence_id},
                {"branch": "bm25", "rank": 1, "raw_score": 4.0, "evidence_chunk_id": evidence_id},
            ],
            query,
            graph,
            retrieval_config,
        )
        self.assertEqual(len(graph_hits), 1)
        self.assertGreater(graph_hits[0]["graph_score"], 0.0)
        self.assertEqual(graph_hits[0]["branch"], "graph")

        fused = registry.reciprocal_rank_fusion(
            {
                "dense": [{"evidence_chunk_id": evidence_id, "rank": 1, "raw_score": 0.9}],
                "bm25": [{"evidence_chunk_id": evidence_id, "rank": 1, "raw_score": 4.0}],
                "graph": graph_hits,
            },
            {"dense": 1.0, "bm25": 1.0, "graph": 0.8},
            60,
        )
        self.assertEqual(len(fused), 1)
        self.assertEqual({trace["branch"] for trace in fused[0]["branch_traces"]}, {"dense", "bm25", "graph"})
        selected = registry.select_evidence(
            fused,
            {evidence_id: 8.0},
            retrieval_config,
        )
        self.assertEqual([row["evidence_chunk_id"] for row in selected["selected"]], [evidence_id])
        self.assertFalse(selected["abstained"])

        abstained = registry.select_evidence(
            fused,
            {evidence_id: -8.0},
            retrieval_config,
        )
        self.assertTrue(abstained["abstained"])
        self.assertEqual(abstained["reason"], "all_candidates_below_minimum_rerank_probability")


class FinalFixEntityCatalogTests(unittest.TestCase):
    def require_core(self):
        self.assertIsNotNone(core)
        return core

    def test_mention_type_is_preserved_in_catalog_entity_type(self):
        registry = self.require_core()
        catalog = registry.build_entity_catalog(
            mention_spans=[{"text": "WHO", "mention_type": "organization", "document_uid": "d", "chunk_id": "c"}]
        )
        entity = next(row for row in catalog["entities"] if row["canonical_name"] == "WHO")
        self.assertEqual(entity["entity_type"], "organization")

    def test_closed_type_ontology_clusters_high_precision_acronyms_but_not_homonyms(self):
        registry = self.require_core()
        mentions = [
            {"text": "WHO", "mention_type": "org", "mention_id": "m1", "start": 0, "end": 3, "document_uid": "d", "chunk_id": "c"},
            {"text": "World Health Organization", "mention_type": "institution", "mention_id": "m2", "start": 4, "end": 31, "document_uid": "d", "chunk_id": "c"},
            {"text": "Jordan", "mention_type": "person", "mention_id": "m3", "start": 32, "end": 38, "document_uid": "d", "chunk_id": "c"},
            {"text": "Jordan", "mention_type": "place", "mention_id": "m4", "start": 39, "end": 45, "document_uid": "d", "chunk_id": "c"},
        ]
        catalog = registry.build_entity_catalog(mention_spans=mentions)
        who_rows = [row for row in catalog["entities"] if "WHO" in row["aliases"] or "World Health Organization" in row["aliases"]]
        self.assertEqual(len(who_rows), 1)
        self.assertEqual(set(who_rows[0]["aliases"]), {"WHO", "World Health Organization"})
        self.assertEqual(who_rows[0]["entity_type"], "organization")
        jordan_rows = [row for row in catalog["entities"] if row["canonical_name"] == "Jordan"]
        self.assertEqual(len(jordan_rows), 2)
        self.assertEqual({row["entity_type"] for row in jordan_rows}, {"person", "place"})
        self.assertTrue(all(row["link_status"] == "ambiguous" for row in jordan_rows))

    def test_catalog_aliases_and_hash_are_order_invariant_with_exact_provenance(self):
        registry = self.require_core()
        first = [
            {"text": "WHO", "mention_type": "organization", "mention_id": "m1", "start": 0, "end": 3, "document_uid": "d", "chunk_id": "c"},
            {"text": "World Health Organization", "mention_type": "organization", "mention_id": "m2", "start": 4, "end": 31, "document_uid": "d", "chunk_id": "c"},
        ]
        second = list(reversed(first))
        left = registry.build_entity_catalog(mention_spans=first)
        right = registry.build_entity_catalog(mention_spans=second)
        self.assertEqual(left, right)
        entity = next(row for row in left["entities"] if "WHO" in row["aliases"])
        self.assertEqual({item["document_uid"] for item in entity["provenance"]}, {"d"})
        self.assertEqual({item["chunk_id"] for item in entity["provenance"]}, {"c"})
        self.assertEqual({item["mention_id"] for item in entity["provenance"]}, {"m1", "m2"})

    def test_target_and_corpus_same_surface_keep_factual_entity_linkable_and_filter_candidates(self):
        registry = self.require_core()
        catalog = registry.build_entity_catalog(
            dataset_targets=["WHO"],
            mention_spans=[{"text": "WHO", "mention_type": "organization", "mention_id": "m1", "start": 0, "end": 3, "document_uid": "d", "chunk_id": "c"}],
        )
        corpus_rows = [row for row in catalog["entities"] if row["namespace"] == "corpus"]
        target_rows = [row for row in catalog["entities"] if row["namespace"] == "target"]
        self.assertEqual(len(corpus_rows), 1)
        self.assertEqual(len(target_rows), 1)
        self.assertTrue(corpus_rows[0]["retrieval_allowed"])
        self.assertEqual(corpus_rows[0]["link_status"], "linked")
        extraction = registry.build_entity_candidates("WHO", catalog, namespace_filter="corpus", factual_only=True)
        query = registry.build_entity_candidates("WHO", catalog, namespace_preference="corpus", allow_target_fallback=True)
        self.assertEqual([row["entity_id"] for row in extraction["candidates"]], [corpus_rows[0]["entity_id"]])
        self.assertEqual([row["namespace"] for row in extraction["candidates"]], ["corpus"])
        self.assertEqual([row["entity_id"] for row in query["candidates"]], [corpus_rows[0]["entity_id"]])
        resolved = registry.resolve_query_signature_entities({"entity_candidate_indices": [{"mention_id": "target", "candidate_index": 0}]}, {"target": query}, catalog)
        self.assertTrue(resolved["valid"])
        self.assertEqual(resolved["entity_ids"], [corpus_rows[0]["entity_id"]])
        self.assertNotEqual(extraction["candidate_set_hash"], registry.build_entity_candidates("WHO", catalog)["candidate_set_hash"])
        self.assertNotEqual(query["candidate_set_hash"], registry.build_entity_candidates("WHO", catalog, allow_target_fallback=True)["candidate_set_hash"])
        stale_policy = dict(query, namespace_preference=None)
        stale_resolution = registry.resolve_query_signature_entities({"entity_candidate_indices": [{"mention_id": "target", "candidate_index": 0}]}, {"target": stale_policy}, catalog)
        self.assertFalse(stale_resolution["valid"])

    def test_extraction_candidate_policy_cannot_select_target_anchor(self):
        registry = self.require_core()
        catalog = registry.build_entity_catalog(
            dataset_targets=["WHO"],
            mention_spans=[{"text": "WHO", "mention_type": "organization", "mention_id": "m1", "start": 0, "end": 3, "document_uid": "d", "chunk_id": "c"}],
        )
        candidate_set = registry.build_entity_candidates("WHO", catalog, namespace_filter="corpus", factual_only=True)
        target_id = next(row["entity_id"] for row in catalog["entities"] if row["namespace"] == "target")
        forged = dict(candidate_set, candidates=[{**candidate_set["candidates"][0], "entity_id": target_id, "namespace": "target"}])
        payload = {"schema_version": "semantic-claims.v1", "claims": [{"mentions": [{"mention_id": "m", "text": "WHO", "start": 0, "end": 3, "candidate_index": 0, "entity_status": "canonical", "canonical_name": "WHO"}], "subject": {"mention_id": "m"}, "predicate": "is", "object": {"value": "an organization", "value_type": "string"}, "polarity": "affirmed", "modality": "asserted", "attribution": None, "evidence_stance": "supports", "model_confidence": 0.8}]}
        result = registry.validate_extraction(payload, "WHO is an organization", {"document_uid": "d", "chunk_id": "c", "entity_catalog": catalog, "candidate_sets": {"m": forged}})
        self.assertTrue(any("candidate_set_hash_mismatch" in item["reasons"] or "candidate_namespace_policy_violation" in item["reasons"] or "candidate_set_authentication_failed" in item["reasons"] for item in result["quarantined"]))

    def test_query_and_extraction_authenticate_full_candidate_rows_not_only_hash(self):
        registry = self.require_core()
        catalog = registry.build_entity_catalog(
            dataset_targets=["WHO"],
            mention_spans=[
                {"text": "WHO", "mention_type": "organization", "mention_id": "m1", "start": 0, "end": 3, "document_uid": "d", "chunk_id": "c"},
                {"text": "UNICEF", "mention_type": "organization", "mention_id": "m2", "start": 4, "end": 10, "document_uid": "d", "chunk_id": "c"},
            ],
        )
        who = registry.build_entity_candidates("WHO", catalog)
        unicef = registry.build_entity_candidates("UNICEF", catalog)["candidates"][0]
        swapped = dict(who, candidates=[unicef, *who["candidates"][1:]])
        query_payload = {"entity_candidate_indices": [{"mention_id": "target", "candidate_index": 0}]}
        query_result = registry.resolve_query_signature_entities(query_payload, {"target": swapped}, catalog)
        self.assertFalse(query_result["valid"])
        self.assertIn("query_candidate_set_authentication_failed", query_result["reasons"])

        extraction_payload = {"schema_version": "semantic-claims.v1", "claims": [{"mentions": [{"mention_id": "m", "text": "WHO", "start": 0, "end": 3, "candidate_index": 0, "entity_status": "canonical", "canonical_name": "WHO"}], "subject": {"mention_id": "m"}, "predicate": "is", "object": {"value": "an organization", "value_type": "string"}, "polarity": "affirmed", "modality": "asserted", "attribution": None, "evidence_stance": "supports", "model_confidence": 0.8}]}
        extraction_result = registry.validate_extraction(extraction_payload, "WHO is an organization", {"document_uid": "d", "chunk_id": "c", "entity_catalog": catalog, "candidate_sets": {"m": swapped}})
        self.assertTrue(any("candidate_set_authentication_failed" in item["reasons"] for item in extraction_result["quarantined"]))

        tampered = dict(who, candidates=[{**who["candidates"][0], "canonical_name": "tampered"}, *who["candidates"][1:]])
        reordered = dict(who, candidates=list(reversed(who["candidates"])))
        for candidate_set in (tampered, reordered):
            result = registry.resolve_query_signature_entities(query_payload, {"target": candidate_set}, catalog)
            self.assertFalse(result["valid"])
            self.assertIn("query_candidate_set_authentication_failed", result["reasons"])
        malformed_policy = dict(who, max_candidates="eight")
        malformed_result = registry.resolve_query_signature_entities(query_payload, {"target": malformed_policy}, catalog)
        self.assertFalse(malformed_result["valid"])
        self.assertIn("query_candidate_set_authentication_failed", malformed_result["reasons"])

    def test_ambiguous_same_type_acronym_expansions_do_not_transitively_merge(self):
        registry = self.require_core()
        mentions = [
            {"text": "ABC", "mention_type": "organization", "mention_id": "m1", "start": 0, "end": 3, "document_uid": "d", "chunk_id": "c"},
            {"text": "Alpha Beta Council", "mention_type": "organization", "mention_id": "m2", "start": 4, "end": 22, "document_uid": "d", "chunk_id": "c"},
            {"text": "Applied Biology Center", "mention_type": "organization", "mention_id": "m3", "start": 23, "end": 46, "document_uid": "d", "chunk_id": "c"},
        ]
        catalog = registry.build_entity_catalog(mention_spans=mentions)
        abc_rows = [row for row in catalog["entities"] if "ABC" in row["aliases"]]
        self.assertEqual(len(abc_rows), 1)
        self.assertEqual(set(abc_rows[0]["aliases"]), {"ABC"})
        self.assertFalse(abc_rows[0]["retrieval_allowed"])
        self.assertEqual(len(catalog["entities"]), 3)
        self.assertEqual(len(registry.build_entity_candidates("ABC", catalog)["candidates"]), 0)

    def test_punctuated_lowercase_and_unknown_type_acronyms_link_only_to_unique_expansions(self):
        registry = self.require_core()
        mentions = [
            {"text": "U.N.", "mention_type": "acronym", "mention_id": "m1", "start": 0, "end": 4, "document_uid": "d", "chunk_id": "c"},
            {"text": "United Nations", "mention_type": "organization", "mention_id": "m2", "start": 5, "end": 19, "document_uid": "d", "chunk_id": "c"},
            {"text": "who", "mention_type": "acronym", "mention_id": "m3", "start": 20, "end": 23, "document_uid": "d", "chunk_id": "c"},
            {"text": "World Health Organization", "mention_type": "organization", "mention_id": "m4", "start": 24, "end": 51, "document_uid": "d", "chunk_id": "c"},
        ]
        catalog = registry.build_entity_catalog(mention_spans=mentions)
        united = [row for row in catalog["entities"] if "U.N." in row["aliases"]]
        who = [row for row in catalog["entities"] if "who" in row["aliases"]]
        self.assertEqual(len(united), 1)
        self.assertEqual(set(united[0]["aliases"]), {"U.N.", "United Nations"})
        self.assertEqual(united[0]["entity_type"], "organization")
        self.assertEqual(len(who), 1)
        self.assertEqual(set(who[0]["aliases"]), {"who", "World Health Organization"})
        self.assertEqual(who[0]["entity_type"], "organization")

    def test_punctuated_acronym_with_multiple_expansions_is_ambiguous(self):
        registry = self.require_core()
        catalog = registry.build_entity_catalog(mention_spans=[
            {"text": "U.N.", "mention_type": "organization", "mention_id": "m1"},
            {"text": "United Nations", "mention_type": "organization", "mention_id": "m2"},
            {"text": "Universal Network", "mention_type": "organization", "mention_id": "m3"},
        ])
        row = next(item for item in catalog["entities"] if "U.N." in item["aliases"])
        self.assertEqual(row["aliases"], ["U.N."])
        self.assertEqual(row["catalog_status"], "ambiguous")
        self.assertEqual(row["link_status"], "ambiguous")
        self.assertFalse(row["retrieval_allowed"])

    def test_closed_catalog_assigns_deterministic_internal_ids_and_rejects_alias_collisions(self):
        registry = self.require_core()
        catalog = registry.build_entity_catalog(
            corpus_rows=[
                {"document_uid": "d1", "manifest_metadata": {"organisation": "WHO"}},
                {"document_uid": "d2", "manifest_metadata": {"organisation": "WHO"}},
            ],
            dataset_targets=["WHO guidance"],
            mention_spans=[
                {"text": "WHO", "entity_type": "organization"},
                {"text": "World Health Organization", "entity_type": "organization"},
                {"text": "WHO", "entity_type": "acronym"},
            ],
        )
        self.assertTrue(catalog["catalog_revision"])
        rows = catalog["entities"]
        self.assertTrue(any(row["entity_id"].startswith("ENT:") for row in rows))
        self.assertTrue(any(row["entity_id"].startswith("TGT:") for row in rows))
        self.assertEqual(catalog, registry.build_entity_catalog(
            corpus_rows=[
                {"document_uid": "d2", "manifest_metadata": {"organisation": "WHO"}},
                {"document_uid": "d1", "manifest_metadata": {"organisation": "WHO"}},
            ],
            dataset_targets=["WHO guidance"],
            mention_spans=[
                {"text": "World Health Organization", "entity_type": "organization"},
                {"text": "WHO", "entity_type": "organization"},
                {"text": "WHO", "entity_type": "acronym"},
            ],
        ))
        who_entities = [row for row in rows if "WHO" in row["aliases"]]
        self.assertEqual(len(who_entities), 1)
        self.assertIn("World Health Organization", who_entities[0]["aliases"])

    def test_candidate_indices_are_the_only_model_link_and_raw_ids_are_rejected(self):
        registry = self.require_core()
        catalog = registry.build_entity_catalog(mention_spans=[{"text": "queer people", "entity_type": "group"}])
        entity = next(row for row in catalog["entities"] if row["entity_id"].startswith("ENT:"))
        candidate_set = registry.build_entity_candidates("queer people", catalog)
        text = "queer people are dangerous"
        payload = {
            "schema_version": "semantic-claims.v1",
            "claims": [{
                "mentions": [{
                    "mention_id": "m1", "text": "queer people", "start": 0, "end": 12,
                    "candidate_index": 0, "entity_status": "canonical", "canonical_name": "queer people",
                }, {
                    "mention_id": "m2", "text": "dangerous", "start": 17, "end": 26,
                    "candidate_index": None, "entity_status": "nil", "canonical_name": "dangerous",
                }],
                "subject": {"mention_id": "m1"}, "predicate": "is", "object": {"mention_id": "m2"},
                "polarity": "affirmed", "modality": "asserted", "attribution": None,
                "evidence_stance": "supports", "model_confidence": 0.9,
            }],
        }
        context = {
            "document_uid": "d", "chunk_id": "c", "entity_catalog": catalog,
            "candidate_sets": {"m1": candidate_set, "m2": registry.build_entity_candidates("dangerous", catalog)},
        }
        accepted = registry.validate_extraction(payload, text, context)
        self.assertEqual(accepted["accepted"], [])
        self.assertTrue(any("unresolved_entity" in item["reasons"] for item in accepted["reviewed"]))
        self.assertEqual(accepted["reviewed"][0]["record"]["mentions"][0]["entity_id"], entity["entity_id"])

        raw_id_payload = json.loads(json.dumps(payload))
        raw_id_payload["claims"][0]["mentions"][0].pop("candidate_index")
        raw_id_payload["claims"][0]["mentions"][0]["entity_id"] = "Q999999"
        rejected = registry.validate_extraction(raw_id_payload, text, context)
        self.assertTrue(any("raw_entity_id_forbidden" in item["reasons"] for item in rejected["quarantined"]))

    def test_unknown_candidate_index_and_stale_candidate_hash_are_quarantined(self):
        registry = self.require_core()
        catalog = registry.build_entity_catalog(mention_spans=[{"text": "WHO", "entity_type": "organization"}])
        candidate_set = registry.build_entity_candidates("WHO", catalog)
        text = "WHO guidance"
        payload = {
            "schema_version": "semantic-claims.v1", "claims": [{
                "mentions": [{"mention_id": "m", "text": "WHO", "start": 0, "end": 3,
                               "candidate_index": 99, "entity_status": "canonical", "canonical_name": "WHO"}],
                "subject": {"mention_id": "m"}, "predicate": "describes", "object": {"value": "guidance", "value_type": "string"},
                "polarity": "affirmed", "modality": "asserted", "attribution": None,
                "evidence_stance": "supports", "model_confidence": 0.8,
            }]}
        bad_index = registry.validate_extraction(payload, text, {"document_uid": "d", "chunk_id": "c", "entity_catalog": catalog, "candidate_sets": {"m": candidate_set}})
        self.assertTrue(any("candidate_index_unknown" in item["reasons"] for item in bad_index["quarantined"]))
        stale = dict(candidate_set); stale["candidate_set_hash"] = "stale"
        stale_result = registry.validate_extraction(payload, text, {"document_uid": "d", "chunk_id": "c", "entity_catalog": catalog, "candidate_sets": {"m": stale}})
        self.assertTrue(any("candidate_set_hash_mismatch" in item["reasons"] for item in stale_result["quarantined"]))

    def test_claimless_seed_is_not_relabelled_as_graph_evidence(self):
        registry = self.require_core()
        graph = {"EvidenceChunk": [{"evidence_chunk_id": "e", "document_uid": "d", "chunk_id": "c", "status": "accepted", "authority_score": 1.0}], "Document": [{"document_uid": "d", "authority_score": 1.0}], "Claim": [], "Entity": [], "edges": []}
        config = {"max_hops": 2, "minimum_dense_score": 0.1, "minimum_graph_score": 0.0, "hop_decay": 0.75, "minimum_rerank_probability": 0.5, "max_evidence": 2, "weights": {key: 1.0 for key in ("query_entity", "predicate", "polarity", "modality", "stance", "review_state", "authority", "extraction_confidence", "seed_score", "hop_decay")}, "review_state_scores": {"accepted": 1.0, "reviewed": 0.5, "unknown": 0.0}}
        self.assertEqual(registry.expand_graph_from_seeds([{"branch": "dense", "rank": 1, "raw_score": 0.9, "evidence_chunk_id": "e"}], {"entity_ids": ["ENT:x"]}, graph, config), [])

    def _hop_zero_claim_probe(self, claim, query):
        registry = self.require_core()
        entity_ids = {claim["subject_entity_id"], claim["object_entity_id"]}
        graph = {
            "EvidenceChunk": [{"evidence_chunk_id": "e", "document_uid": "d", "chunk_id": "c", "status": "accepted", "authority_score": 1.0}],
            "Document": [{"document_uid": "d", "authority_score": 1.0}],
            "Claim": [claim],
            "Entity": [{"entity_id": entity_id} for entity_id in sorted(entity_ids)],
            "edges": [
                {"source_id": "e", "type": "supports", "target_id": "cl"},
                {"source_id": "cl", "type": "has_subject", "target_id": claim["subject_entity_id"], "target_type": "entity"},
                {"source_id": "cl", "type": "has_object", "target_id": claim["object_entity_id"], "target_type": "entity"},
                {"source_id": "cl", "type": "evidenced_by", "target_id": "e"},
            ],
        }
        config = {"max_hops": 1, "minimum_dense_score": 0.1, "minimum_graph_score": 0.0, "hop_decay": 0.75, "minimum_rerank_probability": 0.5, "max_evidence": 2, "weights": {key: 1.0 for key in ("query_entity", "predicate", "polarity", "modality", "stance", "review_state", "authority", "extraction_confidence", "seed_score", "hop_decay")}, "review_state_scores": {"accepted": 1.0, "reviewed": 0.5, "unknown": 0.0}}
        return registry.expand_graph_from_seeds([{"branch": "dense", "rank": 1, "raw_score": 0.9, "evidence_chunk_id": "e"}], query, graph, config)

    def test_hop_zero_polarity_only_match_is_rejected(self):
        claim = {"claim_id": "cl", "subject_entity_id": "ENT:other", "object_entity_id": "ENT:object", "predicate": "unrelated", "polarity": "affirmed", "modality": "possible", "stance": "refutes", "model_confidence": 1.0, "review_status": "accepted"}
        query = {"entity_ids": ["ENT:query"], "predicates": ["supports"], "polarities": ["affirmed"], "modalities": ["asserted"]}
        self.assertEqual(self._hop_zero_claim_probe(claim, query), [])

    def test_hop_zero_modality_only_match_is_rejected(self):
        claim = {"claim_id": "cl", "subject_entity_id": "ENT:other", "object_entity_id": "ENT:object", "predicate": "unrelated", "polarity": "negated", "modality": "asserted", "stance": "refutes", "model_confidence": 1.0, "review_status": "accepted"}
        query = {"entity_ids": ["ENT:query"], "predicates": ["supports"], "polarities": ["affirmed"], "modalities": ["asserted"]}
        self.assertEqual(self._hop_zero_claim_probe(claim, query), [])

    def test_hop_zero_entity_only_match_is_accepted(self):
        claim = {"claim_id": "cl", "subject_entity_id": "ENT:query", "object_entity_id": "ENT:object", "predicate": "unrelated", "polarity": "negated", "modality": "possible", "stance": "refutes", "model_confidence": 1.0, "review_status": "accepted"}
        query = {"entity_ids": ["ENT:query"], "predicates": ["supports"], "polarities": ["affirmed"], "modalities": ["asserted"]}
        self.assertTrue(self._hop_zero_claim_probe(claim, query))

    def test_hop_zero_predicate_only_match_is_accepted(self):
        claim = {"claim_id": "cl", "subject_entity_id": "ENT:other", "object_entity_id": "ENT:object", "predicate": "supports", "polarity": "negated", "modality": "possible", "stance": "refutes", "model_confidence": 1.0, "review_status": "accepted"}
        query = {"entity_ids": ["ENT:query"], "predicates": ["supports"], "polarities": ["affirmed"], "modalities": ["asserted"]}
        self.assertTrue(self._hop_zero_claim_probe(claim, query))

    def test_hop_zero_requires_semantic_query_compatibility(self):
        registry = self.require_core()
        graph = {
            "EvidenceChunk": [{"evidence_chunk_id": "e", "document_uid": "d", "chunk_id": "c", "status": "accepted", "authority_score": 1.0}],
            "Document": [{"document_uid": "d", "authority_score": 1.0}],
            "Claim": [{"claim_id": "cl", "subject_entity_id": "ENT:other", "object_entity_id": "ENT:object", "predicate": "unrelated", "polarity": "negated", "modality": "possible", "stance": "refutes", "model_confidence": 1.0, "review_status": "accepted"}],
            "Entity": [{"entity_id": "ENT:other"}, {"entity_id": "ENT:object"}],
            "edges": [
                {"source_id": "e", "type": "supports", "target_id": "cl"},
                {"source_id": "cl", "type": "has_subject", "target_id": "ENT:other", "target_type": "entity"},
                {"source_id": "cl", "type": "has_object", "target_id": "ENT:object", "target_type": "entity"},
                {"source_id": "cl", "type": "evidenced_by", "target_id": "e"},
            ],
        }
        config = {"max_hops": 1, "minimum_dense_score": 0.1, "minimum_graph_score": 0.0, "hop_decay": 0.75, "minimum_rerank_probability": 0.5, "max_evidence": 2, "weights": {key: 1.0 for key in ("query_entity", "predicate", "polarity", "modality", "stance", "review_state", "authority", "extraction_confidence", "seed_score", "hop_decay")}, "review_state_scores": {"accepted": 1.0, "reviewed": 0.5, "unknown": 0.0}}
        self.assertEqual(registry.expand_graph_from_seeds([{"branch": "dense", "rank": 1, "raw_score": 0.9, "evidence_chunk_id": "e"}], {"entity_ids": ["ENT:query"], "predicates": ["supports"], "polarities": ["affirmed"], "modalities": ["asserted"]}, graph, config), [])


if __name__ == "__main__":
    unittest.main()
