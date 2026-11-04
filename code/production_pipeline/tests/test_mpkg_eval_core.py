import math
import hashlib
import sys
from decimal import Decimal
from pathlib import Path

import pytest


WORK = Path(__file__).resolve().parents[1] / "work"
sys.path.insert(0, str(WORK))

from mpkg_eval_core import (  # noqa: E402
    BoundedLRU,
    benjamini_hochberg,
    bootstrap_mean_ci,
    aggregate_citation_support,
    build_claim_citation_records,
    filter_audit,
    filter_rows_by_category,
    quarantine_missing_references,
    derive_shard_run_name,
    derive_effective_cache_capacity,
    fit_adaptive_prompt_with_evidence,
    fit_prompt_to_budget,
    identity_config,
    normalize_category,
    krippendorff_alpha_ordinal,
    language_match,
    normalize_optional_text,
    paired_variant_tests,
    script_bucket,
    script_match,
    sentence_aligned_windows,
    unicode_chrf,
    weighted_kappa_rows,
    validate_evidence_ledger,
)


def test_multilingual_record_metrics_are_explicit_and_detoxify_is_english_only():
    from mpkg_eval_core import evaluate_multilingual_record

    calls = []
    result = evaluate_multilingual_record(
        {"ID": "ta-1", "Text": "தமிழ் பதிவு", "Counter Narrative": "தமிழ் பதில்", "parsed_counter_narrative": "தமிழ் பதில்", "language": "ta"},
        language_detector=lambda text: "ta",
        detoxify_scorer=lambda text: calls.append(text) or 0.2,
        bertscore_scorer=lambda **kwargs: {"f1": 0.7},
    )
    assert calls == []
    assert result["metrics"]["chrf"]["status"] == "scored"
    assert result["metrics"]["bertscore"]["status"] == "scored"
    assert result["metrics"]["detoxify"]["value"] is None
    assert result["metrics"]["detoxify"]["status"] == "unsupported_language"
    assert result["metrics"]["detoxify"]["reason"]


def test_multilingual_record_metrics_quarantines_missing_reference_and_language_unsupported_metrics():
    from mpkg_eval_core import evaluate_multilingual_record

    result = evaluate_multilingual_record(
        {"ID": "x", "Text": "हिन्दी", "Counter Narrative": float("nan"), "parsed_counter_narrative": "उत्तर", "language": "hi"},
        language_detector=lambda text: "hi",
        bertscore_scorer=None,
    )
    assert result["reference_available"] is False
    assert result["metrics"]["chrf"]["value"] is None
    assert result["metrics"]["chrf"]["status"] == "excluded_missing_reference"
    assert result["metrics"]["bertscore"]["status"] == "excluded_missing_reference"
    assert result["metrics"]["bertscore"]["reason"] == "missing_reference"


def test_pairwise_family_includes_mp_vs_kg_and_corrects_all_declared_metric_comparisons():
    from mpkg_eval_core import pairwise_metric_family

    rows = [
        {"ID": "a", "variant": "mp_kg_rag", "chrf": 0.9, "bertscore": 0.8},
        {"ID": "a", "variant": "kg_rag", "chrf": 0.7, "bertscore": 0.8},
        {"ID": "b", "variant": "mp_kg_rag", "chrf": 0.6, "bertscore": 0.7},
        {"ID": "b", "variant": "kg_rag", "chrf": 0.8, "bertscore": 0.6},
    ]
    result = pairwise_metric_family(rows, metrics=("chrf", "bertscore"), comparisons=(("mp_kg_rag", "kg_rag"),), seed=4, permutations=100)
    assert result["comparisons"][0]["comparison"] == "mp_kg_rag_vs_kg_rag"
    assert result["comparisons"][0]["metrics"]["chrf"]["win_rate"] == pytest.approx(0.5)
    assert result["bh_family_size"] == 2
    assert all("q_value" in metric for metric in result["comparisons"][0]["metrics"].values())


def test_annotation_sampling_is_seeded_bounded_and_keeps_all_variants_per_id():
    from mpkg_eval_core import sample_annotation_records

    rows = [{"ID": str(i), "variant": variant, "stratify_key": "ta" if i < 3 else "en"} for i in range(6) for variant in ("mp_kg_rag", "kg_rag")]
    sample = sample_annotation_records(rows, max_ids=2, seed=7, stratify_key="stratify_key")
    assert sample["selected_id_count"] <= 2
    assert len(sample["rows"]) == sample["selected_id_count"] * 2
    assert set(row["ID"] for row in sample["rows"]) == set(sample["selected_ids"])
    assert sample == sample_annotation_records(rows, max_ids=2, seed=7, stratify_key="stratify_key")


def test_human_agreement_requires_two_distinct_raters_and_reports_overlap_missingness():
    from mpkg_eval_core import human_agreement_report

    with pytest.raises(ValueError, match="two distinct raters"):
        human_agreement_report([{"ID": "a", "rater": "r1", "score": 1}])
    report = human_agreement_report([
        {"ID": "a", "rater": "r1", "score": 1}, {"ID": "a", "rater": "r2", "score": 2},
        {"ID": "b", "rater": "r1", "score": 2}, {"ID": "b", "rater": "r2", "score": 2},
        {"ID": "c", "rater": "r1", "score": 1},
    ])
    assert report["distinct_raters"] == 2
    assert report["overlap_count"] == 2
    assert report["missingness_count"] == 1
    assert "weighted_kappa" in report and "krippendorff_alpha_ordinal" in report


def test_detector_mismatch_blocks_all_metric_callbacks_even_when_record_label_claims_english():
    from mpkg_eval_core import evaluate_multilingual_record
    calls = []
    result = evaluate_multilingual_record(
        {"ID": "m", "Text": "தமிழ் பதிவு", "Counter Narrative": "தமிழ் பதில்", "parsed_counter_narrative": "தமிழ் பதில்", "language": "en"},
        language_detector=lambda text: "ta",
        rouge_scorer=lambda *args: calls.append("rouge") or 1.0,
        meteor_scorer=lambda *args: calls.append("meteor") or 1.0,
        detoxify_scorer=lambda *args: calls.append("detoxify") or 1.0,
    )
    assert calls == []
    assert result["language_status"] == "mismatch"
    assert result["metrics"]["rouge_l"]["status"] == "language_mismatch"


def test_pairwise_directions_make_lower_toxicity_a_win_and_exclude_zero_pairs_from_bh():
    from mpkg_eval_core import pairwise_metric_family
    rows = [
        {"ID": "a", "variant": "mp_kg_rag", "chrf": 0.4, "detoxify": 0.1},
        {"ID": "a", "variant": "kg_rag", "chrf": 0.2, "detoxify": 0.3},
    ]
    result = pairwise_metric_family(rows, metrics=("chrf", "detoxify"), comparisons=(("mp_kg_rag", "kg_rag"),), directions={"detoxify": "lower"})
    assert result["comparisons"][0]["metrics"]["detoxify"]["win_rate"] == 1.0
    empty = pairwise_metric_family([], metrics=("chrf",), comparisons=(("mp_kg_rag", "kg_rag"),))
    metric = empty["comparisons"][0]["metrics"]["chrf"]
    assert metric["n"] == 0 and metric["p_value"] is None and metric["q_value"] is None


def test_pairwise_family_includes_real_citation_support_metrics_and_excludes_non_rag_nulls():
    from mpkg_eval_core import pairwise_metric_family

    rows = [
        {"ID": "a", "variant": "mp_kg_rag", "citation_precision": 0.95, "citation_recall": 1.0, "citation_necessity": 0.8, "citation_entailment": 0.9},
        {"ID": "a", "variant": "kg_rag", "citation_precision": 0.65, "citation_recall": 0.5, "citation_necessity": 0.4, "citation_entailment": 0.6},
        {"ID": "a", "variant": "qwen_zero_shot", "citation_precision": None, "citation_recall": None, "citation_necessity": None, "citation_entailment": None,
         "citation_precision_status": "not_applicable_non_rag",
         "citation_recall_status": "not_applicable_non_rag", "citation_necessity_status": "not_applicable_non_rag", "citation_entailment_status": "not_applicable_non_rag"},
    ]
    result = pairwise_metric_family(rows, metrics=("citation_precision", "citation_recall", "citation_necessity", "citation_entailment"), comparisons=(("mp_kg_rag", "kg_rag"), ("mp_kg_rag", "qwen_zero_shot")), permutations=10)
    metrics = result["comparisons"][0]["metrics"]
    assert set(metrics) == {"citation_precision", "citation_recall", "citation_necessity", "citation_entailment"}
    assert all(metrics[name]["n"] == 1 for name in metrics)
    assert metrics["citation_precision"]["mean_difference"] == pytest.approx(0.30)
    non_rag = result["comparisons"][1]["metrics"]["citation_recall"]
    assert non_rag["n"] == 0 and non_rag["p_value"] is None and non_rag["q_value"] is None
    assert result["bh_family_size"] == 4


def test_nli_calibration_artifact_is_authenticated_to_model_data_language_and_examples():
    from mpkg_eval_core import build_nli_calibration_artifact, verify_nli_calibration_artifact
    artifact = build_nli_calibration_artifact(
        model_id="nli/model", model_revision="model-rev", dataset_id="indicxnli", dataset_revision="data-rev",
        language="ta", split="validation", label_mapping={"entailment": 2}, threshold=0.7,
        n_examples=2, example_ids=["x", "y"], example_content_digest="digest", accuracy=0.9,
        per_label_stats={"entailment": {"n": 2}}, code_hash="code", eval_core_hash="core",
    )
    assert verify_nli_calibration_artifact(artifact, model_id="nli/model", model_revision="model-rev", dataset_id="indicxnli", dataset_revision="data-rev", language="ta")["enabled"]
    forged = dict(artifact, threshold=0.1)
    assert verify_nli_calibration_artifact(forged, model_id="nli/model", model_revision="model-rev", dataset_id="indicxnli", dataset_revision="data-rev", language="ta")["enabled"] is False


def test_ledger_rejects_duplicate_evidence_ids():
    from mpkg_eval_core import validate_evidence_ledger
    item = _citation_ledger()[0]
    with pytest.raises(ValueError, match="duplicate evidence_id"):
        validate_evidence_ledger([item, dict(item)])


def test_calibration_loader_requires_explicit_expected_provenance_and_rejects_forged_model():
    from mpkg_eval_core import load_nli_runtime_evaluator, build_nli_calibration_artifact
    artifact = build_nli_calibration_artifact(model_id='m', model_revision='mr', dataset_id='d', dataset_revision='dr', language='en', split='validation', label_mapping={'entailment': 1}, threshold=.5, n_examples=1, example_ids=['a'], example_content_digest='x', accuracy=.8, per_label_stats={'entailment': {'n': 1}}, code_hash='c', eval_core_hash='e')
    calls = []
    out = load_nli_runtime_evaluator('en', calibration_artifact={'en': artifact}, calibration_provenance={'en': {'model_id': 'wrong', 'model_revision': 'mr', 'dataset_id': 'd', 'dataset_revision': 'dr', 'split': 'validation', 'code_hash': 'c', 'eval_core_hash': 'e'}}, model_loader=lambda **kwargs: calls.append(kwargs))
    assert out['enabled'] is False and calls == []


def test_metric_rows_reject_duplicate_id_variant_and_reference_audit_reports_each_metric_status():
    from mpkg_eval_core import validate_metric_rows_unique, build_reference_metric_audit
    with pytest.raises(ValueError, match='duplicate.*ID.*variant'):
        validate_metric_rows_unique([{'ID': 'a', 'variant': 'mp_kg_rag'}, {'ID': 'a', 'variant': 'mp_kg_rag'}])
    audit = build_reference_metric_audit([
        {'ID': 'a', 'variant': 'mp_kg_rag', 'reference_available': True, 'language': 'en', 'chrf': 1, 'chrf_status': 'scored'},
        {'ID': 'b', 'variant': 'mp_kg_rag', 'reference_available': False, 'language': 'ta', 'chrf': None, 'chrf_status': 'excluded_missing_reference'},
    ], metrics=('chrf',))
    assert audit['chrf']['input'] == 2 and audit['chrf']['scorable'] == 1
    assert audit['chrf']['excluded_by_reason']['missing_reference'] == 1


def test_nli_and_heldout_provenance_are_real_pinned_revisions_for_all_languages():
    from mpkg_eval_core import NLI_MODEL_ID, NLI_MODEL_REVISION, NLI_DATASET_PROVENANCE
    assert NLI_MODEL_ID == 'joeddav/xlm-roberta-large-xnli'
    assert NLI_MODEL_REVISION == '07f8772bf0306314a97e4913cafde2cabf9814a9'
    assert NLI_DATASET_PROVENANCE['en'] == {'dataset_id': 'facebook/xnli', 'dataset_revision': '072e4eb2b447bd887a772a7ab826ce0a7222b782', 'config': 'en', 'split': 'validation'}
    for language in ('hi', 'ta'):
        assert NLI_DATASET_PROVENANCE[language] == {
            'dataset_id': 'mteb/IndicXnliPairClassification',
            'dataset_revision': '027e97b9afe84ea3447b57b7705b8864bb2b3a83',
            'config': language,
            'split': 'test',
            'source_format': 'parquet',
            'columns': {'sentence1': 'premise', 'sentence2': 'hypothesis', 'labels': 'label'},
        }


def test_indicxnli_loader_is_data_only_and_adapts_parquet_columns_with_provenance_hash():
    from mpkg_eval_core import NLI_DATASET_PROVENANCE, load_nli_dataset_rows

    calls = []

    def fake_loader(dataset_id, config, *, split, revision):
        calls.append((dataset_id, config, split, revision))
        return [
            {'sentence1': 'premise', 'sentence2': 'hypothesis', 'labels': 0},
            {'sentence1': 'दावा', 'sentence2': 'उत्तर', 'labels': 1},
        ]

    rows, provenance = load_nli_dataset_rows(fake_loader, NLI_DATASET_PROVENANCE['hi'])
    assert calls == [('mteb/IndicXnliPairClassification', 'hi', 'test', '027e97b9afe84ea3447b57b7705b8864bb2b3a83')]
    assert rows == [
        {'id': '0', 'premise': 'premise', 'hypothesis': 'hypothesis', 'label': 0},
        {'id': '1', 'premise': 'दावा', 'hypothesis': 'उत्तर', 'label': 1},
    ]
    assert provenance['source_format'] == 'parquet'
    assert provenance['dataset_content_hash']
    assert provenance['dataset_content_hash'] == provenance['selected_content_hash']


def test_indicxnli_loader_rejects_legacy_script_repository_and_non_parquet_schema():
    from mpkg_eval_core import load_nli_dataset_rows

    with pytest.raises(ValueError, match='legacy|data-only'):
        load_nli_dataset_rows(lambda *args, **kwargs: [], {
            'dataset_id': 'Divyanshu/indicxnli', 'dataset_revision': 'a' * 40,
            'config': 'hi', 'split': 'test',
        })
    with pytest.raises(ValueError, match='parquet|column'):
        load_nli_dataset_rows(lambda *args, **kwargs: [{'premise': 'p', 'hypothesis': 'h', 'label': 0}], {
            'dataset_id': 'mteb/IndicXnliPairClassification', 'dataset_revision': 'a' * 40,
            'config': 'hi', 'split': 'test', 'source_format': 'parquet',
            'columns': {'sentence1': 'premise', 'sentence2': 'hypothesis', 'labels': 'label'},
        })


def test_nli_label_maps_are_explicit_and_config_mismatch_is_rejected():
    from mpkg_eval_core import validate_nli_label_mapping
    assert validate_nli_label_mapping({'LABEL_0': 'contradiction', 'LABEL_1': 'neutral', 'LABEL_2': 'entailment'}, kind='model')['valid']
    assert validate_nli_label_mapping({'0': 'entailment', '1': 'neutral', '2': 'contradiction'}, kind='dataset')['valid']
    assert validate_nli_label_mapping({'LABEL_0': 'entailment', 'LABEL_1': 'neutral', 'LABEL_2': 'contradiction'}, kind='model')['valid'] is False


def test_threshold_calibration_is_stratified_and_has_untouched_audit_metrics_and_bootstrap_ci():
    from mpkg_eval_core import calibrate_nli_threshold
    rows = [{'id': f'{label}-{i}', 'label': label} for label in (0, 1, 2) for i in range(20)]
    def predictor(row):
        return {'label': row['label'], 'entailment_probability': 0.9 if row['label'] == 2 else 0.1}
    result = calibrate_nli_threshold(rows, predictor, seed=9, min_support=5, n_bootstrap=100)
    assert result['calibration_ids'].isdisjoint(result['audit_ids'])
    assert result['support']['audit'] >= 15
    assert result['threshold'] > 0.1
    assert result['metrics']['audit']['entailment']['f1'] >= 0.9
    assert result['metrics']['audit']['entailment']['f1_ci']['lower'] is not None


def test_accuracy_ci_bootstraps_multiclass_argmax_not_binary_entailment_decision():
    from mpkg_eval_core import calibrate_nli_threshold
    rows = [{'id': f'{label}-{i}', 'label': label} for label in (0, 1, 2) for i in range(4)]
    def predictor(row):
        return {'label': 0, 'entailment_probability': 0.9 if row['label'] == 2 else 0.1}
    result = calibrate_nli_threshold(rows, predictor, seed=5, min_support=1, n_bootstrap=50)
    assert result['metrics']['audit']['accuracy'] < 1.0
    assert result['metrics']['audit']['accuracy_ci']['mean'] == result['metrics']['audit']['accuracy']


def test_dataset_content_digest_verification_rejects_tampered_selected_examples():
    from mpkg_eval_core import build_nli_calibration_artifact, normalized_dataset_content_hash, verify_nli_calibration_artifact
    examples = [{'id': 'a', 'premise': 'p', 'hypothesis': 'h', 'label': 2}]
    artifact = build_nli_calibration_artifact(model_id='m', model_revision='mr', dataset_id='d', dataset_revision='dr', dataset_content_hash=normalized_dataset_content_hash(examples), language='en', split='validation', label_mapping={'entailment': 2}, threshold=.5, n_examples=1, example_ids=['a'], example_content_digest='x', accuracy=.8, per_label_stats={'entailment': {'n': 1, 'precision': 1, 'recall': 1}}, code_hash='c', eval_core_hash='e')
    ok = verify_nli_calibration_artifact(artifact, model_id='m', model_revision='mr', dataset_id='d', dataset_revision='dr', dataset_content_hash=artifact['dataset_content_hash'], language='en')
    bad = verify_nli_calibration_artifact(artifact, model_id='m', model_revision='mr', dataset_id='d', dataset_revision='dr', dataset_content_hash=artifact['dataset_content_hash'], language='en', dataset_examples=[{**examples[0], 'hypothesis': 'tampered'}])
    assert ok['enabled'] and bad['enabled'] is False


def test_production_calibration_shape_has_required_splits_and_lower_bound_quality_gate():
    from mpkg_eval_core import build_nli_calibration_artifact, normalized_dataset_content_hash, stable_identity_hash, nli_calibration_quality
    rows = [{'id': f'{label}-{index}', 'premise': f'p{index}', 'hypothesis': f'h{index}', 'label': label} for label in (0, 1, 2) for index in range(2)]
    calibration_rows = rows[::2]; audit_rows = rows[1::2]
    metrics = {'audit': {'accuracy': .99, 'accuracy_ci': {'lower': .55, 'upper': 1.0}, 'entailment': {'precision': .99, 'recall': .99, 'f1': .99, 'support': 1, 'precision_ci': {'lower': .55, 'upper': 1.0}, 'recall_ci': {'lower': .55, 'upper': 1.0}, 'f1_ci': {'lower': .55, 'upper': 1.0}}}}
    metadata = {'method': 'stratified_calibration_audit_v1', 'seed': 7, 'criterion': 'f1', 'entailment_label': 0, 'calibration_ids': [row['id'] for row in calibration_rows], 'audit_ids': [row['id'] for row in audit_rows], 'calibration_content_digest': normalized_dataset_content_hash(calibration_rows), 'audit_content_digest': normalized_dataset_content_hash(audit_rows), 'support': {'calibration': 3, 'audit': 3, 'per_label': {str(label): {'calibration': 1, 'audit': 1} for label in (0, 1, 2)}}, 'metrics': metrics, 'dataset_content_hash': normalized_dataset_content_hash(rows)}
    artifact = build_nli_calibration_artifact(model_id='m', model_revision='mr', dataset_id='d', dataset_revision='dr', dataset_content_hash=normalized_dataset_content_hash(rows), language='en', split='validation', label_mapping={'0': 'entailment', '1': 'neutral', '2': 'contradiction'}, threshold=.5, n_examples=len(rows), example_ids=[row['id'] for row in rows], example_content_digest=stable_identity_hash([{key: row.get(key) for key in ('id', 'premise', 'hypothesis', 'label')} for row in rows]), accuracy=.99, per_label_stats={'0': {'n_total': 2, 'positive_support': 2}, '1': {'n_total': 2, 'positive_support': 2}, '2': {'n_total': 2, 'positive_support': 2}, 'entailment': {'n_total': 3, 'positive_support': 1, 'precision': .99, 'recall': .99, 'precision_ci': {'lower': .55}, 'recall_ci': {'lower': .55}}}, calibration_metadata=metadata, code_hash='c', eval_core_hash='e')
    assert nli_calibration_quality(artifact, min_accuracy=.8, min_entailment_precision=.8, min_entailment_recall=.8, min_per_label_support=1, min_accuracy_lower=.8, min_entailment_precision_lower=.8, min_entailment_recall_lower=.8)['enabled'] is False
    assert nli_calibration_quality(artifact, min_accuracy=.8, min_entailment_precision=.8, min_entailment_recall=.8, min_per_label_support=1, min_accuracy_lower=.5, min_entailment_precision_lower=.5, min_entailment_recall_lower=.5)['enabled'] is True


def test_calibration_metadata_tampering_is_rejected_after_digest_recompute():
    from mpkg_eval_core import build_nli_calibration_artifact, normalized_dataset_content_hash, stable_identity_hash, verify_nli_calibration_artifact
    rows = [{'id': f'x-{index}', 'premise': f'p{index}', 'hypothesis': f'h{index}', 'label': index % 3} for index in range(6)]
    metadata = {'method': 'stratified_calibration_audit_v1', 'seed': 1, 'criterion': 'f1', 'entailment_label': 0, 'calibration_ids': ['x-0', 'x-1', 'x-2'], 'audit_ids': ['x-3', 'x-4', 'x-5'], 'calibration_content_digest': normalized_dataset_content_hash(rows[:3]), 'audit_content_digest': normalized_dataset_content_hash(rows[3:]), 'support': {'calibration': 3, 'audit': 3, 'per_label': {'0': {'calibration': 1, 'audit': 1}, '1': {'calibration': 1, 'audit': 1}, '2': {'calibration': 1, 'audit': 1}}}, 'metrics': {'audit': {'accuracy': 1.0, 'accuracy_ci': {'lower': 1.0, 'upper': 1.0}, 'entailment': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'precision_ci': {'lower': 1.0, 'upper': 1.0}, 'recall_ci': {'lower': 1.0, 'upper': 1.0}, 'f1_ci': {'lower': 1.0, 'upper': 1.0}}}}, 'dataset_content_hash': normalized_dataset_content_hash(rows)}
    artifact = build_nli_calibration_artifact(model_id='m', model_revision='mr', dataset_id='d', dataset_revision='dr', dataset_content_hash=metadata['dataset_content_hash'], language='en', split='validation', label_mapping={'0': 'entailment', '1': 'neutral', '2': 'contradiction'}, threshold=.5, n_examples=6, example_ids=[row['id'] for row in rows], example_content_digest=stable_identity_hash([{key: row.get(key) for key in ('id', 'premise', 'hypothesis', 'label')} for row in rows]), accuracy=1.0, per_label_stats={'0': {'n_total': 1, 'correct': 1}, '1': {'n_total': 1, 'correct': 1}, '2': {'n_total': 1, 'correct': 1}, 'entailment': {'n_total': 3, 'positive_support': 1, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}}, calibration_metadata=metadata, code_hash='c', eval_core_hash='e')
    ok = verify_nli_calibration_artifact(artifact, model_id='m', model_revision='mr', dataset_id='d', dataset_revision='dr', dataset_content_hash=metadata['dataset_content_hash'], dataset_examples=rows, label_mapping={'0': 'entailment', '1': 'neutral', '2': 'contradiction'}, language='en', split='validation', code_hash='c', eval_core_hash='e')
    assert ok['enabled']
    forged = dict(artifact, calibration={**metadata, 'audit_ids': ['x-0', 'x-4', 'x-5']})
    forged['artifact_digest'] = stable_identity_hash({key: value for key, value in forged.items() if key != 'artifact_digest'})
    assert verify_nli_calibration_artifact(forged, model_id='m', model_revision='mr', dataset_id='d', dataset_revision='dr', dataset_content_hash=metadata['dataset_content_hash'], dataset_examples=rows, label_mapping={'0': 'entailment', '1': 'neutral', '2': 'contradiction'}, language='en', split='validation', code_hash='c', eval_core_hash='e')['enabled'] is False
    missing = {key: value for key, value in artifact.items() if key != 'calibration'}
    missing['artifact_digest'] = stable_identity_hash({key: value for key, value in missing.items() if key != 'artifact_digest'})
    assert verify_nli_calibration_artifact(missing, model_id='m', model_revision='mr', dataset_id='d', dataset_revision='dr', dataset_content_hash=metadata['dataset_content_hash'], dataset_examples=rows, label_mapping={'0': 'entailment', '1': 'neutral', '2': 'contradiction'}, language='en', split='validation', code_hash='c', eval_core_hash='e')['enabled'] is False


def test_existing_artifact_requires_fresh_prediction_recompute_and_rejects_threshold_tamper():
    from mpkg_eval_core import build_nli_calibration_artifact, calibrate_nli_threshold, normalized_dataset_content_hash, stable_identity_hash, verify_nli_calibration_artifact
    rows = [{'id': f'z-{label}-{index}', 'premise': f'p{index}', 'hypothesis': f'h{index}', 'label': label} for label in (0, 1, 2) for index in range(2)]
    def predictor(row):
        return {'label': row['label'], 'entailment_probability': .9 if row['label'] == 0 else .1}
    calibration = calibrate_nli_threshold(rows, predictor, seed=3, min_support=1, n_bootstrap=20, entailment_label=0)
    by_id = {row['id']: row for row in rows}; cal_rows = [row for row in rows if row['id'] in calibration['calibration_ids']]; audit_rows = [row for row in rows if row['id'] in calibration['audit_ids']]
    audit_predictions = [predictor(row)['label'] for row in audit_rows]
    per_label = {str(label): {'n_total': sum(row['label'] == label for row in audit_rows), 'correct': sum(row['label'] == label and pred == label for row, pred in zip(audit_rows, audit_predictions))} for label in (0, 1, 2)}
    per_label['entailment'] = {**calibration['metrics']['audit']['entailment'], 'n_total': len(audit_rows), 'positive_support': calibration['metrics']['audit']['entailment']['support']}
    metadata = {'method': calibration['method'], 'seed': calibration['seed'], 'criterion': calibration['criterion'], 'entailment_label': 0, 'calibration_ids': sorted(calibration['calibration_ids']), 'audit_ids': sorted(calibration['audit_ids']), 'calibration_content_digest': normalized_dataset_content_hash(cal_rows), 'audit_content_digest': normalized_dataset_content_hash(audit_rows), 'support': calibration['support'], 'metrics': calibration['metrics'], 'dataset_content_hash': normalized_dataset_content_hash(rows)}
    artifact = build_nli_calibration_artifact(model_id='m', model_revision='mr', dataset_id='d', dataset_revision='dr', dataset_content_hash=metadata['dataset_content_hash'], language='en', split='validation', label_mapping={'0': 'entailment', '1': 'neutral', '2': 'contradiction'}, threshold=calibration['threshold'], n_examples=len(rows), example_ids=[row['id'] for row in rows], example_content_digest=stable_identity_hash([{key: row.get(key) for key in ('id', 'premise', 'hypothesis', 'label')} for row in rows]), accuracy=calibration['metrics']['audit']['accuracy'], per_label_stats=per_label, calibration_metadata=metadata, code_hash='c', eval_core_hash='e')
    kwargs = dict(model_id='m', model_revision='mr', dataset_id='d', dataset_revision='dr', dataset_content_hash=metadata['dataset_content_hash'], language='en', split='validation', code_hash='c', eval_core_hash='e', dataset_examples=rows, label_mapping={'0': 'entailment', '1': 'neutral', '2': 'contradiction'}, fresh_predictor=predictor, calibration_seed=3, calibration_min_support=1, calibration_bootstrap=20, calibration_criterion='f1', calibration_entailment_label=0)
    assert verify_nli_calibration_artifact(artifact, **kwargs)['enabled']
    forged = dict(artifact, threshold=min(1.0, artifact['threshold'] + .1)); forged['artifact_digest'] = stable_identity_hash({key: value for key, value in forged.items() if key != 'artifact_digest'})
    assert verify_nli_calibration_artifact(forged, **kwargs)['enabled'] is False


def test_category_normalization_filters_whitespace_and_case_with_manifest_rows():
    rows = [
        {"ID": "a", "Category": " Homophobic "},
        {"ID": "b", "Category": "Non-Homophobic\u00a0"},
        {"ID": "c", "Category": "other"},
    ]
    assert normalize_category(rows[0]["Category"]) == "homophobic"
    result = filter_rows_by_category(rows, ["homophobic", "non-homophobic"])
    assert [row["ID"] for row in result["rows"]] == ["a", "b"]
    assert result["manifest"]["input"] == 3
    assert result["manifest"]["kept"] == 2
    assert result["manifest"]["dropped"] == 1
    assert result["manifest"]["reason_counts"] == {"category_not_allowed": 1}
    assert result["manifest"]["rows"][2]["reason"] == "category_not_allowed"


def test_missing_references_are_quarantined_from_scoring_but_preserved_for_generation_audit():
    rows = [
        {"ID": "ok", "Counter Narrative": "reference"},
        {"ID": "missing", "Counter Narrative": float("nan")},
        {"ID": "blank", "Counter Narrative": "   "},
    ]
    result = quarantine_missing_references(rows, reference_key="Counter Narrative")
    assert [row["ID"] for row in result["scorable_rows"]] == ["ok"]
    assert [row["ID"] for row in result["generation_rows"]] == ["ok", "missing", "blank"]
    assert result["manifest"]["reason_counts"] == {"missing_reference": 2}
    assert result["manifest"]["dropped"] == 2
    assert {row["record_id"] for row in result["quarantined_rows"]} == {"missing", "blank"}


@pytest.mark.parametrize("index,count", [(0, 1), (1, 2), (2, 3)])
def test_shard_run_name_is_unique_and_validates_range(index, count):
    assert derive_shard_run_name("experiment", index, count) == f"experiment__shard-{index}-of-{count}"


@pytest.mark.parametrize("index,count", [(-1, 2), (2, 2), (0, 0), (0, -1)])
def test_shard_run_name_rejects_invalid_ranges(index, count):
    with pytest.raises(ValueError, match="shard"):
        derive_shard_run_name("experiment", index, count)


def test_identity_config_keeps_split_identity_shard_independent_but_binds_run_identity_to_shard():
    first = identity_config({"seed": 7, "split_name": "test", "shard_index": 0, "shard_count": 2})
    second = identity_config({"seed": 7, "split_name": "test", "shard_index": 1, "shard_count": 2})
    assert first["split_identity_hash"] == second["split_identity_hash"]
    assert first["run_identity_hash"] != second["run_identity_hash"]
    assert first["shard"] == {"index": 0, "count": 2}


def test_identity_config_explicit_shard_override_replaces_config_fields_before_hashing():
    base = identity_config({"seed": 7, "shard_index": 99, "shard_count": 100}, shard_index=1, shard_count=2)
    equivalent = identity_config({"seed": 7, "shard_index": 1, "shard_count": 2})
    assert base["config"] == equivalent["config"]
    assert base["config_hash"] == equivalent["config_hash"]
    assert base["run_identity_hash"] == equivalent["run_identity_hash"]


def test_effective_cache_capacity_is_deterministic_and_covers_selected_shard_rows():
    assert derive_effective_cache_capacity(16, row_limit=101, shard_count=4) == 26
    assert derive_effective_cache_capacity(64, row_limit=101, shard_count=4) == 64
    with pytest.raises(ValueError, match="cache|shard"):
        derive_effective_cache_capacity(0, row_limit=10, shard_count=1)


def test_bounded_lru_evicts_oldest_and_refreshes_reads():
    cache = BoundedLRU(maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    assert cache["a"] == 1
    cache["c"] = 3
    assert list(cache) == ["a", "c"]
    assert "b" not in cache
    cache["d"] = 4
    assert list(cache) == ["c", "d"]
    assert cache.evictions == 2


def test_bounded_lru_snapshot_is_json_serializable_and_reports_capacity():
    cache = BoundedLRU(maxsize=1)
    cache["a"] = {"nested": [1]}
    cache["b"] = {"nested": [2]}
    snapshot = cache.snapshot()
    assert snapshot["capacity"] == 1
    assert snapshot["size"] == 1
    assert snapshot["evictions"] == 1
    import json
    json.dumps(snapshot)


class _BudgetTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, *, text, truncation=False, **kwargs):
        self.calls.append({"text": text, "truncation": truncation, **kwargs})
        return {"input_ids": [[1] * len(value.split()) for value in text], "attention_mask": [[1] * len(value.split()) for value in text]}


def test_fit_prompt_to_budget_disables_tokenizer_truncation_and_preserves_schema_tail():
    tokenizer = _BudgetTokenizer()
    prompt = fit_prompt_to_budget("context", tokenizer, 5, schema_tail="RETURN JSON SCHEMA", reserve_output_tokens=1)
    assert prompt.endswith("RETURN JSON SCHEMA")
    assert tokenizer.calls[-1]["truncation"] is False
    with pytest.raises(RuntimeError, match="prompt_token_budget_exceeded"):
        fit_prompt_to_budget("one two three four", tokenizer, 4, schema_tail="RETURN JSON SCHEMA", reserve_output_tokens=1)


def _budget_ledger(language):
    texts = {
        "hi": ["यह पहला पूरा साक्ष्य वाक्य है।", "यह दूसरा पूरा साक्ष्य वाक्य है।"],
        "ta": ["இது முதல் முழு ஆதார வாக்கியம்.", "இது இரண்டாவது முழு ஆதார வாக்கியம்."],
    }[language]
    return [
        {"evidence_id": f"E{index}", "displayed_text": text, "span_start": 0, "span_end": len(text), "displayed_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "rank": index}
        for index, text in enumerate(texts, 1)
    ]


def test_adaptive_prompt_reduction_preserves_whole_hindi_tamil_spans_schema_and_order():
    def token_count(prompt):
        return len(prompt)

    for language in ("hi", "ta"):
        ledger = _budget_ledger(language)
        def build(selected, payload):
            return "POST TARGET\n" + "\n".join(item["displayed_text"] for item in selected) + "\n" + payload + "\nSCHEMA_TAIL"
        result = fit_adaptive_prompt_with_evidence(
            ledger,
            prompt_builder=build,
            payload_candidates=("",),
            token_counter=token_count,
            max_input_tokens=70,
            reserve_output_tokens=5,
        )
        assert result["status"] == "fit"
        assert result["selected_evidence_ids"] == ["E1"]
        assert ledger[0]["displayed_text"] in result["prompt"]
        assert ledger[1]["displayed_text"] not in result["prompt"]
        assert result["prompt"].endswith("SCHEMA_TAIL")
        assert result == fit_adaptive_prompt_with_evidence(
            ledger,
            prompt_builder=build,
            payload_candidates=("",),
            token_counter=token_count,
            max_input_tokens=70,
            reserve_output_tokens=5,
        )


def test_adaptive_prompt_compacts_plan_payload_before_quarantining():
    ledger = _budget_ledger("hi")
    def build(selected, payload):
        return "POST TARGET\n" + "\n".join(item["displayed_text"] for item in selected) + "\nPERSPECTIVES=" + payload + "\nSCHEMA_TAIL"
    result = fit_adaptive_prompt_with_evidence(
        ledger,
        prompt_builder=build,
        payload_candidates=("very verbose model-generated perspective payload", "compact"),
        token_counter=len,
        max_input_tokens=47,
        reserve_output_tokens=2,
    )
    assert result["status"] == "fit"
    assert result["payload_index"] == 1
    assert result["payload"] == "compact"
    assert result["selected_evidence_ids"] == []


def test_adaptive_prompt_reports_irreducible_base_without_fabricating_prompt():
    ledger = _budget_ledger("ta")
    result = fit_adaptive_prompt_with_evidence(
        ledger,
        prompt_builder=lambda selected, payload: "POST TARGET SCHEMA_TAIL irreducibly long",
        payload_candidates=("", "compact"),
        token_counter=len,
        max_input_tokens=5,
        reserve_output_tokens=1,
    )
    assert result["status"] == "quarantine"
    assert result["reason"] == "prompt_budget_irreducible"


def _citation_ledger():
    return [
        {"evidence_id": "E1", "displayed_text": "The policy protects equal access.", "span_start": 0, "span_end": 33, "displayed_text_sha256": hashlib.sha256(b"The policy protects equal access.").hexdigest()},
        {"evidence_id": "E2", "displayed_text": "The unrelated document describes rainfall.", "span_start": 0, "span_end": 42, "displayed_text_sha256": hashlib.sha256(b"The unrelated document describes rainfall.").hexdigest()},
    ]


def test_sentence_aligned_windows_are_bounded_and_preserve_exact_source_spans():
    source = "First sentence is here. Second sentence has more words. Third ends here."
    windows = sentence_aligned_windows(source, max_chars=40)
    assert windows
    assert all(len(window["text"]) <= 40 for window in windows)
    assert "".join(window["text"] for window in windows) == source
    assert all(source[window["start_char"] : window["end_char"]] == window["text"] for window in windows)
    assert all(window["text"][-1:] in ".!?" for window in windows[:-1])


def test_claim_records_keep_format_compliance_separate_from_entailment():
    narrative = "The policy protects equal access [E1]. The weather is rainy [E2]."
    records = build_claim_citation_records(narrative, _citation_ledger(), factual_claims=[
        "The policy protects equal access.", "The weather is rainy."
    ])
    result = aggregate_citation_support(
        records,
        evaluator=lambda premise, hypothesis: 1.0 if "equal access" in premise and "equal access" in hypothesis else 0.0,
        require_nli=True,
    )
    assert result["format_compliance"]["valid"] is True
    assert result["entailment_status"] == "scored"
    assert result["factual_claim_count"] == 2
    assert result["supported_claim_count"] == 1
    assert result["citation_precision"] < 1.0
    assert result["syntactic_citation_precision"] == 1.0


def test_claim_support_reports_necessary_citations_and_overcitation():
    records = build_claim_citation_records(
        "The policy protects equal access [E1][E2].",
        _citation_ledger(),
        factual_claims=["The policy protects equal access."],
    )
    def evaluator(premise, hypothesis):
        return 1.0 if "equal access" in premise else 0.0
    result = aggregate_citation_support(records, evaluator=evaluator, require_nli=True)
    assert result["supported_claim_count"] == 1
    assert result["necessary_citation_count"] == 1
    assert result["overcitation_count"] == 1
    assert result["citation_precision"] == pytest.approx(0.5)
    assert result["citation_recall"] == pytest.approx(1.0)
    assert result["citation_necessity"] == pytest.approx(0.5)
    assert result["citation_entailment"] == pytest.approx(1.0)


def test_uncited_factual_claim_fails_closed_but_no_factual_claim_is_safe_abstention():
    uncited = build_claim_citation_records(
        "The policy protects equal access.", _citation_ledger(), factual_claims=["The policy protects equal access."]
    )
    failed = aggregate_citation_support(uncited, evaluator=lambda premise, hypothesis: 1.0, require_nli=True)
    assert failed["claim_citation_recall"] == 0.0
    assert failed["pass"] is False
    abstention = build_claim_citation_records("Please treat people with dignity.", _citation_ledger(), factual_claims=[], abstention_validator=lambda text: True)
    safe = aggregate_citation_support(abstention, evaluator=None, require_nli=True)
    assert safe["abstention"] is True
    assert safe["entailment_status"] == "not_required"
    assert safe["pass"] is True


def test_required_nli_without_callback_fails_closed_without_changing_format_result():
    records = build_claim_citation_records(
        "The policy protects equal access [E1].", _citation_ledger(), factual_claims=["The policy protects equal access."]
    )
    result = aggregate_citation_support(records, evaluator=None, require_nli=True)
    assert result["format_compliance"]["valid"] is True
    assert result["entailment_status"] == "unavailable"
    assert result["pass"] is False


def test_claim_declaration_cannot_hide_hallucinated_narrative_sentences():
    records = build_claim_citation_records(
        "Supported policy claim [E1]. Hallucinated extra claim.",
        _citation_ledger(), factual_claims=[], abstention_validator=None,
    )
    assert [record["is_factual"] for record in records] == [True, True]
    result = aggregate_citation_support(records, evaluator=lambda premise, hypothesis: 1.0, require_nli=True)
    assert result["pass"] is False


def test_safe_abstention_requires_deterministic_validator_and_rejects_any_citation():
    validator = lambda text: text.strip() == "I cannot verify this from the available evidence."
    safe = build_claim_citation_records(
        "I cannot verify this from the available evidence.", _citation_ledger(), factual_claims=[], abstention_validator=validator,
    )
    result = aggregate_citation_support(safe, evaluator=None, require_nli=True)
    assert result["abstention"] is True and result["pass"] is True
    cited = build_claim_citation_records(
        "I cannot verify this from the available evidence. [E1]", _citation_ledger(), factual_claims=[], abstention_validator=validator,
    )
    assert aggregate_citation_support(cited, evaluator=None, require_nli=True)["pass"] is False


def test_unicode_danda_sentence_boundaries_preserve_spans():
    source = "यह दावा गलत है। दूसरा दावा भी गलत है॥ तीसरा वाक्य है।"
    windows = sentence_aligned_windows(source, max_chars=30)
    assert [window["text"].strip() for window in windows] == ["यह दावा गलत है।", "दूसरा दावा भी गलत है॥", "तीसरा वाक्य है।"]
    assert "".join(window["text"] for window in windows) == source


def test_claim_normalization_strips_single_and_double_hindi_danda():
    records = build_claim_citation_records("यह दावा है [E1]।", _citation_ledger(), factual_claims=["यह दावा है॥"])
    assert records[0]["is_factual"] is True


def test_terminal_mark_variants_do_not_create_synthetic_unmatched_claims():
    records = build_claim_citation_records("यह दावा है [E1]।", _citation_ledger(), factual_claims=["यह दावा है॥"])
    assert len(records) == 1


def test_mixed_narrative_requires_exhaustive_safe_nonfactual_accounting():
    validator = lambda sentence: sentence.strip() in {
        "Please treat people with dignity.",
        "कृपया लोगों के साथ गरिमा से व्यवहार करें।",
        "மக்களை கண்ணியத்துடன் நடத்துங்கள்.",
    }
    records = build_claim_citation_records(
        "Please treat people with dignity. The policy protects equal access [E1]. Hallucinated fact.",
        _citation_ledger(), factual_claims=[], safe_non_factual_validator=validator,
    )
    assert [row["is_factual"] for row in records] == [False, True, True]
    result = aggregate_citation_support(records, evaluator=lambda premise, hypothesis: 1.0, require_nli=True)
    assert result["pass"] is False


def test_partial_entailment_is_nullable_and_exposes_only_partial_diagnostic():
    records = build_claim_citation_records(
        "Supported claim [E1]. Uncited factual claim.", _citation_ledger(), factual_claims=None,
    )
    result = aggregate_citation_support(records, evaluator=lambda premise, hypothesis: 1.0, require_nli=True)
    assert result["entailment_status"] == "scored_incomplete"
    assert result["citation_entailment"] is None and result["entailment_mean"] is None
    assert result["evaluated_claim_entailment_mean"] == 1.0
    assert result["evaluated_claim_count"] == 1 and result["incomplete_claim_count"] == 1


def test_unsupported_cited_claim_counts_every_citation_as_unnecessary():
    records = build_claim_citation_records("Unsupported claim [E1][E2].", _citation_ledger(), factual_claims=None)
    result = aggregate_citation_support(records, evaluator=lambda premise, hypothesis: 0.1, require_nli=True)
    assert result["supported_claim_count"] == 0
    assert result["overcitation_count"] == 2
    assert result["necessary_citation_count"] == 0
    assert result["pass"] is False


def test_claim_record_end_char_excludes_separator_whitespace():
    records = build_claim_citation_records("A claim without punctuation   ", _citation_ledger(), factual_claims=None)
    row = records[0]
    assert row["claim"] == "A claim without punctuation"
    assert "A claim without punctuation   "[row["start_char"] : row["end_char"]] == row["claim"]


def test_same_evidence_id_may_be_reused_across_claims_but_not_within_one_claim():
    records = build_claim_citation_records(
        "First supported claim [E1]. Second supported claim [E1].", _citation_ledger(), factual_claims=None,
    )
    assert all(record["citation_format_valid"] for record in records)
    duplicate = build_claim_citation_records("One claim [E1][E1].", _citation_ledger(), factual_claims=None)
    assert duplicate[0]["citation_format_valid"] is False


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_nli_scores_fail_closed_as_evaluator_error(bad_score):
    records = build_claim_citation_records("A factual claim [E1].", _citation_ledger(), factual_claims=None)
    result = aggregate_citation_support(records, evaluator=lambda premise, hypothesis: bad_score, require_nli=True)
    assert result["entailment_status"] == "evaluator_error"
    assert result["pass"] is False


def test_invalid_ledger_offsets_hash_and_source_slice_fail_closed():
    with pytest.raises(ValueError, match="ledger"):
        build_claim_citation_records("A claim [E1].", [{"evidence_id": "E1", "displayed_text": "wrong", "span_start": 3, "span_end": 2, "displayed_text_sha256": "bad"}], factual_claims=None)
    ledger = [{"evidence_id": "E1", "displayed_text": "claim", "span_start": 0, "span_end": 5, "displayed_text_sha256": hashlib.sha256(b"claim").hexdigest(), "source_text": "other"}]
    with pytest.raises(ValueError, match="ledger"):
        build_claim_citation_records("claim [E1].", ledger, factual_claims=None)
    out_of_bounds = [{"evidence_id": "E1", "displayed_text": "claim", "span_start": 0, "span_end": 8, "displayed_text_sha256": hashlib.sha256(b"claim").hexdigest(), "source_text_key": "src", "source_text_sha256": hashlib.sha256(b"claim").hexdigest()}]
    with pytest.raises(ValueError, match="exceed"):
        validate_evidence_ledger(out_of_bounds, {"src": "claim"})


def test_whole_span_budget_never_slices_and_raises_when_first_span_overflows():
    from mpkg_eval_core import select_evidence_within_budget
    ledger = _citation_ledger()
    assert select_evidence_within_budget(ledger, 50)[0]["displayed_text"] == ledger[0]["displayed_text"]
    with pytest.raises(ValueError, match="budget"):
        select_evidence_within_budget(ledger, 5)


def test_missing_reference_stays_missing_instead_of_literal_nan():
    assert normalize_optional_text(float("nan")) is None
    assert normalize_optional_text(None) is None
    assert normalize_optional_text("   ") is None
    assert normalize_optional_text("nan") == "nan"
    audit = filter_audit(
        [{"ID": "ok", "reference": "answer"}, {"ID": "missing", "reference": float("nan")}],
        lambda row: normalize_optional_text(row["reference"]) is not None,
        filter_name="reference",
    )
    assert audit["input"] == 2
    assert audit["kept"] == 1
    assert audit["dropped"] == 1
    assert audit["events"][0]["reason"] == "reference"


def test_numpy_nat_sentinels_stay_missing_without_pandas():
    numpy = pytest.importorskip("numpy")
    assert normalize_optional_text(numpy.datetime64("NaT", "ns")) is None
    assert normalize_optional_text(numpy.timedelta64("NaT", "ns")) is None


def test_tamil_and_devanagari_scripts_are_detected_and_script_match_is_script_aware():
    tamil = "தமிழ் counter narrative"
    hindi = "हिन्दी प्रतिवाद"
    assert script_bucket(tamil) == "mixed"
    assert script_bucket("தமிழ்") == "tamil"
    assert script_bucket(hindi) == "devanagari"
    assert script_match("தமிழ்", "தமிழில் பதில்") is True
    assert script_match("தமிழ்", hindi) is False


def test_language_match_is_label_based_and_unknown_labels_are_nullable():
    assert language_match("en", "English") is True
    assert language_match("en", "fr") is False
    assert language_match("en", "made-up-language") is None
    assert language_match("English text", "English text") is None


def test_unicode_chrf_scores_non_latin_text_without_ascii_tokenizer():
    assert unicode_chrf("தமிழ் பதில்", "தமிழ் பதில்") == pytest.approx(1.0)
    assert unicode_chrf("தமிழ் பதில்", "हिन्दी उत्तर") < 0.2
    assert unicode_chrf(None, "anything") is None
    assert unicode_chrf("a", "a", max_order=6) == pytest.approx(1.0)
    assert unicode_chrf("ab", "ab", max_order=6) == pytest.approx(1.0)


def test_benjamini_hochberg_is_bounded_and_monotone_in_sorted_p_value_order():
    p_values = [0.01, 0.04, 0.03, 0.9]
    q_values = benjamini_hochberg(p_values)
    assert len(q_values) == len(p_values)
    assert all(0.0 <= q <= 1.0 for q in q_values)
    ordered = [q for _, q in sorted(zip(p_values, q_values))]
    assert ordered == sorted(ordered)
    assert benjamini_hochberg([]) == []
    assert benjamini_hochberg([0.01, 0.04, 0.03, 0.9]) == pytest.approx(
        [0.04, 0.05333333333333334, 0.05333333333333334, 0.9]
    )
    with pytest.raises(ValueError):
        benjamini_hochberg([-0.01])
    with pytest.raises(ValueError):
        benjamini_hochberg([float("nan")])


def test_paired_variant_tests_aligns_shared_ids_before_comparison():
    rows = [
        {"ID": "b", "mp_kg_rag": 0.2, "kg_rag": 0.4},
        {"ID": "a", "mp_kg_rag": 0.9, "kg_rag": 0.8},
        {"ID": "only-mp", "mp_kg_rag": 1.0},
    ]
    result = paired_variant_tests(
        rows, variant_a="mp_kg_rag", variant_b="kg_rag", seed=7, permutations=500
    )
    assert result["ids"] == ["a", "b"]
    assert result["n"] == 2
    assert result["mean_difference"] == pytest.approx((0.1 - 0.2) / 2)
    assert result["win_rate"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "rows",
    [
        [{"ID": "dup", "mp_kg_rag": 1.0, "kg_rag": 1.0}, {"ID": "dup", "mp_kg_rag": float("nan"), "kg_rag": 0.0}],
        [{"ID": None, "mp_kg_rag": float("nan"), "kg_rag": 0.0}, {"ID": None, "mp_kg_rag": 0.2, "kg_rag": 0.1}],
        [{"ID": 1, "mp_kg_rag": 1.0, "kg_rag": 0.0}, {"ID": "1", "mp_kg_rag": 0.5, "kg_rag": 0.4}],
    ],
)
def test_paired_variant_tests_rejects_duplicate_or_colliding_raw_ids_before_filtering(rows):
    with pytest.raises(ValueError, match="invalid|duplicate|collision"):
        paired_variant_tests(rows)


@pytest.mark.parametrize("invalid_id", [None, "", "   ", float("nan")])
def test_paired_variant_tests_rejects_lone_invalid_ids_before_score_filtering(invalid_id):
    rows = [{"ID": invalid_id, "mp_kg_rag": float("nan"), "kg_rag": None}]
    with pytest.raises(ValueError, match="invalid paired ID"):
        paired_variant_tests(rows)


def test_paired_variant_tests_rejects_numpy_nat_id_before_score_filtering():
    numpy = pytest.importorskip("numpy")
    rows = [{"ID": numpy.datetime64("NaT", "ns"), "mp_kg_rag": 0.2, "kg_rag": 0.1}]
    with pytest.raises(ValueError, match="invalid paired ID"):
        paired_variant_tests(rows)


@pytest.mark.parametrize("invalid_id", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
def test_paired_variant_tests_rejects_decimal_nonfinite_ids(invalid_id):
    with pytest.raises(ValueError, match="invalid paired ID"):
        paired_variant_tests([{"ID": invalid_id, "mp_kg_rag": 0.2, "kg_rag": 0.1}])


def test_paired_variant_tests_rejects_numpy_nonfinite_numeric_ids_but_keeps_valid_nonnumeric_ids():
    numpy = pytest.importorskip("numpy")
    for invalid_id in (numpy.float32("nan"), numpy.float64("inf")):
        with pytest.raises(ValueError, match="invalid paired ID"):
            paired_variant_tests([{"ID": invalid_id, "mp_kg_rag": 0.2, "kg_rag": 0.1}])
    result = paired_variant_tests([{"ID": ("doc", 1), "mp_kg_rag": 0.2, "kg_rag": 0.1}])
    assert result["n"] == 1


def test_paired_variant_tests_reports_incomplete_and_nonfinite_exclusions():
    rows = [
        {"ID": "complete", "mp_kg_rag": 0.8, "kg_rag": 0.6},
        {"ID": "incomplete", "mp_kg_rag": 0.8},
        {"ID": "nonfinite", "mp_kg_rag": float("nan"), "kg_rag": 0.2},
    ]
    result = paired_variant_tests(rows)
    assert result["n"] == 1
    assert result["excluded_rows"] == 2
    assert result["exclusion_counts"] == {"incomplete": 1, "nonfinite": 1}


def test_paired_variant_tests_seeded_monte_carlo_branch_is_deterministic():
    rows = [
        {"ID": f"id-{index}", "mp_kg_rag": index / 20, "kg_rag": 0.2}
        for index in range(17)
    ]
    first = paired_variant_tests(rows, seed=41, permutations=500)
    second = paired_variant_tests(rows, seed=41, permutations=500)
    assert first == second
    assert 0.0 <= first["p_value"] <= 1.0


def test_bootstrap_mean_ci_drops_nan_values_and_is_seeded():
    first = bootstrap_mean_ci([1.0, float("nan"), 3.0], seed=19, n_resamples=300)
    second = bootstrap_mean_ci([1.0, float("nan"), 3.0], seed=19, n_resamples=300)
    assert first == second
    assert first["n"] == 2
    assert first["mean"] == pytest.approx(2.0)
    assert math.isfinite(first["lower"])
    assert math.isfinite(first["upper"])
    assert bootstrap_mean_ci([float("nan"), None], n_resamples=10)["n"] == 0
    degenerate = bootstrap_mean_ci([3.0, 3.0], n_resamples=10, seed=2)
    assert degenerate["mean"] == degenerate["lower"] == degenerate["upper"] == 3.0


def test_reliability_rejects_fewer_than_two_raters():
    with pytest.raises(ValueError, match="at least two raters"):
        weighted_kappa_rows([[1], [2]])
    with pytest.raises(ValueError, match="at least two raters"):
        krippendorff_alpha_ordinal([[1], [2]])
    with pytest.raises(ValueError, match="exactly two raters"):
        weighted_kappa_rows([[1, 1, 1], [2, 2, 2]])
    with pytest.raises(ValueError, match="exactly two raters"):
        weighted_kappa_rows({"r1": [1, 2], "r2": [1, 2], "r3": [1, 2]})
    with pytest.raises(ValueError, match="at least two raters"):
        weighted_kappa_rows({})
    with pytest.raises(ValueError, match="at least two raters"):
        krippendorff_alpha_ordinal({})


def test_reliability_honors_explicit_rater_columns_and_rejects_missing_columns():
    rows = [{"r1": 0, "r2": 0, "ignored": 2}, {"r1": 1, "r2": 1, "ignored": 0}]
    assert weighted_kappa_rows(rows, rater_columns=("r2", "r1")) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="missing rater column"):
        weighted_kappa_rows(rows, rater_columns=("r1", "missing"))


def test_reliability_reports_perfect_agreement_for_two_raters():
    rows = [[0, 0], [1, 1], [2, 2], [1, 1]]
    assert weighted_kappa_rows(rows) == pytest.approx(1.0)
    assert krippendorff_alpha_ordinal(rows) == pytest.approx(1.0)


def test_ordinal_alpha_uses_coincidence_normalization_for_missing_and_three_raters():
    rows = [[0, 0, 1], [0, 1, None], [1, 2, 2], [2, 2, 2]]
    # Expected from Krippendorff's coincidence matrix with ordinal squared
    # distance and each unit's valid-rater count in its denominator.
    assert krippendorff_alpha_ordinal(rows) == pytest.approx(0.6428571428571429)
