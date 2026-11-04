"""Sequential MP-KG-RAG orchestration with durable stage checkpoints."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Mapping

from .config import DemoConfig, EvidenceRecord, PERSPECTIVES, QueryInput, RetrievalResult
from .backends import load_retrieval_backends
from .history import HistoryStore, InvalidCitationError
from .index_store import LocalRetriever
from .mlx_engine import GenerationEvent, MLXGenerator, _repair_prompt
from .ollama_engine import OllamaGenerator


PERSPECTIVE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["perspective", "rationale", "claims_to_address", "supported_evidence_ids", "response_guidance", "risk_flags", "confidence"], "properties": {"perspective": {"type": "string"}, "rationale": {"type": "string"}, "claims_to_address": {"type": "array", "items": {"type": "string"}}, "supported_evidence_ids": {"type": "array", "items": {"type": "string", "pattern": r"^E[1-9][0-9]*$"}}, "response_guidance": {"type": "array", "items": {"type": "string"}}, "risk_flags": {"type": "array", "items": {"type": "string"}}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}}}
PLAN_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["claim_focus", "selected_evidence_ids", "response_steps", "tone", "factual_constraints", "safety_constraints"], "properties": {"claim_focus": {"type": "string"}, "selected_evidence_ids": {"type": "array", "items": {"type": "string", "pattern": r"^E[1-9][0-9]*$"}}, "response_steps": {"type": "array", "items": {"type": "string"}}, "tone": {"type": "string"}, "factual_constraints": {"type": "array", "items": {"type": "string"}}, "safety_constraints": {"type": "array", "items": {"type": "string"}}}}
FINAL_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["counter_narrative", "cited_evidence_ids", "factual_claims", "safety_notes"], "properties": {"counter_narrative": {"type": "string"}, "cited_evidence_ids": {"type": "array", "items": {"type": "string", "pattern": r"^E[1-9][0-9]*$"}}, "factual_claims": {"type": "array", "items": {"type": "string"}}, "safety_notes": {"type": "array", "items": {"type": "string"}}}}
METADATA_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["target", "category"], "properties": {"target": {"type": "string"}, "category": {"type": "string"}}}
PERSPECTIVE_GOALS = {"fact_checking": "Correct only directly supported claims.", "cultural_context": "Give respectful context.", "harm_reduction": "Avoid amplifying harmful language.", "legal_rights": "Use only source-grounded rights claims.", "persuasion": "Suggest an empathetic response."}
_LGBTQ_TOPIC_TERMS = (
    "lgbt", "lgbtq", "gay", "lesbian", "bisexual", "transgender", "queer",
    "homosexual", "homophobia", "homophobic", "sexual orientation", "gender identity",
    "same-sex", "same sex", "pride", "conversion therapy",
)


def classify_demo_route(text: str, target: str = "", category: str = "") -> str:
    """Choose the cheap local-demo route before any retrieval or generation."""

    haystack = " ".join((str(text), str(target), str(category))).casefold()
    return "mp_kg_rag" if any(term in haystack for term in _LGBTQ_TOPIC_TERMS) else "general_text"


@dataclass(frozen=True, init=False)
class RunRequest:
    text: str = ""
    target: str = ""
    category: str = ""
    conversation_id: str | None = None
    title: str = "New conversation"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __init__(self, text: str = "", target: str = "", category: str = "", conversation_id: str | None = None, title: str = "New conversation", metadata: Mapping[str, Any] | None = None, *, post: str | None = None) -> None:
        object.__setattr__(self, "text", str(post if post is not None and not text else text))
        object.__setattr__(self, "target", str(target or ""))
        object.__setattr__(self, "category", str(category or ""))
        object.__setattr__(self, "conversation_id", conversation_id)
        object.__setattr__(self, "title", str(title or "New conversation"))
        object.__setattr__(self, "metadata", dict(metadata or {}))

    @property
    def post(self) -> str:
        return self.text


@dataclass(frozen=True)
class PipelineEvent:
    status: str
    stage: str
    partial_text: str = ""
    structured_payload: Mapping[str, Any] | None = None
    raw_trace: str = ""
    evidence: tuple[EvidenceRecord, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    conversation_id: str | None = None
    run_id: str | None = None
    error: str | None = None

    @property
    def payload(self) -> Mapping[str, Any] | None:
        return self.structured_payload

    @property
    def stage_name(self) -> str:
        return self.stage


@dataclass(frozen=True)
class _GenerationOutcome:
    payload: Mapping[str, Any] | None
    raw: str
    metrics: Mapping[str, Any]
    error: str | None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


_CITATION_KEYS = {"citations", "citation_ids", "cited_evidence_ids", "supported_evidence_ids", "selected_evidence_ids", "evidence_ids", "evidence_ledger"}


def _citations(payload: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key in _CITATION_KEYS:
                found |= _citations(value)
    elif isinstance(payload, (list, tuple, set)):
        found |= {str(value) for value in payload if isinstance(value, str)}
    elif isinstance(payload, str):
        found.add(payload)
    return found


def _sanitize_citation_fields(payload: Mapping[str, Any], allowed: set[str]) -> dict[str, Any]:
    """Remove hallucinated IDs from explicit citation fields before persistence."""

    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _CITATION_KEYS:
            if isinstance(value, (list, tuple, set)):
                cleaned[key] = list(dict.fromkeys(str(item) for item in value if str(item) in allowed))
            elif isinstance(value, str):
                cleaned[key] = value if value in allowed else ""
            else:
                cleaned[key] = value
        elif isinstance(value, Mapping):
            cleaned[key] = _sanitize_citation_fields(value, allowed)
        elif isinstance(value, list):
            cleaned[key] = [_sanitize_citation_fields(item, allowed) if isinstance(item, Mapping) else item for item in value]
        else:
            cleaned[key] = value
    return cleaned


def _inline_citations(payload: Mapping[str, Any]) -> set[str]:
    text = str(payload.get("counter_narrative", ""))
    return set(re.findall(r"\[(E[1-9][0-9]*)\]", text))


def _inline_citation_list(payload: Mapping[str, Any]) -> list[str]:
    text = str(payload.get("counter_narrative", ""))
    return re.findall(r"\[(E[1-9][0-9]*)\]", text)


def _declared_final_citations(payload: Mapping[str, Any]) -> list[str]:
    value = payload.get("cited_evidence_ids", ())
    return [str(item) for item in value] if isinstance(value, (list, tuple)) else []


def _validate_final_citations(payload: Mapping[str, Any], evidence_ids: set[str]) -> tuple[bool, str]:
    declared = _declared_final_citations(payload)
    inline = _inline_citation_list(payload)
    if len(declared) != len(set(declared)):
        return False, "final declared citations contain duplicates"
    if len(inline) != len(set(inline)):
        return False, "final inline citations contain duplicates"
    if set(declared) != set(inline):
        return False, "final declared citations must exactly match inline [E#] citations"
    all_ids = set(declared) | set(inline)
    if not all_ids <= evidence_ids:
        return False, "final citations are outside the current evidence ledger"
    if payload.get("factual_claims") and not inline:
        return False, "factual final response requires inline [E#] citations"
    return True, ""


def _schema_shape(payload: Any, schema: Mapping[str, Any]) -> bool:
    if schema.get("type") == "object":
        return isinstance(payload, Mapping) and all(key in payload for key in schema.get("required", ())) and (schema.get("additionalProperties") is not False or not (set(payload) - set(schema.get("properties", {})))) and all(_schema_shape(payload[key], spec) for key, spec in schema.get("properties", {}).items() if key in payload)
    if schema.get("type") == "array":
        return isinstance(payload, list) and all(_schema_shape(item, schema.get("items", {})) for item in payload)
    if isinstance(schema.get("type"), list):
        return any(_schema_shape(payload, {"type": kind}) for kind in schema["type"])
    if schema.get("type") == "string":
        import re
        return isinstance(payload, str) and (not schema.get("pattern") or re.fullmatch(schema["pattern"], payload) is not None)
    if schema.get("type") == "number":
        return isinstance(payload, (int, float)) and not isinstance(payload, bool) and schema.get("minimum", float("-inf")) <= payload <= schema.get("maximum", float("inf"))
    return True


def load_generator(config: DemoConfig) -> Any:
    """Load the configured generation backend; never substitute a fake one."""

    backend = str(getattr(config, "generator_backend", "ollama") or "ollama").casefold()
    if backend == "ollama":
        return OllamaGenerator.load(config)
    if backend == "mlx":
        return MLXGenerator.load(config)
    raise RuntimeError(f"unknown generator backend: {backend}")


class MPKGRAGPipeline:
    def __init__(self, config: DemoConfig, retriever: Any | None = None, generator: Any | None = None, history: HistoryStore | None = None) -> None:
        self.config = config
        if retriever is None:
            # Tiny source manifests are the deterministic test fixture and
            # intentionally use injected/hash seams.  Production artifacts
            # load the real BGE backends only when the lazy pipeline resolves.
            fixture = False
            try:
                source_manifest = config.source_manifest_path.read_text(encoding="utf-8")
                fixture = json.loads(source_manifest).get("identity", {}).get("stage_hash") == "tiny"
            except (OSError, TypeError, ValueError, AttributeError):
                pass
            if fixture:
                self.retriever = LocalRetriever(config)
            else:
                embedding, reranker = load_retrieval_backends(config)
                self.retriever = LocalRetriever(config, dense_encoder=embedding, reranker=reranker)
        else:
            self.retriever = retriever
        # Loading the generator (an Ollama connection check, or MLX on Apple
        # silicon) is deferred until a non-cancelled run actually needs
        # generation.  This keeps cancellation side-effect free and makes the
        # pipeline constructible before the model server is running.
        self.generator = generator
        self.history = history or HistoryStore(config.history_path, safe_export_dir=config.safe_export_dir)

    def _generation(self, messages: list[dict[str, str]], schema: Mapping[str, Any], budget: int, temperature: float, cancel: threading.Event, *, enable_thinking: bool = True) -> Iterator[GenerationEvent]:
        if cancel.is_set():
            return
        if self.generator is None:
            if cancel.is_set():
                return
            self.generator = load_generator(self.config)
        stream_fn = self.generator.stream_structured
        # Keep compatibility with tiny test generators while passing the
        # thinking flag to the real generator and to generators that expose it.
        try:
            stream = stream_fn(messages, schema, budget, temperature, cancel=cancel, enable_thinking=enable_thinking)
        except TypeError as exc:
            if "unexpected keyword" not in str(exc) and "keyword argument" not in str(exc):
                raise
            try:
                stream = stream_fn(messages, schema, budget, temperature, cancel=cancel)
            except TypeError as exc2:
                if "unexpected keyword" not in str(exc2) and "keyword argument" not in str(exc2):
                    raise
                stream = stream_fn(messages, schema, budget, temperature)
        yield from stream

    def _generate_live(self, messages: list[dict[str, str]], schema: Mapping[str, Any], budget: int, temperature: float, cancel: threading.Event, *, enable_thinking: bool = True) -> Iterator[GenerationEvent]:
        """Yield model events as the model produces them and return its outcome."""
        events: list[GenerationEvent] = []
        for event in self._generation(messages, schema, budget, temperature, cancel, enable_thinking=enable_thinking):
            if cancel.is_set():
                break
            events.append(event)
            yield event
        final = next((event for event in reversed(events) if event.done), events[-1] if events else GenerationEvent(done=True))
        payload = final.structured if isinstance(final.structured, Mapping) and _schema_shape(final.structured, schema) else None
        raw = final.raw_trace or "".join(event.raw_trace or event.text for event in events)
        metrics = final.metrics
        # The real generators perform their own repair.  A supplied lightweight
        # generator may not, so retain the pipeline's single repair seam.
        if payload is None and not cancel.is_set() and not final.repaired:
            repair_messages = messages + [{"role": "user", "content": _repair_prompt(raw, schema, final.error)}]
            repair_events: list[GenerationEvent] = []
            for event in self._generation(repair_messages, schema, budget, 0.0, cancel, enable_thinking=False):
                if cancel.is_set():
                    break
                repair_events.append(event)
                yield event
            repair_final = next((event for event in reversed(repair_events) if event.done), repair_events[-1] if repair_events else GenerationEvent(done=True))
            payload = repair_final.structured if isinstance(repair_final.structured, Mapping) and _schema_shape(repair_final.structured, schema) else None
            raw = raw + ("\n[repair]\n" + (repair_final.raw_trace or ""))
            metrics = repair_final.metrics
            final = repair_final
        return _GenerationOutcome(payload, raw, metrics, final.error if payload is None else None)

    def _pipeline_generation(self, stage: str, messages: list[dict[str, str]], schema: Mapping[str, Any], budget: int, temperature: float, cancel: threading.Event, evidence: tuple[EvidenceRecord, ...], conversation_id: str, run_id: str) -> Iterator[PipelineEvent]:
        """Adapt the live generator stream into UI-neutral pipeline events."""
        # The interview demo prioritizes a complete, inspectable structured
        # run. Qwen's thinking mode can consume the whole per-stage budget
        # before emitting JSON, so demo stages use direct structured mode;
        # the exact raw JSON output is still preserved in every trace panel.
        generator = self._generate_live(messages, schema, budget, temperature, cancel, enable_thinking=False)
        partial_text = ""
        evidence_ids = tuple(record.evidence_id for record in evidence)
        while True:
            try:
                event = next(generator)
            except StopIteration as stopped:
                outcome = stopped.value
                if not isinstance(outcome, _GenerationOutcome):
                    outcome = _GenerationOutcome(None, "", {}, "generation failed")
                if isinstance(outcome.payload, Mapping):
                    normalized = _sanitize_citation_fields(outcome.payload, set(evidence_ids))
                    if stage == "final":
                        # Unknown inline markers are not allowed to poison an
                        # otherwise usable response. Remove only the invalid
                        # marker; valid current-ledger citations are retained.
                        normalized["counter_narrative"] = re.sub(
                            r"\[(E[1-9][0-9]*)\]",
                            lambda match: match.group(0) if match.group(1) in set(evidence_ids) else "",
                            str(normalized.get("counter_narrative", "")),
                        )
                    if stage == "final":
                        inline = list(dict.fromkeys(_inline_citation_list(normalized)))
                        declared = list(dict.fromkeys(_declared_final_citations(normalized)))
                        if not inline and declared and set(declared) <= set(evidence_ids):
                            normalized["counter_narrative"] = (
                                str(normalized.get("counter_narrative", "")).rstrip()
                                + " Sources: "
                                + " ".join(f"[{citation}]" for citation in declared)
                            )
                            inline = declared
                        if inline and set(inline) <= set(evidence_ids):
                            normalized["cited_evidence_ids"] = inline
                    outcome = _GenerationOutcome(normalized, outcome.raw, outcome.metrics, outcome.error)
                # A model can finish with malformed JSON and no text delta
                # (for example when MLX emits the complete response only on
                # its terminal event). Persist that terminal raw output before
                # the caller validates the stage, so reopening shows both the
                # failed stage and the exact trace that caused it.
                terminal_error = outcome.error or (None if outcome.payload is not None else "schema_invalid")
                terminal_status = "completed" if outcome.payload is not None else ("cancelled" if terminal_error == "cancelled" else "failed")
                self.history.checkpoint_stage(
                    conversation_id, run_id, stage, outcome.payload or {},
                    raw_trace=outcome.raw, evidence_ids=evidence_ids,
                    metrics=outcome.metrics, partial_text=partial_text,
                    status=terminal_status, error=terminal_error,
                )
                return outcome
            if cancel.is_set():
                return _GenerationOutcome(None, event.raw_trace, event.metrics, "cancelled")
            if (event.raw_trace or event.done) and not (stage == "final" and event.done):
                # Keep the latest raw generation even when ``text`` is empty;
                # terminal MLX events commonly carry the only complete trace.
                self.history.checkpoint_stage(
                    conversation_id, run_id, stage, event.structured or {},
                    raw_trace=event.raw_trace, evidence_ids=evidence_ids,
                    metrics=event.metrics, partial_text=partial_text,
                    status="streaming" if not event.done else ("completed" if event.structured is not None else "failed"),
                    error=event.error if event.done else None,
                )
            if event.text:
                # ``event.text`` is a delta in MLX-LM.  Keep the cumulative
                # text in both the UI event and the durable checkpoint so an
                # interrupted run can be reopened with its last visible text.
                partial_text += event.text
                self.history.checkpoint_stage(
                    conversation_id, run_id, stage, event.structured or {}, raw_trace=event.raw_trace,
                    evidence_ids=evidence_ids,
                    metrics=event.metrics, partial_text=partial_text, status="streaming",
                )
                yield self._event("streaming", stage, text=partial_text, raw=event.raw_trace, evidence=evidence, metrics=event.metrics, conversation_id=conversation_id, run_id=run_id)

    def _event(self, status: str, stage: str, *, text: str = "", payload: Mapping[str, Any] | None = None, raw: str = "", evidence: tuple[EvidenceRecord, ...] = (), metrics: Mapping[str, Any] | None = None, conversation_id: str | None = None, run_id: str | None = None, error: str | None = None) -> PipelineEvent:
        return PipelineEvent(status, stage, text, payload, raw, evidence, dict(metrics or {}), conversation_id, run_id, error)

    def run(self, request: RunRequest, cancel: threading.Event | None = None) -> Iterator[PipelineEvent]:
        run_started = time.perf_counter()
        cancel = cancel or threading.Event()
        if cancel.is_set():
            yield self._event("cancelled", "pipeline", error="cancelled")
            return
        text, target, category = str(request.text or "").strip(), str(request.target or "").strip(), str(request.category or "").strip()
        if not text:
            yield self._event("failed", "metadata", error="query text is required")
            return
        conversation_id = request.conversation_id
        if cancel.is_set():
            yield self._event("cancelled", "pipeline", error="cancelled")
            return
        if conversation_id is None:
            conversation = self.history.create_conversation(request.title, input_text=text, target=target, category=category, metadata=request.metadata)
            conversation_id = conversation.id
        if cancel.is_set():
            yield self._event("cancelled", "pipeline", conversation_id=conversation_id, error="cancelled")
            return
        run_id = self.history.mark_run(conversation_id, {"input_text": text, "target": target, "category": category, "evidence_ids": []}, status="running")
        if cancel.is_set():
            self.history.mark_run(conversation_id, {"evidence_ids": []}, run_id=run_id, status="cancelled", error="cancelled")
            yield self._event("cancelled", "pipeline", conversation_id=conversation_id, run_id=run_id, error="cancelled")
            return
        active_stage = "metadata"
        try:
            if classify_demo_route(text, target, category) == "general_text":
                general_metadata = {"target": "general", "category": "general text", "inferred": False, "route": "general_text"}
                self.history.append_stage(conversation_id, run_id, "metadata", general_metadata, partial_text="General text — MP-KG-RAG was not invoked.")
                yield self._event("completed", "metadata", payload=general_metadata, text="General text — MP-KG-RAG was not invoked.", conversation_id=conversation_id, run_id=run_id)
                validation = {"valid": True, "route": "general_text", "label": "General text"}
                self.history.append_stage(conversation_id, run_id, "validation", validation)
                self.history.mark_run(conversation_id, {"route": "general_text"}, run_id=run_id, status="completed")
                yield self._event("completed", "validation", payload=validation, conversation_id=conversation_id, run_id=run_id)
                return
            # Metadata is always checkpointed, even when values were supplied by
            # the caller, so the UI has one stable first stage.
            metadata: dict[str, Any] = {"target": target, "category": category, "inferred": False}
            if not target or not category:
                if cancel.is_set():
                    raise _Cancelled
                messages = [{"role": "system", "content": "Infer missing Target and Category. Return exactly one JSON object."}, {"role": "user", "content": f"Infer Target and Category for: {text}"}]
                active_stage = "metadata"
                inferred_result = yield from self._pipeline_generation("metadata", messages, METADATA_SCHEMA, 256, 0.2, cancel, (), conversation_id, run_id)
                inferred, raw, metrics, error = inferred_result.payload, inferred_result.raw, inferred_result.metrics, inferred_result.error
                if cancel.is_set():
                    raise _Cancelled
                if inferred:
                    target, category = str(inferred.get("target") or target), str(inferred.get("category") or category)
                    metadata.update({"target": target, "category": category, "inferred": True})
                if error and (not target or not category):
                    raise ValueError("metadata inference failed: " + error)
            else:
                raw, metrics = "", {}
            # Persist before the completed event is observable.  This is
            # intentionally true for caller-supplied metadata as well.
            self.history.append_stage(conversation_id, run_id, "metadata", metadata, raw_trace=raw, metrics=metrics)
            yield self._event("completed", "metadata", payload=metadata, raw=raw, metrics=metrics, conversation_id=conversation_id, run_id=run_id)
            self.history.update_conversation_fields(conversation_id, input_text=text, target=target, category=category, metadata=request.metadata)
            if cancel.is_set():
                raise _Cancelled

            active_stage = "retrieval"
            query = QueryInput(text=text, target=target, category=category, query_signature=request.metadata.get("query_signature", {}))
            if cancel.is_set():
                raise _Cancelled
            result: RetrievalResult = self.retriever.retrieve(query, limit=self.config.max_evidence)
            if cancel.is_set():
                raise _Cancelled
            evidence = tuple(result.evidence)
            evidence_ids = tuple(record.evidence_id for record in evidence)
            self.history.mark_run(conversation_id, {"input_text": text, "target": target, "category": category, "evidence_ids": list(evidence_ids), "retrieval_trace": dict(result.trace)}, run_id=run_id, status="running")
            evidence_payload = [dict(asdict(record), routes=list(record.routes), scores=dict(record.scores), metadata=dict(record.metadata)) for record in evidence]
            retrieval_payload = {"evidence_ids": list(evidence_ids), "abstained": result.abstained, "reason": result.reason, "trace": dict(result.trace), "candidates": [_json(getattr(candidate, "metadata", candidate)) for candidate in result.candidates], "evidence": evidence_payload}
            self.history.append_stage(conversation_id, run_id, "retrieval", retrieval_payload, evidence_ids=evidence_ids)
            if cancel.is_set():
                raise _Cancelled
            yield self._event("completed", "retrieval", payload=retrieval_payload, evidence=evidence, conversation_id=conversation_id, run_id=run_id)
            if cancel.is_set():
                raise _Cancelled
            if not evidence:
                raise ValueError(result.reason or "retrieval returned no supported evidence; abstaining")
            if cancel.is_set():
                raise _Cancelled

            ledger = "\n".join(f"[{record.evidence_id}] {record.displayed_text or record.text}" for record in evidence)
            for name in PERSPECTIVES:
                messages = [{"role": "system", "content": "Return exactly one JSON object and no reasoning trace."}, {"role": "user", "content": f"Perspective: {name}\nGoal: {PERSPECTIVE_GOALS[name]}\nPost: {text}\nTarget: {target}\nCategory: {category}\nEvidence (complete frozen ledger):\n{ledger}\nUse only E IDs from this ledger."}]
                if cancel.is_set():
                    raise _Cancelled
                active_stage = name
                outcome = yield from self._pipeline_generation(name, messages, PERSPECTIVE_SCHEMA, 768, 0.6, cancel, evidence, conversation_id, run_id)
                payload, raw, metrics, error = outcome.payload, outcome.raw, outcome.metrics, outcome.error
                if cancel.is_set():
                    raise _Cancelled
                if payload is None or payload.get("perspective") != name or not _citations(payload) <= set(evidence_ids):
                    raise _StageInvalid(name, error or "perspective schema/citation validation failed")
                self.history.append_stage(conversation_id, run_id, name, payload, raw_trace=raw, evidence_ids=evidence_ids, metrics=metrics, partial_text=str(payload.get("rationale") or ""))
                yield self._event("completed", name, payload=payload, raw=raw, evidence=evidence, metrics=metrics, conversation_id=conversation_id, run_id=run_id)
                if cancel.is_set():
                    raise _Cancelled

            perspectives = [stage.payload for stage in self.history.get_conversation(conversation_id).stages if stage.stage_name in PERSPECTIVES]
            plan_messages = [{"role": "system", "content": "Return one JSON evidence plan only."}, {"role": "user", "content": f"Post: {text}\nTarget: {target}\nEvidence (complete frozen ledger):\n{ledger}\nPerspectives: {_json(perspectives)}\nReturn claim_focus, selected_evidence_ids, response_steps, tone, factual_constraints, safety_constraints. Use only E IDs."}]
            if cancel.is_set():
                raise _Cancelled
            active_stage = "plan"
            outcome = yield from self._pipeline_generation("plan", plan_messages, PLAN_SCHEMA, 768, 0.6, cancel, evidence, conversation_id, run_id)
            plan, raw, metrics, error = outcome.payload, outcome.raw, outcome.metrics, outcome.error
            if cancel.is_set():
                raise _Cancelled
            if plan is not None and not plan.get("selected_evidence_ids"):
                supported = [
                    citation
                    for perspective in perspectives
                    for citation in perspective.get("supported_evidence_ids", ())
                    if citation in set(evidence_ids)
                ]
                plan = {**plan, "selected_evidence_ids": list(dict.fromkeys(supported)) or list(evidence_ids)}
            if plan is None or not _citations(plan) <= set(evidence_ids):
                raise _StageInvalid("plan", error or "plan schema/citation validation failed")
            self.history.append_stage(conversation_id, run_id, "plan", plan, raw_trace=raw, evidence_ids=evidence_ids, metrics=metrics)
            yield self._event("completed", "plan", payload=plan, raw=raw, evidence=evidence, metrics=metrics, conversation_id=conversation_id, run_id=run_id)
            if cancel.is_set():
                raise _Cancelled

            final_messages = [{"role": "system", "content": "Write one safe evidence-grounded counter-narrative as JSON only."}, {"role": "user", "content": f"Final response\nPost: {text}\nTarget: {target}\nPlan: {_json(plan)}\nEvidence (complete frozen ledger):\n{ledger}\nFactual claims require inline [E] citations. Use only E IDs."}]
            if cancel.is_set():
                raise _Cancelled
            active_stage = "final"
            outcome = yield from self._pipeline_generation("final", final_messages, FINAL_SCHEMA, 768, 0.6, cancel, evidence, conversation_id, run_id)
            final, raw, metrics, error = outcome.payload, outcome.raw, outcome.metrics, outcome.error
            if cancel.is_set():
                raise _Cancelled
            if final is None:
                raise _StageInvalid("final", error or "final schema/citation validation failed")
            # The inline markers are the authoritative citation record shown
            # to the user. Normalize the declared array to those exact markers
            # when they all belong to the frozen evidence ledger. If Qwen
            # omitted inline markers, append a compact Sources suffix; the
            # verbatim raw model trace remains untouched.
            inline_citations = list(dict.fromkeys(_inline_citation_list(final)))
            declared_citations = list(dict.fromkeys(_declared_final_citations(final)))
            if not declared_citations:
                declared_citations = [
                    citation for citation in plan.get("selected_evidence_ids", ()) if citation in set(evidence_ids)
                ]
            if not inline_citations and declared_citations and set(declared_citations) <= set(evidence_ids):
                final = {
                    **final,
                    "counter_narrative": str(final.get("counter_narrative", "")).rstrip()
                    + " Sources: "
                    + " ".join(f"[{citation}]" for citation in declared_citations),
                }
                inline_citations = declared_citations
            if inline_citations and set(inline_citations) <= set(evidence_ids):
                final = {**final, "cited_evidence_ids": inline_citations}
            citations_valid, citation_error = _validate_final_citations(final, set(evidence_ids))
            if not citations_valid:
                raise _StageInvalid("final", citation_error)
            final_citations = set(_declared_final_citations(final))
            metrics = {
                **dict(metrics or {}),
                "end_to_end_seconds": round(time.perf_counter() - run_started, 3),
            }
            self.history.append_stage(conversation_id, run_id, "final", final, raw_trace=raw, evidence_ids=evidence_ids, metrics=metrics, partial_text=str(final.get("counter_narrative", "")))
            yield self._event("completed", "final", payload=final, text=str(final.get("counter_narrative", "")), raw=raw, evidence=evidence, metrics=metrics, conversation_id=conversation_id, run_id=run_id)
            if cancel.is_set():
                raise _Cancelled
            validation = {"valid": True, "cited_evidence_ids": sorted(final_citations), "unknown_evidence_ids": []}
            self.history.append_stage(conversation_id, run_id, "validation", validation, evidence_ids=evidence_ids)
            self.history.mark_run(conversation_id, {"input_text": text, "target": target, "category": category, "evidence_ids": list(evidence_ids)}, run_id=run_id, status="completed")
            yield self._event("completed", "validation", payload=validation, evidence=evidence, conversation_id=conversation_id, run_id=run_id)
        except _Cancelled:
            self._mark_stage_terminal(conversation_id, run_id, active_stage, "cancelled", "cancelled")
            self.history.mark_run(conversation_id, {"evidence_ids": list(locals().get("evidence_ids", ()))}, run_id=run_id, status="cancelled", error="cancelled")
            yield self._event("cancelled", "pipeline", conversation_id=conversation_id, run_id=run_id, error="cancelled")
        except Exception as exc:
            self._mark_stage_terminal(conversation_id, run_id, getattr(exc, "stage", active_stage), "failed", str(exc))
            self.history.mark_run(conversation_id, {"evidence_ids": list(locals().get("evidence_ids", ()))}, run_id=run_id, status="failed", error=str(exc))
            yield self._event("failed", getattr(exc, "stage", "pipeline"), evidence=tuple(locals().get("evidence", ())), conversation_id=conversation_id, run_id=run_id, error=str(exc))

    def _mark_stage_terminal(self, conversation_id: str, run_id: str, stage_name: str, status: str, error: str) -> None:
        """Upgrade the latest checkpoint so reopened runs show its terminal error."""
        try:
            conversation = self.history.get_conversation(conversation_id)
            stage = next((item for item in conversation.stages if item.stage_name == stage_name), None) if conversation else None
            if stage is None:
                return
            self.history.checkpoint_stage(
                conversation_id, run_id, stage_name, stage.payload,
                raw_trace=stage.raw_trace, metrics=stage.metrics,
                partial_text=stage.partial_text, status=status, error=error,
                evidence_ids=tuple(conversation.run_payload.get("evidence_ids", ())) if conversation else None,
            )
        except (KeyError, ValueError, InvalidCitationError):
            # Run-level status/error remains authoritative if a stage update
            # cannot be written (for example a legacy/failing history adapter).
            return


class _Cancelled(Exception):
    pass


class _StageInvalid(Exception):
    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(message)


__all__ = ["MPKGRAGPipeline", "PipelineEvent", "RunRequest", "FINAL_SCHEMA", "METADATA_SCHEMA", "PERSPECTIVE_SCHEMA", "PLAN_SCHEMA", "classify_demo_route"]
