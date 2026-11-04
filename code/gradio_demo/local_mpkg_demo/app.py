"""Gradio Pipeline Lab for the local MP-KG-RAG demonstration.

The module deliberately keeps Gradio behind a lazy import.  Retrieval and
pipeline tests can therefore run on machines without Gradio, Ollama, Metal, or MLX.
The small :class:`UIState` and rendering helpers are also the test seam for
the browser-facing event flow.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import inspect
import json
import queue
import threading
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import DemoConfig, EvidenceRecord, PERSPECTIVES
from .history import HistoryStore
from .index_store import resolve_current_generation
from .pipeline import MPKGRAGPipeline, PipelineEvent, RunRequest


STAGES = ("metadata", "retrieval", *PERSPECTIVES, "plan", "final", "validation")
EVIDENCE_HEADERS = (
    "Evidence ID", "Passage", "Source", "Page / locator", "Span",
    "Integrity hash", "Routes", "Scores",
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _record_dict(record: EvidenceRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(record, EvidenceRecord):
        data = {
            "evidence_id": record.evidence_id,
            "text": record.text,
            "displayed_text": record.displayed_text or record.text,
            "document_uid": record.document_uid,
            "chunk_id": record.chunk_id,
            "source_id": record.source_id,
            "source_type": record.source_type,
            "locator": record.locator,
            "page": record.page,
            "span_start": record.span_start,
            "span_end": record.span_end,
            "integrity_hash": record.integrity_hash,
            "routes": list(record.routes),
            "scores": dict(record.scores),
        }
    else:
        data = dict(record)
    return data


def evidence_rows(records: Iterable[EvidenceRecord | Mapping[str, Any]]) -> list[list[str]]:
    """Convert immutable evidence records into the provenance table rows."""

    rows: list[list[str]] = []
    for record in records:
        data = _record_dict(record)
        page = data.get("page")
        locator = str(data.get("locator") or "")
        page_locator = locator or (f"page {page}" if page is not None else "")
        span_start, span_end = data.get("span_start"), data.get("span_end")
        span = ""
        if span_start is not None or span_end is not None:
            span = f"{span_start if span_start is not None else '?'}–{span_end if span_end is not None else '?'}"
        source_id = str(data.get("source_id") or "")
        document_uid = str(data.get("document_uid") or "")
        source_type = str(data.get("source_type") or "")
        source_parts = [part for part in (source_id, document_uid, source_type) if part]
        source = " · ".join(dict.fromkeys(source_parts))
        rows.append([
            str(data.get("evidence_id") or ""),
            str(data.get("displayed_text") or data.get("text") or ""),
            source,
            page_locator,
            span,
            str(data.get("integrity_hash") or ""),
            ", ".join(str(item) for item in data.get("routes", ()) if item),
            _json(data.get("scores") or {}),
        ])
    return rows


@dataclass
class UIState:
    """Serializable view model consumed by Gradio callbacks and tests."""

    conversation_id: str | None = None
    run_id: str | None = None
    text: str = ""
    target: str = ""
    category: str = ""
    status: str = "ready"
    error: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    stage_status: dict[str, str] = field(default_factory=lambda: {stage: "pending" for stage in STAGES})
    perspectives: dict[str, dict[str, Any]] = field(default_factory=dict)
    planner: dict[str, Any] = field(default_factory=lambda: {"structured": {}, "raw_trace": "", "metrics": {}, "partial_text": ""})
    final: dict[str, Any] = field(default_factory=lambda: {"structured": {}, "raw_trace": "", "metrics": {}, "partial_text": ""})
    timeline: list[str] = field(default_factory=list)

    @property
    def final_answer(self) -> str:
        return str(self.final.get("structured", {}).get("counter_narrative") or self.final.get("partial_text") or "")

    def copy(self) -> "UIState":
        return UIState(
            conversation_id=self.conversation_id,
            run_id=self.run_id,
            text=self.text,
            target=self.target,
            category=self.category,
            status=self.status,
            error=self.error,
            evidence=[dict(row) for row in self.evidence],
            stage_status=dict(self.stage_status),
            perspectives={name: dict(value) for name, value in self.perspectives.items()},
            planner=dict(self.planner),
            final=dict(self.final),
            timeline=list(self.timeline),
        )


def apply_pipeline_event(state: UIState, event: PipelineEvent) -> UIState:
    """Apply one UI-neutral pipeline event without requiring Gradio."""

    next_state = state.copy()
    if event.conversation_id:
        next_state.conversation_id = event.conversation_id
    if event.run_id:
        next_state.run_id = event.run_id
    if event.evidence:
        next_state.evidence = [_record_dict(record) for record in event.evidence]
    elif event.stage == "retrieval" and event.structured_payload and event.structured_payload.get("evidence"):
        next_state.evidence = [_record_dict(record) for record in event.structured_payload["evidence"]]
    if event.stage in next_state.stage_status and event.status in {"streaming", "completed", "failed", "cancelled"}:
        next_state.stage_status[event.stage] = event.status
    if event.status == "streaming":
        next_state.status = "running"
    elif event.status == "completed":
        # A final answer is still provisional until citation validation has
        # completed. Only validation (or an explicit success event) is
        # terminal success.
        next_state.status = "completed" if event.stage in {"validation", "success"} else "running"
    elif event.status in {"failed", "cancelled"}:
        next_state.status = event.status
        next_state.error = event.error or ("cancelled" if event.status == "cancelled" else "pipeline failed")
    if event.stage == "metadata" and event.structured_payload:
        next_state.target = str(event.structured_payload.get("target") or next_state.target)
        next_state.category = str(event.structured_payload.get("category") or next_state.category)
    elif event.stage in PERSPECTIVES:
        value = next_state.perspectives.setdefault(event.stage, {"structured": {}, "raw_trace": "", "metrics": {}, "partial_text": ""})
        if event.structured_payload is not None:
            value["structured"] = dict(event.structured_payload)
        if event.raw_trace:
            value["raw_trace"] = event.raw_trace
        if event.partial_text:
            value["partial_text"] = event.partial_text
        if event.error:
            value["error"] = event.error
        value["metrics"] = dict(event.metrics or value.get("metrics", {}))
    elif event.stage == "plan":
        if event.structured_payload is not None:
            next_state.planner["structured"] = dict(event.structured_payload)
        if event.raw_trace:
            next_state.planner["raw_trace"] = event.raw_trace
        if event.partial_text:
            next_state.planner["partial_text"] = event.partial_text
        next_state.planner["metrics"] = dict(event.metrics or next_state.planner.get("metrics", {}))
        if event.error:
            next_state.planner["error"] = event.error
    elif event.stage == "final":
        if event.structured_payload is not None:
            next_state.final["structured"] = dict(event.structured_payload)
        if event.raw_trace:
            next_state.final["raw_trace"] = event.raw_trace
        if event.partial_text:
            next_state.final["partial_text"] = event.partial_text
        next_state.final["metrics"] = dict(event.metrics or next_state.final.get("metrics", {}))
        if event.error:
            next_state.final["error"] = event.error
    if event.stage not in next_state.timeline and event.status == "completed":
        next_state.timeline.append(event.stage)
    return next_state


def consume_pipeline_events(
    events: Iterable[PipelineEvent], *, history: HistoryStore | None = None,
    text: str = "", target: str = "", category: str = "",
) -> UIState:
    """Consume a mocked or real event stream and return its complete view model.

    Real pipeline streams already checkpoint stages.  For tests and alternate
    pipeline adapters that omit IDs, this helper creates a small durable run so
    the UI contract still includes persisted history.
    """

    state = UIState(text=text, target=target, category=category)
    pending_events: list[PipelineEvent] = []
    for event in events:
        pending_events.append(event)
        state = apply_pipeline_event(state, event)
    if history is not None and pending_events and not state.conversation_id:
        conversation = history.create_conversation("New conversation", input_text=text, target=state.target, category=state.category)
        state.conversation_id = conversation.id
        run_id = history.mark_run(conversation.id, {"input_text": text, "target": state.target, "category": state.category, "evidence_ids": [row.get("evidence_id") for row in state.evidence]}, status="running")
        state.run_id = run_id
        evidence_ids = [row.get("evidence_id") for row in state.evidence]
        for event in pending_events:
            try:
                history.checkpoint_stage(
                    conversation.id, run_id, event.stage, event.structured_payload or {},
                    raw_trace=event.raw_trace, evidence_ids=evidence_ids,
                    metrics=event.metrics, partial_text=event.partial_text,
                    status=event.status, error=event.error,
                )
            except (KeyError, ValueError):
                continue
        terminal = state.status if state.status in {"failed", "cancelled"} else ("completed" if state.status == "completed" else "running")
        history.mark_run(conversation.id, {"evidence_ids": evidence_ids}, run_id=run_id, status=terminal, error=state.error or None)
    return state


def history_choices(conversations: Iterable[Any]) -> list[tuple[str, str]]:
    return [(f"{item.title} · {item.status}", item.id) for item in conversations]


def state_summary(state: UIState) -> dict[str, Any]:
    """Return a framework-neutral summary useful to smoke tests and logs."""

    return {
        "conversation_id": state.conversation_id,
        "status": state.status,
        "target": state.target,
        "category": state.category,
        "timeline": list(state.timeline),
        "evidence": list(state.evidence),
        "perspectives": {name: value for name, value in state.perspectives.items()},
        "planner": dict(state.planner),
        "final": dict(state.final),
        "final_answer": str(state.final.get("structured", {}).get("counter_narrative") or state.final.get("partial_text") or ""),
        "error": state.error,
    }


def _import_gradio() -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio 6.26.0 is required to launch the Pipeline Lab; install requirements-local-demo.txt") from exc
    return gr


def _component_update(gr: Any, component: Any, **kwargs: Any) -> Any:
    """Build a Gradio 6 component update while retaining old test doubles."""

    cls = component if isinstance(component, type) else type(component)
    try:
        return cls(**kwargs)
    except Exception:
        update = getattr(gr, "update", None)
        return update(**kwargs) if update else kwargs


class LazyPipelineProvider:
    """Resolve a pipeline on demand and cache only successful construction.

    Building the local retriever validates/builds the index, so doing that in
    the Gradio app constructor makes the UI impossible to open on a fresh
    checkout.  The lock also protects the first-run path when Gradio workers
    submit more than one request at once.
    """

    def __init__(self, factory: Any, config: DemoConfig | None = None) -> None:
        self._factory = factory
        self._config = config
        self._pipeline: Any | None = None
        self._lock = threading.Lock()
        self._last_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def resolve(self) -> Any:
        pipeline = self._pipeline
        if pipeline is not None:
            return pipeline
        with self._lock:
            pipeline = self._pipeline
            if pipeline is not None:
                return pipeline
            try:
                pipeline = self._invoke_factory()
                if pipeline is None or not callable(getattr(pipeline, "run", None)):
                    raise TypeError("pipeline provider must return an object with run(request, cancel)")
            except Exception as exc:
                # Do not cache failures: a user can repair setup while the UI
                # remains open and retry Run without rebuilding the app.
                self._last_error = str(exc)
                raise
            self._pipeline = pipeline
            self._last_error = None
            return pipeline

    def _invoke_factory(self) -> Any:
        """Support both zero-argument and config-aware test/application factories."""
        if self._config is None:
            return self._factory()
        try:
            parameters = inspect.signature(self._factory).parameters.values()
        except (TypeError, ValueError):
            return self._factory()
        required_positional = [
            parameter for parameter in parameters
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            and parameter.default is parameter.empty
        ]
        return self._factory(self._config) if required_positional else self._factory()


def _pipeline_factory(config: DemoConfig, supplied: Any | None) -> Any:
    if supplied is None:
        return lambda: MPKGRAGPipeline(config)
    if callable(getattr(supplied, "run", None)):
        return lambda: supplied
    if callable(supplied):
        return supplied
    for method_name in ("resolve", "get_pipeline", "get"):
        method = getattr(supplied, method_name, None)
        if callable(method):
            return method
    raise TypeError("pipeline must be a pipeline instance, factory, or provider")


class _Controller:
    def __init__(self, config: DemoConfig, pipeline: MPKGRAGPipeline | Any | None = None, *, pipeline_factory: Any | None = None) -> None:
        self.config = config
        self.history = getattr(pipeline, "history", None) if pipeline is not None else None
        if self.history is None:
            self.history = HistoryStore(config.history_path, safe_export_dir=config.safe_export_dir)
        supplied = pipeline_factory if pipeline_factory is not None else pipeline
        if pipeline is None and pipeline_factory is None:
            factory = lambda: MPKGRAGPipeline(config, history=self.history)
        elif pipeline_factory is not None:
            factory = pipeline_factory if callable(pipeline_factory) else _pipeline_factory(config, pipeline_factory)
        else:
            factory = _pipeline_factory(config, supplied)
        self.pipeline_provider = LazyPipelineProvider(factory, config)
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def cancel(self, token: str | None) -> bool:
        if not token:
            return False
        with self._lock:
            event = self._cancel.get(token)
        if event is not None:
            event.set()
            return True
        return False

    def readiness(self) -> str:
        try:
            resolve_current_generation(self.config)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            index = f"not ready — {exc}; run `python work/build_local_mpkg_demo_index.py`"
        else:
            index = "ready"
        if self.pipeline_provider.loaded:
            pipeline = "loaded"
        elif self.pipeline_provider.last_error:
            pipeline = f"setup failed — {self.pipeline_provider.last_error}"
        else:
            pipeline = "not loaded — initialized on first Run"
        if getattr(self.config, "generator_backend", "mlx") == "ollama":
            model = f"`{self.config.ollama_model}` via Ollama at `{self.config.ollama_url}` (checked on Run)"
        else:
            model = f"`{self.config.qwen_model}` (loaded on Run)"
        return f"**Local readiness**  \nPipeline: {pipeline}  \nIndex: {index}  \nHistory: `{self.config.history_path}`  \nModel: {model}"

    def history_list(self, search: str = "") -> list[tuple[str, str]]:
        return history_choices(self.history.list_conversations(search.strip()))

    def load_state(self, conversation_id: str | None) -> UIState:
        state = UIState(conversation_id=conversation_id)
        conversation = self.history.get_conversation(conversation_id) if conversation_id else None
        if conversation is None:
            return state
        state.text, state.target, state.category = conversation.input_text, conversation.target, conversation.category
        state.status = conversation.status
        state.error = conversation.run_error or ""
        for stage in conversation.stages:
            event = PipelineEvent(
                stage.status, stage.stage_name, partial_text=stage.partial_text,
                structured_payload=stage.payload, raw_trace=stage.raw_trace,
                metrics=stage.metrics, error=stage.error,
            )
            state = apply_pipeline_event(state, event)
        # The durable run status is authoritative after replay.
        state.status = conversation.status
        state.error = conversation.run_error or state.error
        return state

    def stream(self, text: str, target: str, category: str, conversation_id: str | None):
        token = str(uuid.uuid4())
        cancel = threading.Event()
        with self._lock:
            self._cancel[token] = cancel
        state = UIState(conversation_id=conversation_id, text=str(text or ""), target=str(target or ""), category=str(category or ""), status="running")
        try:
            yield token, state
            request = RunRequest(text=text, target=target, category=category, conversation_id=conversation_id)
            pipeline = self.pipeline_provider.resolve()
            for event in pipeline.run(request, cancel):
                state = apply_pipeline_event(state, event)
                yield token, state
        except Exception as exc:
            state = state.copy()
            state.status = "failed"
            state.error = (
                "Pipeline setup failed: " + str(exc) + ". "
                "Configure the embedding and reranker backends, build a compatible local index, "
                "and ensure the pinned Qwen model is available before running."
            )
            yield token, state
        finally:
            with self._lock:
                self._cancel.pop(token, None)


def _markdown_status(state: UIState) -> str:
    if state.error:
        return f"**{state.status.title()}** — {state.error}"
    if state.status == "completed" and state.category.casefold() == "general text":
        return f"**General text** — MP-KG-RAG not invoked; {len(state.timeline)}/{len(STAGES)} route stages checkpointed"
    elapsed = state.final.get("metrics", {}).get("end_to_end_seconds")
    if state.status == "completed" and isinstance(elapsed, (int, float)):
        minutes, seconds = divmod(float(elapsed), 60.0)
        duration = f"{int(minutes)}m {seconds:.1f}s" if minutes >= 1 else f"{seconds:.1f}s"
        return f"**Completed in {duration}** — {len(state.timeline)}/{len(STAGES)} stages checkpointed"
    return f"**{state.status.title()}** — {len(state.timeline)}/{len(STAGES)} stages checkpointed"


def _timeline_markdown(state: UIState) -> str:
    labels = []
    for stage in STAGES:
        status = state.stage_status.get(stage, "pending")
        icon = {"completed": "✓", "streaming": "◌", "failed": "!", "cancelled": "×"}.get(status, "·")
        labels.append(f"`{icon}` **{stage.replace('_', ' ').title()}** — {status}")
    return "  \n".join(labels)


def _chat_value(state: UIState) -> list[dict[str, str]]:
    if not state.text and not state.final.get("structured") and not state.final.get("partial_text"):
        return []
    answer = str(state.final.get("structured", {}).get("counter_narrative") or state.final.get("partial_text") or "")
    if not answer and state.status == "running":
        answer = "_Pipeline is running — stage updates appear on the right._"
    return ([{"role": "user", "content": state.text}] if state.text else []) + ([{"role": "assistant", "content": answer}] if answer else [])


def _state_values(state: UIState, controls: Mapping[str, Any], gr: Any) -> list[Any]:
    values: list[Any] = [
        _markdown_status(state), _chat_value(state), state.target, state.category,
        _timeline_markdown(state), evidence_rows(state.evidence),
    ]
    for name in PERSPECTIVES:
        value = state.perspectives.get(name, {})
        values += [value.get("structured", {}), value.get("raw_trace", "")]
    values += [state.planner.get("structured", {}), state.planner.get("raw_trace", "")]
    values += [str(state.final.get("structured", {}).get("counter_narrative") or state.final.get("partial_text") or ""), state.final.get("structured", {}), state.final.get("raw_trace", ""), state.final.get("metrics", {})]
    # The radio doubles as the selected conversation input and is refreshed
    # on every streamed state. This makes a newly-created run visible as soon
    # as its first durable event is persisted.
    values += [_component_update(gr, controls["history"], choices=controls["history_choices"](), value=state.conversation_id), None, _component_update(gr, controls["run"], value="Run MP-KG-RAG", interactive=state.status != "running"), _component_update(gr, controls["cancel"], value="Cancel", interactive=state.status == "running")]
    return values


def create_app(config: DemoConfig, pipeline: MPKGRAGPipeline | Any | None = None, *, pipeline_factory: Any | None = None, pipeline_provider: Any | None = None) -> Any:
    """Create the dark responsive MP-only Pipeline Lab Gradio application."""

    gr = _import_gradio()
    if pipeline_factory is not None and pipeline_provider is not None:
        raise TypeError("provide only one of pipeline_factory and pipeline_provider")
    supplied_factory = pipeline_factory if pipeline_factory is not None else pipeline_provider
    controller = _Controller(config, pipeline, pipeline_factory=supplied_factory)
    # MLX Metal streams are thread-local, and a local Ollama server serves one
    # CPU-bound generation at a time. Gradio may resume a streaming generator
    # on different worker threads, so all model work is pinned to one
    # persistent executor thread and bridged back through a queue.
    mlx_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mpkg-model")
    css_path = Path(__file__).with_name("ui.css")
    css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    with gr.Blocks(title="Counter-Narrative Generation for Homophobic Hate Speech in Hindi-Hinglish Text") as demo:
        gr.Markdown("# Counter-Narrative Generation for Homophobic Hate Speech in Hindi-Hinglish Text\nLocal evidence-grounded counter-narrative walkthrough", elem_classes="hero")
        run_token_state = gr.State(None)
        with gr.Row(equal_height=False, elem_classes="shell"):
            with gr.Column(scale=1, min_width=250, elem_classes="sidebar"):
                gr.Markdown("### Workspace")
                readiness = gr.Markdown(controller.readiness(), elem_classes="readiness")
                new_button = gr.Button("＋ New conversation", variant="secondary")
                history_search = gr.Textbox(label="Search history", placeholder="Title or query", lines=1)
                history_list = gr.Radio(choices=controller.history_list(), label="Conversations", type="value", interactive=True)
                rename_text = gr.Textbox(label="Rename selected", lines=1)
                with gr.Row():
                    rename_button = gr.Button("Rename", size="sm")
                    delete_button = gr.Button("Delete", size="sm", variant="stop")
                with gr.Row():
                    export_format = gr.Dropdown(["JSON", "Markdown"], value="JSON", label="Export format", scale=2)
                    export_button = gr.Button("Export", size="sm")
                export_file = gr.File(label="Download", visible=False)
            with gr.Column(scale=2, min_width=420, elem_classes="conversation-pane"):
                gr.Markdown("### Conversation")
                chat = gr.Chatbot(label="", height=420, show_label=False)
                query = gr.Textbox(label="Hate-speech text", placeholder="Paste the text you want to examine…", lines=5)
                with gr.Row():
                    target = gr.Textbox(label="Target (editable)", placeholder="Inferred target")
                    category = gr.Textbox(label="Category (editable)", placeholder="Inferred category")
                with gr.Row():
                    run_button = gr.Button("Run MP-KG-RAG", variant="primary")
                    cancel_button = gr.Button("Cancel", variant="stop", interactive=False)
                status = gr.Markdown("**Ready**", elem_classes="status")
            with gr.Column(scale=2, min_width=460, elem_classes="results-pane"):
                gr.Markdown("### Live pipeline")
                timeline = gr.Markdown(_timeline_markdown(UIState()), elem_classes="timeline")
                evidence = gr.Dataframe(headers=list(EVIDENCE_HEADERS), datatype=["str"] * len(EVIDENCE_HEADERS), value=[], label="Frozen evidence ledger", wrap=True, interactive=False)
                perspective_components: dict[str, tuple[Any, Any]] = {}
                for name in PERSPECTIVES:
                    with gr.Accordion(name.replace("_", " ").title(), open=False):
                        structured = gr.JSON(label="Structured analysis", value={})
                        raw = gr.Textbox(label="Raw model trace", value="", lines=8, max_lines=24, interactive=False)
                        perspective_components[name] = (structured, raw)
                with gr.Accordion("Planner", open=False):
                    planner_json = gr.JSON(label="Structured plan", value={})
                    planner_raw = gr.Textbox(label="Raw model trace", value="", lines=8, max_lines=24, interactive=False)
                with gr.Accordion("Final synthesis", open=False):
                    final_json = gr.JSON(label="Structured final", value={})
                    final_raw = gr.Textbox(label="Raw model trace", value="", lines=8, max_lines=24, interactive=False)
                final_answer = gr.Markdown("", elem_classes="final-answer")
                final_metrics = gr.JSON(label="Final metrics", value={})

        output_components: list[Any] = [status, chat, target, category, timeline, evidence]
        for name in PERSPECTIVES:
            output_components.extend(perspective_components[name])
        output_components.extend([planner_json, planner_raw, final_answer, final_json, final_raw, final_metrics, history_list, run_token_state, run_button, cancel_button])
        controls = {"run": run_button, "cancel": cancel_button, "history": history_list, "history_choices": controller.history_list}

        def run_callback(text: str, target_value: str, category_value: str, selected_id: str | None):
            updates: queue.Queue[tuple[str, Any]] = queue.Queue()

            def produce() -> None:
                try:
                    for token, state in controller.stream(text, target_value, category_value, selected_id):
                        values = _state_values(state, controls, gr)
                        values[-3] = token
                        updates.put(("value", values))
                except BaseException as exc:
                    updates.put(("error", exc))
                finally:
                    updates.put(("done", None))

            future = mlx_executor.submit(produce)
            while True:
                kind, payload = updates.get()
                if kind == "value":
                    yield payload
                elif kind == "error":
                    raise payload
                else:
                    break
            future.result()

        run_button.click(run_callback, inputs=[query, target, category, history_list], outputs=output_components, show_progress="minimal", concurrency_limit=1)

        def cancel_callback(token: str | None):
            controller.cancel(token)
            return _component_update(gr, run_button, value="Stopping…", interactive=False), _component_update(gr, cancel_button, value="Cancel", interactive=False)

        cancel_button.click(cancel_callback, inputs=[run_token_state], outputs=[run_button, cancel_button], concurrency_limit=1)

        def new_callback():
            state = UIState()
            return _state_values(state, controls, gr)

        # Keep New conversation focused on the main workspace.  The history
        # choices are refreshed independently so no stale selection remains.
        # Keep callback return values in exactly the same order as its output
        # components.  Status is part of the shared view-model output too.
        new_outputs = output_components
        new_button.click(new_callback, outputs=new_outputs)

        history_search.change(lambda value: gr.Radio(choices=controller.history_list(value), value=None), inputs=[history_search], outputs=[history_list])

        def select_callback(selected: str | None):
            state = controller.load_state(selected)
            return _state_values(state, controls, gr)

        history_list.change(select_callback, inputs=[history_list], outputs=output_components)

        def rename_callback(selected: str | None, title: str, search: str):
            if selected:
                controller.history.rename_conversation(selected, title)
            return gr.Radio(choices=controller.history_list(search or ""), value=selected), "**Renamed**"

        rename_button.click(rename_callback, inputs=[history_list, rename_text, history_search], outputs=[history_list, status])

        def delete_callback(selected: str | None):
            if selected:
                controller.history.delete_conversation(selected)
            state = UIState()
            return _state_values(state, controls, gr)

        delete_outputs = output_components
        delete_button.click(delete_callback, inputs=[history_list], outputs=delete_outputs)

        def export_callback(selected: str | None, format_name: str):
            if not selected:
                return _component_update(gr, export_file, visible=False)
            suffix = ".json" if str(format_name).casefold() == "json" else ".md"
            export_dir = controller.history.safe_export_dir
            export_dir.mkdir(parents=True, exist_ok=True)
            destination = export_dir / f"conversation-{selected}-{uuid.uuid4().hex}{suffix}"
            controller.history.export_conversation(selected, "json" if suffix == ".json" else "markdown", destination=destination)
            return _component_update(gr, export_file, value=str(destination), visible=True)

        export_button.click(export_callback, inputs=[history_list, export_format], outputs=[export_file])
        demo.load(lambda: controller.readiness(), outputs=[readiness])
        demo.queue(default_concurrency_limit=1, max_size=4)
    # Gradio 6 moved launch styling out of ``Blocks(...)``.  Stashing the
    # values in its supported compatibility slots lets callers use either
    # ``demo.launch()`` or the CLI without losing the lab theme.
    demo._deprecated_css = css
    if hasattr(gr, "themes") and hasattr(gr.themes, "Base"):
        demo._deprecated_theme = gr.themes.Base()
    # Gradio 6 moved CSS to launch(). Wrap the bound method so callers that
    # receive this app and invoke ``app.launch()`` still get the theme, while
    # retaining the normal launch API and allowing an explicit css override.
    launch = demo.launch
    def themed_launch(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("css", css)
        return launch(*args, **kwargs)
    demo.launch = themed_launch
    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the local MP-KG-RAG Pipeline Lab")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root containing semantic_graph_source/")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)
    config = DemoConfig.load(args.project_root)
    app = create_app(config)
    css_path = Path(__file__).with_name("ui.css")
    launch_kwargs = {key: value for key, value in {"server_name": args.host, "server_port": args.port, "share": args.share, "css": css_path.read_text(encoding="utf-8") if css_path.exists() else None}.items() if value is not None}
    if config.safe_export_dir is not None:
        launch_kwargs["allowed_paths"] = [str(config.safe_export_dir.resolve())]
    app.launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EVIDENCE_HEADERS", "STAGES", "UIState", "LazyPipelineProvider", "apply_pipeline_event", "consume_pipeline_events", "create_app", "evidence_rows", "main", "state_summary"]
