"""Transactional private conversation history backed by SQLite."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import Conversation, StageRecord


class InvalidCitationError(ValueError):
    """Raised when a stage cites an evidence ID outside its frozen ledger."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _citation_ids(value: Any) -> set[str]:
    """Collect IDs only from explicit citation/evidence-id fields."""
    found: set[str] = set()
    citation_keys = {
        "citations", "citation_ids", "cited_evidence_ids", "supported_evidence_ids",
        "evidence_ids", "selected_evidence_ids", "supporting_evidence_ids",
        "source_evidence_ids", "cited_ids", "cited_evidence_id", "supported_evidence_id",
        "evidence_ledger", "current_evidence_ledger", "selected_ledger",
        "evidence_id", "evidence_chunk_id",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in citation_keys:
                found.update(_citation_ids(item))
            # Other fields, including prose and arbitrary metadata, are opaque.
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            if isinstance(item, str):
                found.add(item)
            else:
                found.update(_citation_ids(item))
    elif isinstance(value, str):
        found.add(value)
    return found


def _final_citation_error(payload: Mapping[str, Any]) -> str | None:
    """Validate the rendered final citation contract before checkpointing."""
    declared = payload.get("cited_evidence_ids", ())
    if not isinstance(declared, (list, tuple)):
        declared = ()
    declared = [str(item) for item in declared]
    if len(declared) != len(set(declared)):
        return "final declared citations contain duplicates"
    text = str(payload.get("counter_narrative", ""))
    inline = re.findall(r"\[(E[1-9][0-9]*)\]", text)
    if len(inline) != len(set(inline)):
        return "final inline citations contain duplicates"
    if set(declared) != set(inline):
        return "final declared citations must exactly match inline [E#] citations"
    return None


class HistoryStore:
    def __init__(self, path: Path, *, evidence_ledger: Iterable[str] | Mapping[str, Any] | None = None, safe_export_dir: Path | None = None) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if evidence_ledger is None:
            self.evidence_ids: set[str] | None = None
        elif isinstance(evidence_ledger, Mapping):
            rows = evidence_ledger.get("records", evidence_ledger.get("evidence_ids", evidence_ledger))
            self.evidence_ids = {str(row.get("evidence_id") or row.get("evidence_chunk_id")) if isinstance(row, Mapping) else str(row) for row in rows}
        else:
            self.evidence_ids = {str(item) for item in evidence_ledger}
        self.safe_export_dir = Path(safe_export_dir).expanduser() if safe_export_dir is not None else self.path.parent / "exports"
        self.safe_export_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, status TEXT NOT NULL, input_text TEXT NOT NULL DEFAULT '',
                    target TEXT NOT NULL DEFAULT '', category TEXT NOT NULL DEFAULT '', metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    started_at TEXT NOT NULL, ended_at TEXT, status TEXT NOT NULL, payload_json TEXT NOT NULL DEFAULT '{}', error TEXT
                );
                CREATE TABLE IF NOT EXISTS stages (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    stage_name TEXT NOT NULL, sequence INTEGER NOT NULL, created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL, raw_trace TEXT NOT NULL DEFAULT '', UNIQUE(run_id, stage_name)
                );
                CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_runs_conversation ON runs(conversation_id, started_at DESC);
                """
            )
            # Stage checkpoints gained stream metadata after the original demo
            # shipped.  Keep existing private history databases readable and
            # upgrade them in place without dropping any user data.
            columns = {row[1] for row in db.execute("PRAGMA table_info(stages)")}
            migrations = {
                "status": "ALTER TABLE stages ADD COLUMN status TEXT NOT NULL DEFAULT 'completed'",
                "metrics_json": "ALTER TABLE stages ADD COLUMN metrics_json TEXT NOT NULL DEFAULT '{}'",
                "partial_text": "ALTER TABLE stages ADD COLUMN partial_text TEXT NOT NULL DEFAULT ''",
                "error": "ALTER TABLE stages ADD COLUMN error TEXT",
            }
            for name, statement in migrations.items():
                if name not in columns:
                    db.execute(statement)

    def create_conversation(self, title: str = "New conversation", *, input_text: str = "", target: str = "", category: str = "", metadata: Mapping[str, Any] | None = None) -> Conversation:
        conversation_id, timestamp = str(uuid.uuid4()), _now()
        with self._connect() as db:
            db.execute("INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (conversation_id, title or "New conversation", timestamp, timestamp, "new", input_text, target, category, _json(metadata or {})))
        return self.get_conversation(conversation_id)  # type: ignore[return-value]

    def list_conversations(self, search: str = "") -> list[Conversation]:
        with self._connect() as db:
            if search:
                rows = db.execute("SELECT * FROM conversations WHERE title LIKE ? OR input_text LIKE ? ORDER BY updated_at DESC", (f"%{search}%", f"%{search}%")).fetchall()
            else:
                rows = db.execute("SELECT * FROM conversations ORDER BY updated_at DESC").fetchall()
        return [self._conversation(row) for row in rows]

    def _conversation(self, row: sqlite3.Row) -> Conversation:
        with self._connect() as db:
            run = db.execute("SELECT * FROM runs WHERE conversation_id = ? ORDER BY started_at DESC LIMIT 1", (row["id"],)).fetchone()
            stages = db.execute("SELECT * FROM stages WHERE run_id = ? ORDER BY sequence", (run["id"],)).fetchall() if run else []
        return Conversation(
            id=row["id"], title=row["title"], created_at=row["created_at"], updated_at=row["updated_at"],
            status=row["status"], input_text=row["input_text"], target=row["target"], category=row["category"],
            metadata=json.loads(row["metadata_json"]), run_id=run["id"] if run else None,
            stages=tuple(self._stage(item) for item in stages),
            run_status=run["status"] if run else None,
            run_payload=json.loads(run["payload_json"] or "{}") if run else {},
            run_error=run["error"] if run else None,
            run_started_at=run["started_at"] if run else None,
            run_ended_at=run["ended_at"] if run else None,
        )

    def _stage(self, row: sqlite3.Row) -> StageRecord:
        return StageRecord(
            id=row["id"], run_id=row["run_id"], stage_name=row["stage_name"],
            payload=json.loads(row["payload_json"]), raw_trace=row["raw_trace"], created_at=row["created_at"],
            sequence=row["sequence"], status=row["status"] if "status" in row.keys() else "completed",
            metrics=json.loads(row["metrics_json"] or "{}") if "metrics_json" in row.keys() else {},
            partial_text=row["partial_text"] if "partial_text" in row.keys() else "",
            error=row["error"] if "error" in row.keys() else None,
        )

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return self._conversation(row) if row else None

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        with self._connect() as db:
            result = db.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", (title.strip() or "New conversation", _now(), conversation_id))
        return result.rowcount == 1

    def update_conversation_fields(self, conversation_id: str, *, input_text: str | None = None, target: str | None = None, category: str | None = None, metadata: Mapping[str, Any] | None = None) -> bool:
        """Update normalized query fields while retaining existing metadata."""
        with self._connect() as db:
            row = db.execute("SELECT input_text, target, category, metadata_json FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            if row is None:
                return False
            values = (row[0] if input_text is None else input_text, row[1] if target is None else target, row[2] if category is None else category, row[3] if metadata is None else _json(metadata), _now(), conversation_id)
            db.execute("UPDATE conversations SET input_text = ?, target = ?, category = ?, metadata_json = ?, updated_at = ? WHERE id = ?", values)
        return True

    def mark_run(self, conversation_id: str, payload: Mapping[str, Any] | str | None = None, *, status: str = "completed", run_id: str | None = None, error: str | None = None) -> str:
        if isinstance(payload, str) and run_id is None:
            run_id, payload = payload, {}
        run_id = run_id or str(uuid.uuid4())
        timestamp = _now()
        with self._connect() as db:
            exists = db.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            if exists is None:
                raise KeyError(f"unknown conversation: {conversation_id}")
            existing = db.execute("SELECT started_at FROM runs WHERE id = ?", (run_id,)).fetchone()
            if existing:
                owner = db.execute("SELECT conversation_id FROM runs WHERE id = ?", (run_id,)).fetchone()
                if owner is None or owner[0] != conversation_id:
                    raise KeyError(f"run {run_id} belongs to another conversation")
                current_payload = json.loads(db.execute("SELECT payload_json FROM runs WHERE id = ?", (run_id,)).fetchone()[0] or "{}")
                current_payload.update(dict(payload or {}))
                db.execute("UPDATE runs SET ended_at = ?, status = ?, payload_json = ?, error = ? WHERE id = ? AND conversation_id = ?", (timestamp if status != "running" else None, status, _json(current_payload), error, run_id, conversation_id))
            else:
                db.execute("INSERT INTO runs(id, conversation_id, started_at, ended_at, status, payload_json, error) VALUES (?, ?, ?, ?, ?, ?, ?)", (run_id, conversation_id, timestamp, timestamp if status != "running" else None, status, _json(payload or {}), error))
            db.execute("UPDATE conversations SET updated_at = ?, status = ? WHERE id = ?", (timestamp, status, conversation_id))
        return run_id

    def append_stage(
        self, conversation_id: str, run_id: str, stage_name: str, payload: Mapping[str, Any], *,
        raw_trace: str = "", evidence_ids: Iterable[str] | None = None,
        metrics: Mapping[str, Any] | None = None, partial_text: str = "",
        status: str = "completed", error: str | None = None,
    ) -> StageRecord:
        if status not in {"streaming", "completed", "failed", "cancelled"}:
            raise ValueError("stage status must be streaming, completed, failed, or cancelled")
        citations = _citation_ids(payload)
        if status == "completed" and stage_name.casefold() in {"final", "answer", "generation"}:
            citation_error = _final_citation_error(payload)
            if citation_error:
                raise InvalidCitationError(citation_error)
        stage_id, timestamp = str(uuid.uuid4()), _now()
        with self._connect() as db:
            run = db.execute("SELECT conversation_id, payload_json FROM runs WHERE id = ?", (run_id,)).fetchone()
            if run is None or run["conversation_id"] != conversation_id:
                raise KeyError(f"unknown run: {run_id}")
            run_payload = json.loads(run["payload_json"] or "{}")
            persisted = _citation_ids(run_payload)
            ledger_fields = {"evidence_ids", "selected_evidence_ids", "supporting_evidence_ids", "current_evidence_ledger", "evidence_ledger"}
            has_persisted_ledger = bool(set(run_payload) & ledger_fields)
            allowed = set(str(item) for item in evidence_ids) if evidence_ids is not None else (persisted if has_persisted_ledger else self.evidence_ids)
            cite_bearing = stage_name.casefold() in {"perspective", "perspectives", "plan", "final", "answer", "generation"}
            # Streaming/failed checkpoints may have no evidence ledger yet;
            # citation validation is only meaningful for a completed stage.
            if status == "completed" and cite_bearing and not allowed:
                raise InvalidCitationError("citation-bearing stage requires a persisted current-run evidence ledger")
            if allowed is not None:
                invalid = sorted(citations - allowed)
                if invalid:
                    raise InvalidCitationError(f"citations are outside the current evidence ledger: {', '.join(invalid)}")
            existing_stage = db.execute("SELECT id, sequence FROM stages WHERE run_id = ? AND stage_name = ?", (run_id, stage_name)).fetchone()
            if existing_stage:
                stage_id, sequence = existing_stage["id"], existing_stage["sequence"]
                db.execute(
                    "UPDATE stages SET created_at = ?, payload_json = ?, raw_trace = ?, status = ?, metrics_json = ?, partial_text = ?, error = ? WHERE id = ?",
                    (timestamp, _json(payload), raw_trace or "", status, _json(metrics or {}), partial_text or "", error, stage_id),
                )
            else:
                sequence = db.execute("SELECT COALESCE(MAX(sequence), -1) + 1 FROM stages WHERE run_id = ?", (run_id,)).fetchone()[0]
                db.execute(
                    "INSERT INTO stages(id, run_id, stage_name, sequence, created_at, payload_json, raw_trace, status, metrics_json, partial_text, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (stage_id, run_id, stage_name, sequence, timestamp, _json(payload), raw_trace or "", status, _json(metrics or {}), partial_text or "", error),
                )
            db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (timestamp, conversation_id))
        return StageRecord(stage_id, run_id, stage_name, dict(payload), raw_trace or "", timestamp, sequence, status, dict(metrics or {}), partial_text or "", error)

    def checkpoint_stage(self, conversation_id: str, run_id: str, stage_name: str, payload: Mapping[str, Any] | None = None, **kwargs: Any) -> StageRecord:
        """Persist a bounded streaming checkpoint (or terminal stage record)."""
        return self.append_stage(conversation_id, run_id, stage_name, payload or {}, **kwargs)

    def export_conversation(self, conversation_id: str, format: str = "json", destination: Path | None = None) -> str:
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            raise KeyError(f"unknown conversation: {conversation_id}")
        data = {
            "id": conversation.id, "title": conversation.title, "created_at": conversation.created_at,
            "updated_at": conversation.updated_at, "status": conversation.status, "input_text": conversation.input_text,
            "target": conversation.target, "category": conversation.category, "metadata": conversation.metadata,
            "run_id": conversation.run_id, "run_status": conversation.run_status, "run_payload": conversation.run_payload,
            "run_error": conversation.run_error, "run_started_at": conversation.run_started_at,
            "run_ended_at": conversation.run_ended_at,
            "stages": [{
                "id": stage.id, "run_id": stage.run_id, "stage_name": stage.stage_name,
                "payload": stage.payload, "raw_trace": stage.raw_trace, "created_at": stage.created_at,
                "sequence": stage.sequence, "status": stage.status, "metrics": stage.metrics,
                "partial_text": stage.partial_text, "error": stage.error,
            } for stage in conversation.stages],
        }
        if format.lower() == "json":
            output = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
        elif format.lower() in {"md", "markdown"}:
            lines = [f"# {conversation.title}", "", f"Status: {conversation.status}"]
            if conversation.run_error:
                lines.append(f"Error: {conversation.run_error}")
            lines += ["", conversation.input_text, ""]
            for stage in conversation.stages:
                lines += [f"## {stage.stage_name} ({stage.status})", "", "```json", json.dumps(stage.payload, ensure_ascii=False, indent=2, sort_keys=True), "```", ""]
                if stage.metrics:
                    lines += [f"Metrics: `{json.dumps(stage.metrics, ensure_ascii=False, sort_keys=True)}`", ""]
                if stage.partial_text:
                    lines += ["Partial streamed text:", "", stage.partial_text, ""]
                if stage.error:
                    lines += [f"Error: {stage.error}", ""]
                if stage.raw_trace:
                    lines += ["<details><summary>Raw trace</summary>", "", stage.raw_trace, "", "</details>", ""]
            output = "\n".join(lines)
        else:
            raise ValueError("format must be json or markdown")
        if destination is not None:
            destination = Path(destination).expanduser()
            safe_dir = self.safe_export_dir.resolve()
            if destination.exists() and destination.is_symlink():
                raise ValueError("refusing export through symlink")
            try:
                destination.resolve(strict=False).relative_to(safe_dir)
            except ValueError as exc:
                raise ValueError("export destination escapes the configured safe export directory") from exc
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                destination.parent.resolve(strict=True).relative_to(safe_dir)
            except ValueError as exc:
                raise ValueError("export destination escapes the configured safe export directory") from exc
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
            temporary = Path(temporary_name)
            try:
                data = output.encode("utf-8")
                offset = 0
                while offset < len(data):
                    offset += os.write(descriptor, data[offset:])
                os.fsync(descriptor)
                os.close(descriptor)
                descriptor = -1
                # Hard-link publication is atomic and refuses an existing
                # destination, unlike rename/replace.
                os.link(temporary, destination)
                temporary.unlink()
                # Directory descriptors are POSIX-only; Windows journals it.
                if os.name != "nt":
                    directory = os.open(destination.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
            except FileExistsError as exc:
                raise FileExistsError(f"refusing unsafe overwrite: {destination}") from exc
            finally:
                if 'descriptor' in locals() and descriptor >= 0:
                    os.close(descriptor)
                temporary.unlink(missing_ok=True)
        return output

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connect() as db:
            result = db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return result.rowcount == 1
