"""Local, single-user MP-KG-RAG demo primitives."""

from .config import DemoConfig, EvidenceRecord, IndexManifest, QueryInput, RetrievalResult
from .history import Conversation, HistoryStore, InvalidCitationError, StageRecord
from .index_store import LocalRetriever, build_or_validate_index

__all__ = [
    "Conversation", "DemoConfig", "EvidenceRecord", "HistoryStore", "IndexManifest",
    "InvalidCitationError", "LocalRetriever", "QueryInput", "RetrievalResult", "StageRecord",
    "build_or_validate_index",
]
