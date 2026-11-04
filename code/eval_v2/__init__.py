"""Production-faithful MP-arm v2 evaluation tooling.

The package reuses the frozen, pure retrieval/graph core from ``work/mpkg_rag_core.py``
and the pinned retrieval backends from the ``local-mpkg-gradio`` worktree demo, while
never importing the demo's index builder: the eval driver must validate existing
generations, not rebuild or mutate them.

Callers may import :mod:`eval_v2` directly; this module wires the two stable import
roots (``work/`` and the demo worktree package) onto ``sys.path`` exactly once.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_WORK_DIR = REPO_ROOT / "work"
_WORKTREE_DEMO = REPO_ROOT / ".worktrees" / "local-mpkg-gradio"

for _entry in (_WORK_DIR, REPO_ROOT, _WORKTREE_DEMO):
    if _entry.is_dir() and str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))
