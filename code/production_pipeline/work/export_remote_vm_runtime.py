#!/usr/bin/env python3
"""Export the checked production notebook as a headless Python runtime.

The installation cell is intentionally omitted because the remote environment
is provisioned and verified before execution. Keeping code in a physical file
also lets Triton's JIT inspect decorated kernel source.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


LOCKFILE_SHA256 = "19b68a84617c002acd47042274904de7e98f59e5b097c8fcd350bc0cab4c0fb1"


def export_runtime(notebook_path: Path, output_path: Path) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sections = [
        "# Generated from remote_vm_qwen35_mpkg_rag.ipynb; do not edit directly.",
        f'_LOCKFILE_SHA256_EXPECTED = "{LOCKFILE_SHA256}"',
    ]
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code" or index == 1:
            continue
        source = "".join(cell.get("source", []))
        source = source.replace("from __future__ import annotations\n", "")
        sections.extend((f"\n# %% [notebook cell {index}]", source))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


if __name__ == "__main__":
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/remote_vm_qwen35_mpkg_rag.ipynb")
    destination = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/remote_vm_qwen35_mpkg_rag_runtime.py")
    export_runtime(source, destination)
    print(destination.resolve())
