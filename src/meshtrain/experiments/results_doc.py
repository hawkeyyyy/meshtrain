"""Auto-update named sections of docs/v1-results.md.

Sections are delimited by ``<!-- BEGIN:name -->`` / ``<!-- END:name -->``
markers; content between them is replaced, everything else is preserved.
"""

from __future__ import annotations

import platform
import re
from datetime import datetime
from pathlib import Path

import torch

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "docs" / "v1-results.md"


def update_section(name: str, markdown: str, path: str | Path | None = None) -> Path:
    path = Path(path) if path else DEFAULT_PATH
    text = path.read_text() if path.exists() else "# MeshTrain V1 Results\n"
    begin, end = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
    block = f"{begin}\n{markdown.rstrip()}\n{end}"
    if begin in text and end in text:
        text = re.sub(re.escape(begin) + r".*?" + re.escape(end), lambda _: block, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text)
    return path


def environment_line() -> str:
    return (f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} on {platform.node() or 'unknown'} "
            f"({platform.system()} {platform.machine()}, torch {torch.__version__}, "
            f"CUDA available: {torch.cuda.is_available()}, MPS available: {torch.backends.mps.is_available()})._")


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)
