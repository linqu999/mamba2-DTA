"""Local import bootstrap for direct script execution from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path


def add_src_to_path() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
