from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    """Ensure `src/` is importable in test runs without requiring an editable install."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

