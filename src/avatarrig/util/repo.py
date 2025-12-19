from __future__ import annotations
from pathlib import Path

def find_repo_root(start: Path | None = None) -> Path:
    """Find repo root by walking upwards until we see pyproject.toml."""
    cur = (start or Path.cwd()).resolve()
    for _ in range(12):
        if (cur / "pyproject.toml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # Fallback to CWD if not found.
    return (start or Path.cwd()).resolve()
