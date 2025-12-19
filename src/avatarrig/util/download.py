from __future__ import annotations

from pathlib import Path
import os
import time
import urllib.request
from rich.console import Console

console = Console()

def download_if_missing(url: str, dst: Path, *, poll_seconds: float = 0.25, timeout_seconds: float = 120.0) -> Path:
    """Download a file to dst if it's missing or empty.

    Implements a simple cross-platform filesystem lock to avoid multiple
    processes downloading the same model simultaneously (common with ProcessPool).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst

    lock_dir = dst.with_suffix(dst.suffix + ".lock")
    t0 = time.time()

    # Acquire lock (directory creation is atomic across platforms)
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            # Another process is downloading. If the file shows up, we're done.
            if dst.exists() and dst.stat().st_size > 0:
                return dst
            if (time.time() - t0) > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for lock: {lock_dir}")
            time.sleep(poll_seconds)

    try:
        # Re-check after acquiring lock (avoid redundant download).
        if dst.exists() and dst.stat().st_size > 0:
            return dst

        console.print(f"[yellow]Downloading model[/yellow] {url}\n -> {dst}")
        tmp = dst.with_suffix(dst.suffix + f".{os.getpid()}.part")
        if tmp.exists():
            tmp.unlink()

        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            f.write(r.read())

        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError(f"Download failed or produced empty file: {tmp}")

        os.replace(tmp, dst)  # atomic replace
        return dst
    finally:
        # Best-effort lock release.
        try:
            if lock_dir.exists():
                lock_dir.rmdir()
        except Exception:
            pass
