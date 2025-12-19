import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "src").resolve()

if SRC not in (Path(p).resolve() for p in sys.path):
    sys.path.insert(0, str(SRC))
