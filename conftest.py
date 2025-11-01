# conftest.py
import sys
from pathlib import Path

# project root = Ordner mit pyproject.toml
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if SRC.exists() and str(SRC) not in sys.path:
    print(f"[BOOTSTRAP] Adding {SRC} to sys.path")
    sys.path.insert(0, str(SRC))
