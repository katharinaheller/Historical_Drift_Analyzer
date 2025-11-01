# src/core/ingestion/utils/file_utils.py
from __future__ import annotations
from pathlib import Path


def ensure_dir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
