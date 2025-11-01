# src/core/ingestion/metadata_merger.py
from __future__ import annotations

from typing import Dict, Any, List


def merge_metadata(doc_metadata: Dict[str, Any], chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # attaches document-level metadata to every chunk
    merged: List[Dict[str, Any]] = []
    for idx, ch in enumerate(chunks):
        new_ch = {
            "id": f"{doc_metadata.get('source_file', 'doc')}_{idx}",
            "text": ch.get("text", ""),
            "page": ch.get("page"),
            "bbox": ch.get("bbox"),
            "metadata": dict(doc_metadata),
        }
        merged.append(new_ch)
    return merged
