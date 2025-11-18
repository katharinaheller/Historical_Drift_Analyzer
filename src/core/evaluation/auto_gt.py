# src/core/evaluation/auto_gt.py
from __future__ import annotations
import re
from typing import Dict, List, Any

from sentence_transformers import SentenceTransformer, util
from src.core.evaluation.settings import EvaluationSettings, DEFAULT_EVAL_SETTINGS, SimilarityBands

_CIT_PATTERN = re.compile(r"\[(\d+)\]")


class AutoGroundTruth:
    """Automatic answer-conditioned relevance labeling."""

    def __init__(
        self,
        settings: EvaluationSettings = DEFAULT_EVAL_SETTINGS,
        model_name: str | None = None,
        bands: SimilarityBands | None = None,
    ):
        name = model_name or settings.auto_gt.model_name
        self.model = SentenceTransformer(name)

        b = bands or settings.auto_gt.similarity_bands
        self.high_thr = b.high
        self.mid_thr = b.mid
        self.low_thr = b.low

    def _extract_citations(self, out: str) -> List[int]:
        return [int(m.group(1)) for m in _CIT_PATTERN.finditer(out)] if out else []

    def build(self, answer: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        if not answer or not retrieved_chunks:
            return {}

        ans_emb = self.model.encode([answer], normalize_embeddings=True)
        cited = set(self._extract_citations(answer))

        labels: Dict[str, int] = {}
        for rank, ch in enumerate(retrieved_chunks, start=1):
            cid = ch.get("id") or f"auto::{rank}"
            text = ch.get("text", "") or ""

            c_emb = self.model.encode([text], normalize_embeddings=True)
            sim = float(util.cos_sim(ans_emb, c_emb)[0][0])

            cited_here = rank in cited

            if cited_here and sim >= self.high_thr:
                rel = 3
            elif sim >= self.high_thr:
                rel = 2
            elif sim >= self.mid_thr:
                rel = 1
            elif sim >= self.low_thr:
                rel = 0
            else:
                rel = 0

            labels[cid] = rel

        return labels
