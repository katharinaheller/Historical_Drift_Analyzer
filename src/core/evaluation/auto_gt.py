# src/core/evaluation/auto_gt.py
from __future__ import annotations
import re
from typing import Dict, List, Any

from sentence_transformers import SentenceTransformer, util

from src.core.evaluation.settings import (
    EvaluationSettings,
    DEFAULT_EVAL_SETTINGS,
    SimilarityBands,
)


_CIT_PATTERN = re.compile(r"\[(\d+)\]")


class AutoGroundTruth:
    """
    Automatic graded relevance labelling for retrieval evaluation.
    Uses the same embedding model as the retrieval stack for consistency.
    """

    def __init__(
        self,
        settings: EvaluationSettings = DEFAULT_EVAL_SETTINGS,
        model_name: str | None = None,
        bands: SimilarityBands | None = None,
    ):
        # Select model name from explicit argument or global settings
        name = model_name or settings.auto_gt.model_name
        self.model = SentenceTransformer(name)

        # Inject thresholds from shared configuration
        b = bands or settings.auto_gt.similarity_bands
        self.high_thr = b.high
        self.mid_thr = b.mid
        self.low_thr = b.low

    def _extract_citations(self, output: str) -> List[int]:
        # Extract numeric citation markers [n] from model output
        if not output:
            return []
        return [int(m.group(1)) for m in _CIT_PATTERN.finditer(output)]

    def build(self, answer: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        # Compute graded answer-conditioned relevance labels
        if not answer or not retrieved_chunks:
            return {}

        ans_emb = self.model.encode([answer], normalize_embeddings=True)

        labels: Dict[str, int] = {}
        cited = set(self._extract_citations(answer))

        for rank, ch in enumerate(retrieved_chunks, start=1):
            cid = ch.get("id") or f"auto::{rank}"
            text = ch.get("text", "") or ""
            chunk_emb = self.model.encode([text], normalize_embeddings=True)
            sim = float(util.cos_sim(ans_emb, chunk_emb)[0][0])

            cited_here = int(ch.get("rank", rank)) in cited

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

            labels[cid] = int(rel)

        return labels
