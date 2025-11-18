# src/core/evaluation/ground_truth_builder.py
from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import logging

from sentence_transformers import SentenceTransformer, util
from src.core.config.config_loader import ConfigLoader
from src.core.evaluation.settings import EvaluationSettings, DEFAULT_EVAL_SETTINGS, SimilarityBands

logger = logging.getLogger("GroundTruthBuilder")


class GroundTruthBuilder:
    """Builds semantic ground truth using embeddings, independent of final ranking."""

    def __init__(
        self,
        config_path: str = "configs/embedding.yaml",
        settings: EvaluationSettings = DEFAULT_EVAL_SETTINGS,
        bands: SimilarityBands | None = None,
    ):
        cfg = ConfigLoader(config_path).config
        model_name = cfg.get("options", {}).get("embedding_model", "multi-qa-mpnet-base-dot-v1")
        self.model = SentenceTransformer(model_name)

        b = bands or settings.ground_truth.similarity_bands
        self.high_thr = b.high
        self.mid_thr = b.mid
        self.low_thr = b.low

    # ------------------------------------------------------------------
    def build(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, int]:
        if not query or not retrieved_docs:
            return {}

        # deterministic sorting
        stable = sorted(
            retrieved_docs,
            key=lambda x: (x.get("metadata", {}).get("source_file", ""), x.get("raw_rank", 9999))
        )

        q_emb = self.model.encode([query], normalize_embeddings=True)

        truth: Dict[str, int] = {}
        for d in stable:
            text = d.get("text", "") or ""
            doc_id = d.get("id") or f"{d.get('metadata', {}).get('source_file')}"

            d_emb = self.model.encode([text], normalize_embeddings=True)
            sim = float(util.cos_sim(q_emb, d_emb)[0][0])

            if sim >= self.high_thr:
                rel = 3
            elif sim >= self.mid_thr:
                rel = 2
            elif sim >= self.low_thr:
                rel = 1
            else:
                rel = 0

            truth[doc_id] = rel

        logger.info(f"Ground truth built (avg rel={np.mean(list(truth.values())):.3f})")
        return truth
