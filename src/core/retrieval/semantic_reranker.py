# src/core/retrieval/semantic_reranker.py
from __future__ import annotations
import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from src.core.retrieval.interfaces.i_reranker import IReranker

logger = logging.getLogger(__name__)

class SemanticReranker(IReranker):
    """Cross-Encoder-basiertes Re-Ranking nach semantischer Relevanz."""

    def __init__(
        self,
        model_name: str,
        top_k: int = 25,
        semantic_weight: float = 0.75,
        use_gpu: bool = False,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.semantic_weight = semantic_weight
        self.device = "cuda" if use_gpu else "cpu"
        self.model = CrossEncoder(model_name, device=self.device)
        logger.info(f"Semantic Cross-Encoder loaded: {model_name} ({self.device})")

    def rerank(self, docs: List[Dict[str, Any]], top_k: int | None = None) -> List[Dict[str, Any]]:
        """Score documents via Cross-Encoder similarity."""
        if not docs:
            return []

        k = top_k or self.top_k
        pairs = [(d.get("query", ""), d.get("text", "")) for d in docs]
        scores = self.model.predict(pairs)

        for d, score in zip(docs, scores):
            base = float(d.get("score", 0.0))
            d["final_score"] = self.semantic_weight * float(score) + (1 - self.semantic_weight) * base

        docs.sort(key=lambda x: x["final_score"], reverse=True)
        return docs[:k]
