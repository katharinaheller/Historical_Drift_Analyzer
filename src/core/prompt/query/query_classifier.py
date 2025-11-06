# src/core/prompt/query/query_classifier.py 
from __future__ import annotations
import logging
import numpy as np
from typing import Literal, Dict, List, Optional
from sentence_transformers import SentenceTransformer, util

Intent = Literal["chronological", "conceptual", "analytical", "comparative"]

class QueryClassifier:
    """
    Embedding-based intent classifier.
    No heuristics, no language dependency, fully model-driven.
    """
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        label_texts: Optional[Dict[str, str]] = None
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

        self.model = SentenceTransformer(model_name)
        # Canonical label representations (configurable)
        self.label_texts = label_texts or {
            "chronological": "questions about historical development or changes over time",
            "conceptual": "questions asking for definition, explanation or theoretical meaning",
            "analytical": "questions asking for comparison, evaluation or analysis",
            "comparative": "questions asking for contrast or difference between ideas",
        }

        self.labels = list(self.label_texts.keys())
        # Precompute label embeddings for efficient similarity calculation
        self.label_embeddings = self.model.encode(
            list(self.label_texts.values()), normalize_embeddings=True
        )
        self.logger.info(f"Initialized semantic intent classifier with {len(self.labels)} labels")

    # ------------------------------------------------------------------
    def classify(self, query: str) -> Intent:
        """Compute embedding similarity to predefined intent prototypes."""
        if not query or not query.strip():
            return "conceptual"
        # Encode user query
        q_emb = self.model.encode(query, normalize_embeddings=True)
        # Compute cosine similarity between query and label embeddings
        sims = util.cos_sim(q_emb, self.label_embeddings)[0].cpu().numpy()
        # Select intent with highest similarity
        idx = int(np.argmax(sims))
        intent = self.labels[idx]
        self.logger.info(f"Predicted semantic intent='{intent}' (sim={sims[idx]:.3f})")
        return intent  # type: ignore
