# src/core/prompt/query/query_preprocessor.py
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional
from src.core.prompt.query.query_classifier import QueryClassifier

logger = logging.getLogger(__name__)

@dataclass
class QueryPreprocessorConfig:
    lowercase: bool = True
    remove_double_spaces: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"

class QueryPreprocessor:
    """Clean, normalize, and classify queries without heuristics."""
    def __init__(self, cfg: Optional[QueryPreprocessorConfig] = None):
        self.cfg = cfg or QueryPreprocessorConfig()
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
        # Initialize classifier for semantic intent detection
        self.classifier = QueryClassifier(model_name=self.cfg.embedding_model)

    def validate(self, query: Optional[str]) -> str:
        """Ensure query is not empty or invalid."""
        if not query or not query.strip():
            raise ValueError("Empty query is not allowed")
        return query.strip()

    def clean(self, query: str) -> str:
        """Normalize casing and spacing according to configuration."""
        q = query.lower() if self.cfg.lowercase else query
        if self.cfg.remove_double_spaces:
            q = " ".join(q.split())
        return q.strip()

    def process(self, raw_query: Optional[str]) -> dict:
        """Return cleaned text + semantic intent."""
        q = self.validate(raw_query)
        clean_q = self.clean(q)
        intent = self.classifier.classify(clean_q)
        logger.info(f"Processed query='{clean_q}' intent='{intent}'")
        return {"raw_query": q, "processed_query": clean_q, "intent": intent}
