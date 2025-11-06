# src/core/retrieval/reranker_factory.py
from __future__ import annotations
import logging
from typing import Any, Dict, Type
from src.core.retrieval.temporal_reranker import TemporalReranker
from src.core.retrieval.semantic_reranker import SemanticReranker
from src.core.retrieval.interfaces.i_reranker import IReranker

logger = logging.getLogger(__name__)

class RerankerFactory:
    """Factory for deterministic reranker construction."""

    _registry: Dict[str, Type[IReranker]] = {
        "temporal": TemporalReranker,
        "semantic": SemanticReranker,
    }

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> IReranker:
        """Instantiate reranker from configuration dictionary."""
        opts = cfg.get("options", {})
        rtype = str(opts.get("reranker", "semantic")).lower()

        if rtype not in RerankerFactory._registry:
            raise ValueError(
                f"Unsupported reranker type: {rtype}. "
                f"Available: {list(RerankerFactory._registry.keys())}"
            )

        cls = RerankerFactory._registry[rtype]
        logger.info(f"Initializing reranker of type='{rtype}'")

        if cls is TemporalReranker:
            return TemporalReranker(
                lambda_weight=float(opts.get("lambda_weight", 0.55)),
                min_year=int(opts.get("min_year", 1900)),
                enforce_decade_balance=bool(opts.get("enforce_decade_balance", True)),
                age_score_boost=float(opts.get("age_score_boost", 0.25)),
                min_decade_threshold=int(opts.get("min_decade_threshold", 3)),
                nonlinear_boost=str(opts.get("nonlinear_boost", "sigmoid")),
                ignore_years=list(opts.get("ignore_years", [])),
                recency_cutoff_year=opts.get("recency_cutoff_year"),
                allow_legacy_backfill=bool(opts.get("allow_legacy_backfill", True)),
                legacy_backfill_max_ratio=float(opts.get("legacy_backfill_max_ratio", 0.3)),
                must_include=list(opts.get("must_include", [])),
                blacklist_sources=list(opts.get("blacklist_sources", [])),
                zscore_normalization=bool(opts.get("zscore_normalization", False)),
            )

        if cls is SemanticReranker:
            return SemanticReranker(
                model_name=str(
                    opts.get("semantic_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
                ),
                top_k=int(opts.get("top_k_rerank", 25)),
                semantic_weight=float(opts.get("semantic_weight", 0.75)),
                use_gpu=bool(opts.get("use_gpu", False)),
            )
