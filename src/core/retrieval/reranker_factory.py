# src/core/retrieval/reranker_factory.py
from __future__ import annotations
from typing import Any, Dict
from src.core.retrieval.temporal_reranker import TemporalReranker
from src.core.retrieval.interfaces.i_reranker import IReranker

class RerankerFactory:
    """Factory for creating reranker instances from configuration."""

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> IReranker:
        opts = cfg.get("options", {})
        rtype = str(opts.get("reranker", "temporal")).lower()

        if rtype == "temporal":
            return TemporalReranker(
                lambda_weight=float(opts.get("lambda_weight", 0.55)),
                min_year=int(opts.get("min_year", 1900)),
                enforce_decade_balance=bool(opts.get("enforce_decade_balance", True)),
                age_score_boost=float(opts.get("age_score_boost", 0.25)),
                min_decade_threshold=int(opts.get("min_decade_threshold", 3)),
                nonlinear_boost=str(opts.get("nonlinear_boost", "sigmoid")),
                ignore_years=opts.get("ignore_years", []),
                recency_cutoff_year=opts.get("recency_cutoff_year", None),
                allow_legacy_backfill=bool(opts.get("allow_legacy_backfill", True)),
                legacy_backfill_max_ratio=float(opts.get("legacy_backfill_max_ratio", 0.3)),
                must_include=opts.get("must_include", []),
                blacklist_sources=opts.get("blacklist_sources", []),
            )

        raise ValueError(f"Unsupported reranker type: {rtype}")
