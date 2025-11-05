# src/core/retrieval/temporal_reranker.py
from __future__ import annotations
import logging
import math
from collections import defaultdict
from statistics import median
from datetime import datetime
from typing import Any, Dict, List, Set
from src.core.retrieval.interfaces.i_reranker import IReranker

logger = logging.getLogger(__name__)

class TemporalReranker(IReranker):
    """
    Temporal reranker combining semantic similarity and temporal diversity.
    Incorporates recency bias, nonlinear age boost, decade balancing,
    must-include enforcement, and blacklist exclusion.
    """

    def __init__(
        self,
        lambda_weight: float = 0.55,
        min_year: int = 1900,
        enforce_decade_balance: bool = True,
        age_score_boost: float = 0.25,
        min_decade_threshold: int = 3,
        nonlinear_boost: str = "sigmoid",
        ignore_years: List[int] | None = None,
        recency_cutoff_year: int | None = None,
        allow_legacy_backfill: bool = True,
        legacy_backfill_max_ratio: float = 0.3,
        must_include: List[str] | None = None,
        blacklist_sources: List[str] | None = None,
    ):
        # Core weighting and balancing parameters
        self.lambda_weight = lambda_weight
        self.age_score_boost = age_score_boost
        self.enforce_decade_balance = enforce_decade_balance
        self.min_decade_threshold = min_decade_threshold
        self.nonlinear_boost = nonlinear_boost.lower()
        self.min_year = min_year

        # Control sets and filters
        self.ignore_years: Set[int] = set(ignore_years or [])
        self.recency_cutoff_year = recency_cutoff_year
        self.allow_legacy_backfill = allow_legacy_backfill
        self.legacy_backfill_max_ratio = legacy_backfill_max_ratio
        self.must_include = must_include or []
        self.blacklist_sources = blacklist_sources or []

    # ------------------------------------------------------------------
    def rerank(self, results: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """Main entry: apply temporal weighting, decade balancing, and inclusion/exclusion logic."""
        if not results:
            logger.warning("No results to rerank.")
            return []

        # --- Step 1: Clean and filter input ---
        for r in results:
            r["year"] = self._extract_year(r)
        results = [r for r in results if r["year"] not in self.ignore_years]
        results = [r for r in results if not self._is_blacklisted(r)]

        if not results:
            logger.warning("All results filtered by blacklist or invalid years.")
            return []

        # --- Step 2: Compute boosted semantic scores ---
        now = self._current_year()
        for r in results:
            y = r["year"]
            r["score_adjusted"] = self._apply_time_boost(r.get("score", 0.0), y, now)

        results.sort(key=lambda x: x["score_adjusted"], reverse=True)
        decade_groups = self._group_by_decade(results)
        decades = sorted(decade_groups.keys())

        if not decades:
            logger.warning("No valid decades detected; skipping temporal reranking.")
            return results[:top_k]

        # --- Step 3: Adjust lambda adaptively ---
        if len(decades) < self.min_decade_threshold:
            effective_lambda = min(self.lambda_weight, 0.5)
            logger.debug(f"Few decades ({len(decades)}); reducing λ → {effective_lambda}")
        else:
            effective_lambda = self.lambda_weight

        # --- Step 4: Select diverse subset ---
        selected = (
            self._select_balanced(decade_groups, decades, top_k)
            if self.enforce_decade_balance
            else results[:top_k]
        )

        # --- Step 5: Combine semantic + temporal diversity ---
        median_decade = int(median(decades))
        for r in selected:
            sim = r.get("score_adjusted", 0.0)
            decade_diff = abs((r["year"] // 10) * 10 - median_decade)
            temporal_div = 1.0 / (1.0 + decade_diff / 10.0)
            r["final_score"] = effective_lambda * sim + (1 - effective_lambda) * temporal_div

        ranked = sorted(selected, key=lambda x: x["final_score"], reverse=True)

        # --- Step 6: Apply recency cutoff and backfill ---
        ranked = self._apply_recency_filter(ranked, results, top_k)

        # --- Step 7: Inject must-include sources ---
        ranked = self._inject_must_include(ranked, results, top_k)

        logger.info(
            f"Temporal reranking applied | decades={len(decades)} | selected={len(ranked)} | "
            f"λ={effective_lambda:.2f} | age_boost={self.age_score_boost:.2f} | cutoff={self.recency_cutoff_year}"
        )
        return ranked[:top_k]

    # ------------------------------------------------------------------
    def _apply_time_boost(self, base: float, year: int, now: int) -> float:
        """Apply nonlinear recency boost to semantic score."""
        age = max(0, now - year)
        if self.nonlinear_boost == "sigmoid":
            s = 1 / (1 + math.exp((age - 6) / 4.0))
        elif self.nonlinear_boost == "sqrt":
            s = math.sqrt(max(0.0, 1.0 - min(age / 40.0, 1.0)))
        else:
            s = max(0.0, 1.0 - min(age / 30.0, 1.0))
        return base + self.age_score_boost * s

    # ------------------------------------------------------------------
    def _apply_recency_filter(self, ranked: List[Dict[str, Any]], all_results: List[Dict[str, Any]], top_k: int):
        """Favor recent items while allowing limited legacy backfill."""
        if not self.recency_cutoff_year:
            return ranked[:top_k]

        modern = [r for r in ranked if r["year"] >= self.recency_cutoff_year]
        legacy = [r for r in ranked if r["year"] < self.recency_cutoff_year]

        if len(modern) >= top_k:
            return modern[:top_k]
        if not self.allow_legacy_backfill:
            return modern

        max_legacy = int(top_k * self.legacy_backfill_max_ratio)
        return (modern + legacy[:max_legacy])[:top_k]

    # ------------------------------------------------------------------
    def _inject_must_include(self, ranked: List[Dict[str, Any]], all_results: List[Dict[str, Any]], top_k: int):
        """Ensure must-include items appear in final list."""
        required = [r for r in all_results if self._matches_must_include(r)]
        if not required:
            return ranked

        merged, seen = [], set()
        for r in required + ranked:
            key = self._source_key(r)
            if key not in seen:
                seen.add(key)
                merged.append(r)
            if len(merged) >= top_k:
                break
        return merged

    # ------------------------------------------------------------------
    def _extract_year(self, r: Dict[str, Any]) -> int:
        """Safely parse year with fallback to min_year."""
        meta = r.get("metadata", {})
        y = meta.get("year") or r.get("year")
        try:
            y = int(str(y).strip())
            return y if y >= self.min_year else self.min_year
        except Exception:
            return self.min_year

    # ------------------------------------------------------------------
    def _group_by_decade(self, results: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Group results by decade for diversity enforcement."""
        groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for r in results:
            d = (r["year"] // 10) * 10
            groups[d].append(r)
        for d in groups:
            groups[d].sort(key=lambda x: x.get("score_adjusted", 0.0), reverse=True)
        return groups

    # ------------------------------------------------------------------
    def _select_balanced(self, groups: Dict[int, List[Dict[str, Any]]], decades: List[int], top_k: int) -> List[Dict[str, Any]]:
        """Round-robin select to ensure temporal diversity."""
        selected: List[Dict[str, Any]] = []
        # one per decade first
        for d in decades:
            if groups[d]:
                selected.append(groups[d].pop(0))
                if len(selected) >= top_k:
                    return selected
        # then round-robin fill
        i = 0
        while len(selected) < top_k and any(groups.values()):
            d = decades[i % len(decades)]
            if groups[d]:
                selected.append(groups[d].pop(0))
            i += 1
        return selected

    # ------------------------------------------------------------------
    def _source_key(self, r: Dict[str, Any]) -> str:
        meta = r.get("metadata", {})
        return (meta.get("source_file") or meta.get("title") or "unknown").lower()

    # ------------------------------------------------------------------
    def _matches_must_include(self, r: Dict[str, Any]) -> bool:
        key = self._source_key(r)
        return any(m.lower() in key for m in self.must_include)

    # ------------------------------------------------------------------
    def _is_blacklisted(self, r: Dict[str, Any]) -> bool:
        key = self._source_key(r)
        return any(b.lower() in key for b in self.blacklist_sources)

    # ------------------------------------------------------------------
    def _current_year(self) -> int:
        return datetime.utcnow().year
