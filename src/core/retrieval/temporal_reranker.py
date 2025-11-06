# src/core/retrieval/temporal_reranker.py
from __future__ import annotations
import logging, math
from statistics import mean, pstdev, median
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Set
from src.core.retrieval.interfaces.i_reranker import IReranker

logger = logging.getLogger(__name__)

class TemporalReranker(IReranker):
    """Combines semantic scores with temporal diversity balancing."""

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
        zscore_normalization: bool = False,
    ):
        self.lambda_weight = lambda_weight
        self.age_score_boost = age_score_boost
        self.enforce_decade_balance = enforce_decade_balance
        self.min_decade_threshold = min_decade_threshold
        self.nonlinear_boost = nonlinear_boost.lower()
        self.min_year = min_year
        self.ignore_years: Set[int] = set(ignore_years or [])
        self.recency_cutoff_year = recency_cutoff_year
        self.allow_legacy_backfill = allow_legacy_backfill
        self.legacy_backfill_max_ratio = legacy_backfill_max_ratio
        self.must_include = must_include or []
        self.blacklist_sources = blacklist_sources or []
        self.zscore_normalization = zscore_normalization

    # ------------------------------------------------------------------
    def rerank(self, results: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        if not results:
            return []

        # Jahr extrahieren + Filter anwenden
        for r in results:
            r["year"] = self._extract_year(r)
        results = [r for r in results if r["year"] not in self.ignore_years]
        results = [r for r in results if not self._is_blacklisted(r)]
        if not results:
            return []

        # Z-Score-Normalisierung optional
        if self.zscore_normalization:
            self._normalize_scores(results)

        # Zeitliche Gewichtung
        now = datetime.utcnow().year
        for r in results:
            y = r["year"]
            r["adjusted_score"] = self._apply_age_boost(r.get("score", 0.0), y, now)

        results.sort(key=lambda x: x["adjusted_score"], reverse=True)
        decade_groups = self._group_by_decade(results)
        decades = sorted(decade_groups.keys())

        if not decades:
            return results[:top_k]

        λ = self._adaptive_lambda(len(decades))
        selected = (
            self._balanced_decade_selection(decade_groups, decades, top_k)
            if self.enforce_decade_balance
            else results[:top_k]
        )

        median_dec = int(median(decades))
        for r in selected:
            base = r.get("adjusted_score", 0.0)
            dec_diff = abs((r["year"] // 10) * 10 - median_dec)
            temporal_div = 1 / (1 + dec_diff / 10)
            r["final_score"] = λ * base + (1 - λ) * temporal_div

        ranked = sorted(selected, key=lambda x: x["final_score"], reverse=True)
        ranked = self._apply_recency_cutoff(ranked, results, top_k)
        ranked = self._inject_must_include(ranked, results, top_k)

        logger.info(
            f"Temporal reranking complete | decades={len(decades)} | λ={λ:.2f} | age_boost={self.age_score_boost:.2f}"
        )
        return ranked[:top_k]

    # ------------------------------------------------------------------
    def _extract_year(self, r: Dict[str, Any]) -> int:
        meta = r.get("metadata", {})
        y = meta.get("year") or r.get("year")
        try:
            val = int(str(y))
            if val < self.min_year or val > 2100:
                raise ValueError
            return val
        except Exception:
            return self.min_year

    def _group_by_decade(self, results: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        out: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for r in results:
            d = (r["year"] // 10) * 10
            out[d].append(r)
        for d in out:
            out[d].sort(key=lambda x: x.get("adjusted_score", 0.0), reverse=True)
        return out

    def _apply_age_boost(self, base: float, year: int, now: int) -> float:
        age = max(0, now - year)
        if self.nonlinear_boost == "sigmoid":
            s = 1 / (1 + math.exp((age - 6) / 4.0))
        elif self.nonlinear_boost == "sqrt":
            s = math.sqrt(max(0, 1 - min(age / 40, 1)))
        else:
            s = max(0, 1 - min(age / 30, 1))
        return base + self.age_score_boost * s

    def _adaptive_lambda(self, n_decades: int) -> float:
        if n_decades < self.min_decade_threshold:
            return max(0.3, self.lambda_weight * (n_decades / self.min_decade_threshold))
        return min(1.0, self.lambda_weight + 0.05 * math.log1p(n_decades))

    def _balanced_decade_selection(self, groups: Dict[int, List[Dict[str, Any]]], decades: List[int], top_k: int):
        selected: List[Dict[str, Any]] = []
        for d in decades:
            if groups[d]:
                selected.append(groups[d].pop(0))
                if len(selected) >= top_k:
                    return selected
        i = 0
        while len(selected) < top_k and any(groups.values()):
            d = decades[i % len(decades)]
            if groups[d]:
                selected.append(groups[d].pop(0))
            i += 1
        return selected

    def _normalize_scores(self, results: List[Dict[str, Any]]):
        vals = [r.get("score", 0.0) for r in results]
        if len(vals) < 2:
            return
        μ, σ = mean(vals), pstdev(vals)
        if σ < 1e-8:
            return
        for r in results:
            r["score"] = (r["score"] - μ) / σ

    def _apply_recency_cutoff(self, ranked: List[Dict[str, Any]], all_results: List[Dict[str, Any]], top_k: int):
        if not self.recency_cutoff_year:
            return ranked[:top_k]
        recent = [r for r in ranked if r["year"] >= self.recency_cutoff_year]
        legacy = [r for r in ranked if r["year"] < self.recency_cutoff_year]
        if len(recent) >= top_k:
            return recent[:top_k]
        if not self.allow_legacy_backfill:
            return recent
        max_legacy = int(top_k * self.legacy_backfill_max_ratio)
        return (recent + legacy[:max_legacy])[:top_k]

    def _inject_must_include(self, ranked: List[Dict[str, Any]], all_results: List[Dict[str, Any]], top_k: int):
        must = [r for r in all_results if self._matches_must_include(r)]
        if not must:
            return ranked
        merged, seen = [], set()
        for r in must + ranked:
            key = self._src_key(r)
            if key not in seen:
                seen.add(key)
                merged.append(r)
            if len(merged) >= top_k:
                break
        return merged

    def _src_key(self, r: Dict[str, Any]) -> str:
        meta = r.get("metadata", {})
        return (meta.get("source_file") or meta.get("title") or "unknown").lower()

    def _matches_must_include(self, r: Dict[str, Any]) -> bool:
        key = self._src_key(r)
        return any(m.lower() in key for m in self.must_include)

    def _is_blacklisted(self, r: Dict[str, Any]) -> bool:
        key = self._src_key(r)
        return any(b.lower() in key for b in self.blacklist_sources)
