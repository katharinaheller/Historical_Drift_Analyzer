# src/core/retrieval/retrieval_orchestrator.py
from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from src.core.retrieval.faiss_retriever import FAISSRetriever
from src.core.retrieval.query_expander import TemporalQueryExpander
from src.core.retrieval.reranker_factory import RerankerFactory
from src.core.config.config_loader import ConfigLoader


def _safe_float(v: Any, default: float = 0.0) -> float:
    # Robust float parsing with default fallback
    try:
        return float(v)
    except Exception:
        return default


class RetrievalOrchestrator:
    """Coordinates retrieval, query expansion, score fusion, reranking, and deduplication."""

    def __init__(self, config_path: str = "configs/retrieval.yaml"):
        self.logger = logging.getLogger("RetrievalOrchestrator")

        # Load merged config (phase + master) for stable paths and options
        cfg_loader = ConfigLoader(config_path, master_path="configs/config.yaml")
        self.cfg: Dict[str, Any] = cfg_loader.config
        self.paths: Dict[str, Any] = self.cfg.get("paths", {})
        self.opts: Dict[str, Any] = self.cfg.get("options", {})

        # Core options with safe defaults
        log_level = getattr(logging, str(self.opts.get("log_level", "INFO")).upper(), logging.INFO)
        logging.basicConfig(level=log_level, format="%(levelname)s | %(message)s")

        # Query expansion switch and limits
        self.enable_expansion: bool = bool(self.opts.get("enable_query_expansion", True))
        self.max_expansions: int = int(self.opts.get("max_expansions", 4))
        self.use_rrf: bool = bool(self.opts.get("use_rrf_fusion", True))
        self.rrf_k: float = _safe_float(self.opts.get("rrf_k", 60.0), default=60.0)

        # Diversity / constraints
        self.enforce_decade_balance: bool = bool(self.opts.get("enforce_decade_balance", True))
        self.min_decade_threshold: int = int(self.opts.get("min_decade_threshold", 3))
        self.must_include: List[str] = list(self.opts.get("must_include", []))
        self.blacklist_sources: List[str] = list(self.opts.get("blacklist_sources", []))
        self.ignore_years: List[int] = list(self.opts.get("ignore_years", []))
        self.min_year: int = int(self.opts.get("min_year", 1900))

        # Controls for retrieval breadth
        self.pre_rerank_k: int = int(self.opts.get("top_k_retrieve", 50))
        self.final_k_default: int = int(self.opts.get("top_k", 10))

        # Initialize retriever with consistent option names
        self.retriever = FAISSRetriever(
            vector_store_dir=str(self.paths.get("vector_store_dir", "data/vector_store")),
            model_name=str(self.opts.get("embedding_model", "all-MiniLM-L6-v2")),
            top_k_retrieve=self.pre_rerank_k,  # Wide pre-rerank search
            normalize_embeddings=bool(self.opts.get("normalize_embeddings", True)),
            use_gpu=bool(self.opts.get("use_gpu", False)),
            similarity_metric=str(self.opts.get("similarity_metric", "cosine")).lower(),
        )

        # Initialize reranker and temporal query expander
        self.reranker = RerankerFactory.from_config(self.cfg)
        self.expander = TemporalQueryExpander() if self.enable_expansion else None

        self.logger.info("Retriever + reranker pipeline initialized successfully.")

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        # Main entry: retrieval -> expansion -> fusion -> rerank -> constraints -> dedup
        self.logger.info(f"Retrieving for query: {query}")

        # Step 1: Build expanded query set (unique, bounded)
        queries = [query]
        if self.expander:
            expansions = self.expander.expand(query)
            # Keep unique order-preserving subset and respect max_expansions
            seen = set([query.strip().lower()])
            for e in expansions:
                e_norm = e.strip().lower()
                if e_norm not in seen:
                    queries.append(e)
                    seen.add(e_norm)
                if len(queries) - 1 >= self.max_expansions:
                    break
            self.logger.debug(f"Expanded queries: {queries[1:]}")

        # Step 2: Collect raw results per subquery
        per_query_results: List[Tuple[str, List[Dict[str, Any]]]] = []
        total = 0
        for q in queries:
            batch = self.retriever.search(q)
            for r in batch:
                r["__subquery"] = q  # annotate for later fusion/audit
            per_query_results.append((q, batch))
            total += len(batch)

        if total == 0:
            self.logger.warning("No results found.")
            return []

        self.logger.info(f"Initial retrieval produced {total} results across {len(queries)} query variants.")

        # Step 3: Optional RRF/score fusion before reranking to diversify inputs
        fused = self._reciprocal_rank_fusion(per_query_results) if self.use_rrf and len(queries) > 1 \
            else [r for _, batch in per_query_results for r in batch]

        # Step 4: Apply blacklist early to prevent undesired items from dominating
        fused = self._apply_blacklist(fused, self.blacklist_sources)

        # Step 5: Rerank with temporal/semantic fusion (uses config lambda/boost etc.)
        rerank_budget = top_k or self.final_k_default
        reranked = self.reranker.rerank(fused, top_k=len(fused))  # rerank all fused candidates

        # Step 6: Hard must-include injection (if present in pool)
        reranked = self._inject_must_include(reranked, self.must_include)

        # Step 7: Deduplicate and enforce final top-k with optional decade balance
        final = self._deduplicate_with_diversity(reranked, top_k=rerank_budget)

        # Step 8: Log temporal distribution if requested
        if bool(self.opts.get("log_decade_distribution", True)):
            self._log_decade_distribution(final)

        self.logger.info(f"Final results after reranking and deduplication: {len(final)} unique items.")
        return final

    # ------------------------------------------------------------------
    def _reciprocal_rank_fusion(
        self,
        per_query_results: List[Tuple[str, List[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        # RRF: combine per-query rankings to reward items that rank well across variants
        # score_rrf(doc) = sum_q 1 / (k + rank_q(doc)), with k ~ 60 to smooth tails
        rrf_scores: Dict[str, float] = {}
        best_payload: Dict[str, Dict[str, Any]] = {}

        for _, batch in per_query_results:
            for rank, item in enumerate(batch, start=1):
                meta = item.get("metadata", {}) or {}
                key = (meta.get("source_file") or meta.get("title") or "unknown").strip().lower()
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
                # Keep representative payload with highest base score for tie-breaking
                base = item.get("score", 0.0)
                if key not in best_payload or base > best_payload[key].get("score", -1.0):
                    best_payload[key] = item

        fused = []
        for k, payload in best_payload.items():
            p = dict(payload)
            p["score_rrf"] = rrf_scores.get(k, 0.0)
            # Preserve a unified 'score' channel for downstream components
            p["score"] = max(_safe_float(payload.get("score", 0.0)), _safe_float(p["score_rrf"], 0.0))
            fused.append(p)

        # Sort by fused score descending
        fused.sort(key=lambda x: _safe_float(x.get("score", 0.0)), reverse=True)
        return fused

    # ------------------------------------------------------------------
    def _apply_blacklist(self, results: List[Dict[str, Any]], blacklist: List[str]) -> List[Dict[str, Any]]:
        # Remove any result whose source_file/title contains a blacklisted token
        if not blacklist:
            return results
        bl = {b.strip().lower() for b in blacklist if b}
        out = []
        for r in results:
            m = r.get("metadata", {}) or {}
            name = (m.get("source_file") or m.get("title") or "").strip().lower()
            if any(t in name for t in bl):
                continue
            out.append(r)
        return out

    # ------------------------------------------------------------------
    def _inject_must_include(self, results: List[Dict[str, Any]], must: List[str]) -> List[Dict[str, Any]]:
        # Ensure presence of required sources if they exist within results
        if not must:
            return results
        must_lc = {m.strip().lower() for m in must}
        # Promote must-include items to the front while preserving relative order
        must_hits, others = [], []
        for r in results:
            m = r.get("metadata", {}) or {}
            name = (m.get("source_file") or m.get("title") or "").strip().lower()
            if any(t in name for t in must_lc):
                must_hits.append(r)
            else:
                others.append(r)
        return must_hits + others

    # ------------------------------------------------------------------
    def _deduplicate_with_diversity(self, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        # Keep best per (source_file/title) and enforce optional decade-balance
        unique: Dict[str, Dict[str, Any]] = {}
        for r in results:
            meta = r.get("metadata", {}) or {}
            key = (meta.get("source_file") or meta.get("title") or "unknown").strip().lower()
            score = _safe_float(r.get("final_score", r.get("score_adjusted", r.get("score", 0.0))), 0.0)
            if key not in unique or score > _safe_float(unique[key].get("score", -1.0)):
                r["score"] = score
                unique[key] = r

        deduped = list(unique.values())
        # Optional decade-aware selection to mitigate tunnel vision
        if self.enforce_decade_balance:
            buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for r in deduped:
                y = self._safe_year(r)
                decade = f"{(y // 10) * 10}s" if y else "unknown"
                buckets[decade].append(r)
            # Sort each bucket by score desc
            for d in buckets:
                buckets[d].sort(key=lambda x: _safe_float(x.get("score", 0.0)), reverse=True)
            # Round-robin pick across decades
            ordered_decades = sorted(buckets.keys(), key=lambda d: (d == "unknown", d))
            out: List[Dict[str, Any]] = []
            idx = 0
            while len(out) < top_k:
                progressed = False
                for d in ordered_decades:
                    if idx < len(buckets[d]):
                        out.append(buckets[d][idx])
                        if len(out) >= top_k:
                            break
                        progressed = True
                if not progressed:
                    break
                idx += 1
            return out
        else:
            deduped.sort(key=lambda x: _safe_float(x.get("score", 0.0)), reverse=True)
            return deduped[:top_k]

    # ------------------------------------------------------------------
    def _safe_year(self, r: Dict[str, Any]) -> Optional[int]:
        # Extract a plausible year from metadata with guards
        meta = r.get("metadata", {}) or {}
        y = meta.get("year", r.get("year"))
        try:
            y = int(y)
            if y in self.ignore_years or y < self.min_year or y > 2100:
                return None
            return y
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _log_decade_distribution(self, items: List[Dict[str, Any]]) -> None:
        # Print a compact histogram over decades for auditability
        hist: Dict[str, int] = defaultdict(int)
        for r in items:
            y = self._safe_year(r)
            d = f"{(y // 10) * 10}s" if y else "unknown"
            hist[d] += 1
        msg = ", ".join([f"{k}:{v}" for k, v in sorted(hist.items())])
        self.logger.info(f"Decade distribution: {msg}")

    # ------------------------------------------------------------------
    def close(self) -> None:
        # Cleanly close FAISS retriever
        try:
            self.retriever.close()
        except Exception as e:
            self.logger.warning(f"Failed to close retriever: {e}")


# ----------------------------------------------------------------------
def main() -> None:
    # Standalone execution for quick smoke tests
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger = logging.getLogger("RetrievalOrchestrator")

    orchestrator = RetrievalOrchestrator()
    query = "Wie hat sich der Begriff KI im Laufe der Zeit entwickelt?"
    results = orchestrator.retrieve(query, top_k=10)

    logger.info(f"Top {len(results)} temporally diverse and unique results:")
    for i, r in enumerate(results, start=1):
        meta = r.get("metadata", {}) or {}
        title = meta.get("title") or meta.get("source_file") or "Unknown"
        year = meta.get("year") or r.get("year", "n/a")
        score = _safe_float(r.get("final_score", r.get("score_adjusted", r.get("score", 0.0))), 0.0)
        logger.info(f"[{i}] ({year}) {title} | Score={score:.4f}")

    orchestrator.close()


if __name__ == "__main__":
    main()
