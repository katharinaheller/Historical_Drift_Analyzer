# src/core/retrieval/retrieval_orchestrator.py
from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import numpy as np

from src.core.retrieval.faiss_retriever import FAISSRetriever
from src.core.retrieval.reranker_factory import RerankerFactory
from src.core.config.config_loader import ConfigLoader


class RetrievalOrchestrator:
    """
    Unified retrieval orchestrator — controlled externally by prompt intent.
    Pipeline:
      1. Receive (query, intent)
      2. Retrieve chunks from FAISS
      3. Apply reranker (semantic / temporal)
      4. Return exactly top_k results
    """

    def __init__(self, config_path: str = "configs/retrieval.yaml"):
        self.logger = logging.getLogger("RetrievalOrchestrator")
        cfg_loader = ConfigLoader(config_path)
        self.cfg: Dict[str, Any] = cfg_loader.config

        opts = self.cfg.get("options", {})
        paths = self.cfg.get("paths", {})

        self.top_k = int(opts.get("top_k", 10))
        self.vector_store_dir = str(paths.get("vector_store_dir", "data/vector_store"))
        self.embedding_model = opts.get("embedding_model", "all-MiniLM-L6-v2")

        # Initialize FAISS retriever
        self.retriever = FAISSRetriever(
            vector_store_dir=self.vector_store_dir,
            model_name=self.embedding_model,
            top_k_retrieve=max(self.top_k * 8, 80),
            normalize_embeddings=True,
            use_gpu=False,
            similarity_metric="cosine",
            temporal_awareness=False,
        )

        # Embedding model for similarity and diversity computations
        self.embed_model = SentenceTransformer(self.embedding_model)

        self.logger.info(f"RetrievalOrchestrator initialized | top_k={self.top_k}")

    # ------------------------------------------------------------------
    def retrieve(self, query: str, intent: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks according to user intent.
        intent ∈ {'conceptual', 'chronological', 'analytical', 'comparative'}
        """
        if not query or not query.strip():
            self.logger.warning("Empty query ignored.")
            return []

        is_historical = intent == "chronological"
        self.logger.info(f"Retrieval started | intent={intent} | top_k={self.top_k}")

        # Step 1: retrieve raw candidates from FAISS
        try:
            raw_results = self.retriever.search(query, top_k=self.top_k * 8, temporal_mode=is_historical)
        except Exception as e:
            self.logger.exception(f"FAISS retrieval failed: {e}")
            return []

        if not raw_results:
            self.logger.warning("No retrieval results found.")
            return []

        # Step 2: choose reranker (semantic or temporal)
        reranker_type = "temporal" if is_historical else "semantic"
        self.reranker = RerankerFactory.from_config({"options": {"reranker": reranker_type}})
        try:
            reranked = self.reranker.rerank(raw_results, top_k=len(raw_results))
        except Exception as e:
            self.logger.exception(f"Reranking failed ({reranker_type}): {e}")
            reranked = raw_results

        # Sort by reranked scores
        reranked.sort(key=lambda x: x.get("final_score", x.get("score", 0.0)), reverse=True)

        # Step 3: enforce diversity (historical only)
        final = self._ensure_exact_k(
            self._enforce_diversity(reranked, self.top_k, is_historical),
            self.top_k,
        )

        self._log_decade_distribution(final)
        self.logger.info(f"Retrieval finished | returned={len(final)} | mode={intent}")
        return final

    # ------------------------------------------------------------------
    def _enforce_diversity(self, results: List[Dict[str, Any]], k: int, historical: bool) -> List[Dict[str, Any]]:
        """Diversify by decade and source for chronological intent, otherwise maximize precision."""
        if not results:
            return []

        selected, seen = [], set()
        used_sources, used_decades = set(), set()
        kept_embs = []

        for r in results:
            if len(selected) >= k:
                break
            text = (r.get("text") or "").strip()
            if not text:
                continue

            meta = r.get("metadata", {}) or {}
            src = meta.get("source_file", "unknown").lower()
            year = self._safe_year(r)
            decade = (year // 10) * 10 if year else None

            if not historical:
                # For conceptual/analytical queries, avoid duplicates only
                if hash(text) in seen:
                    continue
                seen.add(hash(text))
                selected.append(r)
                continue

            # For historical mode, enforce source/decade diversity
            if src in used_sources and decade in used_decades and len(selected) < k * 0.8:
                continue

            emb = self.embed_model.encode(text, normalize_embeddings=True)
            if kept_embs:
                sims = util.cos_sim(emb, kept_embs)[0]
                if float(sims.max()) > 0.95:
                    continue

            selected.append(r)
            kept_embs.append(emb)
            used_sources.add(src)
            if decade:
                used_decades.add(decade)

        return selected

    # ------------------------------------------------------------------
    def _ensure_exact_k(self, results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Ensure deterministic top-k output length."""
        if not results:
            return []
        if len(results) > k:
            return results[:k]
        if len(results) < k:
            return results + results[-1:] * (k - len(results))
        return results

    # ------------------------------------------------------------------
    def _safe_year(self, r: Dict[str, Any]) -> Optional[int]:
        """Safely extract valid publication year from metadata."""
        meta = r.get("metadata", {}) or {}
        y = meta.get("year", r.get("year"))
        try:
            y = int(y)
            if 1900 <= y <= 2100:
                return y
        except Exception:
            pass
        return None

    def _log_decade_distribution(self, items: List[Dict[str, Any]]) -> None:
        """Log decade distribution for temporal diagnostics."""
        hist: Dict[str, int] = defaultdict(int)
        for r in items:
            y = self._safe_year(r)
            decade = f"{(y // 10) * 10}s" if y else "unknown"
            hist[decade] += 1
        msg = ", ".join(f"{k}:{v}" for k, v in sorted(hist.items()))
        self.logger.info(f"Decade distribution: {msg}")

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close retriever resources gracefully."""
        try:
            self.retriever.close()
        except Exception as e:
            self.logger.warning(f"Failed to close retriever: {e}")
        self.logger.info("RetrievalOrchestrator closed cleanly.")
