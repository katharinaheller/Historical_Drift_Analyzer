# src/core/retrieval/faiss_retriever.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json
import re
import numpy as np
import logging
from math import exp
from src.core.retrieval.interfaces.i_retriever import IRetriever


class FAISSRetriever(IRetriever):
    """FAISS-based semantic retriever with optional temporal modulation."""

    def __init__(
        self,
        vector_store_dir: str,
        model_name: str,
        top_k_retrieve: int = 50,            # # wide initial recall
        normalize_embeddings: bool = True,
        use_gpu: bool = False,
        similarity_metric: str = "cosine",   # # 'cosine' or 'dot'
        temporal_awareness: bool = True,     # # enable temporal prior
        temporal_tau: float = 8.0,           # # smoothing for |year - query_year|
        temporal_weight: float = 0.30,       # # strength of temporal modulation
        valid_year_range: Tuple[int, int] = (1900, 2100),  # # plausible year range
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        try:
            import faiss  # type: ignore
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "faiss-cpu and sentence-transformers are required. "
                "Install via: poetry add faiss-cpu sentence-transformers"
            ) from e

        self.faiss = faiss
        self.vector_store_dir = Path(vector_store_dir).resolve()
        self.index_path = self.vector_store_dir / "index.faiss"
        self.meta_path = self.vector_store_dir / "metadata.jsonl"

        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(f"Incomplete vector store: {self.vector_store_dir}")

        self.model = SentenceTransformer(model_name)
        self.top_k_retrieve = int(top_k_retrieve)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.use_gpu = bool(use_gpu)
        self.similarity_metric = similarity_metric.lower().strip()
        self.temporal_awareness = bool(temporal_awareness)
        self.temporal_tau = float(temporal_tau)
        self.temporal_weight = float(temporal_weight)
        self.valid_year_range = valid_year_range

        if self.similarity_metric not in {"cosine", "dot"}:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")

        # # Load FAISS index (optionally move to GPU)
        self.logger.info(f"Loading FAISS index: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                self.logger.info("FAISS GPU acceleration enabled")
            except Exception as e:
                self.logger.warning(f"GPU mode failed, falling back to CPU: {e}")

        # # Load metadata into memory once
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = [json.loads(line) for line in f]

        self.logger.info(
            f"FAISSRetriever ready | entries={len(self.metadata)} | metric={self.similarity_metric.upper()} "
            f"| temporal_awareness={self.temporal_awareness}"
        )

    # ------------------------------------------------------------------
    def _encode_query(self, query: str) -> np.ndarray:
        # # Encode with optional normalization
        vec = self.model.encode([query], normalize_embeddings=self.normalize_embeddings)
        return np.asarray(vec, dtype="float32")

    # ------------------------------------------------------------------
    def _normalize_scores(self, distances: np.ndarray) -> np.ndarray:
        # # For cosine/dot (IP), higher is better -> use as-is
        if self.similarity_metric in {"cosine", "dot"}:
            return distances
        return 1 - distances

    # ------------------------------------------------------------------
    def _extract_years_from_query(self, query: str) -> List[int]:
        # # Only keep plausible 4-digit years
        yrs = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", query)]
        lo, hi = self.valid_year_range
        return [y for y in yrs if lo <= y <= hi]

    # ------------------------------------------------------------------
    def _temporal_modulate(self, base_score: float, doc_year: Optional[int], query_years: List[int]) -> float:
        # # Exponential distance weighting: nearer years -> higher boost
        if doc_year is None or not query_years:
            return base_score
        nearest = min(abs(doc_year - y) for y in query_years)
        w = exp(-nearest / max(self.temporal_tau, 1e-6))
        return base_score * (1.0 + self.temporal_weight * w)

    # ------------------------------------------------------------------
    def _safe_doc_year(self, entry: Dict[str, Any]) -> Optional[int]:
        # # Parse year from metadata robustly
        meta = entry.get("metadata", {}) or {}
        y = meta.get("year", None)
        try:
            yi = int(str(y))
            lo, hi = self.valid_year_range
            if lo <= yi <= hi:
                return yi
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: Optional[int] = None, temporal_mode: bool = True) -> List[Dict[str, Any]]:
        """Similarity search with optional temporal modulation."""
        # # Toggle temporal behavior per-call for deterministic flows
        self.temporal_awareness = bool(temporal_mode)

        k = int(top_k or self.top_k_retrieve)
        k = max(1, min(k, self.index.ntotal))

        q_vec = self._encode_query(query)
        self.logger.debug(f"FAISS search | k={k} | query='{query[:60]}' | temporal={self.temporal_awareness}")

        D, I = self.index.search(q_vec, k)
        scores = self._normalize_scores(D[0])

        results: List[Dict[str, Any]] = []
        query_years = self._extract_years_from_query(query) if self.temporal_awareness else []

        for score, idx in zip(scores, I[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            entry = self.metadata[idx]
            doc_year = self._safe_doc_year(entry)
            mod_score = (
                self._temporal_modulate(float(score), doc_year, query_years)
                if self.temporal_awareness else float(score)
            )
            results.append({
                "score": float(mod_score),
                "text": (entry.get("text", "") or "")[:500],
                "metadata": entry.get("metadata", {}) or {},
            })

        results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        self.logger.info(
            f"Retrieved {len(results)} candidates | temporal_mode={self.temporal_awareness} "
            f"| years_in_query={query_years if query_years else 'none'}"
        )
        return results

    # ------------------------------------------------------------------
    def close(self) -> None:
        """No-op for symmetry; kept for interface compliance."""
        self.logger.info("FAISS retriever closed")
