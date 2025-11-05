from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import numpy as np
import logging
from src.core.retrieval.interfaces.i_retriever import IRetriever


class FAISSRetriever(IRetriever):
    """Semantic retriever using FAISS vector similarity search with optional cosine/dot mode."""

    def __init__(
        self,
        vector_store_dir: str,
        model_name: str,
        top_k_retrieve: int = 50,        # broad initial search
        normalize_embeddings: bool = True,
        use_gpu: bool = False,
        similarity_metric: str = "cosine",  # new parameter
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        try:
            import faiss
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
            raise FileNotFoundError(f"Vector store incomplete: {self.vector_store_dir}")

        self.model = SentenceTransformer(model_name)
        self.top_k_retrieve = top_k_retrieve
        self.normalize_embeddings = normalize_embeddings
        self.use_gpu = use_gpu
        self.similarity_metric = similarity_metric.lower().strip()

        if self.similarity_metric not in {"cosine", "dot"}:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")

        # Load FAISS index
        self.logger.info(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                self.logger.info("FAISS GPU acceleration enabled")
            except Exception as e:
                self.logger.warning(f"GPU mode failed, falling back to CPU: {e}")

        # Load metadata into memory once
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = [json.loads(line) for line in f]

        self.logger.info(
            f"FAISSRetriever initialized | entries={len(self.metadata)} | metric={self.similarity_metric.upper()}"
        )

    # ------------------------------------------------------------------
    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query to embedding vector with optional normalization."""
        vec = self.model.encode([query], normalize_embeddings=self.normalize_embeddings)
        return np.asarray(vec, dtype="float32")

    # ------------------------------------------------------------------
    def _normalize_scores(self, distances: np.ndarray) -> np.ndarray:
        """Normalize FAISS distances to similarity scores."""
        if self.similarity_metric == "cosine":
            # FAISS inner product (cosine) returns higher = better
            return distances
        elif self.similarity_metric == "dot":
            # dot similarity is equivalent here; optional for clarity
            return distances
        else:
            # fallback normalization
            return 1 - distances

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Perform similarity search for a given text query."""
        k = top_k or self.top_k_retrieve
        k = max(1, min(k, self.index.ntotal))  # keep within safe bounds

        q_vec = self._encode_query(query)
        self.logger.debug(f"FAISS search | k={k} | query='{query[:60]}'")

        distances, indices = self.index.search(q_vec, k)
        scores = self._normalize_scores(distances[0])

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores, indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            entry = self.metadata[idx]
            results.append({
                "score": float(score),
                "text": entry.get("text", "")[:500],
                "metadata": entry.get("metadata", {}),
            })

        self.logger.info(f"Retrieved {len(results)} candidates for query: '{query[:60]}'")
        return results

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Gracefully close retriever."""
        self.logger.info("FAISS retriever closed")
