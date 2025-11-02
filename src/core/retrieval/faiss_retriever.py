from __future__ import annotations
from typing import Any, Dict, List
from pathlib import Path
import json
import numpy as np
import logging
from src.core.retrieval.interfaces.i_retriever import IRetriever


class FAISSRetriever(IRetriever):
    """Retriever that performs semantic similarity search using FAISS."""

    def __init__(self,
                 vector_store_dir: str,
                 model_name: str,
                 top_k: int = 5,
                 normalize_embeddings: bool = True,
                 use_gpu: bool = False):
        self.logger = logging.getLogger(self.__class__.__name__)

        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("faiss-cpu and sentence-transformers are required. "
                              "Install via: poetry add faiss-cpu sentence-transformers") from e

        self.faiss = faiss
        self.vector_store_dir = Path(vector_store_dir).resolve()
        self.index_path = self.vector_store_dir / "index.faiss"
        self.meta_path = self.vector_store_dir / "metadata.jsonl"

        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(f"Vector store incomplete: {self.vector_store_dir}")

        self.model = SentenceTransformer(model_name)
        self.top_k = top_k
        self.normalize = normalize_embeddings
        self.use_gpu = use_gpu

        self.logger.info(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                self.logger.info("FAISS GPU acceleration enabled")
            except Exception as e:
                self.logger.warning(f"GPU mode failed, using CPU fallback: {e}")

        self.logger.info(f"Loading metadata from {self.meta_path}")
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = [json.loads(line) for line in f]

        self.logger.info(f"FAISSRetriever initialized with {len(self.metadata)} entries")

    # ------------------------------------------------------------------
    def _encode_query(self, query: str) -> np.ndarray:
        vec = self.model.encode([query], normalize_embeddings=self.normalize)
        return np.array(vec, dtype="float32")

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        top_k = top_k or self.top_k
        q_vec = self._encode_query(query)
        D, I = self.index.search(q_vec, top_k)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx >= len(self.metadata):
                continue
            entry = self.metadata[idx]
            results.append({
                "score": float(score),
                "text": entry.get("text", "")[:500],
                "metadata": entry.get("metadata", {})
            })
        return results

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.logger.info("Retriever closed")
