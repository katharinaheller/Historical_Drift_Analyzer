# src/core/retrieval/retriever_factory.py
from __future__ import annotations
import logging
from typing import Dict, Any
from src.core.retrieval.faiss_retriever import FAISSRetriever

logger = logging.getLogger(__name__)

class RetrieverFactory:
    """Factory für Intent-spezifische Retriever-Instanzen (vollständig entkoppelte Datenflüsse)."""

    @staticmethod
    def build(intent: str, cfg: Dict[str, Any]) -> FAISSRetriever:
        """Erzeuge deterministischen Retriever für den angegebenen Intent."""
        paths = cfg.get("paths", {})
        opts = cfg.get("options", {})

        # Basisparameter, deterministisch
        common_args = dict(
            model_name=opts.get("embedding_model", "all-MiniLM-L6-v2"),
            normalize_embeddings=True,
            use_gpu=False,
            similarity_metric="cosine",
            temporal_tau=float(opts.get("temporal_tau", 8.0)),
            temporal_weight=float(opts.get("temporal_weight", 0.30)),
            top_k_retrieve=int(opts.get("top_k_retrieve", 80)),
        )

        # Intent → Index-Ordner Mapping
        index_map = {
            "conceptual": paths.get("conceptual_vector_dir", "data/vector_store/conceptual"),
            "chronological": paths.get("chronological_vector_dir", "data/vector_store/chronological"),
            "analytical": paths.get("analytical_vector_dir", "data/vector_store/analytical"),
            "comparative": paths.get("comparative_vector_dir", "data/vector_store/comparative"),
        }

        # Fallback auf globales Standardverzeichnis
        vdir = index_map.get(intent, paths.get("vector_store_dir", "data/vector_store/default"))
        temporal_flag = (intent == "chronological")

        logger.info(f"Building FAISSRetriever for intent='{intent}' | dir={vdir} | temporal={temporal_flag}")

        return FAISSRetriever(
            vector_store_dir=vdir,
            temporal_awareness=temporal_flag,
            **common_args,
        )
