from __future__ import annotations
import logging
from typing import Dict, Any, List
from src.core.config.config_loader import ConfigLoader
from sentence_transformers import SentenceTransformer
from src.core.retrieval.faiss_retriever import FAISSRetriever

from src.core.retrieval.orchestrator.retrieval_pipeline import RetrievalPipeline
from src.core.retrieval.orchestrator.reranking_pipeline import RerankingPipeline
from src.core.retrieval.orchestrator.diversity_pipeline import DiversityPipeline
from src.core.retrieval.orchestrator.relevance_annotator import RelevanceAnnotator
from src.core.retrieval.orchestrator.final_selector import FinalSelector

from src.core.evaluation.utils import make_chunk_id


class RetrievalOrchestrator:
    """Clean multi-stage retrieval orchestrator for RAG with raw-logging support."""

    def __init__(self, config_path: str = "configs/retrieval.yaml"):
        # Load configuration
        self.logger = logging.getLogger("RetrievalOrchestrator")
        cfg_loader = ConfigLoader(config_path)
        self.cfg = cfg_loader.config

        opts = self.cfg["options"]
        paths = self.cfg["paths"]

        # Multi-stage retrieval parameters
        self.final_k = int(opts.get("final_k", 10))
        oversample = int(opts.get("oversample_factor", 15))
        self.initial_k = max(self.final_k * oversample, self.final_k * 8)

        # Embedding model for reranking/diversity
        self.embed_model = SentenceTransformer(opts["embedding_model"])

        # Instantiate FAISS retriever
        self.retriever = FAISSRetriever(
            vector_store_dir=paths["vector_store_dir"],
            model_name=opts["embedding_model"],
            top_k_retrieve=self.initial_k,
            normalize_embeddings=True,
            use_gpu=opts.get("use_gpu", False),
            similarity_metric=opts.get("similarity_metric", "cosine"),
            temporal_awareness=False,
            diversify_sources=opts.get("diversify_sources", True),
        )

        # Build all pipeline components
        self.stage_retrieve = RetrievalPipeline(self.retriever, self.initial_k)
        self.stage_rerank = RerankingPipeline(self.cfg)
        self.stage_diversity = DiversityPipeline(self.embed_model)
        self.stage_label = RelevanceAnnotator()
        self.stage_select = FinalSelector()

    # ------------------------------------------------------------------
    def retrieve(self, query: str, intent: str) -> Dict[str, List[Dict[str, Any]]]:
        # Return both raw and final-ranked retrieval results
        if not query.strip():
            return {"raw": [], "final": []}

        # Determine whether chronological mode is active
        historical = intent == "chronological"

        # Stage 1: FAISS broad retrieval (this is the raw ranking)
        raw = self.stage_retrieve.run(query, historical)

        # Assign stable ids to raw results
        for i, r in enumerate(raw, start=1):
            if not r.get("id"):
                r["id"] = make_chunk_id(r)
            r["raw_rank"] = i

        # Stage 2: Semantic or temporal reranking on full candidate set
        ranked = self.stage_rerank.run(raw, intent)

        # Stage 3: Diversity application
        diversified = self.stage_diversity.apply(ranked, self.final_k, historical)

        # Stage 4: Relevance annotation (quantile-based)
        annotated = self.stage_label.apply(diversified, ranked)

        # Stage 5: Final selection (exact final_k)
        final = self.stage_select.select(annotated, self.final_k)

        # Assign stable ids and final rank to final results
        for i, x in enumerate(final, start=1):
            x["rank"] = i
            if not x.get("id"):
                x["id"] = make_chunk_id(x)

        # Return complete structured output for LLM-logging and evaluation
        return {
            "raw": raw,        # raw FAISS output for NDCG
            "final": final     # final reranked output for LLM
        }

    # ------------------------------------------------------------------
    def close(self):
        # Release FAISS retriever resources
        self.retriever.close()
