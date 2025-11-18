# src/core/evaluation/evaluation_orchestrator.py
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import replace

from src.core.evaluation.interfaces.i_metric import IMetric
from src.core.evaluation.metrics.ndcg_metric import NDCGMetric
from src.core.evaluation.metrics.faithfulness_metric import FaithfulnessMetric
from src.core.evaluation.ground_truth_builder import GroundTruthBuilder
from src.core.evaluation.utils import make_chunk_id
from src.core.evaluation.settings import EvaluationSettings, DEFAULT_EVAL_SETTINGS

logger = logging.getLogger("EvaluationOrchestrator")


class EvaluationOrchestrator:
    """Deterministic evaluation orchestrator adapted to the CURRENT log structure."""

    def __init__(
        self,
        base_output_dir: str = "data/eval_logs",
        model_name: str = "default",
        settings: EvaluationSettings = DEFAULT_EVAL_SETTINGS,
        metrics: Dict[str, IMetric] | None = None,
        gt_builder: GroundTruthBuilder | None = None,
        k: Optional[int] = None,
        bootstrap_iters: Optional[int] = None,
    ):
        # Create output directory
        self.model_name = model_name
        self.out = Path(base_output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        # Store settings with overrides
        overrides: Dict[str, Any] = {}
        if k is not None:
            overrides["ndcg_k"] = int(k)

        self.settings = replace(settings, **overrides) if overrides else settings
        self.bootstrap_iters = bootstrap_iters
        self.k = self.settings.ndcg_k

        # Initialize metrics
        self.metrics = metrics or {
            "ndcg@k": NDCGMetric(k=self.k),
            "faithfulness": FaithfulnessMetric(settings=self.settings),
        }

        # Ground truth builder
        self.gt_builder = gt_builder or GroundTruthBuilder(settings=self.settings)

        logger.info(f"EvaluationOrchestrator ready | model={self.model_name} | k={self.k}")

    # -------------------------------------------------------------
    def _ensure_chunk_ids(self, xs: List[Dict[str, Any]]) -> None:
        # Ensure stable ids are always present
        for ch in xs:
            if not ch.get("id"):
                ch["id"] = make_chunk_id(ch)

    # -------------------------------------------------------------
    def _safe_id(self, s: Optional[str]) -> str:
        # Sanitize file-safe ID
        if not s:
            return "query"
        cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)
        return cleaned[:80] or "query"

    # -------------------------------------------------------------
    def _extract_final(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Accept new key
        final = data.get("retrieved_chunks_final")
        if final:
            return final

        # Accept legacy key for compatibility
        final = data.get("retrieved_chunks")
        if final:
            return final

        return []

    # -------------------------------------------------------------
    def _extract_raw(self, data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        # Prefer newest key
        raw = data.get("raw")
        if raw:
            return raw

        # Accept standard key
        raw = data.get("retrieved_chunks_raw")
        if raw:
            return raw

        # Accept fallback
        return data.get("faiss_raw")

    # -------------------------------------------------------------
    def evaluate_single(
        self,
        query: str,
        retrieved_chunks_final: List[Dict[str, Any]],
        model_output: str,
        prompt_text: Optional[str] = None,
        retrieved_chunks_raw: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:

        # Ensure IDs exist for final chunks
        self._ensure_chunk_ids(retrieved_chunks_final)

        # RAW is preferred for NDCG
        retrieval_for_ndcg = retrieved_chunks_raw if retrieved_chunks_raw else retrieved_chunks_final
        self._ensure_chunk_ids(retrieval_for_ndcg)

        # Build semantic graded ground truth independent of final ranking
        gt_map = self.gt_builder.build(query, retrieval_for_ndcg)

        relevance_scores = [int(gt_map.get(ch["id"], 0)) for ch in retrieval_for_ndcg]

        # Compute NDCG
        ndcg_val = self.metrics["ndcg@k"].compute(relevance_scores=relevance_scores)

        # Compute faithfulness
        faith_val = self.metrics["faithfulness"].compute(
            context_chunks=[c.get("text", "") for c in retrieved_chunks_final],
            answer=model_output,
        )

        qid = self._safe_id(query)

        result = {
            "query_id": qid,
            "model_name": self.model_name,
            "ndcg@k": float(ndcg_val),
            "faithfulness": float(faith_val),
        }

        out_f = self.out / f"{qid}_evaluation.json"
        out_f.write_text(json.dumps(result, indent=2), encoding="utf-8")

        logger.info(f"Evaluation complete → {out_f}")
        return result

    # -------------------------------------------------------------
    def evaluate_batch_from_logs(
        self,
        logs_dir: str = "data/logs",
        pattern: str = "llm_*.json",
    ) -> Dict[str, float]:

        logs_path = Path(logs_dir)
        files = sorted(logs_path.glob(pattern))

        nd_vals, fa_vals = [], []
        evaluated = 0

        for fp in files:
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))

                # Extract query
                query = (
                    data.get("query")
                    or data.get("query_refined")
                    or data.get("processed_query")
                    or ""
                )
                if not query:
                    logger.warning(f"Skipped incomplete log: {fp.name}")
                    continue

                # Extract model answer
                model_output = data.get("model_output") or data.get("answer") or ""

                # Extract final chunks (new + legacy)
                final = self._extract_final(data)
                if not final:
                    logger.warning(f"Skipped incomplete log (no final chunks): {fp.name}")
                    continue

                # Extract raw chunks (new + legacy)
                raw = self._extract_raw(data)

                # Run evaluation
                res = self.evaluate_single(
                    query=query,
                    retrieved_chunks_final=final,
                    model_output=model_output,
                    prompt_text=data.get("prompt_final_to_llm", ""),
                    retrieved_chunks_raw=raw,
                )

                nd_vals.append(float(res["ndcg@k"]))
                fa_vals.append(float(res["faithfulness"]))
                evaluated += 1

            except Exception as e:
                err = self.out / f"{fp.stem}_eval_error.json"
                err.write_text(json.dumps({"error": str(e)}), encoding="utf-8")
                logger.error(f"Evaluation failed for {fp.name}: {e}")

        summary = {
            "model_name": self.model_name,
            "files": len(files),
            "evaluated_files": evaluated,
            "mean_ndcg@k": float(sum(nd_vals) / len(nd_vals)) if nd_vals else 0.0,
            "mean_faithfulness": float(sum(fa_vals) / len(fa_vals)) if fa_vals else 0.0,
        }

        (self.out / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        logger.info(f"Batch evaluation complete | files={len(files)} | evaluated={evaluated}")
        return summary
