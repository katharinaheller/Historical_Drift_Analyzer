from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import replace

from src.core.evaluation.interfaces.i_metric import IMetric
from src.core.evaluation.metrics.ndcg_metric import NDCGMetric
from src.core.evaluation.metrics.faithfulness_metric import FaithfulnessMetric
from src.core.evaluation.ground_truth_builder import GroundTruthBuilder
from src.core.evaluation.utils import make_chunk_id
from src.core.evaluation.settings import EvaluationSettings, DEFAULT_EVAL_SETTINGS

logger = logging.getLogger("EvaluationOrchestrator")


class EvaluationOrchestrator:
    """
    Deterministic evaluation orchestrator.

    Responsibilities:
    - Construct immutable settings (with safe overrides)
    - Evaluate single or batch LLM outputs
    - Persist per-query evaluation results and summary
    """

    def __init__(
        self,
        base_output_dir: str = "data/eval_logs",
        model_name: str = "default",
        settings: EvaluationSettings = DEFAULT_EVAL_SETTINGS,
        metrics: Dict[str, IMetric] | None = None,
        gt_builder: GroundTruthBuilder | None = None,
        k: int | None = None,
        bootstrap_iters: int | None = None,  # kept for compatibility, not in settings
    ):
        # Prepare output directory
        self.model_name = model_name
        self.out = Path(base_output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        # Override only fields that tatsächlich im Dataclass existieren
        settings_overrides: Dict[str, Any] = {}
        if k is not None:
            settings_overrides["ndcg_k"] = int(k)

        if settings_overrides:
            # Create a new immutable settings instance
            self.settings = replace(settings, **settings_overrides)
        else:
            self.settings = settings

        # Store bootstrap_iters separately (used by visualizers, not by settings)
        self.bootstrap_iters = bootstrap_iters

        # Extract effective k
        self.k = self.settings.ndcg_k

        # Metrics subsystem
        self.metrics = metrics or {
            "ndcg@k": NDCGMetric(k=self.k),
            "faithfulness": FaithfulnessMetric(settings=self.settings),
        }

        # Ground truth builder
        self.gt_builder = gt_builder or GroundTruthBuilder(settings=self.settings)

        logger.info(
            f"EvaluationOrchestrator ready | model={self.model_name} | k={self.k} | out={self.out}"
        )

    # -------------------------------------------------------------
    def _ensure_chunk_ids(self, items: List[Dict[str, Any]]) -> None:
        # Ensure each retrieved chunk has a stable ID
        for ch in items:
            if not ch.get("id"):
                ch["id"] = make_chunk_id(ch)

    # -------------------------------------------------------------
    def _safe_id(self, s: str | None) -> str:
        # Generate a filesystem-safe query identifier
        if not s:
            return "query"
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)[:80] or "query"

    # -------------------------------------------------------------
    def _safe_year(self, meta: Dict[str, Any]) -> Optional[int]:
        # Extract a plausible publication year from metadata
        y = meta.get("year")
        try:
            yi = int(str(y))
            if 1900 <= yi <= 2100:
                return yi
        except Exception:
            pass
        return None

    # -------------------------------------------------------------
    def _dominant_decade(self, chunks: List[Dict[str, Any]]) -> Tuple[Optional[int], Dict[str, int]]:
        # Compute decade histogram and dominant decade
        counts: Dict[str, int] = {}
        for ch in chunks:
            y = self._safe_year(ch.get("metadata", {}) or {})
            d = f"{(y // 10) * 10}s" if y else "unknown"
            counts[d] = counts.get(d, 0) + 1

        if not counts:
            return None, {}

        dom_dec_str = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        try:
            dom = int(dom_dec_str[:-1])
        except Exception:
            dom = None

        return dom, counts

    # -------------------------------------------------------------
    def _parse_citation_map_from_prompt(self, prompt_text: str) -> Dict[int, str]:
        # Parse "[n] filename (year)" style lines from the prompt preamble
        if not prompt_text:
            return {}
        import re

        pattern = re.compile(r"^\[(\d+)\]\s+(.+?)\s+\((?:\d{4}|n/a)\)$", re.MULTILINE)
        mapping: Dict[int, str] = {}
        for m in pattern.finditer(prompt_text):
            try:
                mapping[int(m.group(1))] = m.group(2).strip()
            except Exception:
                continue
        return mapping

    # -------------------------------------------------------------
    def _citation_hit_rate(
        self,
        model_output: str,
        retrieved_chunks: List[Dict[str, Any]],
        citation_map: Dict[int, str],
    ) -> float:
        # Compute proportion of citations that point to actually retrieved sources
        if not model_output or not citation_map:
            return 0.0

        import re

        nums = [int(x) for x in re.findall(r"\[(\d+)\]", model_output)]
        if not nums:
            return 0.0

        retrieved_sources = {
            (ch.get("metadata", {}) or {}).get("source_file", "").strip().lower()
            for ch in retrieved_chunks
        }

        hits = sum(
            1
            for n in nums
            if citation_map.get(n, "").strip().lower() in retrieved_sources
        )
        return hits / len(nums)

    # -------------------------------------------------------------
    def evaluate_single(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        model_output: str,
        prompt_text: str | None = None,
    ) -> Dict[str, float]:
        # Evaluate one query-answer pair and persist per-query metrics
        self._ensure_chunk_ids(retrieved_chunks)

        gt_map = self.gt_builder.build(query, retrieved_chunks)
        relevance_scores = [
            int(gt_map.get(ch["id"], ch.get("relevance", 0)))
            for ch in retrieved_chunks
        ]

        ndcg_val = self.metrics["ndcg@k"].compute(relevance_scores=relevance_scores)
        faith_val = self.metrics["faithfulness"].compute(
            context_chunks=[c.get("text", "") for c in retrieved_chunks],
            answer=model_output,
        )

        dom_dec, dec_counts = self._dominant_decade(retrieved_chunks)
        cit_map = self._parse_citation_map_from_prompt(prompt_text or "")
        cit_hit = self._citation_hit_rate(model_output, retrieved_chunks, cit_map)

        qid = self._safe_id(query)

        result = {
            "query_id": qid,
            "ndcg@k": float(ndcg_val),
            "faithfulness": float(faith_val),
            "dominant_decade": int(dom_dec) if dom_dec is not None else None,
            "decade_counts": dec_counts,
            "citation_hit_rate": float(cit_hit),
            "model_name": self.model_name,
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
        # Evaluate all matching LLM logs and write a global summary
        logs_path = Path(logs_dir)
        if not logs_path.exists():
            alt = logs_path.parent / f"{logs_path.name}_{self.model_name}"
            if alt.exists():
                logs_path = alt
                logger.warning(f"Fallback logs directory used: {logs_path}")
            else:
                logger.error(f"Missing log directory: {logs_path}")
                return {
                    "model_name": self.model_name,
                    "files": 0,
                    "evaluated_files": 0,
                    "mean_ndcg@k": 0.0,
                    "mean_faithfulness": 0.0,
                }

        files = sorted(logs_path.glob(pattern))
        if not files:
            logger.error(f"No log files matching '{pattern}' in {logs_path}")
            return {
                "model_name": self.model_name,
                "files": 0,
                "evaluated_files": 0,
                "mean_ndcg@k": 0.0,
                "mean_faithfulness": 0.0,
            }

        nd_vals: List[float] = []
        fa_vals: List[float] = []
        evaluated = 0

        for fp in files:
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))

                query = (
                    data.get("query")
                    or data.get("user_query")
                    or data.get("prompt")
                    or data.get("query_refined")
                    or ""
                )

                model_output = data.get("model_output") or data.get("answer") or ""
                retrieved = data.get("retrieved_chunks") or data.get("context_snippets") or []
                prompt_text = data.get("prompt_final_to_llm") or ""

                for rank, ch in enumerate(retrieved, start=1):
                    ch.setdefault("rank", rank)
                    ch.setdefault("final_score", ch.get("score", 0.0))
                    if "text" not in ch and "snippet" in ch:
                        ch["text"] = ch["snippet"]

                if not query or not retrieved:
                    logger.warning(f"Skipped incomplete log: {fp.name}")
                    continue

                res = self.evaluate_single(query, retrieved, model_output, prompt_text)
                nd_vals.append(float(res["ndcg@k"]))
                fa_vals.append(float(res["faithfulness"]))
                evaluated += 1

            except Exception as e:
                err = self.out / f"{fp.stem}_eval_error.json"
                err.write_text(json.dumps({"error": str(e)}, indent=2), encoding="utf-8")
                logger.error(f"Evaluation failed for {fp.name}: {e}")

        mean_nd = float(sum(nd_vals) / len(nd_vals)) if nd_vals else 0.0
        mean_fa = float(sum(fa_vals) / len(fa_vals)) if fa_vals else 0.0

        summary = {
            "model_name": self.model_name,
            "files": len(files),
            "evaluated_files": evaluated,
            "mean_ndcg@k": mean_nd,
            "mean_faithfulness": mean_fa,
        }

        (self.out / "evaluation_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        logger.info(
            f"Batch evaluation complete | model={self.model_name} | files={len(files)} | evaluated={evaluated}"
        )

        return summary
