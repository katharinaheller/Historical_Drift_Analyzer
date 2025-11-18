from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.core.evaluation.metrics.bert_intrinsic_metric import BertIntrinsicMetric
from src.core.evaluation.plot_style import apply_scientific_style, add_violin_overlay


class BertIntrinsicAnalyzer:
    # Orchestrates computation and visualization of BERT intrinsic scores for one model

    def __init__(
        self,
        logs_dir: str,
        charts_dir: str | None = None,
        pattern: str = "llm_*.json",
        bert_model: str = "bert-base-uncased",
        top_k: int = 10,
        agg: str = "mean",
    ):
        # Resolve input and output directories
        self.logs_dir = Path(logs_dir)
        if charts_dir is None:
            # Default: sibling directory unter data/eval_charts_bert_<modelname>
            parent = self.logs_dir.parent
            stem = f"eval_charts_bert_{self.logs_dir.name.replace('logs_', '')}"
            self.charts_dir = parent / stem
        else:
            self.charts_dir = Path(charts_dir)

        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.pattern = pattern

        # Initialize metric and plotting style
        self.metric = BertIntrinsicMetric(
            model_name=bert_model,
            top_k=top_k,
            agg=agg,
        )
        apply_scientific_style()

    # ------------------------------------------------------------------
    def _iter_logs(self) -> List[Dict[str, Any]]:
        # Iterate over all matching llm_*.json log files
        files = sorted(self.logs_dir.glob(self.pattern))
        for fp in files:
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                yield fp.name, data
            except Exception:
                continue

    # ------------------------------------------------------------------
    def _extract_query_and_chunks(self, data: Dict[str, Any]) -> tuple[str, List[Dict[str, Any]]]:
        # Extract query text and retrieved chunks from a log record
        query = (
            data.get("query")
            or data.get("user_query")
            or data.get("prompt")
            or data.get("query_refined")
            or ""
        )

        retrieved = data.get("retrieved_chunks") or data.get("context_snippets") or []
        # Ensure minimal fields for the metric
        for ch in retrieved:
            if "text" not in ch and "snippet" in ch:
                ch["text"] = ch["snippet"]

        return query, retrieved

    # ------------------------------------------------------------------
    def compute_scores(self) -> pd.DataFrame:
        # Compute BERT intrinsic score for each log file in the directory
        rows: List[Dict[str, Any]] = []

        for fname, data in self._iter_logs():
            query, retrieved = self._extract_query_and_chunks(data)
            if not query or not retrieved:
                continue

            score = self.metric.compute(query=query, retrieved_chunks=retrieved)
            rows.append(
                {
                    "file": fname,
                    "query_id": data.get("query_id") or fname,
                    "bert_intrinsic": score,
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["file", "query_id", "bert_intrinsic"])

        df.to_csv(self.charts_dir / "bert_intrinsic_raw.csv", index=False, encoding="utf-8")
        return df

    # ------------------------------------------------------------------
    def _bootstrap_ci(self, arr: np.ndarray, iters: int = 2000) -> tuple[float, float, float]:
        # Compute mean and 95% bootstrap confidence interval
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return float("nan"), float("nan"), float("nan")
        if arr.size == 1:
            m = float(arr[0])
            return m, m, m

        n = arr.size
        idx = np.random.randint(0, n, size=(iters, n))
        boot = np.mean(arr[idx], axis=1)
        return float(np.mean(boot)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    # ------------------------------------------------------------------
    def _plot_distribution(self, vals: np.ndarray) -> None:
        # Plot histogram + KDE/violin-artige Überlagerung
        if vals.size == 0:
            return

        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(
            vals,
            bins=15,
            range=(0.0, 1.0),
            alpha=0.8,
            edgecolor="black",
        )
        ax.set_xlabel("BERT intrinsic score")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of BERT intrinsic scores")
        ax.set_xlim(0.0, 1.0)

        add_violin_overlay(ax, vals)

        fig.tight_layout()
        fig.savefig(self.charts_dir / "bert_intrinsic_hist.png", dpi=150, bbox_inches="tight")
        fig.savefig(self.charts_dir / "bert_intrinsic_hist.svg", bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    def _plot_run_order(self, vals: np.ndarray) -> None:
        # Simple run-order plot zur Kontrolle von Drift oder Instabilität
        if vals.size == 0:
            return

        x = np.arange(vals.size)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, vals, marker="o", linestyle="-", linewidth=1.1)
        ax.set_xlabel("Run index")
        ax.set_ylabel("BERT intrinsic score")
        ax.set_title("Run-order chart for BERT intrinsic scores")
        ax.set_ylim(0.0, 1.0)
        fig.tight_layout()

        fig.savefig(self.charts_dir / "bert_intrinsic_run_order.png", dpi=150, bbox_inches="tight")
        fig.savefig(self.charts_dir / "bert_intrinsic_run_order.svg", bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    def summarize(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Compute summary statistics and write summary.json
        if df.empty:
            summary = {"files": 0}
            (self.charts_dir / "bert_intrinsic_summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
            return summary

        vals = df["bert_intrinsic"].astype(float).values
        mean, lo, hi = self._bootstrap_ci(vals, iters=2000)

        summary = {
            "files": int(df.shape[0]),
            "bert_intrinsic_mean": mean,
            "bert_intrinsic_ci95_lo": lo,
            "bert_intrinsic_ci95_hi": hi,
            "bert_intrinsic_median": float(np.nanmedian(vals)),
            "bert_intrinsic_std": float(np.nanstd(vals)),
            "top_k": int(self.metric.top_k),
            "bert_model": self.metric.model_name,
            "agg": self.metric.agg,
        }

        (self.charts_dir / "bert_intrinsic_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        # Full pipeline: compute scores, visualize, summarize
        df = self.compute_scores()
        vals = df["bert_intrinsic"].astype(float).values if not df.empty else np.array([])

        self._plot_distribution(vals)
        self._plot_run_order(vals)

        summary = self.summarize(df)
        return summary


def main() -> None:
    # CLI entry point for running the analyzer from the command line
    parser = argparse.ArgumentParser(
        description="Compute and visualize BERT-based intrinsic retrieval scores for one model's logs."
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        required=True,
        help="Directory containing llm_*.json logs for one model (e.g. data/logs_phi3_4b).",
    )
    parser.add_argument(
        "--charts_dir",
        type=str,
        default=None,
        help="Optional output directory for charts and summary. Defaults to data/eval_charts_bert_<model>.",
    )
    parser.add_argument(
        "--bert_model",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace model name for BERT encoder.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top retrieved chunks to consider per query.",
    )
    parser.add_argument(
        "--agg",
        type=str,
        default="mean",
        choices=["mean", "max", "median"],
        help="Aggregation function over chunk similarities.",
    )

    args = parser.parse_args()

    analyzer = BertIntrinsicAnalyzer(
        logs_dir=args.logs_dir,
        charts_dir=args.charts_dir,
        bert_model=args.bert_model,
        top_k=args.top_k,
        agg=args.agg,
    )
    summary = analyzer.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
