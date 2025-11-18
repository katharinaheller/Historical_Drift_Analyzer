from __future__ import annotations

import sys
import os
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.intrinsic import IntrinsicRetrievalOrchestrator
from src.core.intrinsic.intrinsic_visualizer import IntrinsicVisualizer, IntrinsicVizConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run intrinsic retrieval evaluation (NDCG@k + RelevanceSeparationAUC) "
                    "based on existing LLM logs."
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="data/logs",
        help="Directory containing llm_*.json logs",
    )
    parser.add_argument(
        "--intrinsic_dir",
        type=str,
        default="data/intrinsic",
        help="Base directory for intrinsic evaluation outputs",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Cut-off k for NDCG@k",
    )
    parser.add_argument(
        "--bootstrap_iters",
        type=int,
        default=2000,
        help="Number of bootstrap iterations for confidence intervals",
    )
    args = parser.parse_args()

    orch = IntrinsicRetrievalOrchestrator(
        logs_dir=args.logs_dir,
        base_intrinsic_dir=args.intrinsic_dir,
        k=args.k,
        bootstrap_iters=args.bootstrap_iters,
    )

    summary_eval = orch.evaluate_batch()
    print("=== Intrinsic evaluation summary ===")
    print(summary_eval)

    cfg = IntrinsicVizConfig(
        eval_dir=os.path.join(args.intrinsic_dir, "eval_logs"),
        out_dir=os.path.join(args.intrinsic_dir, "charts"),
        bootstrap_iters=args.bootstrap_iters,
    )
    viz = IntrinsicVisualizer(cfg)
    summary_viz = viz.run_all()
    print("=== Intrinsic visualization summary ===")
    print(summary_viz)


if __name__ == "__main__":
    main()
