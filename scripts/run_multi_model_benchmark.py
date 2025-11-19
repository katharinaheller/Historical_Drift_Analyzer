from __future__ import annotations
import sys
import os
import subprocess
import time
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.evaluation.multi_model_report_builder import MultiModelReportBuilder

# ---------------------------------------------------------
FAITH_COLORS = {
    "excellent":  "#006d2c",
    "good":       "#31a354",
    "fair":       "#a1d99b",
    "borderline": "#fed976",
    "poor":       "#fd8d3c",
    "critical":   "#e31a1c",
}

FAITH_LABELS = {
    "excellent":  "Excellent (≥0.95)",
    "good":       "Good (0.85–0.94)",
    "fair":       "Fair (0.75–0.84)",
    "borderline": "Borderline (0.60–0.74)",
    "poor":       "Poor (0.40–0.59)",
    "critical":   "Critical (<0.40)",
}


# ---------------------------------------------------------
def run(cmd: list[str]) -> None:
    print(">>>", " ".join(cmd))
    start = time.time()
    p = subprocess.Popen(cmd)
    p.wait()
    print(f"finished in {time.time() - start:.1f}s\n")


# ---------------------------------------------------------
def load_eval_df(eval_dir: Path) -> pd.DataFrame:
    rows = []
    for fp in eval_dir.glob("*_evaluation.json"):
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue

        rows.append({
            "query_id": d.get("query_id"),
            "ndcg": float(d.get("ndcg@k", np.nan)),
            "faith": float(d.get("faithfulness", np.nan)),
            "model": d.get("model_name", "unknown"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["query_id", "ndcg", "faith", "model"])
    return df


# ---------------------------------------------------------
def faith_band(f: float) -> str:
    if np.isnan(f):
        return "missing"
    v = float(f)
    if v >= 0.95: return "excellent"
    if v >= 0.85: return "good"
    if v >= 0.75: return "fair"
    if v >= 0.60: return "borderline"
    if v >= 0.40: return "poor"
    return "critical"


# ---------------------------------------------------------
def clear_and_prepare(path: Path) -> None:
    if path.exists():
        for f in path.glob("*"):
            if f.is_file():
                f.unlink()
            else:
                import shutil
                shutil.rmtree(f)
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
def plot_global_comparison(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        print("No evaluation data available for global comparison plot.")
        return

    df = df.copy()
    df["band"] = df["faith"].apply(faith_band)

    models = sorted(df["model"].unique())
    bands = ["excellent", "good", "fair", "borderline", "poor", "critical"]

    counts = (
        df.groupby(["model", "band"])["query_id"]
        .count().unstack(fill_value=0)
        .reindex(columns=bands, fill_value=0)
    )

    x = np.arange(len(models))
    width = 0.12

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, b in enumerate(bands):
        offsets = x + (i - (len(bands) - 1) / 2) * width
        ax.bar(
            offsets,
            counts[b].values,
            width=width,
            color=FAITH_COLORS[b],
            edgecolor="black",
            alpha=0.9,
            label=FAITH_LABELS[b],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=10)
    ax.set_ylabel("Number of queries")
    ax.set_title("Faithfulness band comparison across models")
    ax.legend(frameon=False, ncol=3)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


# ---------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-model benchmark with custom output directories."
    )
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--prompt_file", type=str)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--bootstrap_iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    # NEW: allow overriding dirs
    parser.add_argument("--logs_dir", type=str, required=True)
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--charts_dir", type=str, required=True)

    args = parser.parse_args()

    models = {
        "phi3_4b": "phi3-4b",
        "mistral_7b": "mistral-7b-instruct",
        "llama3_8b": "llama3-8b-instruct",
    }

    all_dfs = []

    for profile, model_id in models.items():
        print(f"=== RUNNING MODEL: {profile} ({model_id}) ===")

        # Write model-specific subfolders inside provided base dirs
        logs_dir = Path(args.logs_dir) / f"logs_{profile}"
        eval_dir = Path(args.eval_dir) / f"eval_logs_{profile}"
        charts_dir = Path(args.charts_dir) / f"eval_charts_{profile}"

        clear_and_prepare(logs_dir)
        clear_and_prepare(eval_dir)
        clear_and_prepare(charts_dir)

        cmd = [
            "poetry", "run", "python", "scripts/run_full_benchmark.py",
            "--prompt_file", args.prompt_file,
            "--num_prompts", str(args.num_prompts),
            "--logs_dir", str(logs_dir),
            "--eval_dir", str(eval_dir),
            "--charts_dir", str(charts_dir),
            "--k", str(args.k),
            "--bootstrap_iters", str(args.bootstrap_iters),
            "--seed", str(args.seed),
        ]
        run(cmd)

        df = load_eval_df(eval_dir)
        df["model"] = profile
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)

    outdir = Path(args.charts_dir) / "model_comparison"
    outdir.mkdir(parents=True, exist_ok=True)

    df_all.to_csv(outdir / "all_models_evaluation.csv", index=False)

    if df_all.empty:
        return

    plot_global_comparison(df_all, outdir / "faithfulness_model_comparison")

    rb = MultiModelReportBuilder(base_dir=str(args.charts_dir))
    rb.build(name="multi_model_benchmark_report.pdf")


if __name__ == "__main__":
    main()
