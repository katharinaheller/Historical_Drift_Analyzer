# scripts/calibrate_gt_thresholds.py
# Calibrates similarity thresholds for GroundTruthSettings using real RAW retrieval logs
# # No emojis

from __future__ import annotations
import json
from pathlib import Path
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Load all RAW chunks and compute similarity distribution
# ---------------------------------------------------------------------
def load_raw_pairs(logs_dir: Path) -> list[tuple[str, str]]:
    pairs = []
    for fp in sorted(logs_dir.glob("llm_*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            q = (
                data.get("query")
                or data.get("query_refined")
                or data.get("processed_query")
            )
            raw = (
                data.get("raw")
                or data.get("retrieved_chunks_raw")
                or data.get("faiss_raw")
            )
            if not q or not raw:
                continue

            for ch in raw:
                t = ch.get("text") or ""
                if t.strip():
                    pairs.append((q, t))
        except Exception:
            continue
    return pairs

# ---------------------------------------------------------------------
def compute_similarities(pairs, model_name="multi-qa-mpnet-base-dot-v1"):
    model = SentenceTransformer(model_name)
    sims = []

    for q, t in pairs:
        q_emb = model.encode([q], normalize_embeddings=True)
        t_emb = model.encode([t], normalize_embeddings=True)
        sims.append(float(util.cos_sim(q_emb, t_emb)[0][0]))

    return np.array(sims, float)

# ---------------------------------------------------------------------
def derive_thresholds(sims: np.ndarray) -> dict:
    """Derive high/mid/low thresholds based on percentiles."""
    if sims.size == 0:
        return {"high": 0.90, "mid": 0.80, "low": 0.70}

    high = float(np.percentile(sims, 90))
    mid = float(np.percentile(sims, 70))
    low = float(np.percentile(sims, 50))

    return {
        "high": round(high, 3),
        "mid": round(mid, 3),
        "low": round(low, 3),
    }

# ---------------------------------------------------------------------
def plot_distribution(sims: np.ndarray, out_dir: Path) -> None:
    fig = plt.figure(figsize=(6, 4))
    plt.hist(sims, bins=25, color="#003359", edgecolor="black", alpha=0.85)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.title("Distribution of RAW query–chunk similarities")
    plt.xlim(0.0, 1.0)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "similarity_distribution.png", dpi=150)
    fig.savefig(out_dir / "similarity_distribution.svg")
    plt.close(fig)

# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="multi-qa-mpnet-base-dot-v1")
    parser.add_argument("--out_dir", type=str, default="data/gt_calibration")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)

    print(f"Collecting RAW pairs from {logs_dir} ...")
    pairs = load_raw_pairs(logs_dir)

    sims = compute_similarities(pairs, model_name=args.model)
    print(f"Collected {len(sims)} similarity samples")

    thresholds = derive_thresholds(sims)
    print("\n=== Recommended similarity thresholds ===")
    print(json.dumps(thresholds, indent=2))

    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "thresholds.json").write_text(
        json.dumps(thresholds, indent=2),
        encoding="utf-8"
    )

    plot_distribution(sims, out_dir)
    print(f"\nDistribution plot saved → {out_dir}")

if __name__ == "__main__":
    main()
