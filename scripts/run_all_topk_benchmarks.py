# run_all_topk_benchmarks.py
# Runs TOP-k benchmarks FAST by calling run_full_benchmark.py directly.
# No multi-model aggregation. Clean and predictable folder structure.
# # No emojis

import subprocess
import os
import yaml
from pathlib import Path

BASE_DIR = r"C:\Users\katha\historical-drift-analyzer"
RETRIEVAL_YAML = os.path.join(BASE_DIR, "configs", "retrieval.yaml")
PROMPT_FILE = os.path.join(BASE_DIR, "data", "prompts", "benchmark_prompts.txt")

MODELS = {
    "phi3_4b": "phi3-4b",
    "mistral_7b": "mistral:7b-instruct",
    "llama3_8b": "llama3:8b-instruct"
}

TOPK_VALUES = [3, 5, 7, 10, 20]


def update_retrieval_yaml(final_k: int):
    with open(RETRIEVAL_YAML, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["options"]["final_k"] = int(final_k)
    config["options"]["oversample_factor"] = 15

    with open(RETRIEVAL_YAML, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)


def run_full_model(model_name: str, model_key: str, k: int):
    top_dir = Path(BASE_DIR) / "data" / f"top_{k}"

    logs_dir = top_dir / f"logs_{model_key}"
    eval_dir = top_dir / f"eval_logs_{model_key}"
    charts_dir = top_dir / f"eval_charts_{model_key}"

    logs_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "poetry", "run", "python",
        f"{BASE_DIR}/scripts/run_full_benchmark.py",
        "--prompt_file", PROMPT_FILE,
        "--num_prompts", "10",
        "--logs_dir", str(logs_dir),
        "--eval_dir", str(eval_dir),
        "--charts_dir", str(charts_dir),
        "--k", str(k),
        "--bootstrap_iters", "2000",
        "--seed", "42"
    ]

    print(f"\n=== RUNNING MODEL {model_key} WITH k={k} ===\n")
    subprocess.run(cmd, check=True)


def run_one_topk(k: int):
    print(f"\n===== RUNNING TOP-k = {k} =====\n")

    update_retrieval_yaml(k)

    for model_key, _ in MODELS.items():
        run_full_model(_, model_key, k)

    print(f"\n===== TOP-k {k} DONE =====\n")


def main():
    for k in TOPK_VALUES:
        run_one_topk(k)

    print("\nAll TOP-k benchmarks completed successfully.")


if __name__ == "__main__":
    main()
