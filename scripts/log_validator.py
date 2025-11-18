# scripts/log_validator.py
"""
Validator for LLM logs to ensure they are NDCG-ready.
Checks:
- final ranking exists
- raw ranking exists
- model output exists
- chunk IDs consistent
- warning if all relevance labels identical
"""

from __future__ import annotations
import json
from pathlib import Path
import sys


def load_json(fp: Path):
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception as e:
        return {"__error__": str(e)}


def validate_log(data: dict) -> dict:
    # Extract key fields from log
    raw = (
        data.get("raw")
        or data.get("retrieved_chunks_raw")
        or data.get("faiss_raw")
    )

    final = (
        data.get("retrieved_chunks_final")
        or data.get("retrieved_chunks")
    )

    answer = data.get("model_output") or data.get("answer")

    # Build result dictionary
    result = {
        "has_raw": raw is not None and isinstance(raw, list) and len(raw) > 0,
        "has_final": final is not None and isinstance(final, list) and len(final) > 0,
        "has_answer": isinstance(answer, str) and len(answer.strip()) > 0,
        "raw_len": len(raw) if raw else 0,
        "final_len": len(final) if final else 0,
        "warnings": []
    }

    # Warn if chunk IDs missing
    if final:
        missing_ids = [c for c in final if not c.get("id")]
        if missing_ids:
            result["warnings"].append(f"{len(missing_ids)} final chunks missing IDs")

    if raw:
        missing_ids = [c for c in raw if not c.get("id")]
        if missing_ids:
            result["warnings"].append(f"{len(missing_ids)} raw chunks missing IDs")

    # Check if all relevances identical (problematic GT)
    if final:
        rels = [c.get("relevance") for c in final if c.get("relevance") is not None]
        if rels:
            if len(set(rels)) == 1:
                result["warnings"].append("All relevance labels identical → NDCG may be trivial")

    return result


def main(logs_dir: str = "data/logs", pattern: str = "llm_*.json"):
    logs_path = Path(logs_dir)
    files = sorted(logs_path.glob(pattern))

    if not files:
        print(f"No log files found in {logs_dir}")
        return

    print(f"Validating {len(files)} logs...\n")

    for fp in files:
        data = load_json(fp)
        if "__error__" in data:
            print(f"{fp.name}: ERROR loading JSON → {data['__error__']}")
            continue

        result = validate_log(data)

        print(f"{fp.name}:")
        print(f"  RAW:   {'OK' if result['has_raw'] else 'MISSING'} (count={result['raw_len']})")
        print(f"  FINAL: {'OK' if result['has_final'] else 'MISSING'} (count={result['final_len']})")
        print(f"  ANS:   {'OK' if result['has_answer'] else 'MISSING'}")

        if result["warnings"]:
            for w in result["warnings"]:
                print(f"  WARNING: {w}")

        print("")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        main()
