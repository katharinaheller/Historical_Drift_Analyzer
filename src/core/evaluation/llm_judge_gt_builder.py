# src/core/evaluation/llm_judge_gt_builder.py
from __future__ import annotations
from typing import Dict, Any, List
import json
import logging
import re

from src.core.llm.ollama_llm import OllamaLLM

logger = logging.getLogger("LlmJudgeGTBuilder")


# Strict JSON-only prompt
_JUDGE_PROMPT = """
Return ONLY a JSON exact of the form:
{"relevance": X}

Where X ∈ {0,1,2,3}.
No explanations.
No lists.
No text outside the JSON.

Query:
{query}

Chunk:
{chunk}
""".strip()


class LlmJudgeGTBuilder:
    # Strict, deterministic relevance grader
    def __init__(self, llm_profile: str = "mistral_7b"):
        self.llm_profile = llm_profile

        try:
            self.client = OllamaLLM(config_path="configs/llm.yaml", profile=llm_profile)
        except Exception as e:
            logger.error(f"Judge init failed, fallback to mistral_7b: {e}")
            self.client = OllamaLLM(config_path="configs/llm.yaml", profile="mistral_7b")

    # ----------------------------------------------------------
    def _parse_relevance(self, raw: str) -> int:
        raw = raw.strip()

        # strip code fences if any
        if raw.startswith("```"):
            parts = raw.split("```")
            for p in parts:
                if "{" in p and "}" in p:
                    raw = p.strip()
                    break

        # strict JSON read
        try:
            obj = json.loads(raw)
            rel = obj.get("relevance", None)
            if isinstance(rel, int) and rel in (0, 1, 2, 3):
                return rel
        except Exception:
            pass

        # Fallback: extract first digit in [0..3]
        m = re.search(r"\b([0-3])\b", raw)
        if m:
            return int(m.group(1))

        logger.warning(f"Unparsable judge output → default=0: {raw[:200]}")
        return 0

    # ----------------------------------------------------------
    def _judge_one(self, query: str, chunk_text: str) -> int:
        prompt = _JUDGE_PROMPT.format(query=query, chunk=chunk_text)
        try:
            out = self.client.generate(prompt)
        except Exception as e:
            logger.error(f"Judge call failed → 0: {e}")
            return 0
        return self._parse_relevance(out)

    # ----------------------------------------------------------
    def build(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, int]:
        if not query or not retrieved_docs:
            return {}

        gt: Dict[str, int] = {}

        for doc in retrieved_docs:
            cid = (
                doc.get("id")
                or doc.get("chunk_id")
                or doc.get("doc_id")
                or "unknown"
            )

            text = doc.get("text", "") or ""
            if not text.strip():
                gt[cid] = 0
                continue

            gt[cid] = self._judge_one(query, text)

        return gt
