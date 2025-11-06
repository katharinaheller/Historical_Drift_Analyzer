# src/core/prompt/query/prompt_builder.py
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)
Intent = Literal["chronological", "conceptual", "analytical", "comparative"]


@dataclass
class PromptBuilderConfig:
    snippet_char_limit: int = 700       # allows slightly larger excerpts
    sort_chronologically: bool = True
    include_overview: bool = True
    numeric_citations_only: bool = True


class PromptBuilder:
    """Compose the final LLM prompt using grouped, IEEE-style context references."""

    def __init__(self, cfg: Optional[PromptBuilderConfig] = None):
        self.cfg = cfg or PromptBuilderConfig()
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # ------------------------------------------------------------------
    def _safe_year(self, item: Dict[str, Any]) -> int:
        meta = item.get("metadata", {}) or {}
        year = meta.get("year") or item.get("year") or 0
        try:
            return int(year)
        except Exception:
            return 0

    # ------------------------------------------------------------------
    def _group_by_source(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group chunks belonging to the same PDF and preserve chronological order."""
        grouped: Dict[str, Dict[str, Any]] = {}
        for chunk in items:
            meta = chunk.get("metadata", {}) or {}
            src = meta.get("source_file") or "Unknown.pdf"
            year = meta.get("year") or chunk.get("year", "n/a")
            if src not in grouped:
                grouped[src] = {"year": year, "chunks": []}
            grouped[src]["chunks"].append(chunk.get("text", ""))

        # Sort by year (oldest → newest)
        ordered = sorted(grouped.items(), key=lambda x: self._safe_year({"metadata": {"year": x[1]["year"]}}))
        grouped_list = [
            {"index": i + 1, "source_file": src, "year": meta["year"], "text": " ".join(meta["chunks"])}
            for i, (src, meta) in enumerate(ordered)
        ]
        return grouped_list

    # ------------------------------------------------------------------
    def _context_overview(self, grouped: List[Dict[str, Any]]) -> str:
        lines = [f"[{g['index']}] ({g['year']}) {g['source_file']}" for g in grouped]
        return "Retrieved Context (oldest → newest):\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    def _system_prompt_for(self, intent: Intent) -> str:
        if intent == "chronological":
            return (
                "You are an analytical historian of Artificial Intelligence. "
                "Explain how the concept evolved over time, focusing on paradigm shifts, research trends, and milestones. "
                "Use numeric citations [1], [2], etc., and avoid author names or years in parentheses."
            )
        if intent == "conceptual":
            return (
                "You are an AI expert. Provide a precise definition, core principles, and clear explanation of the concept. "
                "Use numeric citations [1], [2], etc., and avoid author names or years in parentheses."
            )
        return (
            "You are an analytical researcher. Compare, contrast, and evaluate the ideas in a rigorous manner. "
            "Use numeric citations [1], [2], etc., and avoid author names or years in parentheses."
        )

    # ------------------------------------------------------------------
    def _snippets_block(self, grouped: List[Dict[str, Any]]) -> str:
        """Build consolidated context snippets per unique PDF."""
        limit = max(1, self.cfg.snippet_char_limit)
        parts: List[str] = []
        for g in grouped:
            text = g["text"][:limit].replace("\n", " ").strip()
            parts.append(f"[{g['index']}] ({g['year']}) {text}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    def build_prompt(self, query: str, intent: Intent, retrieved_topk: List[Dict[str, Any]]) -> str:
        """Construct a clean, grouped, IEEE-conform LLM prompt."""
        if not query or not query.strip():
            raise ValueError("Empty query passed to PromptBuilder")
        if not isinstance(retrieved_topk, list) or len(retrieved_topk) == 0:
            raise ValueError("Empty retrieved list passed to PromptBuilder")

        grouped = self._group_by_source(retrieved_topk)
        sys_prompt = self._system_prompt_for(intent)
        overview = self._context_overview(grouped) if self.cfg.include_overview else ""
        snippets = self._snippets_block(grouped)

        prompt = f"{sys_prompt}\n\n"
        if overview:
            prompt += f"{overview}\n\n"
        prompt += (
            f"User Question:\n{query.strip()}\n\n"
            f"Context Snippets:\n{snippets}\n\n"
            "Answer requirements:\n"
            "- Be concise, logically structured, and evidence-based.\n"
            "- Base claims only on the Context Snippets.\n"
            "- Cite using only numeric indices like [1], [2], etc.\n"
            "- Do not include author names or explicit years in parentheses.\n"
        )
        return prompt
