from __future__ import annotations
import logging
from typing import Literal

logger = logging.getLogger(__name__)
Intent = Literal["chronological", "conceptual", "analytical", "comparative"]


class PromptBuilder:
    """
    Builds intent-specific system prompts for RAG/LLM orchestration
    and provides intent-guided query reformulation for improved retrieval alignment.
    """

    def __init__(self):
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # ------------------------------------------------------------------
    def _system_prompt_for(self, intent: Intent) -> str:
        """Return intent-specific base instruction with citation control."""
        cite_rule = (
            "Use numeric citations [1], [2], etc. "
            "Do not output a bibliography or any 'References' section. "
            "Do not list full literature items, author names, or publication years in parentheses. "
            "Only use numeric markers to refer to the provided document indices."
        )

        if intent == "chronological":
            return (
                "You are an analytical historian of Artificial Intelligence. "
                "Explain how the concept evolved over time, emphasizing paradigm shifts, milestones, and key research phases. "
                "Present the findings strictly in chronological order of publication years. "
                f"{cite_rule}"
            )

        if intent == "conceptual":
            return (
                "You are an AI expert. Provide a precise definition, describe its theoretical foundations, "
                "and explain its essential principles. "
                f"{cite_rule}"
            )

        if intent == "analytical":
            return (
                "You are a rigorous AI researcher. Analyze mechanisms, trade-offs, and implications "
                "with logically structured reasoning and explicit reference to retrieved sources. "
                f"{cite_rule}"
            )

        # comparative intent
        return (
            "You are an analytical researcher. Compare and contrast positions, frameworks, or definitions, "
            "and evaluate their empirical or theoretical grounding. "
            f"{cite_rule}"
        )

    # ------------------------------------------------------------------
    def reformulate_query(self, query: str, intent: Intent) -> str:
        """
        Intent-guided reformulation of the user's query into a canonical analytical form.

        The goal is to standardize phrasing for better retrieval and prompt conditioning.
        """
        if not query or not query.strip():
            raise ValueError("Empty query cannot be reformulated")

        q = query.strip()

        if intent == "chronological":
            return f"Trace the historical development and evolution of {q} over time."

        if intent == "conceptual":
            return f"Define {q}, describe its theoretical foundations, and explain its core principles."

        if intent == "analytical":
            return f"Analyze the mechanisms, advantages, and limitations of {q}."

        if intent == "comparative":
            return f"Compare and contrast key perspectives, frameworks, or interpretations of {q}."

        return q

    # ------------------------------------------------------------------
    def build_prompt(self, query: str, intent: Intent) -> str:
        """Return the clean base system prompt without context."""
        if not query or not query.strip():
            raise ValueError("Empty query passed to PromptBuilder")

        system_prompt = self._system_prompt_for(intent)
        logger.debug(f"Built system prompt for intent='{intent}' (no context embedded).")
        return system_prompt
