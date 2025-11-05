from __future__ import annotations
import re
from typing import List

class TemporalQueryExpander:
    """Generates temporally diverse query variants to improve historical coverage."""

    def __init__(self):
        # Static synonym sets for common AI terminology
        self.synonym_map = {
            "ai": ["artificial intelligence", "machine intelligence", "intelligent systems"],
            "neural": ["connectionist", "perceptron", "deep learning"],
            "algorithm": ["heuristic", "procedure", "rule-based system"],
        }

        # Temporal cues for encouraging retrieval across decades
        self.temporal_modifiers = [
            "history of", "timeline of", "evolution of", "early developments in",
            "recent trends in", "origins of", "milestones in", "historical overview of"
        ]

    # ------------------------------------------------------------------
    def expand(self, query: str) -> List[str]:
        """Return a small, meaningful set of semantically related and temporally expanded variants."""
        q = query.strip()
        q_lower = q.lower()

        expansions: List[str] = []

        # Synonym substitution (single keyword replacements)
        for key, syns in self.synonym_map.items():
            if re.search(rf"\b{re.escape(key)}\b", q_lower):
                for s in syns:
                    variant = re.sub(rf"\b{re.escape(key)}\b", s, q_lower)
                    expansions.append(variant)

        # Temporal modifier combinations
        for t in self.temporal_modifiers:
            if not any(t in q_lower for t in self.temporal_modifiers):
                expansions.append(f"{t} {q_lower}")

        # Deduplicate while preserving order
        seen = set()
        unique_exp = []
        for e in expansions:
            if e not in seen:
                unique_exp.append(e)
                seen.add(e)

        return unique_exp
