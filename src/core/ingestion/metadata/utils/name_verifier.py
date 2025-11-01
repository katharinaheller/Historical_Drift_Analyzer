import re
from nameparser import HumanName

class LocalNameVerifier:
    """Local deterministic fallback for name validation."""

    TITLE_PATTERN = re.compile(r"^(dr|prof|mr|ms|mrs|herr|frau)\.?\s", re.IGNORECASE)
    NAME_PATTERN = re.compile(r"^[A-Z][a-z]+(?:[-'\s][A-Z][a-z]+)*$")

    def is_person_name(self, text: str) -> bool:
        if not text or len(text) > 60:
            return False
        if any(ch.isdigit() for ch in text):
            return False

        hn = HumanName(text)
        # requires at least one capitalized component
        valid_structure = bool(hn.first or hn.last)
        valid_chars = bool(self.NAME_PATTERN.search(text)) or bool(self.TITLE_PATTERN.search(text))
        invalid_tokens = any(tok.lower() in {
            "university", "institute", "department", "school",
            "machine", "learning", "arxiv", "neural", "transformer",
            "abstract", "introduction", "appendix"
        } for tok in text.split())

        return valid_structure and valid_chars and not invalid_tokens
