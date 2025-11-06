from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class QueryInputConfig:
    prompt_text: str = "> "
    strip_input: bool = True
    allow_empty: bool = False

class QueryInput:
    """Handles raw query intake (interactive or programmatic)."""
    def __init__(self, cfg: Optional[QueryInputConfig] = None):
        self.cfg = cfg or QueryInputConfig()
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    def read_interactive(self) -> Optional[str]:
        """Read query from stdin (interactive mode)."""
        try:
            raw = input(self.cfg.prompt_text)
        except (KeyboardInterrupt, EOFError):
            # Raise further to allow graceful termination at orchestrator level
            print()  # newline for clean CLI exit
            raise
        q = raw.strip() if self.cfg.strip_input else raw
        if not q and not self.cfg.allow_empty:
            logger.warning("Empty query ignored.")
            return None
        return q

    def from_program(self, query: Optional[str]) -> Optional[str]:
        """Receive query from another process or script."""
        if query is None:
            return None
        return query.strip() if self.cfg.strip_input else query
