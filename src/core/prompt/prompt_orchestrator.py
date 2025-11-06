from __future__ import annotations
import logging
from typing import Dict, Any, Optional

from src.core.prompt.query.query_input import QueryInput
from src.core.prompt.query.query_preprocessor import QueryPreprocessor


class PromptOrchestrator:
    """Coordinates the full prompt understanding pipeline:
       (1) user input → (2) preprocessing → (3) intent classification.
    """

    def __init__(self, log_level: str = "INFO"):
        # Initialize logger once
        self.logger = logging.getLogger("PromptOrchestrator")
        if not self.logger.handlers:
            logging.basicConfig(
                level=getattr(logging, log_level.upper(), logging.INFO),
                format="%(levelname)s | %(message)s"
            )

        # Initialize submodules
        self.query_input = QueryInput()
        self.preprocessor = QueryPreprocessor()
        self.logger.info("PromptOrchestrator initialized successfully.")

    # ------------------------------------------------------------------
    def get_prompt_object(self) -> Dict[str, Any]:
        """Execute the complete prompt phase: read, clean, classify.

        Returns
        -------
        dict
            Structured prompt object:
            {
                "raw_query": str,
                "processed_query": str,
                "intent": str,
                "timestamp": str,
                ...
            }
        """
        try:
            # Step 1: Read user input interactively
            raw_query: Optional[str] = self.query_input.read_interactive()
            if not raw_query or not raw_query.strip():
                self.logger.warning("No input provided. Skipping prompt phase.")
                return {}

            # Step 2: Preprocess and classify
            result: Dict[str, Any] = self.preprocessor.process(raw_query)
            if not result or "processed_query" not in result:
                self.logger.warning("Preprocessing failed or returned empty result.")
                return {}

            self.logger.info(f"Prompt phase complete: intent='{result.get('intent', 'unknown')}'")
            return result

        except KeyboardInterrupt:
            # Re-raise to allow graceful termination by outer orchestrator
            self.logger.info("Prompt input cancelled by user (Ctrl+C). Exiting interactive mode.")
            raise

        except EOFError:
            # Handle Ctrl+D or input stream closure
            self.logger.info("Input stream closed (EOF). Terminating prompt phase.")
            raise KeyboardInterrupt  # unify behavior for clean exit

        except Exception as e:
            # Log unexpected runtime errors but keep session alive
            self.logger.exception(f"Unexpected error in prompt phase: {e}")
            return {}
