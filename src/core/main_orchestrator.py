from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from src.core.config.config_loader import ConfigLoader
from src.core.ingestion.ingestion_orchestrator import main as run_ingestion
from src.core.embedding.embedding_orchestrator import main as run_embedding
from src.core.retrieval.retrieval_orchestrator import main as run_retrieval
from src.core.llm.llm_orchestrator import main as run_llm


class MainOrchestrator:
    """Central controller coordinating all pipeline phases."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg_loader = ConfigLoader(config_path)
        self.cfg: Dict[str, Any] = self.cfg_loader.config
        self.logger = self._setup_logger()

    # ------------------------------------------------------------------
    def _setup_logger(self) -> logging.Logger:
        """Initialize consistent logging based on master config."""
        opts = self.cfg.get("global", {})
        level = getattr(logging, opts.get("log_level", "INFO").upper(), logging.INFO)
        logging.basicConfig(level=level, format="%(levelname)s | %(message)s")
        logger = logging.getLogger("MainOrchestrator")
        logger.info("Initialized main orchestrator")
        return logger

    # ------------------------------------------------------------------
    def run_phase(self, phase: str) -> None:
        """Dispatch to the correct pipeline phase."""
        self.logger.info(f"Starting phase: {phase.upper()}")

        base_dir = Path(self.cfg["global"]["base_dir"]).resolve()
        sys.path.append(str(base_dir / "src"))  # ensure relative imports work

        try:
            if phase == "ingestion":
                run_ingestion()

            elif phase == "embedding":
                run_embedding()

            elif phase == "retrieval":
                run_retrieval()

            elif phase == "llm":
                run_llm()

            elif phase == "all":
                self.logger.info("Running full pipeline (ingestion → embedding → retrieval → llm)")
                run_ingestion()
                run_embedding()
                run_retrieval()
                run_llm()

            else:
                self.logger.error(f"Unknown phase: {phase}")
                sys.exit(1)

            self.logger.info(f"Phase '{phase}' completed successfully.")

        except Exception as e:
            self.logger.error(f"Phase '{phase}' failed: {e}")
            raise

    # ------------------------------------------------------------------
    def run(self, args: argparse.Namespace) -> None:
        """Entrypoint wrapper for CLI execution."""
        if args.phase:
            self.run_phase(args.phase)
        else:
            self.logger.warning("No phase specified. Use --phase <name> or --phase all")


# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Define command-line interface."""
    parser = argparse.ArgumentParser(description="Historical Drift Analyzer – Main Orchestrator")
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["ingestion", "embedding", "retrieval", "llm", "evaluation", "all"],
        help="Select which phase of the pipeline to execute",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the master configuration YAML file",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    orchestrator = MainOrchestrator(config_path=args.config)
    orchestrator.run(args)
