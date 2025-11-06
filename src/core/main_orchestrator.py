# src/core/main_orchestrator.py
from __future__ import annotations
import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Any, Dict, List

from src.core.config.config_loader import ConfigLoader
from src.core.ingestion.ingestion_orchestrator import main as run_ingestion
from src.core.embedding.embedding_orchestrator import main as run_embedding
from src.core.retrieval.retrieval_orchestrator import RetrievalOrchestrator
from src.core.prompt.prompt_orchestrator import PromptOrchestrator
from src.core.llm.llm_orchestrator import LLMOrchestrator


class MainOrchestrator:
    """Central controller coordinating all pipeline phases of the Historical Drift Analyzer."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        # Ensure UTF-8 runtime consistency
        os.environ["PYTHONIOENCODING"] = "utf-8"
        os.environ["PYTHONUTF8"] = "1"
        os.environ["LC_ALL"] = "C.UTF-8"
        os.environ["LANG"] = "C.UTF-8"

        # Reconfigure stdout/stderr for UTF-8
        for stream_name in ("stdout", "stderr"):
            stream = getattr(sys, stream_name, None)
            if hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass

        # Load global configuration and setup logging
        self.cfg_loader = ConfigLoader(config_path)
        self.cfg: Dict[str, Any] = self.cfg_loader.config
        self.logger = self._setup_logger()

    # ------------------------------------------------------------------
    def _setup_logger(self) -> logging.Logger:
        """Initialize global logger."""
        opts = self.cfg.get("global", {})
        level = getattr(logging, opts.get("log_level", "INFO").upper(), logging.INFO)
        logging.basicConfig(level=level, format="%(levelname)s | %(message)s")
        logger = logging.getLogger("MainOrchestrator")
        logger.info("Initialized main orchestrator")
        return logger

    # ------------------------------------------------------------------
    def run_phase(self, phase: str) -> None:
        """Dispatch the orchestrator to the specified phase."""
        self.logger.info(f"Starting phase: {phase.upper()}")

        base_dir = Path(self.cfg["global"]["base_dir"]).resolve()
        sys.path.append(str(base_dir / "src"))

        try:
            if phase == "ingestion":
                run_ingestion()

            elif phase == "embedding":
                run_embedding()

            elif phase == "retrieval":
                self._run_prompt_retrieval_chain()

            elif phase == "llm":
                self._run_llm_interactive()

            elif phase == "all":
                self.logger.info("Running full pipeline (ingestion → embedding → interactive LLM phase)")
                run_ingestion()
                run_embedding()
                self._run_llm_interactive()

            else:
                self.logger.error(f"Unknown phase: {phase}")
                sys.exit(1)

            self.logger.info(f"Phase '{phase}' completed successfully.")

        except UnicodeDecodeError as ue:
            self.logger.error(f"Unicode decoding failed: {ue}. Retrying with UTF-8 replacement.")
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
            raise

        except KeyboardInterrupt:
            self.logger.info("Execution manually interrupted by user.")
            sys.exit(0)

        except Exception as e:
            self.logger.exception(f"Phase '{phase}' failed: {e}")
            raise

    # ------------------------------------------------------------------
    def _run_prompt_retrieval_chain(self) -> None:
        """Execute prompt → retrieval → automatic LLM generation (single-shot mode)."""
        self.logger.info("Executing prompt → retrieval → LLM phase")

        prompt_orch = PromptOrchestrator()
        prompt_data = prompt_orch.get_prompt_object()
        if not prompt_data or "processed_query" not in prompt_data:
            self.logger.warning("Prompt phase returned no valid query. Aborting retrieval.")
            return

        query = prompt_data["processed_query"]
        intent = prompt_data["intent"]

        retrieval = RetrievalOrchestrator(config_path="configs/retrieval.yaml")
        self.logger.info(f"Query intent='{intent}' → executing retrieval flow")
        retrieved: List[Dict[str, Any]] = retrieval.retrieve(query, intent)
        retrieval.close()

        if not retrieved:
            self.logger.warning("No documents retrieved.")
            return

        print("\n" + "=" * 80)
        print(f"Retrieved Top-{len(retrieved)} Chunks (intent={intent})")
        for i, r in enumerate(retrieved, start=1):
            meta = r.get("metadata", {}) or {}
            year = meta.get("year", "n/a")
            src = meta.get("source_file") or meta.get("title") or "Unknown"
            print(f"[{i}] ({year}) {src} | score={r.get('score', 0.0):.3f}")
        print("=" * 80 + "\n")

        try:
            self.logger.info("Launching Ollama model for contextual generation...")
            llm_orch = LLMOrchestrator()
            output = llm_orch.run_with_context(query, intent, retrieved)
            print("\n=== MODEL OUTPUT ===\n")
            print(output)
            print("\n====================\n")
            llm_orch.close()
        except Exception as e:
            self.logger.error(f"Automatic LLM generation failed: {e}")

    # ------------------------------------------------------------------
    def _run_llm_interactive(self) -> None:
        """Start an interactive prompt → retrieval → LLM loop until Ctrl+C."""
        self.logger.info("Starting interactive LLM session. Press Ctrl+C to exit.")
        llm_orch = LLMOrchestrator()
        try:
            llm_orch.run_interactive()
        except KeyboardInterrupt:
            self.logger.info("Interactive session terminated by user.")
        finally:
            llm_orch.close()

    # ------------------------------------------------------------------
    def run(self, args: argparse.Namespace) -> None:
        """Entrypoint for orchestrator execution."""
        if args.phase:
            self.run_phase(args.phase)
        else:
            self.logger.warning("No phase specified. Use --phase <name> or --phase all.")


# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """CLI argument parser."""
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
