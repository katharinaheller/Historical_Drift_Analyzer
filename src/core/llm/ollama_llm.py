# src/core/llm/ollama_llm.py
from __future__ import annotations
import subprocess
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from src.core.llm.interfaces.i_llm import ILLM
from src.core.config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class OllamaLLM(ILLM):
    """Local Ollama backend (configurable via YAML profile, logs structured JSON)."""

    def __init__(self, config_path: str = "configs/llm.yaml", profile: str | None = None):
        cfg = ConfigLoader(config_path).config
        global_cfg = cfg.get("global", {})
        profiles = cfg.get("profiles", {})
        self.profile = profile or "default"
        profile_cfg = profiles.get(self.profile, {})

        self.model = profile_cfg.get("model", "mistral:7b-instruct")
        self.temperature = float(profile_cfg.get("temperature", 0.2))
        self.max_tokens = int(profile_cfg.get("max_tokens", 512))
        self.auto_pull = bool(profile_cfg.get("auto_pull", True))

        log_level = global_cfg.get("log_level", "INFO").upper()
        logging.basicConfig(level=getattr(logging, log_level), format="%(levelname)s | %(message)s")

        self.log_dir = Path("data/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"OllamaLLM ready (profile={self.profile}, model={self.model})")

        self._ensure_model_available()

    # ------------------------------------------------------------------
    def _ensure_model_available(self) -> None:
        """Ensure selected Ollama model exists (pull if missing)."""
        try:
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True,
                encoding="utf-8", errors="replace"
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip())

            if self.model.lower() not in result.stdout.lower():
                if not self.auto_pull:
                    logger.warning(f"Model '{self.model}' missing (auto_pull disabled).")
                    return
                logger.info(f"Pulling model '{self.model}' ...")
                pull_result = subprocess.run(
                    ["ollama", "pull", self.model],
                    capture_output=True, text=True,
                    encoding="utf-8", errors="replace"
                )
                if pull_result.returncode != 0:
                    raise RuntimeError(pull_result.stderr.strip())
                logger.info(f"Successfully pulled '{self.model}'.")
            else:
                logger.info(f"Model '{self.model}' available locally.")
        except Exception as e:
            logger.error(f"Model check failed: {e}")

    # ------------------------------------------------------------------
    def generate(self, prompt: str, context: List[Dict[str, Any]]) -> str:
        """Run Ollama model, logging prompt and context as structured JSON."""
        n_ctx = len(context)
        snippets = [
            {
                "index": i + 1,
                "source_file": (c.get("metadata", {}) or {}).get("source_file"),
                "year": (c.get("metadata", {}) or {}).get("year"),
                "text": c.get("text", "")
            }
            for i, c in enumerate(context[:n_ctx])
        ]

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        log_path = self.log_dir / f"llm_{timestamp}.json"
        log_data = {
            "timestamp": timestamp,
            "profile": self.profile,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "prompt": prompt,
            "context_snippets": snippets
        }

        try:
            with log_path.open("w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Logged LLM input to {log_path}")
        except Exception as e:
            logger.warning(f"Failed to write input log: {e}")

        cmd = ["ollama", "run", self.model, prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

        output = result.stdout.strip()
        log_data["model_output"] = output

        try:
            with log_path.open("w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Updated log with model output → {log_path}")
        except Exception as e:
            logger.warning(f"Failed to append model output: {e}")

        return output

    # ------------------------------------------------------------------
    def close(self) -> None:
        """No persistent connections to close."""
        pass
