from __future__ import annotations
import logging
import os
import json
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Any, Dict
from src.core.logging.formatters import PlainFormatter, ColorFormatter, JSONFormatter

try:
    # Initialize colorama if available for colored console output
    from colorama import init as colorama_init
    colorama_init()
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


class StructuredLogger:
    # Singleton-based, configurable logger for all pipeline phases
    _instance: logging.Logger | None = None

    @classmethod
    def get_logger(cls, name: str = "HDA", config: Dict[str, Any] | None = None) -> logging.Logger:
        # Reuse existing logger if already created
        if cls._instance:
            return cls._instance

        cfg = config or {}
        level = getattr(logging, cfg.get("level", "INFO").upper(), logging.INFO)
        log_to_file = cfg.get("log_to_file", True)
        log_dir = Path(cfg.get("log_dir", "logs"))
        file_name = cfg.get("file_name", "hda.log")
        rotate = cfg.get("rotate_logs", True)
        json_log = cfg.get("json_log", False)
        console_color = cfg.get("console_color", True)

        # Create and configure logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False

        # Console handler setup
        console_handler = logging.StreamHandler()
        if console_color and COLORAMA_AVAILABLE:
            console_handler.setFormatter(ColorFormatter())
        else:
            console_handler.setFormatter(PlainFormatter())
        logger.addHandler(console_handler)

        # File handler setup
        if log_to_file:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_path = log_dir / file_name
            if rotate:
                handler = RotatingFileHandler(file_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
            else:
                handler = logging.FileHandler(file_path, encoding="utf-8")
            handler.setFormatter(JSONFormatter() if json_log else PlainFormatter())
            logger.addHandler(handler)

        # Store singleton instance
        cls._instance = logger
        return logger
