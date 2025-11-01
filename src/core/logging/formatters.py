from __future__ import annotations
import logging
import json
from datetime import datetime

try:
    # Use colorama for colored console output if available
    from colorama import Fore, Style
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


class PlainFormatter(logging.Formatter):
    # Simple, deterministic log format without colors
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        return f"{ts} | {record.levelname:<8} | {record.getMessage()}"


class ColorFormatter(logging.Formatter):
    # Colored console formatter for better readability
    COLORS = {
        "DEBUG": Fore.BLUE,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.MAGENTA,
    }

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        color = self.COLORS.get(record.levelname, "")
        reset = Style.RESET_ALL if COLORAMA_AVAILABLE else ""
        return f"{color}{ts} | {record.levelname:<8} | {record.getMessage()}{reset}"


class JSONFormatter(logging.Formatter):
    # JSON-based formatter for structured, machine-readable logs
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        return json.dumps(entry, ensure_ascii=False)
