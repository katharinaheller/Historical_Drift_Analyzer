from __future__ import annotations
import os
import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    """
    Loads YAML configuration files and replaces ${base_dir} / ${PROJECT_ROOT}
    placeholders only when they are explicitly present.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f) or {}

        # Detect project root (directory above 'configs')
        self.project_root = self._detect_project_root()

        # Determine base_dir (may contain placeholders)
        global_section = self._raw.get("global", {})
        base_dir_value = global_section.get("base_dir", "${PROJECT_ROOT}")
        self.base_dir = self._expand_single_var(base_dir_value)

        # Expand all placeholders recursively
        self.config = self._expand_vars(self._raw)

        # Debug info
        print("\n[DEBUG] Expanded paths:")
        for k, v in self.config.get("paths", {}).items():
            print(f"  {k}: {v}")
        print()

    # ------------------------------------------------------------------
    def _detect_project_root(self) -> Path:
        """Return the root directory of the project."""
        p = self.path.resolve()
        if "configs" in p.parts:
            idx = p.parts.index("configs")
            return Path(*p.parts[:idx])
        return p.parent

    # ------------------------------------------------------------------
    def _expand_single_var(self, value: str) -> str:
        """Replace placeholders only if ${...} patterns are present."""
        if not isinstance(value, str) or "${" not in value:
            return value  # do not modify normal strings like "auto"

        replacements = {
            "${PROJECT_ROOT}": str(self.project_root),
            "${project_root}": str(self.project_root),
            "${BASE_DIR}": str(self.project_root),
            "${base_dir}": str(self.project_root),
        }

        for placeholder, real in replacements.items():
            value = value.replace(placeholder, real)
        return str(Path(value).resolve())

    # ------------------------------------------------------------------
    def _expand_vars(self, data: Any) -> Any:
        """Recursively replace placeholders in nested structures."""
        if isinstance(data, dict):
            return {k: self._expand_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._expand_vars(v) for v in data]
        elif isinstance(data, str):
            return self._expand_single_var(data)
        else:
            return data

    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Access top-level config sections."""
        return self.config.get(key, default)


# ----------------------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    """Convenience wrapper returning parsed and expanded configuration dict."""
    return ConfigLoader(path).config
