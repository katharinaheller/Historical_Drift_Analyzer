from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    """
    Universal YAML configuration loader for all pipeline phases.
    - Supports ${PROJECT_ROOT} / ${base_dir} placeholders
    - Can inherit from a master config (for global settings)
    - Provides safe defaults for missing sections
    """

    def __init__(self, path: str, master_path: str | None = "configs/config.yaml"):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.path}")

        # Load phase-specific YAML (e.g. ingestion.yaml)
        with open(self.path, "r", encoding="utf-8") as f:
            phase_cfg = yaml.safe_load(f) or {}

        # Load master config if available
        master_cfg: Dict[str, Any] = {}
        if master_path:
            master_file = Path(master_path)
            if master_file.exists():
                with open(master_file, "r", encoding="utf-8") as f:
                    master_cfg = yaml.safe_load(f) or {}

        # Merge configurations (phase overrides master)
        self._raw = self._merge_dicts(master_cfg, phase_cfg)

        # Detect project root
        self.project_root = self._detect_project_root()

        # Resolve all placeholders like ${base_dir}
        self.config = self._expand_vars(self._raw)

        # Ensure minimal safe defaults
        for section in ["paths", "options", "chunking"]:
            self.config.setdefault(section, {})

    # ------------------------------------------------------------------
    def _detect_project_root(self) -> Path:
        """Infer project root (the directory above 'configs')."""
        p = self.path.resolve()
        if "configs" in p.parts:
            idx = p.parts.index("configs")
            return Path(*p.parts[:idx])
        return p.parent

    # ------------------------------------------------------------------
    def _merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge dicts, with override taking precedence."""
        merged = base.copy()
        for k, v in override.items():
            if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
                merged[k] = self._merge_dicts(merged[k], v)
            else:
                merged[k] = v
        return merged

    # ------------------------------------------------------------------
    def _expand_single_var(self, value: Any) -> Any:
        """Replace placeholders only when explicitly present."""
        if not isinstance(value, str) or "${" not in value:
            return value

        replacements = {
            "${PROJECT_ROOT}": str(self.project_root),
            "${project_root}": str(self.project_root),
            "${BASE_DIR}": str(self.project_root),
            "${base_dir}": str(self.project_root),
        }
        for ph, real in replacements.items():
            value = value.replace(ph, real)
        return str(Path(value).resolve())

    # ------------------------------------------------------------------
    def _expand_vars(self, data: Any) -> Any:
        """Recursively expand placeholders in nested structures."""
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
        """Safely access top-level config sections."""
        return self.config.get(key, default)

    # ------------------------------------------------------------------
    @property
    def raw(self) -> Dict[str, Any]:
        """Return unexpanded raw YAML structure."""
        return self._raw


# ----------------------------------------------------------------------
def load_config(path: str, master_path: str | None = "configs/config.yaml") -> Dict[str, Any]:
    """Convenience function: directly load and expand a config dictionary."""
    return ConfigLoader(path, master_path).config
