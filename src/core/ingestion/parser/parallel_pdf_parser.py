# src/core/ingestion/parser/parallel_pdf_parser.py
from __future__ import annotations
import concurrent.futures
import logging
import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from src.core.ingestion.parser.parser_factory import ParserFactory


class ParallelPdfParser:
    """CPU-based parallel parser orchestrator using multiple processes."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.parser_factory = ParserFactory(config, logger=self.logger)

        parallel_cfg = config.get("options", {}).get("parallelism", "auto")
        if isinstance(parallel_cfg, int) and parallel_cfg > 0:
            self.num_workers = parallel_cfg
        else:
            self.num_workers = max(1, multiprocessing.cpu_count() - 1)

        self.logger.info(f"Initialized ParallelPdfParser with {self.num_workers} worker(s)")

    # ------------------------------------------------------------------
    def parse_all(self, pdf_dir: str | Path, output_dir: str | Path) -> List[Dict[str, Any]]:
        pdf_dir, output_dir = Path(pdf_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            self.logger.warning(f"No PDFs found in {pdf_dir}")
            return []

        results: List[Dict[str, Any]] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_pdf = {executor.submit(self._parse_single, str(pdf)): pdf for pdf in pdf_files}
            for future in concurrent.futures.as_completed(future_to_pdf):
                pdf = future_to_pdf[future]
                try:
                    result = future.result()
                    results.append(result)
                    out_path = output_dir / f"{pdf.stem}.parsed.json"
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    self.logger.info(f"✓ Parsed {pdf.name}")
                except Exception as e:
                    self.logger.error(f"✗ Failed to parse {pdf.name}: {e}")

        return results

    # ------------------------------------------------------------------
    def _parse_single(self, pdf_path: str) -> Dict[str, Any]:
        parser = self.parser_factory.create_parser()
        return parser.parse(pdf_path)
