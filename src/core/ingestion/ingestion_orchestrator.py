from __future__ import annotations
import logging
import json
from pathlib import Path
from typing import Any, Dict

from src.core.config.config_loader import ConfigLoader
from src.core.ingestion.utils.file_utils import ensure_dir
from src.core.ingestion.parser.parser_factory import ParserFactory
from src.core.ingestion.metadata.metadata_extractor_factory import MetadataExtractorFactory
from src.core.ingestion.cleaner.rag_text_cleaner import RagTextCleaner
from src.core.ingestion.chunking.chunking_orchestrator import ChunkingOrchestrator


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load configuration (merge phase + master)
    # ------------------------------------------------------------------
    cfg_loader = ConfigLoader("configs/ingestion.yaml", master_path="configs/config.yaml")
    cfg = cfg_loader.config

    opts: Dict[str, Any] = cfg.get("options", {})
    log_level = getattr(logging, opts.get("log_level", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(levelname)s | %(message)s")
    logger = logging.getLogger("IngestionOrchestrator")
    logger.info("Starting ingestion pipeline")

    # ------------------------------------------------------------------
    # 2. Initialize core components
    # ------------------------------------------------------------------
    parser_factory = ParserFactory(cfg, logger=logger)
    metadata_factory = MetadataExtractorFactory.from_config(cfg)
    cleaner = RagTextCleaner.default()
    chunking_orchestrator = ChunkingOrchestrator(config=cfg)

    active_metadata_fields = opts.get("metadata_fields", [])
    parallelism = opts.get("parallelism", "auto")

    # ------------------------------------------------------------------
    # 3. Resolve all paths
    # ------------------------------------------------------------------
    paths = cfg.get("paths", {})
    raw_dir = Path(paths.get("raw_pdfs", "data/raw_pdfs")).resolve()
    parsed_dir = Path(paths.get("parsed", "data/processed/parsed")).resolve()
    cleaned_dir = Path(paths.get("cleaned", "data/processed/cleaned")).resolve()
    metadata_dir = Path(paths.get("metadata", "data/processed/metadata")).resolve()
    chunks_dir = Path(paths.get("chunks", "data/processed/chunks")).resolve()

    for d in [parsed_dir, cleaned_dir, metadata_dir, chunks_dir]:
        ensure_dir(d)

    pdf_files = sorted(raw_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {raw_dir}")
        return
    logger.info(f"Found {len(pdf_files)} PDF(s) in {raw_dir}")

    # ------------------------------------------------------------------
    # 4. Parallel or sequential mode
    # ------------------------------------------------------------------
    use_parallel = (
        isinstance(parallelism, int) and parallelism > 1
    ) or (isinstance(parallelism, str) and parallelism.lower() == "auto")

    if use_parallel:
        try:
            parallel_parser = parser_factory.create_parallel_parser()
            logger.info("Running ingestion in parallel mode ...")
            results = parallel_parser.parse_all(raw_dir, parsed_dir)

            for res in results:
                if "text" not in res:
                    continue
                res["text"] = cleaner.clean(res["text"])
                res["chunks"] = chunking_orchestrator.process(res["text"], metadata=res.get("metadata", {}))

                pdf_name = Path(res["metadata"]["source_file"]).stem
                pdf_path = raw_dir / f"{pdf_name}.pdf"
                all_meta = metadata_factory.extract_all(str(pdf_path))
                filtered_meta = {k: v for k, v in all_meta.items()
                                 if not active_metadata_fields or k in active_metadata_fields}
                res["metadata"].update(filtered_meta)

                # Save outputs
                (parsed_dir / f"{pdf_name}.parsed.json").write_text(
                    json.dumps({"text": res["text"]}, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (cleaned_dir / f"{pdf_name}.cleaned.json").write_text(
                    json.dumps({"cleaned_text": res["text"]}, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (metadata_dir / f"{pdf_name}.metadata.json").write_text(
                    json.dumps(res["metadata"], ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (chunks_dir / f"{pdf_name}.chunks.json").write_text(
                    json.dumps({"chunks": res["chunks"]}, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            logger.info(f"Parallel ingestion completed ({len(results)} file(s)).")
            return
        except Exception as e:
            logger.error(f"Parallel ingestion failed → fallback to sequential: {e}")

    # ------------------------------------------------------------------
    # 5. Sequential fallback mode
    # ------------------------------------------------------------------
    parser = parser_factory.create_parser()
    for pdf in pdf_files:
        logger.info(f"Parsing {pdf.name} ...")
        try:
            parsed_result = parser.parse(str(pdf))
            if "text" not in parsed_result:
                continue

            parsed_result["text"] = cleaner.clean(parsed_result["text"])
            parsed_result["chunks"] = chunking_orchestrator.process(
                parsed_result["text"], metadata=parsed_result.get("metadata", {})
            )

            base_metadata = parsed_result.get("metadata", {})
            all_metadata = metadata_factory.extract_all(str(pdf))
            if active_metadata_fields:
                all_metadata = {k: v for k, v in all_metadata.items() if k in active_metadata_fields}
            base_metadata.update(all_metadata)
            parsed_result["metadata"] = base_metadata

            # Write outputs
            (parsed_dir / f"{pdf.stem}.parsed.json").write_text(
                json.dumps({"text": parsed_result["text"]}, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (cleaned_dir / f"{pdf.stem}.cleaned.json").write_text(
                json.dumps({"cleaned_text": parsed_result["text"]}, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (metadata_dir / f"{pdf.stem}.metadata.json").write_text(
                json.dumps(base_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (chunks_dir / f"{pdf.stem}.chunks.json").write_text(
                json.dumps({"chunks": parsed_result["chunks"]}, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            logger.info(f"Completed {pdf.name}")
        except Exception as e:
            logger.error(f"Failed to parse {pdf.name}: {e}")

    # ------------------------------------------------------------------
    # 6. Finish
    # ------------------------------------------------------------------
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()
