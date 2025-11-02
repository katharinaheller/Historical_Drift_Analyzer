from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.core.config.config_loader import ConfigLoader
from src.core.embedding.embedder_factory import EmbedderFactory
from src.core.embedding.vector_store_factory import VectorStoreFactory

logger = logging.getLogger("EmbeddingOrchestrator")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_chunk_files(chunks_dir: Path):
    for p in chunks_dir.glob("*.json"):
        if p.is_file():
            yield p


def _extract_chunks(chunk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "chunks" in chunk_data and isinstance(chunk_data["chunks"], list):
        return chunk_data["chunks"]
    elif "text" in chunk_data:
        return [{"text": chunk_data["text"]}]
    return []


def _resolve_metadata_for_chunk(chunk_file: Path, metadata_dir: Path) -> Dict[str, Any]:
    """Try to locate matching metadata JSON for a given chunk file."""
    base = chunk_file.stem.replace(".chunks", "")
    candidates = [
        metadata_dir / f"{chunk_file.name}",
        metadata_dir / f"{base}.json",
        metadata_dir / f"{base}.metadata.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return _load_json(candidate)
    logger.warning(f"No metadata found for {chunk_file.name}")
    return {}


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load merged configuration (embedding + master)
    # ------------------------------------------------------------------
    cfg_loader = ConfigLoader("configs/embedding.yaml", master_path="configs/config.yaml")
    cfg = cfg_loader.config

    opts: Dict[str, Any] = cfg.get("options", {})
    log_level = getattr(logging, opts.get("log_level", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(levelname)s | %(message)s")
    logger.info("Starting embedding pipeline")

    paths: Dict[str, Any] = cfg.get("paths", {})
    chunks_dir = Path(paths.get("chunks_dir", "data/processed/chunks")).resolve()
    metadata_dir = Path(paths.get("metadata_dir", "data/processed/metadata")).resolve()

    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory does not exist: {chunks_dir}")
    if not metadata_dir.exists():
        logger.warning(f"Metadata directory does not exist: {metadata_dir} (metadata will be empty)")

    # ------------------------------------------------------------------
    # 2. Initialize embedding backend + vector store
    # ------------------------------------------------------------------
    embedder = EmbedderFactory.from_config(cfg)
    logger.info(f"Initialized embedder with dimension={embedder.dimension}")

    vector_store = VectorStoreFactory.from_config(cfg, dimension=embedder.dimension)
    logger.info("Initialized vector store")

    batch_size = int(opts.get("batch_size", 16))
    texts_batch: List[str] = []
    metas_batch: List[Dict[str, Any]] = []

    try:
        for chunk_file in _iter_chunk_files(chunks_dir):
            chunk_json = _load_json(chunk_file)
            chunks = _extract_chunks(chunk_json)
            if not chunks:
                logger.warning(f"No chunks found in {chunk_file.name}")
                continue

            meta_data = _resolve_metadata_for_chunk(chunk_file, metadata_dir)

            for ch in chunks:
                text = ch.get("text", "").strip()
                if not text:
                    continue

                merged_meta = {
                    "source_file": meta_data.get("source_file", chunk_file.stem),
                    "title": meta_data.get("title"),
                    "authors": meta_data.get("authors"),
                    "year": meta_data.get("year"),
                    "detected_language": meta_data.get("detected_language"),
                    "page_count": meta_data.get("page_count"),
                    "origin_chunk_file": str(chunk_file.name),
                }

                texts_batch.append(text)
                metas_batch.append(merged_meta)

                if len(texts_batch) >= batch_size:
                    try:
                        vectors = embedder.embed_batch(texts_batch, batch_size=batch_size)
                        vector_store.add_vectors(vectors, texts_batch, metas_batch)
                        logger.info(f"Embedded and stored batch of size {len(texts_batch)}")
                    except Exception as e:
                        logger.error(f"Error during embedding or storing batch: {e}")
                    finally:
                        texts_batch.clear()
                        metas_batch.clear()

        if texts_batch:
            vectors = embedder.embed_batch(texts_batch, batch_size=batch_size)
            vector_store.add_vectors(vectors, texts_batch, metas_batch)
            logger.info(f"Embedded and stored final batch of size {len(texts_batch)}")

        vector_store.persist()
        logger.info("Embedding pipeline finished successfully.")

    finally:
        embedder.close()
        vector_store.close()


if __name__ == "__main__":
    main()
