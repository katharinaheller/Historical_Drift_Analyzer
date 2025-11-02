from __future__ import annotations
import logging
from typing import List, Dict
from src.core.retrieval.faiss_retriever import FAISSRetriever


class RetrievalOrchestrator:
    """Encapsulates vector retrieval as a callable module (no interactive prompt)."""

    def __init__(self,
                 vector_store_dir: str = "data/vector_store",
                 model_name: str = "all-MiniLM-L6-v2",
                 top_k: int = 5):
        self.logger = logging.getLogger("RetrievalOrchestrator")
        self.retriever = FAISSRetriever(
            vector_store_dir=vector_store_dir,
            model_name=model_name,
            top_k=top_k,
        )
        self.logger.info("Retriever initialized successfully.")

    # ------------------------------------------------------------------
    def retrieve(self, query: str) -> List[Dict[str, any]]:
        """Perform semantic retrieval given a query string."""
        self.logger.debug(f"Retrieving for query: {query}")
        results = self.retriever.search(query)
        if not results:
            self.logger.warning("No results found.")
            return []
        self.logger.info(f"Retrieved {len(results)} results for query '{query}'.")
        return results

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Gracefully close retriever resources."""
        self.retriever.close()


# ----------------------------------------------------------------------
def main() -> None:
    """Standalone execution for quick tests (not interactive)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger = logging.getLogger("RetrievalOrchestrator")

    orchestrator = RetrievalOrchestrator()
    # Minimal test query to validate integration
    query = "How did the term Artificial Intelligence evolve over time?"
    results = orchestrator.retrieve(query)
    logger.info(f"Top {len(results)} results retrieved.")
    for i, r in enumerate(results[:3], start=1):
        meta = r["metadata"]
        title = meta.get("title") or meta.get("source_file") or "Unknown source"
        year = meta.get("year", "n/a")
        logger.info(f"[{i}] ({year}) {title} | Score={r['score']:.4f}")
    orchestrator.close()


if __name__ == "__main__":
    main()
