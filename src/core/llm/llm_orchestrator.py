from __future__ import annotations
import logging
from typing import Dict, Any, List
from src.core.config.config_loader import ConfigLoader
from src.core.retrieval.retrieval_orchestrator import RetrievalOrchestrator
from src.core.llm.ollama_llm import OllamaLLM


class LLMOrchestrator:
    """Coordinates retrieval and LLM generation phases for contextual answering."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = ConfigLoader(config_path).config
        self.logger = logging.getLogger("LLMOrchestrator")
        self._setup_logging()
        self.retriever = self._init_retriever()
        self.llm = self._init_llm()

    # ------------------------------------------------------------------
    def _setup_logging(self) -> None:
        """Initialize consistent logging based on configuration."""
        log_level = self.cfg.get("global", {}).get("log_level", "INFO").upper()
        logging.basicConfig(level=getattr(logging, log_level), format="%(levelname)s | %(message)s")
        self.logger.info("Initialized LLM orchestrator")

    # ------------------------------------------------------------------
    def _init_retriever(self) -> RetrievalOrchestrator:
        """Initialize retrieval phase handler."""
        retrieval_cfg = self.cfg.get("retrieval", {}).get("retriever", {})
        top_k = retrieval_cfg.get("top_k", 5)
        retriever = RetrievalOrchestrator(top_k=top_k)
        self.logger.info(f"Retriever initialized with top_k={top_k}")
        return retriever

    # ------------------------------------------------------------------
    def _init_llm(self) -> OllamaLLM:
        """Initialize local LLM backend."""
        llm_cfg: Dict[str, Any] = self.cfg.get("generation", {}).get("llm", {})
        llm = OllamaLLM(
            model=llm_cfg.get("model", "mistral:7b-instruct"),
            temperature=llm_cfg.get("temperature", 0.2),
            max_tokens=llm_cfg.get("max_tokens", 1024),
        )
        self.logger.info(f"LLM initialized: {llm_cfg.get('model', 'mistral:7b-instruct')}")
        return llm

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Interactive session for analytical question answering."""
        self.logger.info("LLM phase ready for analytical queries.")
        print("\n🧠 Ask something like:")
        print('   "How did the term Artificial Intelligence evolve between 1950 and 2020?"')
        print("   Type 'exit' to quit.\n")

        while True:
            query = input("> ").strip()
            if query.lower() in {"exit", "quit"}:
                self.logger.info("Exiting LLM phase.")
                break
            if not query:
                continue

            # --- Retrieval ---
            retrieved_docs: List[Dict[str, Any]] = self.retriever.retrieve(query)
            if not retrieved_docs:
                print("No relevant documents found.\n")
                continue

            # --- LLM Generation ---
            answer = self.llm.generate(query, retrieved_docs)

            # --- Structured Output ---
            print("\n" + "=" * 100)
            print("📚 Retrieved Context:")
            for i, doc in enumerate(retrieved_docs, start=1):
                meta = doc.get("metadata", {})
                title = meta.get("title") or meta.get("source_file", "Unknown Source")
                year = meta.get("year", "n/a")
                print(f"[{i}] ({year}) {title}")
            print("=" * 100)
            print("\n💬 Model Output:\n")
            print(answer)
            print("\n" + "=" * 100 + "\n")

        self.close()

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Gracefully release resources."""
        self.retriever.close()
        self.llm.close()
        self.logger.info("LLM generation phase finished successfully.")


# ----------------------------------------------------------------------
def main() -> None:
    """Entry point for standalone execution."""
    orchestrator = LLMOrchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()
