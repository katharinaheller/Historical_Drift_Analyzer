from __future__ import annotations
import logging
from typing import Dict, Any, List
from src.core.config.config_loader import ConfigLoader
from src.core.retrieval.retrieval_orchestrator import RetrievalOrchestrator
from src.core.llm.ollama_llm import OllamaLLM


class LLMOrchestrator:
    """Coordinates retrieval, chronological context ordering, and LLM-based analytical QA."""

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
        self.logger.info("Initialized LLM orchestrator (chronological mode).")

    # ------------------------------------------------------------------
    def _init_retriever(self) -> RetrievalOrchestrator:
        """Initialize retrieval orchestrator."""
        try:
            retriever = RetrievalOrchestrator()
            self.logger.info("Retriever initialized successfully from config.")
            return retriever
        except Exception as e:
            self.logger.error(f"Failed to initialize retriever: {e}")
            raise

    # ------------------------------------------------------------------
    def _init_llm(self) -> OllamaLLM:
        """Initialize local Ollama-based LLM backend."""
        llm_cfg: Dict[str, Any] = self.cfg.get("generation", {}).get("llm", {})
        model = llm_cfg.get("model", "mistral:7b-instruct")
        temperature = float(llm_cfg.get("temperature", 0.2))
        max_tokens = int(llm_cfg.get("max_tokens", 1024))

        try:
            llm = OllamaLLM(model=model, temperature=temperature, max_tokens=max_tokens)
            self.logger.info(f"LLM initialized: {model}")
            return llm
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise

    # ------------------------------------------------------------------
    def _format_retrieved_context(self, results: List[Dict[str, Any]], top_k: int = 10) -> str:
        """Format retrieved results sorted by publication year (oldest first)."""
        seen = set()
        formatted_lines = []

        # sort ascending → oldest first
        results_sorted = sorted(
            results,
            key=lambda x: int(x.get("metadata", {}).get("year") or x.get("year") or 0),
        )[:top_k]

        for i, r in enumerate(results_sorted, start=1):
            meta = r.get("metadata", {})
            title = meta.get("title") or meta.get("source_file") or "Unknown"
            year = meta.get("year") or r.get("year", "n/a")
            if title in seen:
                continue
            seen.add(title)
            formatted_lines.append(f"[{i}] ({year}) {title}")

        context_header = "Retrieved Context (chronological order: oldest → newest):\n" + "\n".join(formatted_lines)
        context_header += "\n" + "=" * 100 + "\n"
        return context_header

    # ------------------------------------------------------------------
    def _build_prompt(self, query: str, retrieved: List[Dict[str, Any]]) -> str:
        """Construct chronological, citation-aware prompt for LLM generation."""
        # enforce ascending temporal order
        retrieved_sorted = sorted(
            retrieved,
            key=lambda r: int(r.get("metadata", {}).get("year") or r.get("year") or 0),
        )

        # formatted overview block
        context_block = self._format_retrieved_context(retrieved_sorted)

        # extract text snippets for each source
        snippets = "\n\n".join(
            f"[{i+1}] ({r.get('metadata', {}).get('year', 'n/a')}) "
            f"{r.get('text', '').encode('utf-8', errors='ignore').decode('utf-8')[:500]}"
            for i, r in enumerate(retrieved_sorted[:10])
        )

        # precise system prompt for timeline reasoning
        system_prompt = (
            "You are an analytical historian of Artificial Intelligence. "
            "Use the following chronological sources to explain how the concept evolved over time. "
            "Begin with the earliest developments and progress step-by-step toward the most recent events. "
            "Clearly mark transitions between decades or paradigm shifts, and reference each source [1]-[10] only once."
        )

        # combine final structured prompt
        prompt = (
            f"{system_prompt}\n\n"
            f"{context_block}\n"
            f"User Question:\n{query}\n\n"
            f"Context Snippets:\n{snippets}\n\n"
            "Now provide an analytical narrative that starts from the oldest sources and ends with the newest. "
            "Ensure the answer maintains chronological coherence and factual grounding."
        )
        return prompt

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Interactive analytical QA session with chronological reasoning."""
        self.logger.info("LLM phase ready for chronological analytical queries.")
        print("\nAsk something like:")
        print('   "How did the term Artificial Intelligence evolve over time?"')
        print("   Type 'exit' to quit.\n")

        while True:
            try:
                query = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                self.logger.info("Exiting LLM phase.")
                break

            if query.lower() in {"exit", "quit"}:
                self.logger.info("Session terminated by user.")
                break
            if not query:
                continue

            # --- Retrieval Phase ---
            self.logger.info(f"Retrieving context for query: {query}")
            retrieved_docs: List[Dict[str, Any]] = self.retriever.retrieve(query, top_k=10)
            if not retrieved_docs:
                print("No relevant documents found.\n")
                continue

            # --- Context Display ---
            context_block = self._format_retrieved_context(retrieved_docs)
            print("\n" + context_block)

            # --- LLM Generation ---
            try:
                full_prompt = self._build_prompt(query, retrieved_docs)
                answer = self.llm.generate(full_prompt, context=retrieved_docs)
            except Exception as e:
                self.logger.error(f"LLM generation failed: {e}")
                print("LLM generation failed. See logs for details.\n")
                continue

            # --- Output ---
            print("\nModel Output:\n")
            print(answer)
            print("\n" + "=" * 100 + "\n")

        self.close()

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Gracefully close all components."""
        try:
            self.retriever.close()
        except Exception as e:
            self.logger.warning(f"Error closing retriever: {e}")

        try:
            self.llm.close()
        except Exception as e:
            self.logger.warning(f"Error closing LLM: {e}")

        self.logger.info("LLM generation phase finished successfully.")


# ----------------------------------------------------------------------
def main() -> None:
    """Standalone execution for RAG + chronological LLM QA."""
    orchestrator = LLMOrchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()
