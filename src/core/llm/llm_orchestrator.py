# src/core/llm/llm_orchestrator.py
from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import time

from src.core.config.config_loader import ConfigLoader
from src.core.retrieval.retrieval_orchestrator import RetrievalOrchestrator
from src.core.llm.ollama_llm import OllamaLLM
from src.core.prompt.prompt_orchestrator import PromptOrchestrator
from src.core.prompt.query.prompt_builder import PromptBuilder


class LLMOrchestrator:
    """Refined query → Retrieval → Prompt assembly → LLM"""

    def __init__(self, config_path: str = "configs/llm.yaml"):
        self.cfg = ConfigLoader(config_path).config
        self.logger = logging.getLogger("LLMOrchestrator")
        self._setup_logging()
        self.prompt_phase = PromptOrchestrator()
        self.retriever = self._init_retriever()
        self.prompt_builder = PromptBuilder()
        self.llm = self._init_llm()
        self.logger.info("LLMOrchestrator initialized successfully.")

    def _setup_logging(self) -> None:
        log_level = self.cfg.get("global", {}).get("log_level", "INFO").upper()
        logging.basicConfig(level=getattr(logging, log_level), format="%(levelname)s | %(message)s")
        self.logger.info("Logging configured.")

    def _init_retriever(self) -> RetrievalOrchestrator:
        retriever = RetrievalOrchestrator()
        self.logger.info("Retriever initialized.")
        return retriever

    def _init_llm(self) -> OllamaLLM:
        profile_name = self.cfg.get("generation", {}).get("llm", {}).get("profile", "default")
        llm = OllamaLLM(config_path="configs/llm.yaml", profile=profile_name)
        self.logger.info(f"LLM backend ready (profile='{profile_name}').")
        return llm

    def process_query(self, query_obj: Optional[Dict[str, Any]]) -> str:
        """Full pipeline execution."""
        if not query_obj or not query_obj.get("refined_query"):
            self.logger.warning("Invalid or empty refined query object.")
            return ""

        query = query_obj["refined_query"]
        intent = query_obj.get("intent", "conceptual")

        self.logger.info(f"Retrieving context for refined query='{query}' (intent='{intent}')")
        try:
            retrieved_docs = self.retriever.retrieve(query, intent)
        except Exception as e:
            self.logger.exception(f"Retrieval failed: {e}")
            return ""
        if not retrieved_docs:
            self.logger.warning("No relevant documents retrieved.")
            return ""

        try:
            final_prompt = self._compose_full_prompt(query, intent, retrieved_docs)
        except Exception as e:
            self.logger.exception(f"Prompt construction failed: {e}")
            return ""

        llm_input = {
            "system_prompt": final_prompt,
            "query_refined": query.strip(),
            "intent": intent,
            "context_chunks": retrieved_docs,
        }
        self._log_llm_input(llm_input)

        try:
            output = self.llm.generate(final_prompt.strip())
            qid = self._log_llm_run(query, intent, retrieved_docs, output, final_prompt)
            self.logger.info(f"LLM generation successful. Run logged (query_id={qid}).")
            return output.strip()
        except Exception as e:
            self.logger.exception(f"LLM generation failed: {e}")
            return ""

    # ---------------------------------------------------------------
    def _compose_full_prompt(
        self,
        refined_query: str,
        intent: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> str:
        """Builds final prompt with refined query (no raw user query)."""
        system_prompt = self.prompt_builder.build_prompt(refined_query, intent)

        if intent == "chronological":
            def safe_year(meta: Dict[str, Any]) -> int:
                y = meta.get("year")
                try:
                    return int(y)
                except Exception:
                    return 9999
            retrieved_chunks = sorted(
                retrieved_chunks,
                key=lambda c: safe_year(c.get("metadata", {}))
            )
            self.logger.info("Chunks sorted chronologically (ascending).")

        lines: List[str] = []
        lines.append(system_prompt.strip())
        lines.append("")
        lines.append(f"Refined query:\n{refined_query.strip()}\n")
        lines.append(
            "You are given the following context snippets from historical AI-related documents. "
            "Each snippet is associated with a numeric source id in square brackets. "
            "When you answer, cite these sources using their numeric ids, e.g. [1], [2]. "
            "Do not invent new references or sources that are not listed below."
        )
        lines.append("")
        lines.append("Context snippets:")

        for idx, chunk in enumerate(retrieved_chunks, start=1):
            meta = chunk.get("metadata", {}) or {}
            src = meta.get("source_file", "Unknown.pdf")
            year = meta.get("year", "n/a")
            header = f"[{idx}] {src} ({year})"
            lines.append(header)
            lines.append(chunk.get("text", ""))
            lines.append("")

        lines.append(
            "Now answer the refined query above using ONLY the information from the context snippets. "
            "Use numeric citations like [1], [3] that refer to the snippet ids. "
            "Do not add a bibliography or 'References' section; use inline numeric citations only."
        )
        return "\n".join(lines)

    # ---------------------------------------------------------------
    def _log_llm_input(self, llm_input: Dict[str, Any]) -> Path:
        ts = time.strftime("%Y-%m-%dT%H-%M-%S")
        log_dir = Path("data/logs_llm")
        log_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": ts,
            "query_refined": llm_input["query_refined"],
            "intent": llm_input["intent"],
            "prompt_final_to_llm": llm_input["system_prompt"],
            "chunks_final_to_llm": llm_input["context_chunks"],
        }
        out_path = log_dir / f"llm_input_{ts}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info(f"Logged refined LLM input → {out_path}")
        return out_path

    def _log_llm_run(
        self,
        query: str,
        intent: str,
        retrieved: List[Dict[str, Any]],
        output: str,
        final_prompt: str,
    ) -> str:
        ts = time.strftime("%Y-%m-%dT%H-%M-%S")
        qid = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in query)[:80] or "query"
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": ts,
            "query_id": qid,
            "query_refined": query,
            "intent": intent,
            "prompt_final_to_llm": final_prompt,
            "retrieved_chunks": retrieved,
            "model_output": output,
        }
        (log_dir / f"llm_{ts}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return qid
