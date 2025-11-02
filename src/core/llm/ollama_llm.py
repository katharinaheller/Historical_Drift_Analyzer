from __future__ import annotations
import subprocess
from typing import List, Dict
from src.core.llm.interfaces.i_llm import ILLM

class OllamaLLM(ILLM):
    """Local interface to Ollama (e.g., mistral:7b-instruct)."""

    def __init__(self, model: str = "mistral:7b-instruct",
                 temperature: float = 0.2, max_tokens: int = 512):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, context: List[Dict[str, str]]) -> str:
        # Build context text
        context_text = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(context[:5]))
        full_prompt = (
            f"System: You are an analytical assistant summarizing semantic change in terminology.\n"
            f"Parameters: temperature={self.temperature}, max_tokens={self.max_tokens}\n"
            f"User: {prompt}\n\nContext:\n{context_text}\n\n"
            "Answer analytically and cite document indices like [1], [2], etc."
        )

        # “ollama run” with prompt as positional arg
        cmd = ["ollama", "run", self.model, full_prompt]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"Ollama execution failed (code {result.returncode}):\n{result.stderr.strip()}"
            )

        return result.stdout.strip()

    def close(self) -> None:
        pass
