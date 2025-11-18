from __future__ import annotations
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from src.core.evaluation.interfaces.i_metric import IMetric


class _BertEncoder:
    # Lightweight wrapper around a BERT model for CLS embeddings

    def __init__(self, model_name: str = "bert-base-uncased", device: Optional[str] = None):
        # Select device automatically if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: List[str]) -> np.ndarray:
        # Encode a list of texts into L2-normalized CLS embeddings
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)

        with torch.no_grad():
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            cls = out.last_hidden_state[:, 0, :]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
        return cls.cpu().numpy()


class BertIntrinsicMetric(IMetric):
    """
    Intrinsische Retrieval-Metrik auf Basis von BERT-Ähnlichkeit.

    Idee:
    - Verwende BERT-CLS-Embedding des Queries.
    - Berechne Cosine Similarity zu den ersten k Retrieved Chunks.
    - Aggregiere zu einem Score in [0, 1] (negatives wird auf 0 gecappt).
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: Optional[str] = None,
        top_k: int = 10,
        agg: str = "mean",
    ):
        # Store configuration for later inspection
        self.model_name = model_name
        self.top_k = int(top_k)
        self.agg = agg.lower()
        self.encoder = _BertEncoder(model_name=model_name, device=device)

    def _aggregate(self, sims: np.ndarray) -> float:
        # Aggregate similarity values into a single scalar
        if sims.size == 0:
            return 0.0

        sims = np.clip(sims, 0.0, 1.0)
        if self.agg == "max":
            return float(np.max(sims))
        if self.agg == "median":
            return float(np.median(sims))
        return float(np.mean(sims))

    def compute(self, **kwargs: Any) -> float:
        """
        Compute intrinsic BERT score.

        Erwartete Argumente:
        - query: str
        - retrieved_chunks: List[Dict[str, Any]]
        """
        query = kwargs.get("query", "")
        retrieved_chunks: List[Dict[str, Any]] = kwargs.get("retrieved_chunks", [])

        if not query or not retrieved_chunks:
            return 0.0

        # Limit to top_k chunks (falls mehr vorhanden sind)
        chunks = retrieved_chunks[: self.top_k]
        texts = [c.get("text") or c.get("snippet") or "" for c in chunks]
        texts = [t for t in texts if t.strip()]

        if not texts:
            return 0.0

        # Encode query and chunks
        q_vec = self.encoder.encode([query])
        c_vecs = self.encoder.encode(texts)

        if q_vec.shape[0] == 0 or c_vecs.shape[0] == 0:
            return 0.0

        q = q_vec[0]
        sims = np.dot(c_vecs, q)  # Cosine similarity, da Embeddings L2-normalisiert
        score = self._aggregate(sims)
        return float(score)

    def describe(self) -> Dict[str, str]:
        # Provide minimal metadata about this metric
        return {
            "name": "BERT-Intrinsic",
            "type": "intrinsic",
            "description": (
                "Query-chunk similarity metric using BERT CLS embeddings and cosine similarity, "
                "aggregated over the top-k retrieved chunks."
            ),
        }
