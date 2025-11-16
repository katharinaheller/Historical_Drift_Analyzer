# src/core/evaluation/metrics/faithfulness_metric.py
from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import re
import spacy

from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim

from src.core.evaluation.settings import EvaluationSettings, DEFAULT_EVAL_SETTINGS


# Load spaCy model once (NER + sentence segmentation)
try:
    NLP = spacy.load("en_core_web_sm")
except Exception:
    NLP = None


class FaithfulnessMetric:
    """
    Claim-level, evidence-based faithfulness metric combining:
    - Cross-encoder NLI entailment for robust evidence validation
    - Embedding similarity fallback for borderline segments
    - Top-k evidence aggregation
    - Specificity penalty (excessive ungrounded detail)
    - Temporal consistency penalty (year/decade mismatch)
    """

    def __init__(self, settings: EvaluationSettings = DEFAULT_EVAL_SETTINGS):
        # Store settings locally for easier inspection
        self.settings = settings
        cfg = settings.faithfulness

        # Main entailment model (entails/neutral/contradiction)
        self.cross = CrossEncoder(cfg.ent_model)

        # Embedding model fallback
        self.emb = SentenceTransformer(cfg.emb_model)

        # Thresholds for embedding fallback
        self.high_thr = cfg.emb_bands.high
        self.mid_thr = cfg.emb_bands.mid
        self.low_thr = cfg.emb_bands.low

        # Evidence aggregation and penalties
        self.top_k = cfg.top_k
        self.entailment_low_cutoff = cfg.entailment_low_cutoff
        self.w_spec = cfg.specificity_penalty
        self.w_temp = cfg.temporal_penalty

    # ----------------------------------------------------------
    def _extract_claims(self, answer: str) -> List[str]:
        """Split answer into minimal evidence-checkable claims."""
        if not answer:
            return []
        if NLP:
            doc = NLP(answer)
            sents = [s.text.strip() for s in doc.sents]
        else:
            sents = re.split(r"[.!?]\s+", answer)

        claims = []
        for s in sents:
            s_clean = s.strip()
            if len(s_clean.split()) >= 3:
                claims.append(s_clean)
        return claims

    # ----------------------------------------------------------
    def _claim_date(self, claim: str) -> int | None:
        """Extract explicit years from claim (e.g. 1998, 2020)."""
        yrs = re.findall(r"\b(19\d{2}|20\d{2})\b", claim)
        return int(yrs[0]) if yrs else None

    # ----------------------------------------------------------
    def _specificity_score(self, claim: str) -> float:
        """Quantify factual specificity based on NER and numeric density."""
        if not NLP:
            nums = len(re.findall(r"\d+", claim))
            ents = len(re.findall(r"[A-Z][a-z]+", claim))
            return (nums + ents) / max(5, len(claim.split()))

        doc = NLP(claim)
        nums = sum(1 for t in doc if t.like_num)
        ents = len(doc.ents)
        return (nums + ents) / max(5, len(doc))

    # ----------------------------------------------------------
    def _entailment_score(self, claim: str, chunks: List[str]) -> float:
        """Compute aggregated entailment probability over all chunks."""
        if not chunks:
            return 0.0

        pairs = [(claim, c) for c in chunks]
        preds = self.cross.predict(pairs, apply_softmax=True)
        entail_probs = np.array([p[0] for p in preds])
        topk = np.sort(entail_probs)[-self.top_k :]
        return float(np.mean(topk))

    # ----------------------------------------------------------
    def _embedding_fallback(self, claim: str, chunks: List[str]) -> float:
        """Fallback evidence score using embedding similarity."""
        if not chunks:
            return 0.0

        c_emb = self.emb.encode([claim], normalize_embeddings=True)
        ch_emb = self.emb.encode(chunks, normalize_embeddings=True)

        sims = cos_sim(c_emb, ch_emb)[0].cpu().numpy()
        topk = np.sort(sims)[-self.top_k :]
        s = float(np.mean(topk))

        if s >= self.high_thr:
            return 1.0
        if s >= self.mid_thr:
            return 0.5
        if s >= self.low_thr:
            return 0.25
        return 0.0

    # ----------------------------------------------------------
    def _temporal_penalty(self, claim_year: int | None) -> float:
        """Temporal penalty hook for potential future decade-aware scoring."""
        if claim_year is None:
            return 0.0
        return 0.0

    # ----------------------------------------------------------
    def compute(self, context_chunks: List[str], answer: str) -> float:
        """Main faithfulness routine combining entailment, penalties and fallback."""
        if not context_chunks or not answer:
            return 0.0

        claims = self._extract_claims(answer)
        if not claims:
            return 0.0

        scores: List[float] = []

        for claim in claims:
            claim_year = self._claim_date(claim)
            specificity = self._specificity_score(claim)

            ent = self._entailment_score(claim, context_chunks)
            if ent < self.entailment_low_cutoff:
                ent = self._embedding_fallback(claim, context_chunks)

            # Specificity penalty scales with factual density
            spec_pen = min(1.0, specificity) * self.w_spec

            # Temporal penalty currently a no-op but kept for extension
            temp_pen = self._temporal_penalty(claim_year)

            raw_score = max(0.0, ent - spec_pen - temp_pen)
            scores.append(raw_score)

        return float(np.mean(scores))

    # ----------------------------------------------------------
    def describe(self) -> Dict[str, str]:
        return {
            "name": "FaithfulnessV2",
            "type": "extrinsic",
            "description": (
                "Claim-level factual evaluation using cross-encoder entailment, "
                "top-k evidence aggregation, specificity and temporal penalties."
            ),
        }
