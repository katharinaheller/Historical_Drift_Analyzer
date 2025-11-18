# src/core/evaluation/settings.py
from __future__ import annotations
from dataclasses import dataclass, field


# ============================================================
# Generic similarity band container (shared by GT + Faithfulness)
# ============================================================
@dataclass(frozen=True)
class SimilarityBands:
    high: float
    mid: float
    low: float


# ============================================================
# Faithfulness Metric Settings
# ============================================================
@dataclass(frozen=True)
class FaithfulnessSettings:
    # Cross-encoder for textual entailment
    ent_model: str = "cross-encoder/nli-deberta-base"

    # Embedding model for fallback similarity-based evidence scoring
    emb_model: str = "multi-qa-mpnet-base-dot-v1"

    # Minimum entailment probability to accept a hypothesis as supported
    entailment_low_cutoff: float = 0.10

    # Empirically tuned thresholds for embedding similarity evidence
    # Calibrated for MiniLM/MPNet distributions.
    emb_bands: SimilarityBands = field(
        default_factory=lambda: SimilarityBands(
            high=0.50,   # reliable semantic match
            mid=0.32,    # moderate contextual alignment
            low=0.18     # weak but non-random signal
        )
    )

    # Only the strongest evidence is used per claim
    top_k: int = 3

    # Penalties for overly specific or temporally inconsistent statements
    specificity_penalty: float = 0.08
    temporal_penalty: float = 0.08


# ============================================================
# Ground Truth Settings for NDCG (pure retrieval relevance)
# ============================================================
@dataclass(frozen=True)
class GroundTruthSettings:
    # Thresholds calibrated on MPNet cosine similarity distributions.
    #
    # Interpretation:
    #   ≥ high → relevance 3
    #   ≥ mid  → relevance 2
    #   ≥ low  → relevance 1
    #   < low  → relevance 0
    #
    # These values prevent "all chunks = high relevance" and make NDCG meaningful.
    similarity_bands: SimilarityBands = field(
        default_factory=lambda: SimilarityBands(
            high=0.62,   # strong semantic alignment
            mid=0.42,    # moderate alignment
            low=0.24     # weak but non-random signal
        )
    )


# ============================================================
# Auto Ground Truth Settings (answer-conditioned relevance)
# ============================================================
@dataclass(frozen=True)
class AutoGroundTruthSettings:
    # Embedding model for answer-conditioned relevance
    model_name: str = "multi-qa-mpnet-base-dot-v1"

    # Looser thresholds because answer-conditioned similarity is noisier.
    similarity_bands: SimilarityBands = field(
        default_factory=lambda: SimilarityBands(
            high=0.48,   # answer strongly supported
            mid=0.30,    # partial support
            low=0.15     # minimal or weak support
        )
    )


# ============================================================
# Visualization Settings
# ============================================================
@dataclass(frozen=True)
class VisualizationSettings:
    # Faithfulness color thresholds
    faith_band_high: float = 0.50
    faith_band_mid: float = 0.25

    # Sampling and bootstrap configuration
    bootstrap_iters: int = 2000
    random_seed: int = 42

    # Outlier detection
    iqr_k: float = 1.5
    z_thresh: float = 3.0


# ============================================================
# Unified Evaluation Settings Object
# ============================================================
@dataclass(frozen=True)
class EvaluationSettings:
    # Rank cutoff for NDCG@k
    ndcg_k: int = 10

    # Subsystems
    faithfulness: FaithfulnessSettings = field(default_factory=FaithfulnessSettings)
    ground_truth: GroundTruthSettings = field(default_factory=GroundTruthSettings)
    auto_gt: AutoGroundTruthSettings = field(default_factory=AutoGroundTruthSettings)
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)


# Shared default instance
DEFAULT_EVAL_SETTINGS = EvaluationSettings()
