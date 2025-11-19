# src/core/evaluation/settings.py
from __future__ import annotations
from dataclasses import dataclass, field

# ============================================================
# Generic similarity band container
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
    ent_model: str = "cross-encoder/nli-deberta-base"
    emb_model: str = "multi-qa-mpnet-base-dot-v1"
    entailment_low_cutoff: float = 0.10

    emb_bands: SimilarityBands = field(
        default_factory=lambda: SimilarityBands(
            high=0.80,   # strong semantic agreement
            mid=0.55,    # moderate alignment
            low=0.30     # weak partial evidence
        )
    )

    top_k: int = 3
    specificity_penalty: float = 0.08
    temporal_penalty: float = 0.08

# ============================================================
# Ground Truth Settings for NDCG
# ============================================================
@dataclass(frozen=True)
class GroundTruthSettings:
    similarity_bands: SimilarityBands = field(
        default_factory=lambda: SimilarityBands(
            high=0.80,   # relevance = 3
            mid=0.55,    # relevance = 2
            low=0.30     # relevance = 1
        )
    )

# ============================================================
# Auto Ground Truth (answer-conditioned relevance)
# ============================================================
@dataclass(frozen=True)
class AutoGroundTruthSettings:
    model_name: str = "multi-qa-mpnet-base-dot-v1"

    similarity_bands: SimilarityBands = field(
        default_factory=lambda: SimilarityBands(
            high=0.70,
            mid=0.45,
            low=0.25
        )
    )

# ============================================================
# Visualization Settings
# ============================================================
@dataclass(frozen=True)
class VisualizationSettings:
    faith_band_high: float = 0.50
    faith_band_mid: float = 0.25
    bootstrap_iters: int = 2000
    random_seed: int = 42
    iqr_k: float = 1.5
    z_thresh: float = 3.0

# ============================================================
# Unified Evaluation Settings Object
# ============================================================
@dataclass(frozen=True)
class EvaluationSettings:
    ndcg_k: int = 10
    faithfulness: FaithfulnessSettings = field(default_factory=FaithfulnessSettings)
    ground_truth: GroundTruthSettings = field(default_factory=GroundTruthSettings)
    auto_gt: AutoGroundTruthSettings = field(default_factory=AutoGroundTruthSettings)
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)

DEFAULT_EVAL_SETTINGS = EvaluationSettings()
