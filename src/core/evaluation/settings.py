from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SimilarityBands:
    # Shared container for high/mid/low similarity thresholds
    high: float
    mid: float
    low: float


@dataclass(frozen=True)
class FaithfulnessSettings:
    # Configuration for the faithfulness metric
    ent_model: str = "cross-encoder/nli-deberta-base"
    emb_model: str = "multi-qa-mpnet-base-dot-v1"
    entailment_low_cutoff: float = 0.20
    emb_bands: SimilarityBands = field(
        default_factory=lambda: SimilarityBands(high=0.55, mid=0.35, low=0.0)
    )
    top_k: int = 3
    specificity_penalty: float = 0.15
    temporal_penalty: float = 0.15


@dataclass(frozen=True)
class GroundTruthSettings:
    # Configuration for semantic ground truth construction
    similarity_bands: SimilarityBands = field(
        default_factory=lambda: SimilarityBands(high=0.40, mid=0.25, low=0.10)
    )


@dataclass(frozen=True)
class AutoGroundTruthSettings:
    # Configuration for automatic answer-conditioned ground truth
    model_name: str = "multi-qa-mpnet-base-dot-v1"
    similarity_bands: SimilarityBands = field(
        default_factory=lambda: SimilarityBands(high=0.30, mid=0.15, low=0.07)
    )


@dataclass(frozen=True)
class VisualizationSettings:
    # Configuration for evaluation visualizations and band definitions
    faith_band_high: float = 0.70
    faith_band_mid: float = 0.40
    bootstrap_iters: int = 2000
    random_seed: int = 42
    iqr_k: float = 1.5
    z_thresh: float = 3.0


@dataclass(frozen=True)
class EvaluationSettings:
    # Global evaluation configuration passed into orchestrators and metrics
    ndcg_k: int = 10
    faithfulness: FaithfulnessSettings = field(default_factory=FaithfulnessSettings)
    ground_truth: GroundTruthSettings = field(default_factory=GroundTruthSettings)
    auto_gt: AutoGroundTruthSettings = field(default_factory=AutoGroundTruthSettings)
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)


# Default singleton style settings object
DEFAULT_EVAL_SETTINGS = EvaluationSettings()
