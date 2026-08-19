"""Runtime state owned by a single ANTS evaluation session."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class ANTSRuntimeState:
    """Mutable test-time memory separated from algorithm configuration."""

    batch_index: int = 0
    ensemble_index: int = 0
    adaptive_threshold: float = 0.8
    upper_interval: Optional[float] = None

    all_confidences: List[torch.Tensor] = field(default_factory=list)
    activation_confidences: List[torch.Tensor] = field(default_factory=list)
    far_confidences: List[torch.Tensor] = field(default_factory=list)
    near_confidences: List[torch.Tensor] = field(default_factory=list)

    candidate_paths: List[str] = field(default_factory=list)
    candidate_predictions: List[int] = field(default_factory=list)
    predictions: List[int] = field(default_factory=list)

    far_negative_features: Optional[torch.Tensor] = None
    near_negative_features: Optional[torch.Tensor] = None
    near_negative_labels: List[str] = field(default_factory=list)
    similar_label_cache: Dict[int, List[str]] = field(default_factory=dict)

    def reset(
        self, initial_far_features: Optional[torch.Tensor] = None
    ) -> None:
        """Clear all test-time memory for a new OOD dataset."""

        self.batch_index = 0
        self.ensemble_index = 0
        self.adaptive_threshold = 0.8
        self.upper_interval = None
        self.all_confidences.clear()
        self.activation_confidences.clear()
        self.far_confidences.clear()
        self.near_confidences.clear()
        self.candidate_paths.clear()
        self.candidate_predictions.clear()
        self.predictions.clear()
        self.far_negative_features = initial_far_features
        self.near_negative_features = None
        self.near_negative_labels.clear()
        self.similar_label_cache.clear()
