"""Ensemble negative-space (ENS) state transitions."""

from typing import Iterable, Optional, Tuple

import numpy as np
import torch


def update_adaptive_threshold(
    confidences: Iterable[torch.Tensor], percentile: float
) -> Tuple[float, float]:
    """Estimate the low-confidence cutoff used to select ENS candidates."""

    values = np.asarray(
        [float(value.detach().cpu()) for value in confidences],
        dtype=np.float32,
    )
    if values.size == 0:
        raise ValueError('cannot update a threshold without confidences')
    if not 0.0 <= percentile <= 1.0:
        raise ValueError('percentile must be between zero and one')

    bins = np.arange(0.0, 1.1, 0.1)
    counts, _ = np.histogram(values, bins)
    upper_interval = float(bins[np.argmax(np.abs(np.diff(counts))) + 1])
    low_confidences = values[values < upper_interval]
    if low_confidences.size == 0:
        low_confidences = values
    threshold = float(np.percentile(low_confidences, percentile * 100.0))
    return threshold, upper_interval


def append_feature_queue(
    queue: Optional[torch.Tensor],
    new_features: Optional[torch.Tensor],
    max_size: int,
) -> Optional[torch.Tensor]:
    """Prepend new ``[labels, dim]`` features and bound the queue size."""

    if new_features is None or new_features.numel() == 0:
        return queue
    if max_size <= 0:
        raise ValueError('max_size must be positive')

    combined = (
        new_features
        if queue is None
        else torch.cat((new_features, queue), dim=0)
    )
    return combined[-max_size:]
