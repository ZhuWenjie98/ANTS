"""Visual-similar negative label (VSNL) cache helpers."""

from collections import Counter
from typing import Dict, Iterable, List, Sequence

from .labels import normalize_labels


def most_frequent_predictions(
    predictions: Iterable[int], limit: int = 40
) -> List[int]:
    """Return the most frequent predicted class indices."""

    return [index for index, _ in Counter(predictions).most_common(limit)]


def update_similar_label_cache(
    cache: Dict[int, List[str]],
    active_predictions: Sequence[int],
    generated_predictions: Sequence[int],
    generated_labels: Sequence[Sequence[str]],
) -> List[str]:
    """Update the VSNL cache and return its flattened unique labels."""

    for prediction, labels in zip(generated_predictions, generated_labels):
        cache[prediction] = normalize_labels(labels)

    active = set(active_predictions)
    for prediction in list(cache):
        if prediction not in active:
            del cache[prediction]

    return normalize_labels(
        label for labels in cache.values() for label in labels
    )
