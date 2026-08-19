"""Helpers for normalizing free-form labels produced by MLLMs."""

import re
from typing import Iterable, List


_NUMBERED_LABEL = re.compile(r'^\s*\d+[\.\):\-]\s*(.+?)\s*$')


def normalize_labels(labels: Iterable[str]) -> List[str]:
    """Clean labels and remove duplicates while preserving their order."""

    normalized = []
    seen = set()
    for label in labels:
        if label is None:
            continue
        cleaned = ' '.join(str(label).strip().rstrip('.').split())
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key not in seen:
            normalized.append(cleaned)
            seen.add(key)
    return normalized


def parse_numbered_labels(text: str, limit: int = 5) -> List[str]:
    """Parse a numbered MLLM response into at most ``limit`` labels."""

    labels = []
    for line in text.splitlines():
        match = _NUMBERED_LABEL.match(line)
        if match:
            labels.append(match.group(1))
    return normalize_labels(labels)[:limit]
