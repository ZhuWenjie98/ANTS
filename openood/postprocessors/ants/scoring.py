"""Pure confidence-scoring functions used by ANTS."""

from typing import Optional, Tuple

import torch


def positive_probability_score(
    logits: torch.Tensor, num_id_classes: int
) -> torch.Tensor:
    """Return the softmax mass assigned to in-distribution classes."""

    _validate_logits(logits, num_id_classes)
    return logits.softmax(dim=-1)[:, :num_id_classes].sum(dim=-1)


def find_activation_threshold(scores: torch.Tensor) -> float:
    """Find the two-cluster threshold used by TANL's online adaptation."""

    values = scores.detach().float().flatten()
    if values.numel() == 0:
        raise ValueError('cannot estimate a threshold without scores')

    best_threshold = float(values.median())
    best_variance = float('inf')
    for threshold in torch.arange(
        0.0, 1.0, 0.01, device=values.device
    ):
        high = values >= threshold
        if not high.any() or high.all():
            continue
        low_values = values[~high]
        high_values = values[high]
        weighted_variance = (
            low_values.numel() * low_values.var(unbiased=False)
            + high_values.numel() * high_values.var(unbiased=False)
        ) / values.numel()
        variance = float(weighted_variance)
        if variance < best_variance:
            best_variance = variance
            best_threshold = float(threshold)
    return best_threshold


def cumulative_negative_score(
    logits: torch.Tensor,
    num_id_classes: int,
    step: int,
) -> torch.Tensor:
    """Average ID mass while incrementally adding ranked negatives."""

    _validate_logits(logits, num_id_classes)
    if step <= 0:
        raise ValueError('activation step must be positive')
    negative_count = logits.size(1) - num_id_classes
    if negative_count == 0:
        raise ValueError('cumulative scoring requires negative labels')

    shifted_logits = logits.float() - logits.float().amax(
        dim=1, keepdim=True
    )
    exponentials = shifted_logits.exp()
    id_sum = exponentials[:, :num_id_classes].sum(dim=1, keepdim=True)
    negative_cumsum = exponentials[:, num_id_classes:].cumsum(dim=1)
    endpoints = torch.arange(
        step - 1,
        negative_count,
        step,
        device=logits.device,
    )
    if endpoints.numel() == 0 or int(endpoints[-1]) != negative_count - 1:
        endpoints = torch.cat((
            endpoints,
            torch.tensor(
                [negative_count - 1],
                device=logits.device,
                dtype=endpoints.dtype,
            ),
        ))
    denominators = id_sum + negative_cumsum[:, endpoints]
    return (id_sum / denominators).mean(dim=1)


def activation_selected_score(
    logits: torch.Tensor,
    num_id_classes: int,
    reference_confidence: torch.Tensor,
    threshold: float,
    gap: float,
    max_negatives: int,
    step: int,
) -> Tuple[Optional[torch.Tensor], int, int, int]:
    """Rank ENS negatives by low-vs-high confidence activation and score."""

    _validate_logits(logits, num_id_classes)
    if reference_confidence.ndim != 1:
        raise ValueError('reference confidence must be one-dimensional')
    if reference_confidence.size(0) != logits.size(0):
        raise ValueError('confidence count must match the batch size')
    if not 0.0 <= threshold <= 1.0:
        raise ValueError('activation threshold must be between zero and one')
    if not 0.0 <= gap <= 1.0:
        raise ValueError('activation gap must be between zero and one')
    if max_negatives <= 0:
        raise ValueError('max negatives must be positive')

    high_cutoff = threshold + gap * (1.0 - threshold)
    low_cutoff = threshold - gap * threshold
    high_mask = reference_confidence > high_cutoff
    low_mask = reference_confidence < low_cutoff
    high_count = int(high_mask.sum())
    low_count = int(low_mask.sum())
    if high_count == 0 or low_count == 0:
        return None, 0, low_count, high_count

    probabilities = logits.float().softmax(dim=1)
    negative_probabilities = probabilities[:, num_id_classes:]
    relevance = (
        negative_probabilities[low_mask].mean(dim=0)
        - negative_probabilities[high_mask].mean(dim=0)
    )
    selected_count = min(max_negatives, relevance.numel())
    selected_indices = relevance.topk(selected_count).indices
    selected_logits = torch.cat((
        logits[:, :num_id_classes],
        logits[:, num_id_classes:][:, selected_indices],
    ), dim=1)
    score = cumulative_negative_score(
        selected_logits, num_id_classes, step
    )
    return score, selected_count, low_count, high_count


def grouped_negative_score(
    logits: torch.Tensor,
    num_id_classes: int,
    group_size: int,
    random_permute: bool = False,
    seed: int = 0,
) -> torch.Tensor:
    """Average ID probability over groups of negative labels.

    A local random generator is used so test-time grouping does not reset the
    application's global PyTorch random state.
    """

    _validate_logits(logits, num_id_classes)
    if group_size <= 0:
        raise ValueError('group_size must be a positive integer')

    positive_logits = logits[:, :num_id_classes]
    negative_logits = logits[:, num_id_classes:]
    negative_count = negative_logits.size(1)
    if negative_count == 0:
        raise ValueError('grouped scoring requires at least one negative label')

    group_count = max(1, negative_count // group_size)
    remainder = negative_count % group_count
    if remainder:
        negative_logits = negative_logits[:, :-remainder]

    if random_permute:
        generator = _make_generator(seed)
        indices = torch.randperm(
            negative_logits.size(1),
            generator=generator,
        ).to(negative_logits.device)
        negative_logits = negative_logits[:, indices]

    grouped_logits = negative_logits.reshape(
        positive_logits.size(0), group_count, -1
    )
    positive_logits = positive_logits.unsqueeze(1).expand(
        -1, group_count, -1
    )
    full_logits = torch.cat((positive_logits, grouped_logits), dim=-1)
    probabilities = full_logits.softmax(dim=-1)
    return probabilities[:, :, :num_id_classes].sum(dim=-1).mean(dim=-1)


def score_with_negative_features(
    image_features: torch.Tensor,
    id_text_features: torch.Tensor,
    negative_features: Optional[torch.Tensor],
    logit_scale: torch.Tensor,
    group_size: int,
    random_permute: bool,
) -> Optional[torch.Tensor]:
    """Score images against ID text and an optional negative feature bank."""

    if negative_features is None or negative_features.numel() == 0:
        return None

    text_features = torch.cat((id_text_features, negative_features), dim=0)
    logits = logit_scale * image_features @ text_features.t()
    if negative_features.size(0) <= group_size:
        return positive_probability_score(logits, id_text_features.size(0))
    return grouped_negative_score(
        logits,
        id_text_features.size(0),
        group_size,
        random_permute,
    )


def _validate_logits(logits: torch.Tensor, num_id_classes: int) -> None:
    if logits.ndim != 2:
        raise ValueError('logits must have shape [batch, classes]')
    if num_id_classes <= 0 or num_id_classes > logits.size(1):
        raise ValueError('num_id_classes is incompatible with logits')


def _make_generator(seed: int) -> torch.Generator:
    # The original implementation generated the permutation on CPU. Keeping
    # that sequence preserves historical scores without mutating global RNG.
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    return generator
