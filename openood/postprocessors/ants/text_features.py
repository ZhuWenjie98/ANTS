"""CLIP text-feature encoding for generated negative labels."""

from typing import Iterable, Optional

import torch

from openood.networks.clip import clip


def encode_text_features(
    net,
    labels: Iterable[str],
    prompt_template: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """Encode labels as a feature matrix with shape ``[dim, labels]``."""

    label_list = list(labels)
    if not label_list:
        return None

    device = next(net.model.parameters()).device
    features = []
    with torch.no_grad():
        for label in label_list:
            text = prompt_template.format(label) if prompt_template else label
            tokens = clip.tokenize(text).to(device)
            embeddings = net.model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            feature = embeddings.mean(dim=0)
            features.append(feature / feature.norm())
    return torch.stack(features, dim=1)
