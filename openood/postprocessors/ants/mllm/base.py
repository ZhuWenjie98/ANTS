"""Common interface for ANTS multimodal language-model backends."""

from abc import ABC, abstractmethod
from typing import List, Sequence

from PIL import Image


class MLLMBackend(ABC):
    """Backend contract consumed by the ANTS orchestration layer."""

    supports_similar_labels = True

    @abstractmethod
    def load(self) -> None:
        """Load model resources once."""

    @abstractmethod
    def describe_images(
        self, images: Sequence[Image.Image], id_classes: Sequence[str]
    ) -> List[str]:
        """Generate concise candidate labels for images."""

    def suggest_similar_classes(
        self, class_names: Sequence[str]
    ) -> List[List[str]]:
        """Generate visually similar class names."""

        if class_names:
            raise NotImplementedError(
                'this MLLM backend does not support similar-label generation'
            )
        return []
