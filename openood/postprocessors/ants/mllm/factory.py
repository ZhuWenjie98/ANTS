"""Factory for lazily imported ANTS MLLM adapters."""

from .base import MLLMBackend


SUPPORTED_MLLM_TYPES = ('QWEN', 'LLAVA', 'BLIP', 'BLIP2')


def create_mllm_backend(model_type: str) -> MLLMBackend:
    """Create an MLLM backend from a case-insensitive config value."""

    normalized_type = str(model_type).upper()
    if normalized_type == 'QWEN':
        from .qwen import QwenBackend

        return QwenBackend()
    if normalized_type == 'LLAVA':
        from .llava import LlavaBackend

        return LlavaBackend()
    if normalized_type in ('BLIP', 'BLIP2'):
        from .blip import BlipBackend

        return BlipBackend(normalized_type)
    choices = ', '.join(SUPPORTED_MLLM_TYPES)
    raise ValueError(
        f'unsupported mllm_model_type {model_type!r}; choose one of {choices}'
    )
