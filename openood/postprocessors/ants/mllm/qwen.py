"""Qwen2-VL adapter for ANTS."""

from typing import List, Sequence

import torch
from PIL import Image

from ..labels import parse_numbered_labels
from .base import MLLMBackend


class QwenBackend(MLLMBackend):
    """Generate image descriptions and similar labels with Qwen2-VL."""

    model_id = 'Qwen/Qwen2-VL-2B-Instruct'

    def __init__(self) -> None:
        self.processor = None
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            attn_implementation='flash_attention_2',
            device_map='auto',
        )

    def describe_images(
        self, images: Sequence[Image.Image], id_classes: Sequence[str]
    ) -> List[str]:
        del id_classes
        self.load()
        device = _model_device(self.model)
        prompt = self.processor.apply_chat_template(
            [{
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {
                        'type': 'text',
                        'text': (
                            'Describe this image in fewer than eight words. '
                            'Answer:'
                        ),
                    },
                ],
            }],
            tokenize=False,
            add_generation_prompt=True,
        )

        descriptions = []
        with torch.no_grad():
            for start in range(0, len(images), 16):
                batch_images = [
                    image.resize((56, 56), Image.BILINEAR)
                    for image in images[start:start + 16]
                ]
                inputs = self.processor(
                    text=[prompt] * len(batch_images),
                    images=batch_images,
                    padding=True,
                    return_tensors='pt',
                ).to(device)
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=10
                )
                trimmed_ids = [
                    output_ids[len(input_ids):]
                    for input_ids, output_ids in zip(
                        inputs.input_ids, generated_ids
                    )
                ]
                descriptions.extend(
                    self.processor.batch_decode(
                        trimmed_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                )
        return descriptions

    def suggest_similar_classes(
        self, class_names: Sequence[str]
    ) -> List[List[str]]:
        self.load()
        device = _model_device(self.model)
        placeholder = Image.new('RGB', (28, 28))
        results = []
        with torch.no_grad():
            for class_name in class_names:
                conversation = [{
                    'role': 'user',
                    'content': [
                        {'type': 'image'},
                        {
                            'type': 'text',
                            'text': (
                                'Suggest five different class names that share '
                                f'visual features with {class_name}. Put each '
                                'numbered class on a separate line.'
                            ),
                        },
                    ],
                }]
                prompt = self.processor.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = self.processor(
                    text=[prompt],
                    images=[placeholder],
                    padding=True,
                    return_tensors='pt',
                ).to(device)
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=50
                )
                trimmed_ids = [
                    output_ids[len(input_ids):]
                    for input_ids, output_ids in zip(
                        inputs.input_ids, generated_ids
                    )
                ]
                output = self.processor.batch_decode(
                    trimmed_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                results.append(parse_numbered_labels(output))
        return results


def _model_device(model) -> torch.device:
    return next(model.parameters()).device
