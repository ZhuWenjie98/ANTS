"""LLaVA adapter for ANTS."""

from typing import List, Sequence

import torch
from PIL import Image

from ..labels import parse_numbered_labels
from .base import MLLMBackend


class LlavaBackend(MLLMBackend):
    """Generate ANTS labels with LLaVA 1.5."""

    model_id = 'llava-hf/llava-1.5-7b-hf'

    def __init__(self) -> None:
        self.processor = None
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map='auto',
        )

    def describe_images(
        self, images: Sequence[Image.Image], id_classes: Sequence[str]
    ) -> List[str]:
        self.load()
        results = []
        for image, id_class in zip(images, id_classes):
            prompt = self._prompt(
                'Describe this image in fewer than eight words. '
                f'Do not include the class name {id_class}.'
            )
            results.append(self._generate(image, prompt, max_new_tokens=30))
        return results

    def suggest_similar_classes(
        self, class_names: Sequence[str]
    ) -> List[List[str]]:
        self.load()
        placeholder = Image.new('RGB', (28, 28))
        results = []
        for class_name in class_names:
            prompt = self._prompt(
                'Give five different class names that share visual features '
                f'with {class_name}, excluding {class_name}. Put each numbered '
                'class on a separate line.'
            )
            response = self._generate(
                placeholder, prompt, max_new_tokens=100
            )
            results.append(parse_numbered_labels(response))
        return results

    def _prompt(self, text: str) -> str:
        conversation = [{
            'role': 'user',
            'content': [
                {'type': 'image'},
                {'type': 'text', 'text': text},
            ],
        }]
        return self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

    def _generate(
        self, image: Image.Image, prompt: str, max_new_tokens: int
    ) -> str:
        device = next(self.model.parameters()).device
        inputs = self.processor(
            images=image, text=prompt, return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        new_tokens = output[0, inputs.input_ids.size(1):]
        return self.processor.decode(
            new_tokens, skip_special_tokens=True
        ).strip()
