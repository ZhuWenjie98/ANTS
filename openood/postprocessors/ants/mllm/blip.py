"""BLIP and BLIP-2 adapters for ANTS."""

from typing import List, Sequence

import torch
from PIL import Image

from .base import MLLMBackend


class BlipBackend(MLLMBackend):
    """Image-description backend; BLIP models do not run the VSNL branch."""

    supports_similar_labels = False

    def __init__(self, model_type: str) -> None:
        self.model_type = model_type
        self.processor = None
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        if self.model_type == 'BLIP2':
            from transformers import (
                AutoProcessor,
                Blip2ForConditionalGeneration,
            )

            model_id = 'Salesforce/blip2-opt-2.7b'
            self.processor = AutoProcessor.from_pretrained(
                model_id, use_fast=True
            )
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map='auto',
            )
        else:
            from transformers import (
                BlipForConditionalGeneration,
                BlipProcessor,
            )

            model_id = 'Salesforce/blip-image-captioning-base'
            self.processor = BlipProcessor.from_pretrained(model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16
            )
            if torch.cuda.is_available():
                self.model = self.model.to(torch.cuda.current_device())

    def describe_images(
        self, images: Sequence[Image.Image], id_classes: Sequence[str]
    ) -> List[str]:
        del id_classes
        self.load()
        device = next(self.model.parameters()).device
        prompts = None
        if self.model_type == 'BLIP2':
            prompts = [
                'Question: Describe this image in fewer than eight words. '
                'Answer:'
            ] * len(images)

        inputs = self.processor(
            images=list(images),
            text=prompts,
            return_tensors='pt',
            padding=True,
        ).to(device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=30
            )
        outputs = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return [output.split('Answer:')[-1].strip() for output in outputs]
