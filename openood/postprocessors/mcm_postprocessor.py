from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor
import pdb
from tools.classname import imagenet_classes
from PIL import Image
from torchvision import transforms
import requests
import re
import time

class MCMPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(MCMPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tau = self.args.tau
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, model: Any, processor: Any):
        output = net(data)
        score = torch.softmax(output / self.tau, dim=1)
        conf, pred = torch.max(score, dim=1)
        conf = []
        top3_indices = torch.topk(score, k=3, dim=1).indices
        top3_classes = [[imagenet_classes[idx] for idx in indices] for indices in top3_indices.tolist()]


        # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
        # Each value in "content" has to be a list of dicts with types ("text", "image") 
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": "Your task is to classify the image into one classes: {ID classes, none of these classes} and assign confidence to each class."},
                {"type": "text", "text": "You can classify the image into 'none of these classes': if you cannot classify the image into ID classes, if you are not sure whether the image belongs to one of the ID classes, or if you think you need other classes other than the ID classes."},
                {"type": "text", "text": "The following are guidelines for your response. Please respond according to these guidelines. You should provide your confidence for each class between 0.00 and 100.00. The sum of each class confidence should be 100.0. Strictly follow the guidelines above."},
                {"type": "text", "text": "Here is example of your response. Please respond with the following examples format: Prediction: car Confidence: {airplane: 6.34, car: 73.07, bird: 12.72, none of these classes: 1.29}"},
                {"type": "image"},
                ],
            },
        ]

        # image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
        # raw_image = Image.open(requests.get(image_file, stream=True).raw)
        data = data.cpu()
        for i in range(len(top3_classes)):
            # 将值转换为 [0, 255] 范围并转换为 byte 类
            data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())  # 归一化到 [0, 1]
            data[i] = (data[i] * 255).clamp(0, 255).byte()  # 确保在 [0, 255] 范围内
            raw_image = transforms.ToPILImage()(data[i])
            # raw_image.save('/data1/wenjie/projects/OpenOOD-VLM/image.png')
            top3_classes[i] = ", ".join(top3_classes[i])
            conversation[0]["content"][0]["text"] = conversation[0]["content"][0]["text"].replace("ID classes", top3_classes[i])
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
            time2 = time.time()
            output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            time3= time.time()
            answer = processor.decode(output[0][2:], skip_special_tokens=True)
            matches = re.findall(r'none of these classes:\s*([0-9]*\.?[0-9]+)', answer)
            if matches:
                none_confidence = matches[-1]  # 获取最后一个匹配项   
            conf.append(100.0 - float(none_confidence))    
        conf = torch.tensor(conf)

        return pred, conf

    def set_hyperparam(self, hyperparam: list):
        self.tau = hyperparam[0]

    def get_hyperparam(self):
        return self.tau
