from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor, LlavaForConditionalGeneration, Blip2ForConditionalGeneration, BlipForConditionalGeneration, Qwen2VLForConditionalGeneration
from transformers import LlavaOnevisionForConditionalGeneration
from transformers import BlipProcessor, BlipForQuestionAnswering, Blip2Processor
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import AutoModelForVision2Seq
# from vllm import LLM

import openood.utils.comm as comm
import pdb
import random

class BasePostprocessor:
    def __init__(self, config):
        self.config = config
        self.config.model_type = 'BLIP'

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        pred_list, conf_list, label_list = [], [], []
        conf_near_list, conf_far_list = [], []

        # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
        # model = Qwen2VLForConditionalGeneration.from_pretrained(
        # "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.float16).to(device)

        # model_id = "llava-hf/llava-1.5-7b-hf"
        # model = LlavaForConditionalGeneration.from_pretrained(
        # model_id,
        # # load_in_4bit=True,
        # torch_dtype=torch.float16,
        # low_cpu_mem_usage=True,
        # ).to(device)

        processor, model = self.get_model(self.config.model_type)
        
        for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            path = batch['path']

            #pred, conf = self.postprocess(net, data)
            pred, conf = self.postprocess(net, data, path, processor, model)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list


    def get_model(self, type):
        device = torch.cuda.current_device()
        if type=='BLIP':
            #processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            #model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            #processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", revision="51572668da0eb669e01a189dc22abe6088589a24", torch_dtype=torch.float16)
            #processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", revision="51572668da0eb669e01a189dc22abe6088589a24", load_in_8bit=True)
            #model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", revision="51572668da0eb669e01a189dc22abe6088589a24", torch_dtype=torch.float16)
            processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
            model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto")
            model = model.to(device)
        elif type=='InstructBLIP':
            processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl", torch_dtype=torch.float16)
            model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl", torch_dtype=torch.float16)
        elif type=='QWEN':
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels, torch_dtype=torch.float16)
            model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.float16)
        elif type=='LLAVA':
            model_id = "llava-hf/llava-1.5-7b-hf"
            #model_id = "bczhou/tiny-llava-v1-hf"

            # model_id = "unsloth/llava-1.5-7b-hf-bnb-4bit"
            model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)

            #model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
            # model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            #     model_id, 
            #     torch_dtype=torch.float16, 
            #     low_cpu_mem_usage=True, 
            # )

            #model = LLM(model=model_id, dtype=torch.bfloat16, max_model_len=4096, trust_remote_code=True, quantization="bitsandbytes", load_format="bitsandbytes", tensor_parallel_size=1)
            #model = LLM(model=model_id, dtype=torch.bfloat16, max_model_len=4096, trust_remote_code=True, tensor_parallel_size=1)
            # only llava have chat template
            #model_id = "llava-hf/llava-1.5-7b-hf"
            processor = AutoProcessor.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit=True, use_flash_attention_2=True)
        elif type=='SmolVLM':
            device = torch.cuda.current_device()
            processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
            model = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceTB/SmolVLM-256M-Instruct",
                torch_dtype=torch.bfloat16,
                _attn_implementation="flash_attention_2",
            ).to(device)
        elif type=='InternVL2':
            processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL2-2B", torch_dtype=torch.float16)
            model = AutoModelForVision2Seq.from_pretrained("OpenGVLab/InternVL2-2B", torch_dtype=torch.float16)
        return processor, model

    

