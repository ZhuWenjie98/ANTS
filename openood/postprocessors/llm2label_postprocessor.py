from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from transformers import AutoProcessor, LlavaForConditionalGeneration, Blip2ForConditionalGeneration, BlipForConditionalGeneration, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers import pipeline

from .base_postprocessor import BasePostprocessor
from tools.classname import imagenet_classes, imagenet_near_classnames, food_names, pet_names
from collections import Counter
from PIL import Image
import requests
import pdb
import time
import re
import os
import random
import math
from transformers.image_utils import load_image
from vllm import SamplingParams
import concurrent.futures
import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp

# imagenet_classes = imagenet_near_classnames


############################ following nnguide!!  besides single points, using its neighbor images. 
class Llm2Labelprocessor(BasePostprocessor):
    def __init__(self, config):
        super(Llm2Labelprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tau = self.args.tau
        self.beta = int(self.args.beta)
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.in_score = self.args.in_score # sum | max
        self.setup_flag = False
        self.proj_flag = False
        self.group_len = int(self.args.group_len)
        self.random_permute = self.args.random_permute
        self.class_num = None
        # self.lock = threading.Lock()
        self.slow_thread = None
        #self.executor = ThreadPoolExecutor(max_workers=1)
        # self.executor = ProcessPoolExecutor(max_workers=1)
        # self.current_process = None
    
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        return

    def reset_group_num(self, group_num):
        self.group_num = group_num      

    def grouping_score(self, output, group_len=1000):
        pos_logit = output[:, :self.class_num] ## B*C
        neg_logit = output[:, self.class_num:] ## B*total_neg_num
        group_num = int(neg_logit.size(1)/group_len)
        drop = neg_logit.size(1) % group_num
        if drop > 0:
            neg_logit = neg_logit[:, :-drop]

        if self.random_permute:
            # print('use random permute')
            SEED=0
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            idx = torch.randperm(neg_logit.shape[1]).to(output.device)
            neg_logit = neg_logit.T ## total_neg_num*B
            # pdb.set_trace()
            neg_logit = neg_logit[idx].T.reshape(pos_logit.shape[0], group_num, -1).contiguous()
        else:
            neg_logit = neg_logit.reshape(pos_logit.shape[0], group_num, -1).contiguous()
        scores = []
        for i in range(group_num):
            full_sim = torch.cat([pos_logit, neg_logit[:, i, :]], dim=-1) 
            full_sim = full_sim.softmax(dim=-1)
            pos_score = full_sim[:, :pos_logit.shape[1]].sum(dim=-1)
            scores.append(pos_score.unsqueeze(-1))
        scores = torch.cat(scores, dim=-1)
        conf_in = scores.mean(dim=-1)
        return conf_in

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, path: Any,  processor: Any, model: Any,):
        net.eval()
        net.batch_idx = net.batch_idx + 1
        class_num = net.n_cls
        self.class_num = class_num
        # pdb.set_trace()
        image_features, text_features, logit_scale = net(data, return_feat=True)
        # image_classifier = self.image_classifier

        output = logit_scale * image_features @ text_features.t() # batch * class.
        # output_eoe = logit_scale * image_features @ net.eoe_text_features # batch * class.

        output_only_in = output[:, :class_num]
        score_only_in = torch.softmax(output_only_in, dim=1)

        # for visualize
        # score = torch.softmax(output, dim=1)
        # top5_first_1000_values, top5_first_1000_indices = torch.topk(score[:, :1000], k=5, dim=1)

        # top5_rest_values, top5_rest_indices = torch.topk(score[:, 1000:], k=5, dim=1)
        # top10_rest_values, top10_rest_indices = torch.topk(score[:, 1000:], k=10, dim=1)

        # top5_first_idx_np = top5_first_1000_indices.cpu().numpy()
        # top5_rest_idx_np = top5_rest_indices.cpu().numpy()
        # top10_rest_idx_np = top10_rest_indices.cpu().numpy()

        # first_1000_elements = np.array(imagenet_classes)[top5_first_idx_np]  # shape [512, 5]
        # rest_elements = np.array(net.far_nts_list)[top10_rest_idx_np]  # shape [512, 5]

        _, pred_in = torch.max(score_only_in, dim=1)
        _, pred_out = torch.max(output[:, class_num:], dim=1)
        # _, pred_eoe = torch.max(output_eoe, dim=1)
        pred_neglabel = [net.neglabel_list[i] for i in pred_out]
        # pred_eoe = [net.eoe_list[i] for i in pred_eoe]

        pred_in_list = [pred.item() for pred in pred_in]
        net.add_pred_list(pred_in_list)
        
        # #if net.far_nts_features == None and net.near_nts_features==None:

        #use MCM
        #conf_in, _ = torch.max(score_only_in, dim=1)
        #use NegLabel
        conf_in = self.grouping_score(output, int(self.group_len))

        net.all_conf_list.extend(conf_in)
        net.ants_conf_list.extend(conf_in)

        #for ENS generation
        bins = np.arange(0, 1.1, 0.1)
        net.all_conf_list = [i.cpu() for i in net.all_conf_list]
        net.ants_conf_list = [i.cpu() for i in net.ants_conf_list]

        net.ada_threshold = 0.4
        threshold = net.ada_threshold
        print("net.ada_threshold", net.ada_threshold)
        for i in range(len(conf_in)):
            if conf_in[i] < threshold:  # 判断分数是否低于 threshold
                net.path_list.append(path[i])  # 存储对应的路径
                net.far_pred_list.append(pred_in[i])
                #net.batch_image_features.append(image_features[i])
            # if conf_in[i] > 0.9:
            #     net.id_noise_path_list.append(path[i])
            #     net.id_noise_pred_list.append(pred_in[i])

        print("len(net.path_list)", len(net.path_list))
        if len(net.path_list) > 200:
            # counts, _ = np.histogram(net.all_conf_list, bins)
            # differences = np.abs(np.diff(counts))
            # max_diff_index = np.argmax(differences)
            # upper_interval = bins[max_diff_index + 1] 
            # net.upper_interval = upper_interval
            # conf_neg = [conf.cpu() for conf in net.all_conf_list if conf < torch.tensor(upper_interval)]
            # #conf_neg = [x for x in net.conf_list if x < upper_interval]
            # percentile = 50
            # #percentile = 30 use to calculate lambda
            # # update net.fg_thresold
            # n_threshold = np.percentile(conf_neg, percentile)
            # net.ada_threshold = n_threshold
            
            self.far_canlabel_generation(net, processor, model)

        #for VSNL generation
        # self.get_high_pred_simlabel(net, processor, model)
        # net.get_near_nts_text_features()

        if net.far_nts_features!=None and net.far_nts_features.shape[1]>self.group_len:
            far_text_features = torch.cat((net.imagenet_features, net.far_nts_features), dim=1)
            far_text_features = far_text_features.transpose(0,1)
            output_far = logit_scale * image_features @ far_text_features.t() # batch * class.
            conf_in_far = self.grouping_score(output_far)
        else:
            conf_in_far = None
             

        if net.near_nts_features!=None and net.near_nts_features.shape[1]>self.group_len:
            n_text_features = net.near_nts_features.shape[1]
            remaining_features = net.negative_features[:, -n_text_features:]
            combine_near_text_features = torch.cat((remaining_features, net.near_nts_features), dim=1)
            near_text_features = torch.cat((net.imagenet_features, combine_near_text_features), dim=1)
            near_text_features = near_text_features.transpose(0,1)
            output_near = logit_scale * image_features @ near_text_features.t() # batch * class.
            conf_in_near = self.grouping_score(output_near)
        else:
            conf_in_near = None 

        # print("conf_in_far", conf_in_far)
  
        conf = []
        # max in prob - max out prob
        if self.in_score == 'oodscore' or self.in_score == 'sum':
            if conf_in_far is None:
                conf_in_far = conf_in

            if conf_in_near is None:
                conf_in_near = conf_in   
            
            
            net.conf_near_list.extend(conf_in_near)    
            net.conf_far_list.extend(conf_in_far)  


            conf_in_his = torch.tensor(net.ants_conf_list)
            conf_in_near_his = torch.tensor(net.conf_near_list)
            conf_in_far_his = torch.tensor(net.conf_far_list)

            # must len(conf_in_his)==len(conf_in_near_his)==len(conf_in_far_his)
            low_threshold = net.ada_threshold
            #indices = torch.nonzero((conf_in_his >= low_threshold) & (conf_in_his <= high_threshold)).squeeze()
            indices = torch.nonzero((conf_in_his >= low_threshold ) & (conf_in_his <= low_threshold + 0.2)).squeeze()

            conf_in_selected = conf_in_his[indices]
            conf_in_near_selected = conf_in_near_his[indices]
            conf_in_far_selected = conf_in_far_his[indices]
            
            #1.using fraction ada_weight
            a = 1 - conf_in_far_selected.mean()
            b = 1 - conf_in_near_selected.mean()
            ada_weight = a/(a + b)
            print("ada_weight", ada_weight)

            # ada weight
            #conf = ada_weight*conf_in_far + (1-ada_weight)*conf_in_near

            #only conf_in_far
            conf = conf_in_far
            
            #2. only use ens to get the adaptive weight
            # a = conf_in_far_selected.mean()
            # b = 1 - a
            # ada_weight = a
            # print("ada_weight", a)
            # conf = ada_weight*conf_in_far + (1-ada_weight)*conf_in_near

            #3. only using vsnl to get the adaptive weight
            # b = conf_in_near_selected.mean()
            # a = 1 - b
            # ada_weight = a
            # print("ada_weight", a)
            # conf = ada_weight*conf_in_far + (1-ada_weight)*conf_in_near

            #4. using log function
            # a = (1 - conf_in_far_selected.mean()) / (2 - conf_in_far_selected.mean() - conf_in_near_selected.mean())
            # b = 1 - a
            # ada_weight = a
            # print("ada_weight", a)
            # conf = ada_weight*conf_in_far + (1-ada_weight)*conf_in_near

            #5. using exponent function
            # a = math.exp(1 - conf_in_far_selected.mean()) / (math.exp(1 - conf_in_near_selected.mean()) + math.exp(1 - conf_in_far_selected.mean()))
            # b = 1 - a
            # ada_weight = a
            # print("ada_weight", a)
            # conf = ada_weight*conf_in_far + (1-ada_weight)*conf_in_near

            conf = torch.tensor(conf)
    
        else:
            raise NotImplementedError
        if torch.isnan(conf).any():
            pdb.set_trace()

        return pred_in, conf

    def set_hyperparam(self, hyperparam: list):
        self.tau = hyperparam[0]

    def get_hyperparam(self):
        return self.tau

    def far_canlabel_generation(self, net, processor, model):
        # indices = list(range(len(net.id_noise_path_list)))
        # # Randomly select 20 unique indices
        # selected_indices = random.sample(indices, 20)

        # # Filter the lists based on the selected indices
        # id_paths = [net.id_noise_path_list[i] for i in selected_indices]
        # id_preds = [net.id_noise_pred_list[i] for i in selected_indices]

        print("开始generation")
        batch_far_labels = self.get_far_canlabel(net, net.path_list, net.far_pred_list, processor, model)  # 执行方法
        #batch_far_labels = self.get_far_canlabel_vllm(net.batch_image_features, model)
        net.far_nts_list.extend(batch_far_labels)
        net.path_list.clear()  # 清空 path_list
        net.far_pred_list.clear()  # 清空 path_list
        #net.batch_image_features.clear()
        batch_text_features = net.get_far_nts_text_features(batch_far_labels)

        if net.far_nts_features == None:
            net.far_nts_features = batch_text_features
        else:
            net.far_nts_features = torch.cat((net.far_nts_features, batch_text_features), dim=1)     
            # queue update
            # n_text_features = batch_text_features.shape[1]
            # remaining_features = net.far_nts_features[:, n_text_features:]
            # net.far_nts_features = torch.cat((remaining_features, batch_text_features), dim=1)
        print("generation完成")

    def get_far_canlabel(self, net, path, far_pred_list, processor, model):
        filter_images = [Image.open(image_path) for image_path in path]
        id_classses = [imagenet_classes[pred] for pred in far_pred_list]
        

        if len(filter_images)!=0:
            time1 = time.time()
            #candidate_label_list = self.get_candidate_label_list_qwen(filter_images, id_classses, processor, model)
            #candidate_label_list = self.get_candidate_label_list_llava(filter_images, id_classses, processor, model)
            candidate_label_list = self.get_candidate_label_list_blip2(filter_images, id_classses, processor, model)
            #candidate_label_list = self.get_candidate_label_list_blip(filter_images, id_classses, processor, model)
            #candidate_label_list = self.get_candidate_label_list_smolvlm(filter_images, id_classses, processor, model)
            #candidate_label_list = self.get_far_canlabel_vllm(filter_images, id_classses, model)
            #candidate_label_list = list(dict.fromkeys(candidate_label_list))
            time2 = time.time()
            print("mllm time", time2-time1)
            # candidate_label_list = list(set(label.rstrip('.') for label in candidate_label_list))
            return candidate_label_list
        else:
            return []

    def get_far_canlabel_vllm(self, images, id_classses, model):
        candidate_label_list = []
        
        for i in range(len(images)):
            image = images[i]
            prompt = "USER: <image>\nProvide a short and concise description of this image with no more than eight words, don't include ###.\nASSISTANT:"
            prompt = prompt.replace("###", id_classses[i])
            llm_input = {"prompt": prompt, "multi_modal_data": {"image": image}}
            outputs = model.generate([llm_input], use_tqdm=False)

            for o in outputs:
                generated_text = o.outputs[0].text
                candidate_label_list.append(generated_text)
        return candidate_label_list       

    def get_high_pred_simlabel(self, net, processor, model):
        sim_id_classes_num = 80
        class_counts = Counter(net.pred)
        all_counts = len(net.pred)

        filtered_counts = class_counts.most_common(sim_id_classes_num)
        high_freq_pred = [pred for pred, _ in filtered_counts]
        
        save_high_freq_pred = [pred for pred in high_freq_pred if pred not in net.high_freq_pred_dict.keys()]
        

        if net.batch_idx%20==0:
            print("save_high_freq_pred",save_high_freq_pred)
            simlabel_list = self.get_simlabel_list_llava(save_high_freq_pred, processor, model)
            #simlabel_list = self.get_simlabel_list_tinyllava(save_high_freq_pred, processor, model)
            #simlabel_list = self.get_simlabel_list_qwen(save_high_freq_pred, processor, model)
            #simlabel_list = self.get_simlabel_list_blip2(save_high_freq_pred, processor, model)
            #simlabel_list = self.get_simlabel_list_smolvlm(save_high_freq_pred, processor, model)
            #simlabel_list = self.get_simlabel_list_instructblip(save_high_freq_pred, processor, model)
            #simlabel_list = self.get_simlabel_list_llava_vllm(save_high_freq_pred, processor, model)
            new_items = []
            keys_to_remove = []
            for i in range(len(save_high_freq_pred)):
                new_items.append((save_high_freq_pred[i], simlabel_list[i]))
            for key, value in new_items:
                net.high_freq_pred_dict[key] = value    
            for pre_pred in net.high_freq_pred_dict.keys():
                if pre_pred not in high_freq_pred:
                    keys_to_remove.append(pre_pred)
            for key in keys_to_remove:
                net.high_freq_pred_dict.pop(key)
            net.near_nts_list = [label for sublist in net.high_freq_pred_dict.values() for label in sublist] 
            net.near_nts_list = list(dict.fromkeys(net.near_nts_list))
            print("net.near_nts_list", net.near_nts_list)          
        
    def get_candidate_label_list_blip2(self, images, id_classses, processor, model):
        device = torch.cuda.current_device()
        
        #model = model.to(device)
        ood_candidate_label_list = []

        with torch.no_grad():
            for i in range(len(images)):
                prompt = "Question: Describe this image less than eight words. Answer:"
                #prompt = "Question: Provide a short and concise description of this image less than eight words. Answer:"
                raw_image = images[i]
                #classname = id_classses[i]
                inputs = processor(raw_image, return_tensors="pt").to(device, torch.float16)
                generated_ids = model.generate(**inputs, max_length=40)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                ood_candidate_label_list.append(generated_text)
                #ood_candidate_label_list = processor.batch_decode(generated_ids, skip_special_tokens=True)
        # acc
        #return
        return ood_candidate_label_list

    def get_candidate_label_list_blip(self, images, id_classses, processor, model):
        device = torch.cuda.current_device()
        ood_candidate_label_list = []
        model = model.to(device)
        with torch.no_grad():
            for i in range(len(images)):
                #prompt = "Question: Briefly describe this image. Answer:"
                raw_image = images[i]
                # raw_image.save('/data1/wenjie/projects/OpenOOD-VLM/image.png')
                inputs = processor(raw_image, return_tensors="pt").to(device, torch.float16)
                generated_ids = model.generate(**inputs, max_length=40)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                ood_candidate_label_list.append(generated_text)  
        # acc
        #return
        return ood_candidate_label_list
    
    def get_candidate_label_list_smolvlm(self, images, id_classses, processor, model):
        device = torch.cuda.current_device()
        model = model.to(device)
        candidate_list = []
        ood_candidate_label_list = []

        with torch.no_grad():
            for i in range(len(images)):
                raw_image = images[i]
                #raw_image = raw_image.convert('RGB')
                classname = id_classses[i]
                conversation = [
                        {
                            "role": "user",
                            "content": [
                            {"type": "image"},
                            {"type": "text", "text": "Describe this image less than eight words."},
                            ],
                        }
                    ]

                #conversation[0]["content"][1]["text"] = conversation[0]["content"][1]["text"].replace("###", classname)
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(images=raw_image, text=prompt, size={"longest_edge": 5*384}, return_tensors='pt').to(device, torch.float16)
                generated_ids = model.generate(**inputs, max_new_tokens=200)
                generated_texts = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
                generated_texts = generated_texts[0]
                ood_candidate_label = generated_texts.split('Assistant: ')[1].strip()
                ood_candidate_label_list.append(ood_candidate_label)    
                
        return ood_candidate_label_list

    def get_candidate_label_list_qwen(self, images, id_classses, processor, model):
        device = torch.cuda.current_device()
        # conversation = "Give me one fine-grained image label to the image, no more than five words."
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        ood_candidate_label_list = []
        with torch.no_grad():
            for i in range(len(images)):
                raw_image = images[i]
                classname = id_classses[i]
                conversation = [
                    {
                        "role": "user",
                        "content": [
                        {"type": "image",},
                        {"type": "text", "text": "Briefly describe this image, don’t include the ###. Answer:"},
                        ],
                    }
                ]
                conversation[0]["content"][1]["text"] = conversation[0]["content"][1]["text"].replace("###", classname)
                text = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                text=[text], images=[raw_image], padding=True, return_tensors="pt"
                )
                # image_inputs, video_inputs = process_vision_info(conversation)
                inputs = inputs.to(device, torch.float16)

                # Inference: Generation of the output
                generated_ids = model.generate(**inputs, max_new_tokens=15)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                ood_candidate_label_list.extend(output_text)

        # acc
        #return
        return ood_candidate_label_list    


    def get_candidate_label_list_llava(self, images, id_classses, processor, model):
        device = torch.cuda.current_device()
        model = model.to(device)
        candidate_list = []
        ood_candidate_label_list = []

        with torch.no_grad():
            for i in range(len(images)):
                raw_image = images[i]
                classname = id_classses[i]
                conversation = [
                        {
                            "role": "user",
                            "content": [
                            {"type": "image",},
                            {"type": "text", "text": "Provide a short and concise description of this image less than eight words, don't include ###."},
                            ],
                        }
                    ]

                conversation[0]["content"][1]["text"] = conversation[0]["content"][1]["text"].replace("###", classname)
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device, torch.float16)
                output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
                assistant_response = processor.decode(output[0][2:], skip_special_tokens=True)
                #ood_candidate_label = assistant_response.split('ASSISTANT: ')[1].strip()
                ood_candidate_label = assistant_response.split('assistant\n')[1].strip()
                ood_candidate_label_list.append(ood_candidate_label)    
        # acc
        #return
        return ood_candidate_label_list  
      


    def get_simlabel_list_llava(self, high_freq_pred, processor, model):
        candidate_list = []
        device = torch.cuda.current_device()
        model = model.to(device)
        #classnames = [pet_names[i] for i in high_freq_pred]
        classnames = [imagenet_classes[i] for i in high_freq_pred]
        with torch.no_grad():
            image_file = "/data1/wenjie/projects/ANTS/blank.jpeg"
            raw_image = Image.open(image_file)

            for classname in tqdm(classnames):
                conversation = [
                    {
                    "role": "user",
                    "content": [
                        {"type":"image",},
                        {"type": "text", "text": "Give me five different class names that share similar visual features with ###, don't contain ###."},
                        ],
                    },
                ]
                conversation[0]["content"][1]["text"] = conversation[0]["content"][1]["text"].replace("###", classname)
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device, torch.float16)
                output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
                answer = processor.decode(output[0][2:], skip_special_tokens=True)
                assistant_response = answer.split('ASSISTANT: ')[1].strip()
                # 按行分割字符串
                lines = assistant_response.split('\n')
                # 提取类名
                simclass_list = []
                for line in lines:
                    # 检查行是否以数字开头并包含类名
                    if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                        # 提取类名并去除前面的编号
                        class_name = line.split('. ')[1].strip() if len(line.split('. ')) > 1 else None
                        simclass_list.append(class_name)      
                candidate_list.append(simclass_list)   
        return candidate_list    
    
    

    def get_simlabel_list_qwen(self, high_freq_pred, processor, model):
        candidate_list = []
        device = torch.cuda.current_device()
        model = model.to(device)
        # conversation = "Give me one fine-grained image label to the image, no more than five words."
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        classnames = [imagenet_classes[i] for i in high_freq_pred]
        image_file = "/data1/wenjie/projects/ANTS/blank.jpeg"
        raw_image = Image.open(image_file)
        with torch.no_grad():
            for i in range(len(classnames)):
                conversation = [
                    {
                        "role": "user",
                        "content": [
                        {"type": "image",},
                        {"type": "text", "text": "please suggest five different class names that share visual features with ###, don't include the ###. "},
                        ],
                    }
                ]
                conversation[0]["content"][1]["text"] = conversation[0]["content"][1]["text"].replace("###", classnames[i])
                text = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                text=[text], images=[raw_image], padding=True, return_tensors="pt"
                )
                # image_inputs, video_inputs = process_vision_info(conversation)
                inputs = inputs.to(device)

                # Inference: Generation of the output
                generated_ids = model.generate(**inputs, max_new_tokens=100)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                lines = output_text[0].strip().split('\n')
                simclass_list = [line.split('. ', 1)[1] for line in lines]      
                candidate_list.append(simclass_list)  
                    
        return candidate_list 
    

    def get_simlabel_list_smolvlm(self, high_freq_pred, processor, model):
        candidate_list = []
        device = torch.cuda.current_device()
        #classnames = [pet_names[i] for i in high_freq_pred]
        classnames = [imagenet_classes[i] for i in high_freq_pred]
        with torch.no_grad():
            image_file = "/data1/wenjie/projects/ANTS/blank.jpeg"
            raw_image = Image.open(image_file)

            for classname in tqdm(classnames):
                conversation = [
                    {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please suggest me five different class names that share similar visual features with ###."},
                        ],
                    },
                ]
                conversation[0]["content"][0]["text"] = conversation[0]["content"][0]["text"].replace("###", classname)
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(text=prompt, images=None, return_tensors="pt").to(device, torch.float16)
                generated_ids = model.generate(**inputs, max_new_tokens=500)
                generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

                # 提取类名
                simclass_list = []
                for line in generated_texts:
                    # 检查行是否以数字开头并包含类名
                    if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                        # 提取类名并去除前面的编号
                        class_name = line.split('. ')[1].strip()
                        simclass_list.append(class_name)      
                candidate_list.append(simclass_list)   
        return candidate_list  
    
    def get_simlabel_list_instructblip(self, high_freq_pred, processor, model):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        device = torch.cuda.current_device()
        candidate_list = []
        classnames = [imagenet_classes[i] for i in high_freq_pred]

        model = model.to(device)
        with torch.no_grad():
            candidate_list = []
            for classname in tqdm(classnames):
                # context = [
                # ("Give me five different class names share visual features with donut", "1.bagle 2.pastry 3.bread 4.cake 5.cookie"),
                # ]
                # template = "Question: {} Answer: {}."

                conversation = " Question: Give me five different class names share visual features with ###. Answer:"
                conversation = conversation.replace("###", classname)
                #prompt = " ".join([template.format(context[i][0], context[i][1]) for i in range(len(context))]) + conversation
                inputs = processor(images=image, text=conversation, return_tensors="pt").to("cuda")
                outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
                )
                generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                simclass_list = []
                candidate_list.extend(simclass_list)   
        return candidate_list


    def get_simlabel_list_llava_vllm(self, high_freq_pred, processor, model):
        candidate_list = []
        device = torch.cuda.current_device()
        #classnames = [pet_names[i] for i in high_freq_pred]
        classnames = [imagenet_classes[i] for i in high_freq_pred]
        sampling_params = SamplingParams(max_tokens=200)
        with torch.no_grad():
            image_file = "/data1/wenjie/projects/ANTS/blank.jpeg"
            raw_image = Image.open(image_file)

            for classname in tqdm(classnames):
                conversation = [
                    {
                    "role": "user",
                    "content": [
                        {"type":"image",},
                        {"type": "text", "text": "Give me five different class names that share similar visual features with ###, don't contain ###."},
                        ],
                    },
                ]
                conversation[0]["content"][1]["text"] = conversation[0]["content"][1]["text"].replace("###", classname)
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                # prompt = "USER: <image>\nGive me five distinct class names that share similar visual features with ###, don't contain ###. Use this format:1.  2.  3.  4.  5. \nASSISTANT:"
                # prompt = prompt.replace("###", classname)
                llm_input = {"prompt": prompt, "multi_modal_data": {"image": raw_image}}
                outputs = model.generate([llm_input], sampling_params=sampling_params, use_tqdm=False)

                for o in outputs:
                    generated_text = o.outputs[0].text
                    matches = re.findall(r'\d+\.\s*(.*?)(?=\s*\d+\.|$)', generated_text.strip())
                    candidate_list.append(match.strip() for match in matches)
        return candidate_list      
    
    


      
        
                
 