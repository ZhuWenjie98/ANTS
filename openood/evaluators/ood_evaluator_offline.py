import csv
import os
from typing import Dict, List
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoProcessor, LlavaForConditionalGeneration, Blip2ForConditionalGeneration, BlipForConditionalGeneration, BlipTextModel, Qwen2VLForConditionalGeneration
from transformers import BlipProcessor, BlipForQuestionAnswering
from transformers import pipeline
import clip
from PIL import Image

from openood.postprocessors import BasePostprocessor
from openood.utils import Config
from tools.classname import imagenet_classes, sun_llm_label_list, places_llm_label_list, dtd_llm_label_list, inaturalist_llm_label_list
from collections import Counter

from .base_evaluator import BaseEvaluator
from .metrics import compute_all_metrics
from .neglabel_metrics import evaluate_all

import pdb
from tqdm import tqdm
import re
import time
from PIL import Image

def get_candidate_label_list_blip2(images):
        device = torch.cuda.current_device()
        
        processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", revision="51572668da0eb669e01a189dc22abe6088589a24")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", revision="51572668da0eb669e01a189dc22abe6088589a24", torch_dtype=torch.float16).to(device)
        ood_candidate_label_list = []

        with torch.no_grad():
            for image in tqdm(images):
                prompt = "Question: Briefly describe this image with no more than eight words. Answer:"
                raw_image = image
                # raw_image.save('/data1/wenjie/projects/OpenOOD-VLM/image.png')
                inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device, torch.float16)
                generated_ids = model.generate(**inputs, max_length=40)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                ood_candidate_label_list.append(generated_text)  
        # acc
        #return
        return ood_candidate_label_list

def get_ID_aware_candidate_label_list(model, loader, conf_indices, pred_list):
        model_id = "llava-hf/llava-1.5-7b-hf"
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        ).to(0)
        processor = AutoProcessor.from_pretrained(model_id)
        tqdm_object = tqdm(loader, total=len(loader))
        candidate_list = []

        train_dataiter = iter(loader)
        #far ood
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": "Give me one fine-grained image label to the image, no more than five words, don't provide me ### and its superclass or synonyms."},
                {"type": "image"},
                ],
            },
        ]

        ood_candidate_label_list = []
        time_list = []
        for batch_idx, batch in enumerate(tqdm_object):  
            # time_list.append(time.time())

            bz = batch['data'].shape[0]
            batch_list = batch['index']
            existing_elements = [elem for elem in conf_indices if elem in batch_list]
            batch_indices_list = [elem % bz for elem in existing_elements]
            #if batch_idx in conf_indices:
            if existing_elements!=[]:
                with torch.no_grad():
                    for i in range(len(existing_elements)):
                        pred_indice = np.where(conf_indices == existing_elements[i])[0][0]
                        classname = imagenet_classes[int(pred_list[pred_indice])]
                        #替换 conversation 中的 ### 为 classname
                        conversation[0]["content"][0]["text"] = conversation[0]["content"][0]["text"].replace("###", classname)
                        images = batch['data'][batch_indices_list[i]].cuda()
                        images = (images - images.min()) / (images.max() - images.min())  # 归一化到 [0, 1]
                        images = (images * 255).clamp(0, 255).byte()  # 确保在 [0, 255] 范围内
                        images = images.squeeze()
                        raw_image = transforms.ToPILImage()(images)
                        #raw_image.save('/data1/wenjie/projects/OpenOOD-VLM/image.png')
                        # raw_image.save('/data1/wenjie/projects/OpenOOD-VLM/image.png')
                        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                        inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
                        output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
                        answer = processor.decode(output[0][2:], skip_special_tokens=True)
                        assistant_response = answer.split('ASSISTANT: ')[1].strip()
                        ood_candidate_label_list.append(assistant_response)          
        # acc
        #return
        return ood_candidate_label_list

def clipscope_get_candidate_label_list(model, loader, pred_list, high_freq_pred):
        pred_list = list(map(int, pred_list))
        high_freq_pred = list(map(int, high_freq_pred))
        model_id = "llava-hf/llava-1.5-7b-hf"
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        ).to(0)
        processor = AutoProcessor.from_pretrained(model_id)
        tqdm_object = tqdm(loader, total=len(loader))
        candidate_list = []

        train_dataiter = iter(loader)

        #near ood
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": "Give me one fine-grained image label to the image, no more than three words, don't provide me ### and its superclass or synonyms."},
                {"type": "image"},
                ],
            },
        ]
        ood_candidate_label_list = []
        for batch_idx, batch in enumerate(tqdm_object):   
            bz = batch['data'].shape[0]
            start = batch_idx * bz
            end = start + bz
            batch_list = list(range(start, end))
            batch_pred = [pred_list[i] for i in batch_list]
            # elements_indices = [elem for elem in batch_pred if elem in high_freq_pred]
            # batch_indices_list = [elem % bz for elem in existing_elements]
            #if batch_idx in conf_indices:
            existing_elements = [elem for elem in batch_pred if elem in high_freq_pred]
            batch_indices_list = [elem % bz for elem in existing_elements]
            if existing_elements!=[]:
                with torch.no_grad():
                    for i in range(len(existing_elements)):
                        # index = np.where(conf_indices==existing_elements[i])[0][0]
                        # pred = int(low_conf_pred[index])
                        # classname = imagenet_classes[pred]
                        # 替换 conversation 中的 ### 为 classname
                        # conversation[0]["content"][0]["text"] = conversation[0]["content"][0]["text"].replace("###", classname)
                        classname = imagenet_classes[existing_elements[i]]
                        conversation[0]["content"][0]["text"] = conversation[0]["content"][0]["text"].replace("###", classname)
                        images = batch['data'][i].cuda()
                        images = (images - images.min()) / (images.max() - images.min())  # 归一化到 [0, 1]
                        images = (images * 255).clamp(0, 255).byte()  # 确保在 [0, 255] 范围内
                        images = images.squeeze()
                        raw_image = transforms.ToPILImage()(images)
                        #raw_image.save('/data1/wenjie/projects/OpenOOD-VLM/image.png')
                        # raw_image.save('/data1/wenjie/projects/OpenOOD-VLM/image.png')
                        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                        inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
                        output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
                        answer = processor.decode(output[0][2:], skip_special_tokens=True)
                        assistant_response = answer.split('ASSISTANT: ')[1].strip()
                        ood_candidate_label_list.append(assistant_response) 
        # acc
        return ood_candidate_label_list

def get_simlabel_list(high_freq_pred):
    device = torch.cuda.current_device()
    high_freq_pred = list(map(int, high_freq_pred))
    candidate_list = []
    classnames = [imagenet_classes[i] for i in high_freq_pred]
    with torch.no_grad():
        model_id = "llava-hf/llava-1.5-7b-hf"
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        candidate_list = []

        #pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-V3", max_new_tokens=100, device_map="auto")
        for classname in tqdm(classnames):
            #near ood
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type":"image",},
                        {"type": "text", "text": "Give me five different class names share visual features with ### and have same superclass with ###, don't give me description, reasons, its superclass name and don't contain ###."},
                        ],
                    },
            ]
            image_file = "/data1/wenjie/projects/OpenOOD-VLM/blank.jpeg"
            raw_image = Image.open(image_file)
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
                    class_name = line.split('. ')[1].strip()
                    simclass_list.append(class_name)      
            candidate_list.extend(simclass_list)
    return candidate_list

# def get_simlabel_list(high_freq_pred):
#     high_freq_pred = list(map(int, high_freq_pred))
#     candidate_list = []
#     classnames = [imagenet_classes[i] for i in high_freq_pred]
#     with torch.no_grad():
#         pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-V3", max_new_tokens=100, device_map="auto")
#         for classname in tqdm(classnames):
#             messages = [
#             {"role": "user", "content": "Give me 5 class names that looks like ### and has the same superclass with ###, don't give me description and reasons"},
#             ]
#             messages[0]["content"] = messages[0]["content"].replace("###", classname)
#             response = pipe(messages)
#             assistant_content = response[0]['generated_text'][1]['content']
#             # 按行分割字符串
#             lines = assistant_content.split('\n')
#             # 提取类名
#             simclass_list = []
#             for line in lines:
#                 # 检查行是否以数字开头并包含类名
#                 if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
#                     # 提取类名并去除前面的编号
#                     class_name = line.split('. ')[1].strip()
#                     simclass_list.append(class_name)      
#             candidate_list.extend(simclass_list)
#     return candidate_list
           

class OODOfflineEvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        """OOD Evaluator.

        Args:
            config (Config): Config file from
        """
        super(OODOfflineEvaluator, self).__init__(config)
        self.id_pred = None
        self.id_conf = None
        self.id_gt = None

    def eval_ood(self,
                 net: nn.Module,
                 id_data_loaders: Dict[str, DataLoader],
                 ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                 postprocessor: BasePostprocessor,
                 fsood: bool = False,
                 llm2label: bool = False):
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()
        assert 'test' in id_data_loaders, \
            'id_data_loaders should have the key: test!'
        dataset_name = self.config.dataset.name
        
        if self.config.postprocessor.APS_mode:
            assert 'val' in id_data_loaders
            assert 'val' in ood_data_loaders
            self.hyperparam_search(net, id_data_loaders['val'],
                                   ood_data_loaders['val'], postprocessor)

        print(f'Performing inference on {dataset_name} dataset...', flush=True)
        id_pred, id_conf, id_gt = postprocessor.inference(
            net, id_data_loaders['test'])
        if self.config.recorder.save_scores:
            self._save_scores(id_pred, id_conf, id_gt, dataset_name)

        if fsood:
            # load csid data and compute confidence
            for dataset_name, csid_dl in ood_data_loaders['csid'].items():
                print(f'Performing inference on {dataset_name} dataset...',
                      flush=True)
                csid_pred, csid_conf, csid_gt = postprocessor.inference(
                    net, csid_dl)
                if self.config.recorder.save_scores:
                    self._save_scores(csid_pred, csid_conf, csid_gt,
                                      dataset_name)
                id_pred = np.concatenate([id_pred, csid_pred])
                id_conf = np.concatenate([id_conf, csid_conf])
                id_gt = np.concatenate([id_gt, csid_gt])

        # load nearood data and compute ood metrics
        print(u'\u2500' * 70, flush=True)
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='nearood',
                       llm2label=llm2label)

        # load farood data and compute ood metrics
        print(u'\u2500' * 70, flush=True)
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='farood',
                       llm2label=llm2label)

    def _eval_ood(self,
                  net: nn.Module,
                  id_list: List[np.ndarray],
                  id_data_loaders:  Dict[str, Dict[str, DataLoader]],
                  ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                  postprocessor: BasePostprocessor,
                  ood_split: str = 'nearood',
                  llm2label: bool = False):
        print(f'Processing {ood_split}...', flush=True)
        [id_pred, id_conf, id_gt] = id_list
        metrics_list = []
        # pdb.set_trace()
        for dataset_name, ood_dl in ood_data_loaders[ood_split].items():
            print(f'Performing inference on {dataset_name} dataset...',
                  flush=True)
            # ood_pred, ood_conf, ood_gt = postprocessor.inference(net, ood_dl)
            # # 保存为文本文件
            # combined_array = np.column_stack((ood_pred, ood_conf, ood_gt))
            # np.savetxt('/data1/wenjie/projects/OpenOOD-VLM/scorefiles/' + dataset_name +'_conf.txt', combined_array, fmt='%.5f', header='ood_pred, ood_conf, ood_gt', delimiter=',')

            # if llm2label:
            # print("load ood result")
            # loaded_data = np.loadtxt('/data1/wenjie/projects/OpenOOD-VLM/scorefiles/'+dataset_name+ '_conf.txt', delimiter=',', skiprows=1)
            # # 分割成三个数组
            # ood_pred = loaded_data[:, 0]
            # ood_conf = loaded_data[:, 1]
            # ood_gt = loaded_data[:, 2]

            # # pdb.set_trace()
            # ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            # if self.config.recorder.save_scores:
            #     self._save_scores(ood_pred, ood_conf, ood_gt, dataset_name)

            # pred = np.concatenate([id_pred, ood_pred])
            # conf = np.concatenate([id_conf, ood_conf])
            
            # topk=6000
            # print("ens generation")
            # negative_labels = self.get_noisy_ood_canlabel(dataset_name, topk, id_conf, ood_conf, id_pred, ood_pred, net, id_data_loaders, ood_dl)
            # # negative_labels = self.get_high_pred_simlabel(id_pred, ood_pred, dataset_name)
            # # pdb.set_trace()

            # far_nts_text_features = net.get_far_nts_text_features(negative_labels)
            # net.negative_features = far_nts_text_features

            print("infer with ens")
            # id_pred, id_conf, id_gt = postprocessor.inference(
            #     net, id_data_loaders['test'])
            ood_pred, ood_conf, ood_gt = postprocessor.inference(net, ood_dl)
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])        
            label = np.concatenate([id_gt, ood_gt])
            print(f'Computing metrics on {dataset_name} dataset...')

            #ood_metrics = compute_all_metrics(conf, label, pred)
            ood_metrics = evaluate_all(id_conf, ood_conf, label, pred)
            if self.config.recorder.save_csv:
                self._save_csv(ood_metrics, dataset_name=dataset_name)
            metrics_list.append(ood_metrics)

        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0)
        if self.config.recorder.save_csv:
            self._save_csv(metrics_mean, dataset_name=ood_split)
   

    def eval_ood_val(self, net: nn.Module, id_data_loaders: Dict[str,
                                                                 DataLoader],
                     ood_data_loaders: Dict[str, DataLoader],
                     postprocessor: BasePostprocessor):
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()
        assert 'val' in id_data_loaders
        assert 'val' in ood_data_loaders
        if self.config.postprocessor.APS_mode:
            val_auroc = self.hyperparam_search(net, id_data_loaders['val'],
                                               ood_data_loaders['val'],
                                               postprocessor)
        else:
            id_pred, id_conf, id_gt = postprocessor.inference(
                net, id_data_loaders['val'])
            ood_pred, ood_conf, ood_gt = postprocessor.inference(
                net, ood_data_loaders['val'])
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            val_auroc = ood_metrics[1]
        return {'auroc': 100 * val_auroc}

    def eval_ood_val_accname(self, net: nn.Module, id_data_loaders: Dict[str,
                                                                 DataLoader],
                     ood_data_loaders: Dict[str, DataLoader],
                     postprocessor: BasePostprocessor, epoch_idx: int = -1):
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()
        assert 'val' in id_data_loaders
        assert 'val' in ood_data_loaders
        if self.config.postprocessor.APS_mode:
            val_auroc = self.hyperparam_search(net, id_data_loaders['val'],
                                               ood_data_loaders['val'],
                                               postprocessor)
        else:
            id_pred, id_conf, id_gt = postprocessor.inference(
                net, id_data_loaders['val'])
            ood_pred, ood_conf, ood_gt = postprocessor.inference(
                net, ood_data_loaders['val'])
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            val_auroc = ood_metrics[1]
        
        metrics = {}
        metrics['acc'] = val_auroc
        metrics['epoch_idx'] = epoch_idx
        return metrics

    def _save_csv(self, metrics, dataset_name):
        [fpr, auroc, aupr_in, aupr_out, accuracy] = metrics

        write_content = {
            'dataset': dataset_name,
            'FPR@95': '{:.2f}'.format(100 * fpr),
            'AUROC': '{:.2f}'.format(100 * auroc),
            'AUPR_IN': '{:.2f}'.format(100 * aupr_in),
            'AUPR_OUT': '{:.2f}'.format(100 * aupr_out),
            'ACC': '{:.2f}'.format(100 * accuracy)
        }

        fieldnames = list(write_content.keys())

        # print ood metric results
        print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
              end=' ',
              flush=True)
        print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
            100 * aupr_in, 100 * aupr_out),
              flush=True)
        print('ACC: {:.2f}'.format(accuracy * 100), flush=True)
        print(u'\u2500' * 70, flush=True)

        csv_path = os.path.join(self.config.output_dir, 'ood.csv')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)

    def _save_scores(self, pred, conf, gt, save_name):
        save_dir = os.path.join(self.config.output_dir, 'scores')
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, save_name),
                 pred=pred,
                 conf=conf,
                 label=gt)

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1,
                 fsood: bool = False,
                 csid_data_loaders: DataLoader = None):
        """Returns the accuracy score of the labels and predictions.

        :return: float
        """
        if type(net) is dict:
            net['backbone'].eval()
        else:
            net.eval()
        self.id_pred, self.id_conf, self.id_gt = postprocessor.inference(
            net, data_loader)

        if fsood:
            assert csid_data_loaders is not None
            for dataset_name, csid_dl in csid_data_loaders.items():
                csid_pred, csid_conf, csid_gt = postprocessor.inference(
                    net, csid_dl)
                self.id_pred = np.concatenate([self.id_pred, csid_pred])
                self.id_conf = np.concatenate([self.id_conf, csid_conf])
                self.id_gt = np.concatenate([self.id_gt, csid_gt])

        metrics = {}
        metrics['acc'] = sum(self.id_pred == self.id_gt) / len(self.id_pred)
        metrics['epoch_idx'] = epoch_idx
        return metrics

    def report(self, test_metrics):
        print('Completed!', flush=True)

    def hyperparam_search(
        self,
        net: nn.Module,
        id_data_loader,
        ood_data_loader,
        postprocessor: BasePostprocessor,
    ):
        print('Starting automatic parameter search...')
        aps_dict = {}
        max_auroc = 0
        hyperparam_names = []
        hyperparam_list = []
        count = 0
        for name in postprocessor.args_dict.keys():
            hyperparam_names.append(name)
            count += 1
        for name in hyperparam_names:
            hyperparam_list.append(postprocessor.args_dict[name])
        hyperparam_combination = self.recursive_generator(
            hyperparam_list, count)
        for hyperparam in hyperparam_combination:
            postprocessor.set_hyperparam(hyperparam)
            id_pred, id_conf, id_gt = postprocessor.inference(
                net, id_data_loader)
            ood_pred, ood_conf, ood_gt = postprocessor.inference(
                net, ood_data_loader)
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            index = hyperparam_combination.index(hyperparam)
            aps_dict[index] = ood_metrics[1]
            print('Hyperparam:{}, auroc:{}'.format(hyperparam,
                                                   aps_dict[index]))
            if ood_metrics[1] > max_auroc:
                max_auroc = ood_metrics[1]
        for key in aps_dict.keys():
            if aps_dict[key] == max_auroc:
                postprocessor.set_hyperparam(hyperparam_combination[key])
        print('Final hyperparam: {}'.format(postprocessor.get_hyperparam()))
        return max_auroc

    def recursive_generator(self, list, n):
        if n == 1:
            results = []
            for x in list[0]:
                k = []
                k.append(x)
                results.append(k)
            return results
        else:
            results = []
            temp = self.recursive_generator(list, n - 1)
            for x in list[n - 1]:
                for y in temp:
                    k = y.copy()
                    k.append(x)
                    results.append(k)
            return results


    def get_high_pred_simlabel(self, id_pred, ood_pred, dataset_name):
        pred = np.concatenate([id_pred, ood_pred])
        id_class_counts = Counter(id_pred)
        ood_class_counts = Counter(ood_pred)
        class_counts = Counter(pred)
        # # 按照频率从高到低排序
        sorted_counts = class_counts.most_common(80)
        high_freq_pred = [pred for pred, _ in sorted_counts] 

        simlabel_list = get_simlabel_list(high_freq_pred)

        #return simlabel_list
        file_path = '/data1/wenjie/projects/OpenOOD-VLM/tools/classname.py'
        # 将列表以 ood_candidate_label_list = 的方式写入文件末尾
        with open(file_path, 'a') as file:
            file.write('\n' + dataset_name  + '_llm_simlabel_list = ' + repr(simlabel_list) + '\n')


    def get_noisy_ood_canlabel(self, dataset_name, topk, id_conf, ood_conf, id_pred, ood_pred, net, id_data_loaders, ood_dl):
        conf = np.concatenate([id_conf, ood_conf])
        sorted_conf = np.sort(conf)
        threshold = sorted_conf[topk]

        # sorted_id_conf = np.sort(id_conf)
        # sorted_ood_conf = np.sort(ood_conf)
        # value_id_threshold = sorted_conf[500]
        # value_ood_threshold = sorted_conf[500]

        id_path_list = []
        id_data_loader = id_data_loaders['test']
        for batch in tqdm(id_data_loader):
            path = batch['path']
            id_path_list.extend(path)

        ood_path_list = []
        for batch in tqdm(ood_dl):
            path = batch['path']
            ood_path_list.extend(path)    

        id_conf_indices = np.where(id_conf <= threshold)[0]
        ood_conf_indices = np.where(ood_conf <= threshold)[0]

        filter_id_path_list = [id_path_list[i] for i in id_conf_indices]
        filter_ood_path_list = [ood_path_list[i] for i in ood_conf_indices]

        filter_id_images = []
        filter_ood_images = []
        for image_path in filter_id_path_list:
            with Image.open(image_path) as img:  # Auto-closes when block exits
                filter_id_images.append(img.copy())

        filter_ood_images = []
        for image_path in filter_ood_path_list:
            with Image.open(image_path) as img:  # Auto-closes when block exits
                filter_ood_images.append(img.copy())

        if len(id_conf_indices) != 0:
            #id_candidate_label_list = get_ID_aware_candidate_label_list(net, id_data_loaders['test'], id_conf_indices, id_pred_list)
            id_candidate_label_list = get_candidate_label_list_blip2(filter_id_images)
            id_cleaned_list = list(set(label.rstrip('.') for label in id_candidate_label_list))
        else:
            id_cleaned_list = []      

        #ood_candidate_label_list = get_ID_aware_candidate_label_list(net, ood_dl, ood_conf_indices, ood_pred_list)
        ood_candidate_label_list = get_candidate_label_list_blip2(filter_ood_images)
        ood_cleaned_list = list(set(label.rstrip('.') for label in ood_candidate_label_list))
        ood_cleaned_list = id_cleaned_list + ood_cleaned_list

        #return ood_cleaned_list
        # 指定文件路径
        # file_path = '/data1/wenjie/projects/OpenOOD-VLM/tools/classname.py'
        # # 将列表以 ood_candidate_label_list = 的方式写入文件末尾
        # with open(file_path, 'a') as file:
        #     file.write('\n' + dataset_name + '_top' + str(topk) + '_llm_label_list = ' + repr(ood_cleaned_list) + '\n')
        # return ood_cleaned_list