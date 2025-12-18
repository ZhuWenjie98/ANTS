import csv
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from tqdm import tqdm
import openood.utils.comm as comm
import torch.nn.functional as F
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

from openood.postprocessors import BasePostprocessor

from .ood_evaluator import OODEvaluator
from .ood_evaluator_offline import OODOfflineEvaluator
from .metrics import compute_all_metrics
from .neglabel_metrics import evaluate_all
from tools.classname import imagenet_classes
import pdb
import re

class FSOODEvaluatorClip(OODEvaluator):
    def eval_csid_acc(self, net: nn.Module,
                      csid_loaders: Dict[str, Dict[str, DataLoader]], postprocessor):
        # ensure the networks in eval mode
        net.eval()
        # pdb.set_trace()

        for dataset_name, csid_dl in csid_loaders.items():
            print(f'Computing accuracy on {dataset_name} dataset...')
            correct = 0
            with torch.no_grad():
                for batch in csid_dl:
                    data = batch['data'].cuda()
                    target = batch['label'].cuda()
                    # forward
                    # output = net(data)
                    pred, _ = postprocessor.postprocess(net, data)
                    # # accuracy
                    # pred = output.data.max(1)[1]
                    correct += pred.eq(target.data).sum().item()
            acc = correct / len(csid_dl.dataset)
            if self.config.recorder.save_csv:
                self._save_acc_results(acc, dataset_name)
        print(u'\u2500' * 70, flush=True)

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1):
        net.eval()

        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Eval: ',
                              position=0,
                              leave=True,
                              disable=not comm.is_main_process()):
                # prepare data
                data = batch['data'].cuda()
                target = batch['label'].cuda()

                # forward
                # output = net(data)
                pred, _ = postprocessor.postprocess(net, data)

                # loss = F.cross_entropy(output, target)

                # # accuracy
                # pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                # loss_avg += float(loss.data)

        loss = loss_avg / len(data_loader)
        acc = correct / len(data_loader.dataset)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss)
        metrics['acc'] = self.save_metrics(acc)
        return metrics

    def _save_acc_results(self, acc, dataset_name):
        write_content = {
            'dataset': dataset_name,
            'FPR@95': '-',
            'AUROC': '-',
            'AUPR_IN': '-',
            'AUPR_OUT': '-',
            'ACC': '{:.2f}'.format(100 * acc),
        }
        fieldnames = list(write_content.keys())
        # print csid metric results
        print('CSID[{}] accuracy: {:.2f}%'.format(dataset_name, 100 * acc),
              flush=True)
        csv_path = os.path.join(self.config.output_dir, 'csid.csv')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)

    def eval_ood(self, net: nn.Module, id_data_loader: List[DataLoader],
                 ood_data_loaders: List[DataLoader],
                 postprocessor: BasePostprocessor):
        # ensure the networks in eval mode
        net.eval()
        # load training in-distribution data
        assert 'test' in id_data_loader, \
            'id_data_loaders should have the key: test!'
        dataset_name = self.config.dataset.name
        print(f'Performing inference on {dataset_name} dataset...', flush=True)
        id_pred, id_conf, id_gt = postprocessor.inference(
            net, id_data_loader['test'])
        if self.config.recorder.save_scores:
            self._save_scores(id_pred, id_conf, id_gt, dataset_name)

        # load csid data and compute confidence
        for dataset_name, csid_dl in ood_data_loaders['csid'].items():
            print(f'Performing inference on {dataset_name} dataset...',
                  flush=True)
            csid_pred, csid_conf, csid_gt = postprocessor.inference(net, csid_dl)
            if self.config.recorder.save_scores:
                self._save_scores(csid_pred, csid_conf, csid_gt, dataset_name)
            id_pred = np.concatenate([id_pred, csid_pred])
            id_conf = np.concatenate([id_conf, csid_conf])
            id_gt = np.concatenate([id_gt, csid_gt])

        # compute accuracy on csid
        print(u'\u2500' * 70, flush=True)
        self.eval_csid_acc(net, ood_data_loaders['csid'], postprocessor)

        # load nearood data and compute ood metrics
        print(u'\u2500' * 70, flush=True)
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='nearood')

        # load farood data and compute ood metrics
        print(u'\u2500' * 70, flush=True)
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='farood')

# class OODEvaluatorClip(OODOfflineEvaluator):
class OODEvaluatorClip(OODEvaluator):
    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1):
        net.eval()

        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Eval: ',
                              position=0,
                              leave=True,
                              disable=not comm.is_main_process()):
                # prepare data
                data = batch['data'].cuda()
                target = batch['label'].cuda()
                # forward
                # output = net(data)
                pred, _ = postprocessor.postprocess(net, data)
                # # accuracy
                # pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()
                # test loss average
                # loss_avg += float(loss.data)

        loss = loss_avg / len(data_loader)
        acc = correct / len(data_loader.dataset)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss)
        metrics['acc'] = self.save_metrics(acc)
        return metrics

    def eval_ood(self, net: nn.Module, id_data_loaders: List[DataLoader],
                 ood_data_loaders: List[DataLoader],
                 postprocessor: BasePostprocessor,  fsood: bool = False):
        # ensure the networks in eval mode
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()
        # load training in-distribution data
        assert 'test' in id_data_loaders, \
            'id_data_loaders should have the key: test!'
        dataset_name = self.config.dataset.name
        # if self.config.postprocessor.APS_mode:
        #     assert 'val' in id_data_loaders
        #     assert 'val' in ood_data_loaders
        #     self.hyperparam_search(net, id_data_loaders['val'],
        #                            ood_data_loaders['val'], postprocessor)
        print(f'Performing inference on {dataset_name} dataset...', flush=True)

        id_pred, id_conf, id_gt = postprocessor.inference(
            net, id_data_loaders['test'])

        #保存为文本文件
        # combined_array = np.column_stack((id_pred, id_conf, id_gt))
        # np.savetxt('/data1/wenjie/projects/OpenOOD-VLM/scorefiles/id_conf.txt', combined_array, fmt='%.5f', header='id_pred,id_conf,id_gt', delimiter=',')


        # loaded_data = np.loadtxt('/data1/wenjie/projects/OpenOOD-VLM/scorefiles/id_conf.txt', delimiter=',', skiprows=1)
        # # 分割成三个数组
        # id_pred = loaded_data[:, 0]
        # id_conf = loaded_data[:, 1]
        # id_gt = loaded_data[:, 2]

        # simlabel_list = self.get_simlabel_list()
        # file_path = '/data1/wenjie/projects/OpenOOD-VLM/tools/classname.py'
        # # 将列表以 ood_candidate_label_list = 的方式写入文件末尾
        # with open(file_path, 'a') as file:
        #     file.write('\n' + 'eoe_near_llm_label_list = ' + repr(simlabel_list) + '\n')
  

        # id_pred, id_conf, id_gt = postprocessor.inference(
        # net, id_data_loaders['test'])
      
        if self.config.recorder.save_scores:
            self._save_scores(id_pred, id_conf, id_gt, dataset_name)

        # print(u'\u2500' * 70, flush=True)
        # self._eval_ood(net, [id_pred, id_conf, id_gt],
        #                id_data_loaders,
        #                ood_data_loaders,
        #                postprocessor,
        #                ood_split='nearood')

        #load farood data and compute ood metrics
        print(u'\u2500' * 70, flush=True)
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       id_data_loaders,
                       ood_data_loaders,
                       postprocessor,
                       ood_split='farood')   




class OODEvaluatorClipTTA(OODEvaluator):
    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1, fsood=True,
                 csid_data_loaders=None):
        postprocessor.reset_memory()  ## reset the memory for the ID evaluation.
        net.eval()

        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Eval: ',
                              position=0,
                              leave=True,
                              disable=not comm.is_main_process()):
                # prepare data
                data = batch['data'].cuda()
                target = batch['label'].cuda()

                # forward
                # output = net(data)
                pred, _ = postprocessor.postprocess(net, data)

                # loss = F.cross_entropy(output, target)

                # # accuracy
                # pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                # loss_avg += float(loss.data)

        loss = loss_avg / len(data_loader)
        acc = correct / len(data_loader.dataset)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss)
        metrics['acc'] = self.save_metrics(acc)
        return metrics

    def eval_ood(self, net: nn.Module, id_data_loaders: List[DataLoader],
                 ood_data_loaders: List[DataLoader],
                 postprocessor: BasePostprocessor,  fsood: bool = False):
        # ensure the networks in eval mode
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()
        # load training in-distribution data
        assert 'test' in id_data_loaders, \
            'id_data_loaders should have the key: test!'
        dataset_name = self.config.dataset.name
        if self.config.postprocessor.APS_mode:
            assert 'val' in id_data_loaders
            assert 'val' in ood_data_loaders
            self.hyperparam_search(net, id_data_loaders['val'],
                                   ood_data_loaders['val'], postprocessor)
            

        # print(f'Performing inference on {dataset_name} dataset...', flush=True)
        # print(u'\u2500' * 70, flush=True)
        # self._eval_ood(net, id_data_loaders['test'],
        #                ood_data_loaders,
        #                postprocessor,
        #                ood_split='nearood', fsood=fsood)

        #load farood data and compute ood metrics
        print(u'\u2500' * 70, flush=True)
        self._eval_ood(net, id_data_loaders['test'],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='farood', fsood=fsood)
        

    ## calculate the id pred/conf/gt online, not offline as the default setting. Therefore, different data order may lead to different results. 
    def _eval_ood(self,
                  net: nn.Module,
                  id_loader: DataLoader,
                  ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                  postprocessor: BasePostprocessor,
                  ood_split: str = 'nearood', fsood=False):
        print(f'Processing {ood_split}...', flush=True)
        # [id_pred, id_conf, id_gt] = id_list
        metrics_list = []
        # postprocessor.reset_memory()  ## here, we inherit the memory with the same near/far OOD group; using more information, not fair
        # net.reset_all_nts_list()
        # net.reset_all_pred_list()
        for dataset_name, ood_dl in ood_data_loaders[ood_split].items():
            #postprocessor.reset_memory()  ## here, we reset the memory for each OOD datasets.
            print(f'Performing inference on {dataset_name} dataset...', flush=True)
            # merging the id dataloader and ood dataloader! 
            combined_dataset = ConcatDataset([id_loader.dataset, ood_dl.dataset])
            #combined_dataset = ConcatDataset([ood_dl.dataset, id_loader.dataset])
            #combined_dataset = id_loader.dataset
            #combined_dataset = ood_dl.dataset
            if fsood and 'csid' in ood_data_loaders.keys():
                print(f'concating ID, CSID, and OOD dataset', flush=True)
                for dataset_name_csid, csid in ood_data_loaders['csid'].items():
                    combined_dataset = ConcatDataset([combined_dataset, csid.dataset])
            print(f'Generating combined dataset with ID and OOD dataset of {dataset_name}, total size {len(combined_dataset)}')
            # pdb.set_trace()
            # Create a new DataLoader from the combined dataset. The shuffle operation is verified
            combined_dataloader = DataLoader(combined_dataset, batch_size=id_loader.batch_size, num_workers=id_loader.num_workers, shuffle=True)
            pred, conf, label = postprocessor.inference(net, combined_dataloader) 

            print(f'Computing metrics on {dataset_name} dataset...')

            ood_metrics = compute_all_metrics(conf, label, pred)
            if self.config.recorder.save_csv:
                self._save_csv(ood_metrics, dataset_name=dataset_name)
            metrics_list.append(ood_metrics)


        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0)
        if self.config.recorder.save_csv:
            self._save_csv(metrics_mean, dataset_name=ood_split)
