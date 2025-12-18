import time

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger

import os
import torch
import pdb
from tqdm import tqdm

import numpy as np
import pandas as pd
# from tsne import bh_sne
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoProcessor, LlavaForConditionalGeneration
import re
import time

def tsne_plot(save_dir,  outputs, targets):
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)
    tsne = TSNE(random_state=0, perplexity=100)

    # pca = PCA(n_components=50)
    # outputs1 = pca.fit_transform(outputs1)
    # outputs2 = pca.fit_transform(outputs2)
    tsne_output1 = tsne.fit_transform(outputs)
    # tsne_output2 = tsne.fit_transform(outputs2)

    df1 = pd.DataFrame(tsne_output1, columns=['x', 'y'])
    # df2 = pd.DataFrame(tsne_output2, columns=['x', 'y'])

    df1['targets'] = targets
    # df2['targets'] = targets2

    palette = [
    (133/255, 90/255, 52/255, 0.2),  # #855A34, 透明度 0.7
    (133/255, 61/255, 243/255, 0.8), # #853DF3, 透明度 0.6
    (113/255, 200/255, 226/255, 0.2), # #71C8E2, 透明度 0.5
    (228/255, 179/255, 2/255, 0.2),   # #E4B302, 透明度 0.4
    (67/255, 191/255, 135/255, 0.8),  # #43BF87, 透明度 0.3
    (251/255, 105/255, 121/255, 0.8)  # #FB6979, 透明度 0.2
    ]


    plt.rcParams['figure.figsize'] = 16, 10
    sns.scatterplot(
        x='x', y='y',
        hue='targets',
        palette=palette,
        #palette=['#855A34', '#853DF3', '#71C8E2', '#E4B302', '#43BF87', '#FB6979'],
        data=df1,
        marker='o',
        legend="full",
    )

    # sns.scatterplot(
    #     x='x', y='y',
    #     hue='targets',
    #     palette=['blue'],
    #     data=df2,
    #     marker='o',
    #     legend="full",
    #     alpha=0.5
    # )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(fontsize=20, markerscale=2.5)

    plt.savefig(os.path.join(save_dir,'ants.png'), bbox_inches='tight')
    print('done!')


class TestOODPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        id_loader_dict = get_dataloader(self.config)
        ood_loader_dict = get_ood_dataloader(self.config)

        # init network
        net = get_network(self.config.network)

        # init ood evaluator
        evaluator = get_evaluator(self.config)

        # init ood postprocessor
        postprocessor = get_postprocessor(self.config)
        # setup for distance-based methods
        postprocessor.setup(net, id_loader_dict, ood_loader_dict)
        print('\n', flush=True)
        print(u'\u2500' * 70, flush=True)

        # start calculating accuracy
        print('\nStart evaluation...', flush=True)
        #neg_sun_sim_list = self.get_neg_sim(net, ood_loader_dict['farood']['sun'])
        # if self.config.evaluator.ood_scheme == 'fsood':
        #     acc_metrics = evaluator.eval_acc(
        #         net,
        #         id_loader_dict['test'],
        #         postprocessor,
        #         fsood=True,
        #         csid_data_loaders=ood_loader_dict['csid'])
        # else:
        #     acc_metrics = evaluator.eval_acc(net, id_loader_dict['test'],
        #                                      postprocessor)
        # print('\nAccuracy {:.2f}%'.format(100 * acc_metrics['acc']),
        #       flush=True)
        # print(u'\u2500' * 70, flush=True)

        # with torch.no_grad():
        #     text_features = net.text_features
        #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #     text_features = text_features.cpu()

        # cls_num = net.n_cls 
        # ood_cls_num =  text_features.shape[1]-cls_num  
        # ID_text_embedding = [f"ID" for i in range(cls_num)]
        # OOD_text_embedding = [f"Neglabel" for i in range(ood_cls_num)]
        # text_embedding = ID_text_embedding + OOD_text_embedding

        # tsne_plot('/data1/wenjie/projects/OpenOOD-VLM/', text_features.t(), text_embedding, text_features[:,cls_num:].t(), OOD_text_embedding)
        # pdb.set_trace()

        #start evaluating ood detection methods
        #id_image_features = self.get_features(net, id_loader_dict['test'])
        #ood_image_features = self.get_features(net, ood_loader_dict['farood']['dtd'])
        #torch.save(id_image_features, '/data1/wenjie/projects/ANTS/scorefiles/id_image_features.pt')
        id_image_features = torch.load('/data1/wenjie/projects/ANTS/scorefiles/id_image_features.pt')
        ood_image_features = torch.load('/data1/wenjie/projects/ANTS/scorefiles/dtd_image_features.pt')
        id_image_features = id_image_features[:1000,:]
        ood_image_features = ood_image_features[:1000,:] #only select 1000 ood

        with torch.no_grad():
            id_text_features = net.imagenet_features
            neglabel_features = net.text_features[:,3000:3500]
            eoe_features = net.eoe_text_features
            dtd_gt_features = net.dtd_gt_features
            ants_features = net.ants_text_features

        text_features = torch.cat((id_image_features, ood_image_features, neglabel_features.t(), eoe_features.t(), dtd_gt_features.t(), ants_features.t()), dim=0)
        text_features = text_features.cpu()

        cls_num = net.n_cls 
        ood_cls_num =  net.text_features.shape[1]-cls_num  
        ID_Images = [f"ID images" for i in range(id_image_features.shape[0])]
        OOD_Images = [f"OOD images" for i in range(ood_image_features.shape[0])]
        #ID_text = [f"ID" for i in range(id_text_features.shape[1])]
        Neglabel_text= [f"NegLabel" for i in range(neglabel_features.shape[1])]
        EOE_text= [f"EOE" for i in range(eoe_features.shape[1])]
        OOD_gt_text= [f"OOD GT" for i in range(dtd_gt_features.shape[1])]
        ANTS_text= [f"ANTS(ENS)" for i in range(ants_features.shape[1])]

        #texts = Neglabel_text + EOE_text + OOD_gt_text + ANTS_text
        texts = ID_Images + OOD_Images + Neglabel_text + EOE_text + OOD_gt_text + ANTS_text

        tsne_plot('/data1/wenjie/projects/ANTS/visualization/', text_features, texts)
        pdb.set_trace()


        timer = time.time()
        if self.config.evaluator.ood_scheme == 'fsood':
            evaluator.eval_ood(net,
                               id_loader_dict,
                               ood_loader_dict,
                               postprocessor,
                               fsood=True)
        else:
            evaluator.eval_ood(net, id_loader_dict, ood_loader_dict,
                               postprocessor)
        print('Time used for eval_ood: {:.0f}s'.format(time.time() - timer))
        print('Completed!', flush=True)

    def get_features(self, model, loader):
        tqdm_object = tqdm(loader, total=len(loader))
        image_features_list = []

        cls_num = model.n_cls
        train_dataiter = iter(loader)
        # for train_step in tqdm(range(1,
        #                              len(train_dataiter) + 1)):
        for batch_idx, batch in enumerate(tqdm_object):
            with torch.no_grad():
                #batch = next(train_dataiter)
                images = batch['data'].cuda()
                images = images.cuda()
                targets = batch['label'].long().cuda()
                image_features = model.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                image_features_list.append(image_features)
        image_features = torch.cat(image_features_list, dim=0)


        return image_features    

    def get_neg_sim(self, model, loader):
        tqdm_object = tqdm(loader, total=len(loader))
        sim_list = []
        all_targets = []
        result = {
            'scores': None,
            'acc': None,
        }

        with torch.no_grad():
            text_features = model.text_features.t()
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

        text_features = text_features[1000:,:]
        train_dataiter = iter(loader)
        # for train_step in tqdm(range(1,
        #                              len(train_dataiter) + 1)):
        for batch_idx, batch in enumerate(tqdm_object):
            with torch.no_grad():
                #batch = next(train_dataiter)
                images = batch['data'].cuda()
                targets = batch['label'].cuda()
                targets = targets.long().cuda()
                image_features = model.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                sim = image_features @ text_features.t()
                sim = sim.mean(dim=1)
                sim = sim.detach().cpu()
                sim_list.append(sim)
        sim_list = torch.cat(sim_list, dim=0).float()
        
        return sim_list    
