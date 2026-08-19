"""Base inference loop for path-aware ANTS postprocessors."""

from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm


class ANTSBasePostprocessor:
    """Minimal postprocessor protocol that also forwards image paths."""

    def __init__(self, config) -> None:
        self.config = config

    def setup(
        self, net: nn.Module, id_loader_dict: Any, ood_loader_dict: Any
    ) -> None:
        """Prepare state before inference."""

    @torch.no_grad()
    def postprocess(
        self,
        net: nn.Module,
        data: Any,
        path: Optional[Sequence[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return predictions and confidence for one batch."""

        del path
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(
        self,
        net: nn.Module,
        data_loader: DataLoader,
        progress: bool = True,
    ):
        """Run postprocessing and return NumPy predictions, scores, labels."""

        pred_list, conf_list, label_list = [], [], []
        for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            path = batch.get('path')
            pred, conf = self.postprocess(net, data, path)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list
