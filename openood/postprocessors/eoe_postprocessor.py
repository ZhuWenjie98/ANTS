from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor
import pdb

class EOEPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(EOEPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tau = self.args.tau
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        class_num = net.n_cls
        # output = output[:1000]
        score = torch.softmax(output / self.tau, dim=1)
        smax = torch.max(score[:, :class_num], dim=1)[0] - self.args.beta * torch.max(score[:, class_num:], dim=1)[0]
        _, pred = torch.max(score, dim=1)
        return pred, smax

    def set_hyperparam(self, hyperparam: list):
        self.tau = hyperparam[0]

    def get_hyperparam(self):
        return self.tau
