import torch
import torch.nn as nn


class PreprocessingModel(nn.Module):
    def __init__(self, net, preproc=None):
        super().__init__()
        self.net = net
        self.preproc_args = preproc

    def preprocess(self, x):
        if self.preproc_args is None:
            return x
        else:
            return (x-self.preproc_args['mean']) / self.preproc_args['std']

    def forward(self, x):
        y = self.preprocess(x)
        return self.net(y)
