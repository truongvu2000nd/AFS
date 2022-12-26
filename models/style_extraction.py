import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import Union, List


class Highway(nn.Module):
    def __init__(self, size, num_layers, act="relu", momentum=0.1, share_weights=False):

        super(Highway, self).__init__()

        self.num_layers = num_layers
        self.share_weights = share_weights
        if share_weights:
            self.nonlinear = nn.Sequential(
                nn.Linear(size, size), nn.BatchNorm1d(size, momentum=momentum)
            )
            self.linear = nn.Linear(size, size)
            self.gate = nn.Linear(size, size)
        else:
            self.nonlinear = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(size, size), nn.BatchNorm1d(size, momentum=momentum)
                    )
                    for _ in range(num_layers)
                ]
            )
            self.linear = nn.ModuleList(
                [nn.Linear(size, size) for _ in range(num_layers)]
            )
            self.gate = nn.ModuleList(
                [nn.Linear(size, size) for _ in range(num_layers)]
            )

        if act == "relu":
            self.act = nn.ReLU()
        elif act == "lrelu":
            self.act = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise NotImplementedError

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """
        if self.share_weights:
            for _ in range(self.num_layers):
                gate = torch.sigmoid(self.gate(x))
                nonlinear = self.act(self.nonlinear(x))
                linear = self.linear(x)
                x = gate * nonlinear + (1 - gate) * linear

            return x

        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = self.act(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear

        return x


class StyleExtractionNet(nn.Module):
    def __init__(
        self,
        size=256,
        n_latent=18,
        num_layers=5,
        act="relu",
        momentum=0.1,
        share_weights=False,
    ):
        super().__init__()
        self.fc1s = nn.ModuleList([nn.Linear(512, size) for _ in range(n_latent)])
        self.fc2s = nn.ModuleList([nn.Linear(size, 512) for _ in range(n_latent)])
        self.highways = nn.ModuleList(
            [
                Highway(
                    size,
                    num_layers,
                    act=act,
                    momentum=momentum,
                    share_weights=share_weights,
                )
                for _ in range(n_latent)
            ]
        )

    def forward(self, w):
        outputs = []
        for i in range(w.size(1)):
            out = self.fc1s[i](w[:, i])
            out = self.highways[i](out)
            out = self.fc2s[i](out)
            outputs.append(out.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs
