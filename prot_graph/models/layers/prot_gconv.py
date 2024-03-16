
import math

import torch
from torch import nn


class ProtGConv(nn.Module):

    def __init__(self, d_in: int, d_out: int, device: str, bias: bool = True):

        super(ProtGConv, self).__init__()

        self.d_in = d_in
        self.d_out = d_out

        self.weight = nn.Parameter(torch.FloatTensor(d_in, d_out)).to(device)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(d_out)).to(device)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):

        sigma = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-sigma, sigma)
        if self.bias is not None:
            self.bias.data.uniform_(-sigma, sigma)

    def forward(self, x, adj_mat):

        x = x.reshape(x.size()[0] * x.size()[1], x.size()[2])
