
from typing import List, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

from ..layers import ProtGConv


class ProtRGCN(nn.Module):

    """
    https://arxiv.org/pdf/1609.02907.pdf
    """

    def __init__(
        self, d_input: int, d_hidden: Union[int, List[int]],
        batch_norm=False, concat_hidden=False
    ):

        super(ProtRGCN, self).__init__()

        if not isinstance(d_hidden, List):
            d_hidden = [d_hidden]

        self.d_input = d_input
        self.d_output = sum(d_hidden) if concat_hidden else d_hidden[-1]
        self.d = [d_input] + d_hidden

        self.batch_norm = batch_norm
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        self.n_layers = len(self.d) - 1

        for i in range(self.n_layers):
            self.layers.append(ProtGConv(self.d[i], self.d[i + 1]))

        return

    def forward(
        self, x: Tensor, edge_index: Tensor, relations: Tensor, batch: Tensor
    ) -> Tensor:

        xs = []
        x_i = x
        for i in range(self.n_layers):
            x_i = self.layers[i](x_i, edge_index)
            x_i = F.relu(x_i)
            xs.append(x_i)

        if self.concat_hidden:
            x = torch.cat(xs, dim=-1)
        else:
            x = xs[-1]

        x = global_add_pool(x, batch)

        return x
