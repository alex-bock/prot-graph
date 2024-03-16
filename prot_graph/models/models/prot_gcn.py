
from typing import Sequence, Union

from torch import nn, Tensor
from torch_geometric.typing import Adj

# from torch_geometric.nn.conv import GCNConv

from ..layers import ProtGConv


class ProtGCN(nn.Module):

    def __init__(
        self, d_input: int, d_hidden: Union[int, Sequence[int]], d_output: int,
        d_latent: int, interactions: Sequence[str], batch_norm: bool = False,
        layer_norm: bool = False, activation: str = "relu"
    ):

        super().__init__()

        if not isinstance(d_hidden, Sequence):
            d_hidden = [d_hidden]

        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.d_latent = d_latent
        self.dims = (
            [self.d_latent if self.d_latent > 0 else self.d_input] +
            self.d_hidden + [self.d_output]
        )

        self.interactions = interactions
        self.n_interactions = len(interactions)

        if self.d_latent > 0:
            self.linear = nn.Linear(self.d_input, self.d_latent)
            self.latent_batch_norm = nn.BatchNorm1d(self.d_latent)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(ProtGConv())

        return

    def reset_parameters(self):

        return
