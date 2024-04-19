
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class ProtGConv(nn.Module):

    def __init__(self, d_input: int, d_output: int, batch_norm: bool = False):

        super(ProtGConv, self).__init__()

        self.d_input = d_input
        self.d_output = d_output
        self.linear = nn.Linear(d_input, d_output)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.d_output)
        else:
            self.batch_norm = None

        return

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:

        n_nodes, n_edges = x.shape[0], edge_index.shape[1]
        weights = torch.ones(n_edges + n_nodes)

        u, v = edge_index
        u_loops, v_loops = (
            torch.cat([u, torch.arange(n_nodes)]),
            torch.cat([v, torch.arange(n_nodes)])
        )
        u_degree, v_degree = torch.bincount(u_loops), torch.bincount(v_loops)

        norm_weights = weights / (u_degree[u_loops] * v_degree[v_loops]).sqrt()
        adj_mat = torch.sparse_coo_tensor(
            torch.stack([u_loops, v_loops]), norm_weights
        )

        output = self.linear(torch.sparse.mm(adj_mat.t(), x))
        if self.batch_norm is not None:
            output = self.batch_norm(output)
        output = F.relu(output)

        return output
