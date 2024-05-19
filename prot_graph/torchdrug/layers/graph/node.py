
import torch
from torch import nn

from torchdrug import core, data
from torchdrug.core import Registry as R
from torchdrug.layers.geometry import AlphaCarbonNode


@R.register("layers.geometry.CentroidNode")
class CentroidNode(nn.Module, core.Configurable):

    def forward(self, graph: data.Protein):

        res_graph = AlphaCarbonNode()(graph)

        # hacky but move alpha carbons to centroid coordinates
        res_graph.node_position = torch.stack(
            [
                graph.node_position[graph.atom2residue == x].mean(axis=0)
                for x in res_graph.atom2residue
            ]
        )

        return res_graph