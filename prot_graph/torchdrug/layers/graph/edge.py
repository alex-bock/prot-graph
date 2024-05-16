
import torch
from torch import nn, Tensor

from torchdrug import core, data
from torchdrug.core import Registry as R
from torchdrug.layers.geometry import SpatialEdge

from ....graphs.constants import HB_ATOMS, PEP_ATOMS, DB_ATOMS


@R.register("layers.geometry.HydrogenBondEdge")
class HydrogenBondEdge(nn.Module, core.Configurable):

    def __init__(
        self, radius: float = 3.5, min_distance: int = 4,
        max_distance: int = None
    ):

        super(HydrogenBondEdge, self).__init__()

        self.radius = radius
        self.min_distance = min_distance
        self.max_distance = max_distance

        self.spatial_edge_layer = SpatialEdge(
            radius=radius, min_distance=min_distance, max_distance=max_distance
        )

        return

    def forward(self, graph: data.Protein):

        hb_atom_ids = list(map(lambda x: graph.atom_name2id[x], HB_ATOMS))
        is_hb_atom = torch.isin(graph.atom_name, Tensor(hb_atom_ids))
        hb_atom_is = is_hb_atom.nonzero().squeeze()

        edge_list, i = self.spatial_edge_layer(graph.subgraph(hb_atom_is))

        return torch.stack(
            [hb_atom_is[edge_list[:, 0]],
             hb_atom_is[edge_list[:, 1]],
             edge_list[:, 2]]
        ).t(), i


@R.register("layers.geometry.PeptideBondEdge")
class PeptideBondEdge(nn.Module, core.Configurable):

    def __init__(
        self, radius: float = 1.5, min_distance: int = 1, max_distance: int = 1
    ):

        super(PeptideBondEdge, self).__init__()

        self.radius = radius
        self.min_distance = min_distance
        self.max_distance = max_distance

        self.spatial_edge_layer = SpatialEdge(
            radius=radius, min_distance=min_distance, max_distance=max_distance
        )

        return

    def forward(self, graph: data.Protein):

        pep_atom_ids = list(map(lambda x: graph.atom_name2id[x], PEP_ATOMS))
        is_pep_atom = torch.isin(graph.atom_name, Tensor(pep_atom_ids))
        pep_atom_is = is_pep_atom.nonzero().squeeze()

        edge_list, i = self.spatial_edge_layer(graph.subgraph(pep_atom_is))

        return torch.stack(
            [pep_atom_is[edge_list[:, 0]],
             pep_atom_is[edge_list[:, 1]],
             edge_list[:, 2]]
        ).t(), i


@R.register("layers.geometry.DisulfideBridgeEdge")
class DisulfideBridgeEdge(nn.Module, core.Configurable):

    pass
