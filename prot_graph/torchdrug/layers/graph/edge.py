
import networkx as nx
import numpy as np

import torch
from torch import nn, Tensor

from torchdrug import core, data
from torchdrug.core import Registry as R
from torchdrug.layers.geometry import SpatialEdge

from ....graphs.constants import HB_ATOMS, PEP_ATOMS, DB_ATOMS
from ....util import CONTACT2ID


@R.register("layers.geometry.HydrogenBondEdge")
class HydrogenBondEdge(nn.Module, core.Configurable):

    atom2res = True

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
        is_hb_atom = torch.isin(
            graph.atom_name, Tensor(hb_atom_ids).to(graph.device)
        )
        hb_atom_is = is_hb_atom.nonzero().squeeze()

        edge_list, i = self.spatial_edge_layer(graph.subgraph(hb_atom_is))

        return torch.stack(
            [hb_atom_is[edge_list[:, 0]],
             hb_atom_is[edge_list[:, 1]],
             edge_list[:, 2]]
        ).t(), i


@R.register("layers.geometry.PeptideBondEdge")
class PeptideBondEdge(nn.Module, core.Configurable):

    atom2res = True

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
        is_pep_atom = torch.isin(
            graph.atom_name, Tensor(pep_atom_ids).to(graph.device)
        )
        pep_atom_is = is_pep_atom.nonzero().squeeze()

        edge_list, i = self.spatial_edge_layer(graph.subgraph(pep_atom_is))

        return torch.stack(
            [pep_atom_is[edge_list[:, 0]],
             pep_atom_is[edge_list[:, 1]],
             edge_list[:, 2]]
        ).t(), i


@R.register("layers.geometry.DisulfideBridgeEdge")
class DisulfideBridgeEdge(nn.Module, core.Configurable):

    atom2res = True

    def __init__(
        self, radius: float = 2.2, min_distance: int = 4,
        max_distance: int = None
    ):

        super(DisulfideBridgeEdge, self).__init__()

        self.radius = radius
        self.min_distance = min_distance
        self.max_distance = max_distance

        self.spatial_edge_layer = SpatialEdge(
            radius=radius, min_distance=min_distance, max_distance=max_distance
        )

        return

    def forward(self, graph: data.Protein):

        db_atom_ids = list(map(lambda x: graph.atom_name2id[x], DB_ATOMS))
        is_db_atom = torch.isin(
            graph.atom_name, Tensor(db_atom_ids).to(graph.device)
        )
        db_atom_is = is_db_atom.nonzero().squeeze()

        edge_list, i = self.spatial_edge_layer(graph.subgraph(db_atom_is))

        return torch.stack(
            [db_atom_is[edge_list[:, 0]],
             db_atom_is[edge_list[:, 1]],
             edge_list[:, 2]]
        ).t(), i


@R.register("layers.geometry.GetContactsEdge")
class GetContactsEdge(nn.Module, core.Configurable):

    atom2res = True

    def __init__(self, contact_type: str):

        super(GetContactsEdge, self).__init__()

        self.contact_type = CONTACT2ID[contact_type]

        return

    def forward(self, graph: data.Protein):

        edge_list = graph.edge_list[graph.edge_list[:, 2] == self.contact_type]
        edge_list = torch.cat(
            (
                edge_list[:, :2],
                torch.zeros(
                    len(edge_list), device=edge_list.device
                ).t().unsqueeze(dim=1)
            ), dim=1
        ).long()

        return edge_list, 1


@R.register("layers.geometry.MSTEdge")
class MSTEdge(nn.Module, core.Configurable):

    atom2res = False

    def __init__(
        self, base_edge_layer: nn.Module, p: float = 0.0, **mst_params
    ):

        super(MSTEdge, self).__init__()

        self.base_edge_layer = base_edge_layer
        self.p = p
        self.mst_params = mst_params

        return

    def forward(self, graph: data.Protein):

        base_graph_edge_list, i = self.base_edge_layer(graph)

        if len(base_graph_edge_list) == 0:
            return base_graph_edge_list, i

        base_graph = nx.from_edgelist(base_graph_edge_list[:, :2].cpu().numpy())
        mst = nx.minimum_spanning_tree(base_graph, **self.mst_params)
        mst_edge_list = torch.cat(
            [
                Tensor([[u, v] for (u, v) in mst.edges]),
                torch.zeros(len(mst.edges)).unsqueeze(dim=1)
            ], dim=1
        ).to(graph.device).to(int)
        mst_size = len(mst_edge_list) / len(base_graph_edge_list)
        remainder_size = max(0.0, self.p - mst_size)

        n_base_edges = len(base_graph.edges)
        base_graph.remove_edges_from(mst.edges)
        base_graph_edge_list = torch.cat(
            [
                Tensor([[u, v] for (u, v) in base_graph.edges]),
                torch.zeros(len(base_graph.edges)).unsqueeze(dim=1)
            ], dim=1
        ).to(int)
        remainder_edge_list = base_graph_edge_list[
            np.random.choice(
                len(base_graph_edge_list),
                size=min(
                    len(base_graph_edge_list),
                    int(n_base_edges * remainder_size)
                ),
                replace=False
            )
        ]

        if len(remainder_edge_list) == 0:
            final_edge_list = mst_edge_list
        else:
            final_edge_list = torch.cat([mst_edge_list, remainder_edge_list])

        return final_edge_list, i


@R.register("layers.geometry.SampleEdge")
class SampleEdge(nn.Module, core.Configurable):

    atom2res = False

    def __init__(self, base_edge_layer: nn.Module, p: float = 1.0):

        super(SampleEdge, self).__init__()

        self.base_edge_layer = base_edge_layer
        self.sampler = np.random.default_rng(0)
        self.p = p

        return

    def forward(self, graph: data.Protein):

        base_graph_edge_list, i = self.base_edge_layer(graph)
        n_base_edges = len(base_graph_edge_list)

        return base_graph_edge_list[
            self.sampler.choice(
                n_base_edges,
                size=min(int(n_base_edges * self.p), n_base_edges),
                replace=False
            )
        ], i
