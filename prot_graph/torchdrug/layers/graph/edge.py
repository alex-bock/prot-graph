
from typing import Callable, Tuple

import math
import numpy as np
from scipy.stats import norm, halfnorm

import torch
from torch import nn, Tensor

from torchdrug.core import Configurable, Registry as R
from torchdrug.data import Protein
from torchdrug.layers.geometry import SpatialEdge

from ....graphs.constants import HB_ATOMS, PEP_ATOMS, DB_ATOMS
from ....util import CONTACT2ID


@R.register("layers.geometry.PeptideBondEdge")
class PeptideBondEdge(nn.Module, Configurable):

    atom2res = True

    def __init__(
        self, radius: float = 1.5, min_distance: int = 1, max_distance: int = 1
    ):

        super(PeptideBondEdge, self).__init__()

        self.spatial_edge_layer = SpatialEdge(
            radius=radius, min_distance=min_distance, max_distance=max_distance
        )

        return

    def forward(self, graph: Protein) -> Tuple[torch.Tensor, int]:

        pep_atom_ids = list(map(lambda x: graph.atom_name2id[x], PEP_ATOMS))
        is_pep_atom = torch.isin(
            graph.atom_name, Tensor(pep_atom_ids).to(graph.device)
        )
        pep_atom_is = is_pep_atom.nonzero().squeeze()
        pep_atom_subgraph = graph.subgraph(pep_atom_is)

        atom_edge_list, i = self.spatial_edge_layer(pep_atom_subgraph)
        atom_edge_list = torch.stack(
            [
                pep_atom_is[atom_edge_list[:, 0]],
                pep_atom_is[atom_edge_list[:, 1]],
                atom_edge_list[:, 2]
            ]
        ).t()

        res_edge_list = to_res_edges(atom_edge_list, graph)

        return res_edge_list, i


@R.register("layers.geometry.GetContactsEdge")
class GetContactsEdge(nn.Module, Configurable):

    atom2res = True

    def __init__(self, contact_type: str):

        super(GetContactsEdge, self).__init__()

        self.contact_type = CONTACT2ID[contact_type]

        return

    def forward(self, graph: Protein) -> Tuple[torch.Tensor, int]:

        atom_edge_list = graph.edge_list[
            graph.edge_list[:, 2] == self.contact_type
        ]
        edge_list = to_res_edges(atom_edge_list, graph)
        edge_list = add_reverse_edges(edge_list)

        return torch.cat(
            (
                edge_list[:, :2],
                torch.zeros(
                    len(edge_list), device=edge_list.device
                ).t().unsqueeze(dim=1)
            ), dim=1
        ).long(), 1


@R.register("layers.geometry.CompleteEdge")
class CompleteEdge(nn.Module, Configurable):

    atom2res = False

    def __init__(self):

        super(CompleteEdge, self).__init__()

        return

    def forward(self, graph: Protein) -> Tuple[torch.Tensor, int]:

        node_is = torch.arange(len(graph.residue2graph)).to(graph.device)
        pairs = torch.cat(
            [
                torch.combinations(node_is[graph.residue2graph == i])
                for i in torch.unique(graph.residue2graph)
            ]
        )
        edge_list = torch.cat(
            (
                pairs,
                torch.zeros(
                    len(pairs), device=pairs.device
                ).t().unsqueeze(dim=1)
            ), dim=1
        ).long()
        edge_list = add_reverse_edges(edge_list)

        return edge_list, 1


@R.register("layers.geometry.SampleEdge")
class SampleEdge(nn.Module, Configurable):

    atom2res = False

    def __init__(
        self, base_edge_layer: nn.Module, p: float = 1.0, fn: Callable = None
    ):

        super(SampleEdge, self).__init__()

        self.base_edge_layer = base_edge_layer
        self.sampler = np.random.default_rng(0)

        self._fn = fn
        if self._fn == "sqrt":
            self._fn = lambda n: math.sqrt(n)
        elif self._fn is None:
            self._fn = lambda n: n
        self.fn = lambda n: p * self._fn(n)

        return

    def forward(self, graph: Protein) -> Tuple[torch.Tensor, int]:

        base_graph_edge_list, i = self.base_edge_layer(graph)

        edge_list = self.sample(base_graph_edge_list)
        edge_list = add_reverse_edges(edge_list)

        return torch.cat(
            (
                edge_list[:, :2],
                torch.zeros(
                    len(edge_list), device=edge_list.device
                ).t().unsqueeze(dim=1)
            ), dim=1
        ).long(), i

    def sample(
        self, base_graph_edge_list: Tensor, weights: Tensor = None
    ) -> Tensor:

        if weights is None:
            weights = torch.ones(len(base_graph_edge_list))

        base_graph_edge_list_unique_idx = (
            base_graph_edge_list[:, 0] < base_graph_edge_list[:, 1]
        ).nonzero().squeeze()
        base_graph_edge_list_unique = base_graph_edge_list[
            base_graph_edge_list_unique_idx
        ]
        n_base_edges = len(base_graph_edge_list_unique)

        weights = weights[base_graph_edge_list_unique_idx.cpu()]
        weights /= weights.sum()

        sample_size = min(int(self.fn(n_base_edges)), n_base_edges)
        sample_idx = self.sampler.choice(
            n_base_edges, size=sample_size, replace=False, p=np.array(weights)
        )
        edge_list = base_graph_edge_list[sample_idx]

        return edge_list


@R.register("layers.geometry.GaussianDistanceSampleEdge")
class GaussianDistanceSampleEdge(SampleEdge):

    def __init__(
        self, m: float, s: float = 1.0, is_half: bool = True, p: float = 1.0,
        fn: Callable = None
    ):

        super().__init__(base_edge_layer=CompleteEdge(), p=p, fn=fn)

        self.m = m
        self.s = s

        if is_half:
            self.model = halfnorm
        else:
            self.model = norm

    def forward(self, graph: Protein) -> Tuple[Tensor, int]:

        base_graph_edge_list, i = self.base_edge_layer(graph)

        distances = get_edge_distances(base_graph_edge_list, graph)
        weights = self.model.pdf(distances.cpu(), loc=self.m, scale=self.s)

        edge_list = self.sample(base_graph_edge_list, weights=weights)
        edge_list = add_reverse_edges(edge_list)

        return torch.cat(
            (
                edge_list[:, :2],
                torch.zeros(
                    len(edge_list), device=edge_list.device
                ).t().unsqueeze(dim=1)
            ), dim=1
        ).long(), i


# ---------------------------- utility functions -------------------------------


def to_res_edges(edge_list: Tensor, protein: Protein) -> Tensor:

    res_edge_list = torch.stack(
        [
            protein.atom2residue[edge_list[:, 0]],
            protein.atom2residue[edge_list[:, 1]],
            edge_list[:, 2]
        ]
    ).t()

    return res_edge_list


def remove_duplicates(edge_list: Tensor) -> Tensor:

    return torch.unique(edge_list, dim=0)


def remove_self_loops(edge_list: Tensor) -> Tensor:

    return edge_list[edge_list[:, 0] != edge_list[:, 1]]


def add_reverse_edges(edge_list: Tensor) -> Tensor:

    return torch.cat(
        (
            edge_list,
            torch.cat(
                (
                    torch.flip(edge_list[:, :2], dims=[1]),
                    edge_list[:, 2].unsqueeze(dim=1)
                ), dim=1
            )
        )
    )


def get_edge_distances(edge_list: Tensor, graph: Protein) -> Tensor:

    dist_mat = torch.cdist(graph.node_position, graph.node_position)

    return dist_mat[edge_list[:, 0], edge_list[:, 1]]


# -------------------------------- obsolete ------------------------------------


@R.register("layers.geometry.HydrogenBondEdge")
class HydrogenBondEdge(nn.Module, Configurable):

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

    def forward(self, graph: Protein) -> Tuple[torch.Tensor, int]:

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


@R.register("layers.geometry.DisulfideBridgeEdge")
class DisulfideBridgeEdge(nn.Module, Configurable):

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

    def forward(self, graph: Protein) -> Tuple[torch.Tensor, int]:

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
