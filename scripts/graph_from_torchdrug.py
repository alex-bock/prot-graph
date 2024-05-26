
import os
import sys

sys.path.append(os.getcwd())

import networkx as nx

import torch
from torch import Tensor

from torchdrug.data import PackedProtein, Graph
from torchdrug.layers.geometry import AlphaCarbonNode, IdentityNode
from torchdrug.layers.geometry import SpatialEdge
from torchdrug.layers import GraphConstruction

from prot_graph.torchdrug.layers.graph.edge import *
from prot_graph.torchdrug.layers.graph.graph import BondNetworkConstruction

from prot_graph.graphs import ProtGraph


if __name__ == "__main__":

    pdb_fp = sys.argv[1]
    pack = PackedProtein.from_pdb([pdb_fp])
    protein = pack[0]

    node_layer = AlphaCarbonNode()

    for radius in range(5, 10):
        edge_layers = [
            SpatialEdge(radius=radius, min_distance=0),
            MSTEdge(
                base_edge_layer=SpatialEdge(radius=radius, min_distance=0),
                p=0.0
            ),
            HydrogenBondEdge()
        ]
        bond_net_constructor = BondNetworkConstruction(
            node_layers=[node_layer], edge_layers=edge_layers
        )
        bond_net = bond_net_constructor(pack)[0]
        import pdb; pdb.set_trace()
        ProtGraph(bond_net).visualize(hide_nodes=True)