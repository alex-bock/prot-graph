
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
    edge_layers = [
        MSTEdge(
            base_edge_layer=SpatialEdge(radius=10.0, min_distance=0),
            p=0.0
        )
    ]
    edge_layers = [SpatialEdge(radius=10.0, min_distance=0)]

    for construction in [BondNetworkConstruction, GraphConstruction]:
        constructor = construction(node_layers=[node_layer], edge_layers=edge_layers)
        network = constructor(pack)[0]
        ProtGraph(network).visualize(hide_nodes=False)
