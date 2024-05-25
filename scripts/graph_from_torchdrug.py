
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
    print(protein)

    node_layer = AlphaCarbonNode()

    for p in range(5):
        edge_layers = [
            MSTEdge(
                base_edge_layer=SpatialEdge(radius=8, min_distance=0),
                p=(p / 5)
            )
        ]
        graph_constructor = GraphConstruction(
            node_layers=[node_layer], edge_layers=edge_layers
        )
        protein = graph_constructor(pack)[0]
        ProtGraph(protein).visualize(hide_nodes=True)

    for radius in range(5, 10):
        edge_layers = [
            MSTEdge(
                base_edge_layer=SpatialEdge(radius=radius, min_distance=0),
                p=0.0
            )
        ]
        graph_constructor = GraphConstruction(
            node_layers=[node_layer], edge_layers=edge_layers
        )
        protein = graph_constructor(pack)[0]
        ProtGraph(protein).visualize(hide_nodes=True)