
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
    edge_layers = [PeptideBondEdge(), SampleEdge(SpatialEdge(radius=10.0, min_distance=5))]

    for construction in [BondNetworkConstruction]:
        constructor = construction(
            node_layers=[node_layer], edge_layers=edge_layers
        )
        bond_net = constructor(pack)[0]
        ProtGraph(bond_net).visualize(hide_nodes=True)