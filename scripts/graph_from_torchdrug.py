
import os
import sys

sys.path.append(os.getcwd())

from torchdrug.data import PackedProtein
from torchdrug.layers.geometry import AlphaCarbonNode, IdentityNode

from prot_graph.torchdrug.layers.graph.edge import *
from prot_graph.torchdrug.layers.graph.graph import BondNetworkConstruction

from prot_graph.graphs import ProtGraph


if __name__ == "__main__":

    pdb_fp = sys.argv[1]
    pack = PackedProtein.from_pdb([pdb_fp])
    protein = pack[0]
    print(protein)

    node_layer = AlphaCarbonNode()
    edge_layers = [PeptideBondEdge(), HydrogenBondEdge(), DisulfideBridgeEdge()]
    graph_constructor = BondNetworkConstruction(
        node_layers=[node_layer], edge_layers=edge_layers
    )
    print(graph_constructor)
    protein = graph_constructor(pack)[0]
    print(protein)

    graph = ProtGraph(protein)
    graph.visualize(color_node_by="residue_type")