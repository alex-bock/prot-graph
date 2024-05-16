
import os
import sys

sys.path.append(os.getcwd())

from torchdrug.data import PackedProtein
from torchdrug.layers.geometry import AlphaCarbonNode
from torchdrug.layers.geometry import SequentialEdge

from prot_graph.torchdrug.layers.graph.edge import *
from prot_graph.torchdrug.layers.graph.graph import BondNetworkConstruction

from prot_graph.datasets import PDB

from prot_graph.graphs import ResGraph


if __name__ == "__main__":

    pdb_fp = sys.argv[1]
    protein = PackedProtein.from_pdb([pdb_fp])
    print(protein)

    node_layer = AlphaCarbonNode()
    edge_layers = [PeptideBondEdge()]
    graph_constructor = BondNetworkConstruction(
        node_layers=[node_layer], edge_layers=edge_layers
    )
    print(graph_constructor)

    graph = graph_constructor(protein)
    print(graph[0])

    pdb_db = PDB("/".join(pdb_fp.split("/")[:-1]))
    pdb_struct = pdb_db.load_structure(pdb_fp.split("/")[-1].split(".")[0])
    res_graph_0 = ResGraph(pdb_struct)
    res_graph_0.add_peptide_bonds()
    print(res_graph_0.edge_df)
