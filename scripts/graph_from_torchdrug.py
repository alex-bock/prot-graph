
import os
import sys

sys.path.append(os.getcwd())

from torchdrug.data import Protein, PackedProtein
from torchdrug.layers.geometry import AlphaCarbonNode

from prot_graph.torchdrug.layers.graph.edge import *
from prot_graph.torchdrug.layers.graph.graph import BondNetworkConstruction

from prot_graph.util import load_contacts, visualize


if __name__ == "__main__":

    pdb_fp = sys.argv[1]
    pack = PackedProtein.from_pdb([pdb_fp])
    protein = pack[0]

    contacts_fp = sys.argv[2]
    protein = load_contacts(protein, contacts_fp)
    visualize(protein, color_node_by="atom_name")

    pack = Protein.pack([protein])
    graph_constructor = BondNetworkConstruction(
        node_layers=[AlphaCarbonNode()],
        edge_layers=[
            PeptideBondEdge(), GetContactsEdge("hb"), GetContactsEdge("sb"),
            GetContactsEdge("hp"), GetContactsEdge("pc"),
            GetContactsEdge("ts"), GetContactsEdge("ps"), GetContactsEdge("vdw")
        ]
    )
    graph_pack = graph_constructor(pack)
    graph = graph_pack[0]
    visualize(graph)
