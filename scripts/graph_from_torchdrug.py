
import os
import sys

sys.path.append(os.getcwd())

from torchdrug.data import Protein, PackedProtein
from torchdrug.data.util import load_contacts
from torchdrug.layers.geometry import AlphaCarbonNode
from torchdrug.layers.geometry import SpatialEdge

from prot_graph.layers.graph.edge import PeptideBondEdge, GetContactsEdge
from prot_graph.layers.graph.graph import BondNetworkConstruction

from prot_graph.util import visualize


if __name__ == "__main__":

    pdb_fp = sys.argv[1]
    pack = PackedProtein.from_pdb([pdb_fp])
    protein = pack[0]

    contacts_fp = sys.argv[2]
    protein = load_contacts(protein, contacts_fp)
    visualize(protein, color_node_by="atom_type")

    pack = Protein.pack([protein])
    graph_constructor = BondNetworkConstruction(
        node_layers=[AlphaCarbonNode()],
        edge_layers=[
            SpatialEdge(
                radius=10.0, min_distance=0, max_num_neighbors=int(1e10)
            ),
            PeptideBondEdge(),
            GetContactsEdge("hb"),
            GetContactsEdge("hp"),
            GetContactsEdge("vdw"),
            GetContactsEdge("sb"),
            GetContactsEdge("pc"),
            GetContactsEdge("ps"),
            GetContactsEdge("ts")
        ]
    )
    graph_pack = graph_constructor(pack)
    graph = graph_pack[0]
    visualize(graph, color_node_by="residue_type")
