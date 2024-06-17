
import os
import sys

sys.path.append(os.getcwd())

from torchdrug.data import PackedProtein
from torchdrug.layers.geometry import AlphaCarbonNode

from prot_graph.torchdrug.layers.graph.edge import *
from prot_graph.torchdrug.layers.graph.graph import BondNetworkConstruction

from prot_graph.util import load_contacts, visualize


if __name__ == "__main__":

    pdb_fp = sys.argv[1]
    pack = PackedProtein.from_pdb([pdb_fp])
    protein = pack[0]

    contacts_fp = os.path.join(
        os.path.split(pdb_fp)[0],
        os.path.basename(pdb_fp).replace(".pdb", ".tsv")
    )
    protein = load_contacts(protein, contacts_fp)
    visualize(protein, color_node_by="atom_name")
