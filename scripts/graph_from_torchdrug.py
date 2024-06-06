
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

    protein = load_contacts(protein, "./data/sandbox/fixed/1708455578/6JXR.tsv")
    visualize(protein, hide_nodes=True)
