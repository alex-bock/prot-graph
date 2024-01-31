
import sys

import networkx as nx

from prot_graph.datasets import PDB
from prot_graph.prot_graph import ProtGraph

if __name__ == "__main__":

    pdb_id = sys.argv[1]
    prot_db = PDB()
    print("Downloading...")
    prot_db.download_pdb_record(pdb_id)
    print("Loading...")
    pdb_struct = prot_db.load_pdb_structure(pdb_id)

    print("Building graph...")
    graph = ProtGraph(pdb_struct)
    graph.add_sequence_edges()
    graph.add_radius_edges(r=8.0)
    graph.add_hydrogen_bonds()
    graph.add_hydrophobic_interactions()
    graph.add_ionic_bonds()
    graph.add_salt_bridges()
    graph.add_disulfide_bridges()
    graph.add_pi_cation_bonds()
    graph.visualize(color_residue_by="chain", hide_residues=True)

    for cc in nx.connected_components(graph.graph):
        print(len(cc))
