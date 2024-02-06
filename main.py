
import sys

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
    print(f" - {len(graph.connected_components)} components")

    graph.add_hydrophobic_interactions()
    print(f" - {len(graph.connected_components)} components")

    graph.add_hydrogen_bonds()
    print(f" - {len(graph.connected_components)} components")

    graph.add_ionic_bonds()
    print(f" - {len(graph.connected_components)} components")

    graph.add_pi_cation_bonds()
    print(f" - {len(graph.connected_components)} components")

    graph.add_salt_bridges()
    print(f" - {len(graph.connected_components)} components")

    graph.add_disulfide_bridges()
    print(f" - {len(graph.connected_components)} components")

    graph.visualize(color_residue_by="chain", hide_residues=False)
    
    print(graph.calculate_bond_overlap(show=True))
