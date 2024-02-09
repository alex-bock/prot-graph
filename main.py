
import sys

from prot_graph.datasets import PDB
from prot_graph.structures import PDBStructure
from prot_graph.graphs import ResGraph


if __name__ == "__main__":

    pdb_id = sys.argv[1]
    
    pdb = PDB()
    pdb.download_record(pdb_id)

    pdb_struct = PDBStructure(pdb_id, pdb)
    res_graph = ResGraph(pdb_struct)
    res_graph.add_peptide_bonds()
    res_graph.add_hydrogen_bonds()
    res_graph.visualize(color_node_by="chain")

    # print(res_graph.struct.get_atom_pairs(5.0, types=["NH1", "NH2", "NZ", "OD1", "OD2", "OE1", "OE2"]))
