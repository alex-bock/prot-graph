
import os
import sys

from prot_graph.datasets import PDB
from prot_graph.graphs import ResGraph


if __name__ == "__main__":

    sys.path.append(os.getcwd())
    pdb_db = PDB(pdb_dir=sys.argv[1])

    for pdb_id in pdb_db[:10]:
        struct = pdb_db.load_structure(pdb_id)
        graph = ResGraph(struct)
        graph.add_peptide_bonds()
        graph.add_hydrogen_bonds()
        graph.visualize(color_node_by="type")
