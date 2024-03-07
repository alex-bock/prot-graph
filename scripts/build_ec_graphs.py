
import os
import sys
import tqdm

sys.path.append(os.getcwd())

from prot_graph.datasets import PDB
from prot_graph.graphs import ResGraph


if __name__ == "__main__":

    pdb_db = PDB(pdb_dir=sys.argv[1])
    pdb_db.load_metadata(sys.argv[2])
    figs = []

    for pdb_id in tqdm.tqdm(pdb_db.filter_by_metadata("ec", "2.7.10.-")):
        try:
            struct = pdb_db.load_structure(pdb_id)
            graph = ResGraph(struct)
            graph.add_peptide_bonds()
            graph.add_hydrogen_bonds()
            figs.append(graph.visualize(color_node_by="type"))
        except:
            continue
