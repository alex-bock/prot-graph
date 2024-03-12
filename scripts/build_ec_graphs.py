
import functools
import os
import sys
from tqdm.contrib.concurrent import process_map

sys.path.append(os.getcwd())

from prot_graph.datasets import PDB
from prot_graph.graphs import ResGraph


def build_graph(pdb_db: PDB, pdb_id: str) -> ResGraph:

    try:
        struct = pdb_db.load_structure(pdb_id)
        graph = ResGraph(struct)
        graph.add_peptide_bonds()
        graph.add_hydrogen_bonds()
    except:
        graph = None

    return graph


if __name__ == "__main__":

    pdb_db = PDB(pdb_dir=sys.argv[1])
    # pdb_db.load_metadata(sys.argv[2])
    figs = []

    graphs = process_map(functools.partial(build_graph, pdb_db), pdb_db.ids)
    # process_map(
    #     functools.partial(build_graph, pdb_db),
    #     pdb_db.filter_by_metadata("ec", "2.7.10.-")
    # )
