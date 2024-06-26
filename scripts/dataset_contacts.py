
import glob
import os
import sys
from tqdm.contrib.concurrent import process_map

sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from torchdrug.data import Protein
from torchdrug.data.util import load_contacts, CONTACT2ID
from torchdrug.layers.geometry import AlphaCarbonNode

from prot_graph.torchdrug.layers.graph.edge import *
from prot_graph.torchdrug.layers.graph.graph import BondNetworkConstruction


ID2ATOM_CONTACT = {v: k for k, v in CONTACT2ID.items()}
ID2RES_CONTACT = {
    i: c for i, c in enumerate(
        ["pep", "hb", "sb", "hp", "pc", "ts", "ps", "vdw"]
    )
}


def count_atom_contacts(pdb_fp: str, contact_fp: str) -> pd.Series:

    try:
        protein = Protein.from_pdb(pdb_fp)
        protein = load_contacts(protein, contact_fp)
    except:
        return None

    edge_types = protein.edge_list[:, 2]
    contact_edges = protein.edge_list[edge_types >= 4]

    counts = pd.Series(contact_edges[:, 2]).apply(
        lambda x: ID2ATOM_CONTACT[x]
    ).value_counts().to_dict()
    counts["n_node"] = len(protein.atom2residue)

    return counts


def count_residue_contacts(pdb_fp: str, contact_fp: str) -> pd.Series:

    try:
        protein = Protein.from_pdb(pdb_fp)
        protein = load_contacts(protein, contact_fp)
    except:
        return None

    pack = Protein.pack([protein])
    graph_constructor = BondNetworkConstruction(
        node_layers=[AlphaCarbonNode()],
        edge_layers=[
            PeptideBondEdge(),
            GetContactsEdge("hb"),
            GetContactsEdge("sb"),
            GetContactsEdge("hp"),
            GetContactsEdge("pc"),
            GetContactsEdge("ts"),
            GetContactsEdge("ps"),
            GetContactsEdge("vdw")
        ]
    )
    graph_pack = graph_constructor(pack)
    graph = graph_pack[0]

    counts = pd.Series(
        torch.unique(graph.edge_list, dim=0)[:, 2]
    ).apply(lambda x: ID2RES_CONTACT[x]).value_counts().to_dict()
    counts["n_node"] = len(graph.residue_type)

    return counts


if __name__ == "__main__":

    dataset_path = sys.argv[1]

    pdb_fps = []
    contact_fps = []
    for split in ("train", "test", "valid"):
        split_pdb_fps = glob.glob(os.path.join(dataset_path, split, "*.pdb"))
        pdb_fps.extend(split_pdb_fps)
        split_pdb_fns = [os.path.basename(fp) for fp in split_pdb_fps]
        split_contact_fps = [
            os.path.join(dataset_path, "contacts", fn)
            for fn in split_pdb_fns
        ]
        contact_fps.extend(split_contact_fps)

    contact_counts = process_map(count_residue_contacts, pdb_fps, contact_fps)
    contact_counts = [x for x in contact_counts if x is not None]
    contact_counts = pd.DataFrame(contact_counts)
    contact_counts.replace(np.nan, 0, inplace=True)

    zeros = contact_counts.apply(
        lambda col: 1 - np.count_nonzero(col.values) / len(col)
    )
    print(zeros)

    fig = go.Figure()
    for contact in contact_counts.columns:
        if contact == "n_node":
            continue
        fig.add_trace(
            go.Histogram(
                x=contact_counts[contact].values, name=contact,
                xbins=dict(size=1)
            )
        )

    fig.update_layout(barmode="stack")
    fig.show()
