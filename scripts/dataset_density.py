
import os
import sys
from tqdm.contrib.concurrent import process_map

sys.path.append(os.getcwd())

import math
import plotly.graph_objects as go
from torch import Tensor

from torchdrug.data import Protein
from torchdrug.datasets import EnzymeCommission
from torchdrug.transforms import ProteinView
from torchdrug.layers.geometry import *

from prot_graph.torchdrug.layers.graph.edge import *
from prot_graph.torchdrug.layers.graph.graph import BondNetworkConstruction


baseline_edges = [
    CompleteEdge(), SampleEdge(CompleteEdge(), fn=lambda n: math.sqrt(n)),
    SampleEdge(CompleteEdge(), fn=lambda n: math.sqrt(n), p=3.2),
    SpatialEdge(radius=10.0, min_distance=0, max_num_neighbors=int(1e10))
]
baseline_edge_names = [
    "Complete", "Complete (root(n))", "Complete (h-bond approx.)",
    f"Spatial ({10.0} Å)"
]
topline_edges = [
    PeptideBondEdge(), GetContactsEdge("hb"), GetContactsEdge("hp"),
    GetContactsEdge("vdw"), GetContactsEdge("sb"), GetContactsEdge("pc"),
    GetContactsEdge("ps"), GetContactsEdge("ts")
]
topline_edge_names = [
    "Peptide bonds", "Hydrogen bonds", "Hydrophobic interactions",
    "Van der Waals forces", "Salt bridges", "π-cation bonds", "π-stacking",
    "t-stacking"
]


def count_edges(protein: Protein):

    constructor = BondNetworkConstruction(
        node_layers=[AlphaCarbonNode()],
        edge_layers=baseline_edges + topline_edges
    )
    graph = constructor(Protein.pack([protein]))[0]
    n_nodes = len(graph.node2graph)
    edge_types = graph.edge_list[:, 2]

    return [n_nodes] + [
        len(edge_types[edge_types == i])
        for i in range(len(baseline_edges) + len(topline_edges))
    ]


if __name__ == "__main__":

    dataset_path = sys.argv[1]
    dataset = EnzymeCommission(
        dataset_path, transform=ProteinView(view="residue")
    )

    X = Tensor(
        process_map(count_edges, [protein["graph"] for protein in dataset])
    )

    fig = go.Figure()
    x = X[:, 0].to(int)
    for i in range(len(baseline_edges) + len(topline_edges)):
        fig.add_trace(
            go.Scatter(
                x=x, y=X[:, i + 1], mode="markers",
                name=(baseline_edge_names + topline_edge_names)[i]
            )
        )
    fig.update_layout(
        title="Edge count by number of residues", xaxis_title="# residues",
        yaxis_title="# edges"
    )
    fig.show()
