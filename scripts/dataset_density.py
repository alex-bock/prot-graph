
import os
import sys
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

sys.path.append(os.getcwd())

import math
import plotly.graph_objects as go
from torch import Tensor

from torchdrug.data import Protein
from torchdrug.datasets import EnzymeCommission
from torchdrug.transforms import ProteinView
from torchdrug.layers.geometry import AlphaCarbonNode
from torchdrug.layers.geometry import SpatialEdge

from prot_graph.torchdrug.layers.graph.edge import (
    CompleteEdge, SampleEdge, PeptideBondEdge, GetContactsEdge, DelaunayEdge
)
from prot_graph.torchdrug.layers.graph.graph import BondNetworkConstruction


layers = [
    DelaunayEdge(),
    CompleteEdge(),
    SampleEdge(CompleteEdge(), fn=lambda n: math.sqrt(n), p=6.8),
    SpatialEdge(radius=10.0, min_distance=0, max_num_neighbors=int(1e10)),
    PeptideBondEdge(),
    GetContactsEdge("hb"),
    GetContactsEdge("hp"),
    GetContactsEdge("vdw"),
    GetContactsEdge("sb"),
    GetContactsEdge("pc"),
    GetContactsEdge("ps"),
    GetContactsEdge("ts")
]
layer_names = [
    "Delaunay",
    "Complete",
    "Complete (h-bond approx.)",
    "Spatial (r=10 Å)",
    "Peptide bonds",
    "Hydrogen bonds",
    "Hydrophobic interactions",
    "Van der Waals forces",
    "Salt bridges",
    "π-cation bonds",
    "π-stacking",
    "t-stacking"
]


def count_edges(protein: Protein):

    constructor = BondNetworkConstruction(
        node_layers=[AlphaCarbonNode()], edge_layers=layers
    )
    graph = constructor(Protein.pack([protein]))[0]
    n_nodes = len(graph.node2graph)
    edge_types = graph.edge_list[:, 2]

    return [n_nodes] + [
        len(edge_types[edge_types == i]) for i in range(len(layers))
    ]


if __name__ == "__main__":

    dataset_path = sys.argv[1]
    dataset = EnzymeCommission(
        dataset_path, transform=ProteinView(view="residue")
    )

    X = Tensor(
        [count_edges(protein["graph"]) for protein in tqdm(dataset)]
    ).to(int)
    print(X)

    scatter_plot = go.Figure()
    x = X[:, 0]
    for i in range(len(layers)):
        scatter_plot.add_trace(
            go.Scatter(
                x=x, y=X[:, i + 1], mode="markers", name=(layer_names)[i]
            )
        )
    scatter_plot.update_layout(
        title="Edge count by number of residues", xaxis_title="# residues",
        yaxis_title="# edges"
    )
    scatter_plot.show()

    histogram = go.Figure()
    for i in range(len(layers)):
        histogram.add_trace(
            go.Histogram(
                x=X[:, i + 1], name=layer_names[i], xbins=dict(size=50)
            )
        )

    histogram.add_trace(
        go.Histogram(x=X[:, 0], name="Residue count", xbins=dict(size=50))
    )
    histogram.update_layout(title="Edge count distribution", barmode="stack")
    histogram.show()
