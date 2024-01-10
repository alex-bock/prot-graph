
from dataclasses import dataclass
import itertools

from Bio.PDB.Structure import Structure
from networkx import MultiGraph
import numpy as np
from plotly.colors import sample_colorscale
import plotly.graph_objects as go
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import minmax_scale
from scipy.spatial.distance import euclidean


C_ALPHA = "CA"
DEFAULT_NODE_RADIUS = 10.0
DEFAULT_K = 10


@dataclass
class Residue:

    res_id: str
    chain: str
    chain_i: int
    aa_id: str
    pos: np.ndarray


@dataclass
class Edge:

    type: str
    weight: float


class ProtGraph:

    def __init__(self, pdb_struct: Structure):

        self.pdb_id = pdb_struct.get_id()
        self.graph = MultiGraph()
        self.add_nodes(pdb_struct)

        return

    @property
    def __len__(self):

        return len(self.graph)

    def add_nodes(self, pdb_struct: Structure):

        i = 0
        for chain in pdb_struct.get_chains():
            chain_id = chain.get_id().capitalize()
            for chain_i, res in enumerate(chain.get_residues()):
                c_alpha = next(
                    (
                        atom for atom in res.get_atoms()
                        if atom.get_name() == C_ALPHA
                    ),
                    None
                )
                if c_alpha is None:
                    continue
                self.graph.add_node(
                    i,
                    data=Residue(
                        res_id=chain_id + str(chain_i),
                        chain=chain_id,
                        chain_i=chain_i,
                        aa_id=res.get_resname(),
                        pos=c_alpha.get_coord()
                    )
                )
                i += 1

        print(f"Added {len(self.graph)} residue nodes")

        return

    def add_sequence_edges(self):

        n_edges = 0
        for (u, v) in itertools.combinations(self.graph.nodes(data=True), 2):
            if u[1]["data"].chain == v[1]["data"].chain and abs(
                u[1]["data"].chain_i - v[1]["data"].chain_i
               ) == 1:
                self.graph.add_edge(
                    u[0], v[0], data=Edge(type="seq", weight=None)
                )
                n_edges += 1

        print(f"Added {n_edges} sequence edges")

        return

    def add_radius_edges(self, r: float, seq_gap: int = 0):

        n_edges = 0
        for (u, v) in itertools.combinations(self.graph.nodes(data=True), 2):
            if u[1]["data"].chain == v[1]["data"].chain and abs(
                u[1]["data"].chain_i - v[1]["data"].chain_i
               ) < seq_gap:
                continue
            elif euclidean(u[1]["data"].pos, v[1]["data"].pos) <= r:
                self.graph.add_edge(
                    u[0], v[0], data=Edge(type="radius", weight=None)
                )
                n_edges += 1

        print(f"Added {n_edges} radius edges")

        return

    def add_knn_edges(self, k: int, seq_gap: int = 0):

        k_adj_mat = kneighbors_graph(
            [n[1]["data"].pos for n in self.graph.nodes(data=True)],
            k,
            mode="distance",
            metric="euclidean"
        )

        n_edges = 0
        for (u, v) in itertools.combinations(self.graph.nodes(data=True), 2):
            if k_adj_mat[u[0], v[0]] > 0:
                self.graph.add_edge(
                    u[0], v[0], data=Edge(type="knn", weight=None)
                )
                n_edges += 1

        print(f"Added {n_edges} knn edges")

        return

    def visualize(
        self, color_node_by: str = "chain", color_edge_by: str = "type"
    ):

        fig = go.Figure()

        self._plot_nodes(fig, color_node_by)
        self._draw_edges(fig, color_edge_by)

        fig.show()

        return

    def _plot_nodes(self, fig: go.Figure, color_field: str):

        field_vals = set([
            getattr(node[1]["data"], color_field)
            for node in self.graph.nodes(data=True)
        ])
        color_scale = sample_colorscale(
            "viridis", minmax_scale(range(len(field_vals)))
        )
        val_color_map = {
            val: color_scale[i] for i, val in enumerate(field_vals)
        }

        for val in field_vals:
            fig.add_trace(
                go.Scatter3d(
                    x=[
                        node[1]["data"].pos[0]
                        for node in self.graph.nodes(data=True)
                        if getattr(node[1]["data"], color_field) == val
                    ],
                    y=[
                        node[1]["data"].pos[1]
                        for node in self.graph.nodes(data=True)
                        if getattr(node[1]["data"], color_field) == val
                    ],
                    z=[
                        node[1]["data"].pos[2]
                        for node in self.graph.nodes(data=True)
                        if getattr(node[1]["data"], color_field) == val
                    ],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=3,
                        color=val_color_map[val]
                    ),
                    text=[
                        node[1]["data"].res_id
                        for node in self.graph.nodes(data=True)
                    ],
                    hoverinfo="text",
                    name=f"{color_field}: {val}",
                    legend="legend1"
                )
            )

        return

    def _draw_edges(self, fig: go.Figure, color_field: str):

        field_vals = set([
            getattr(edge[2]["data"], color_field)
            for edge in self.graph.edges(data=True)
        ])
        val_color_map = {val: i for i, val in enumerate(field_vals)}

        for val in field_vals:
            fig.add_trace(
                go.Scatter3d(
                    x=[
                        c for edge in self.graph.edges(data=True)
                        for c in [
                            self.graph.nodes[edge[0]]["data"].pos[0],
                            self.graph.nodes[edge[1]]["data"].pos[0],
                            None
                        ]
                        if getattr(edge[2]["data"], color_field) == val
                    ],
                    y=[
                        c for edge in self.graph.edges(data=True)
                        for c in [
                            self.graph.nodes[edge[0]]["data"].pos[1],
                            self.graph.nodes[edge[1]]["data"].pos[1],
                            None
                        ]
                        if getattr(edge[2]["data"], color_field) == val
                    ],
                    z=[
                        c for edge in self.graph.edges(data=True)
                        for c in [
                            self.graph.nodes[edge[0]]["data"].pos[2],
                            self.graph.nodes[edge[1]]["data"].pos[2],
                            None
                        ]
                        if getattr(edge[2]["data"], color_field) == val
                    ],
                    mode="lines",
                    line=dict(color=val_color_map[val]),
                    name=val,
                    hoverinfo="text",
                    legend="legend2",
                    opacity=0.5
                )
            )

        return
