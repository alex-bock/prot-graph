
from dataclasses import dataclass
from typing import Dict, List

from Bio.PDB.Structure import Structure
import numpy as np
from plotly.express.colors import sample_colorscale
import plotly.graph_objects as go
from sklearn.neighbors import radius_neighbors_graph
from sklearn.preprocessing import minmax_scale


C_ALPHA = "CA"
DEFAULT_NODE_RADIUS = 10.0


@dataclass
class Residue:

    id: int
    chain: str
    aa_id: str
    pos: np.ndarray


@dataclass
class Edge:

    u: str
    v: str
    weight: str


class ProtGraph:

    def __init__(
        self,
        pdb_struct: Structure,
        radius: float = DEFAULT_NODE_RADIUS
    ):

        self.pdb_id = pdb_struct.get_id()
        self.nodes, self.edges = ProtGraph.build_graph(pdb_struct, radius)
        self.n = len(self.nodes.keys())

        return

    @staticmethod
    def build_graph(pdb_struct: Structure, radius: float):

        nodes = ProtGraph.populate_nodes(pdb_struct)
        edges = ProtGraph.create_edges(nodes, radius)

        return nodes, edges

    @staticmethod
    def populate_nodes(pdb_struct: Structure) -> Dict[str, Residue]:

        nodes = dict()
        for chain in pdb_struct.get_chains():
            chain_id = chain.get_id().capitalize()
            for i, res in enumerate(chain.get_residues()):
                res_id = chain_id + str(i)
                c_alpha = next(
                    (
                        atom for atom in res.get_atoms()
                        if atom.get_name() == C_ALPHA
                    ),
                    None
                )
                if c_alpha is None:
                    continue
                nodes[res_id] = Residue(
                    id=res_id,
                    chain=chain_id,
                    aa_id=res.get_resname(),
                    pos=c_alpha.get_coord()
                )

        return nodes

    @staticmethod
    def create_edges(nodes: Dict[str, Residue], radius: float) -> List[Edge]:

        res_list = list(nodes.values())
        adj_mat = radius_neighbors_graph(
            [res.pos for res in res_list],
            radius,
            mode="distance",
            metric="euclidean"
        )

        edges = list()
        for i, n1 in enumerate(res_list):
            for j, n2 in enumerate(res_list[(i + 1):]):
                dist = adj_mat[i, (i + 1) + j]
                if dist > 0:
                    edges.append(Edge(u=n1.id, v=n2.id, weight=dist))

        return edges

    @property
    def __len__(self) -> int:

        return self.n

    def get_neighbors(self, res_id: str) -> List[str]:

        neighbors = list()
        for edge in self.edges:
            if edge.u == res_id:
                neighbors.append(edge.v)
            elif edge.v == res_id:
                neighbors.append(edge.u)

        return neighbors

    def visualize(self, color_field: str = "chain"):

        fig = go.Figure()

        vals = set([getattr(res, color_field) for res in self.nodes.values()])
        color_scale = sample_colorscale(
            "viridis", minmax_scale(range(len(vals)))
        )
        val_colors = {val: color_scale[i] for i, val in enumerate(vals)}

        for val in vals:
            fig.add_trace(
                go.Scatter3d(
                    x=[
                        res.pos[0] for res in self.nodes.values()
                        if getattr(res, color_field) == val
                    ],
                    y=[
                        res.pos[1] for res in self.nodes.values()
                        if getattr(res, color_field) == val
                    ],
                    z=[
                        res.pos[2] for res in self.nodes.values()
                        if getattr(res, color_field) == val
                    ],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=6,
                        color=val_colors[val]
                    ),
                    text=[res.id for res in self.nodes.values()],
                    hoverinfo="text",
                    name=f"{color_field}: {val}",
                    legend="legend1"
                )
            )

        fig.add_trace(
            go.Scatter3d(
                x=[
                    c for edge in self.edges
                    for c in [
                        self.nodes[edge.u].pos[0],
                        self.nodes[edge.v].pos[0],
                        None
                    ]
                ],
                y=[
                    c for edge in self.edges
                    for c in [
                        self.nodes[edge.u].pos[1],
                        self.nodes[edge.v].pos[1],
                        None
                    ]
                ],
                z=[
                    c for edge in self.edges
                    for c in [
                        self.nodes[edge.u].pos[2],
                        self.nodes[edge.v].pos[2],
                        None
                    ]
                ],
                mode="lines",
                line=dict(color="grey"),
                text=[str(edge.weight) for edge in self.edges],
                hoverinfo="text",
                legend="legend2",
                opacity=0.5
            )
        )

        fig.show()

        return
