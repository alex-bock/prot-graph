
from dataclasses import dataclass
from typing import Dict, List

from Bio.PDB.Structure import Structure
import numpy as np
from plotly.express.colors import sample_colorscale
import plotly.graph_objects as go
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph
from sklearn.preprocessing import minmax_scale


C_ALPHA = "CA"
DEFAULT_NODE_RADIUS = 10.0
DEFAULT_K = 10


@dataclass
class Residue:

    id: int
    chain: str
    chain_i: int
    aa_id: str
    pos: np.ndarray


@dataclass
class Edge:

    u: str
    v: str
    weight: float
    type: str


class ProtGraph:

    def __init__(
        self,
        pdb_struct: Structure,
        radius: float = 0.0,
        k: int = 0,
        seq: bool = False
    ):

        self.pdb_id = pdb_struct.get_id()
        self.nodes, self.edges = ProtGraph.build_graph(
            pdb_struct, radius=radius, k=k, seq=seq
        )
        self.n = len(self.nodes.keys())

        return

    @staticmethod
    def build_graph(
        pdb_struct: Structure,
        radius: float = 0.0,
        k: int = 0,
        seq: bool = False
    ):

        nodes = ProtGraph.populate_nodes(pdb_struct)
        edges = ProtGraph.add_edges(nodes, radius=radius, k=k, seq=seq)

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
                    chain_i=i,
                    aa_id=res.get_resname(),
                    pos=c_alpha.get_coord()
                )

        return nodes

    @staticmethod
    def add_edges(
        nodes: Dict[str, Residue],
        radius: float = 0.0,
        k: int = 0,
        seq: bool = False
    ) -> List[Edge]:

        res_list = list(nodes.values())
        edges = list()

        radius_adj_mat = radius_neighbors_graph(
            [res.pos for res in res_list],
            radius,
            mode="distance",
            metric="euclidean"
        )
        if k > 0:
            k_adj_mat = kneighbors_graph(
                [res.pos for res in res_list],
                k,
                mode="distance",
                metric="euclidean"
            )
        else:
            k_adj_mat = np.zeros((len(res_list), len(res_list)))

        for i, n1 in enumerate(res_list):
            for j, n2 in enumerate(res_list[(i + 1):]):
                radius_dist = radius_adj_mat[i, (i + 1) + j]
                if radius_dist > 0:
                    edges.append(
                        Edge(
                            u=n1.id, v=n2.id, weight=radius_dist, type="radius"
                        )
                    )
                k_dist = k_adj_mat[i, (i + 1) + j]
                if k_dist > 0:
                    edges.append(
                        Edge(u=n1.id, v=n2.id, weight=k_dist, type="k")
                    )
                if seq:
                    if n1.chain == n2.chain and n1.chain_i == n2.chain_i - 1:
                        edges.append(
                            Edge(u=n1.id, v=n2.id, weight=1, type="seq")
                        )

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

    def visualize(self, node_color: str = "chain", edge_color: str = "type"):

        fig = go.Figure()

        node_vals = set(
            [getattr(res, node_color) for res in self.nodes.values()]
        )
        node_color_scale = sample_colorscale(
            "viridis", minmax_scale(range(len(node_vals)))
        )
        val_colors = {
            val: node_color_scale[i] for i, val in enumerate(node_vals)
        }

        for val in node_vals:
            fig.add_trace(
                go.Scatter3d(
                    x=[
                        res.pos[0] for res in self.nodes.values()
                        if getattr(res, node_color) == val
                    ],
                    y=[
                        res.pos[1] for res in self.nodes.values()
                        if getattr(res, node_color) == val
                    ],
                    z=[
                        res.pos[2] for res in self.nodes.values()
                        if getattr(res, node_color) == val
                    ],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=3,
                        color=val_colors[val]
                    ),
                    text=[res.id for res in self.nodes.values()],
                    hoverinfo="text",
                    name=f"{node_color}: {val}",
                    legend="legend1"
                )
            )

        edge_vals = set([getattr(edge, edge_color) for edge in self.edges])
        edge_colors = {val: i for i, val in enumerate(edge_vals)}
        for val in edge_vals:
            fig.add_trace(
                go.Scatter3d(
                    x=[
                        c for edge in self.edges
                        for c in [
                            self.nodes[edge.u].pos[0],
                            self.nodes[edge.v].pos[0],
                            None
                        ]
                        if getattr(edge, edge_color) == val
                    ],
                    y=[
                        c for edge in self.edges
                        for c in [
                            self.nodes[edge.u].pos[1],
                            self.nodes[edge.v].pos[1],
                            None
                        ]
                        if getattr(edge, edge_color) == val
                    ],
                    z=[
                        c for edge in self.edges
                        for c in [
                            self.nodes[edge.u].pos[2],
                            self.nodes[edge.v].pos[2],
                            None
                        ]
                        if getattr(edge, edge_color) == val
                    ],
                    mode="lines",
                    line=dict(color=edge_colors[val]),
                    text=[str(edge.weight) for edge in self.edges],
                    name=val,
                    hoverinfo="text",
                    legend="legend2",
                    opacity=0.5
                )
            )

        fig.show()

        return
