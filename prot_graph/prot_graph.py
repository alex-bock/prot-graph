
from dataclasses import dataclass
import itertools

from Bio.PDB.Structure import Structure
from networkx import MultiGraph
import numpy as np
import pandas as pd
from plotly.colors import sample_colorscale
import plotly.graph_objects as go
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import minmax_scale
from scipy.spatial.distance import euclidean


C_ALPHA = "CA"


@dataclass
class Residue:

    id: str
    chain: str
    chain_i: int
    aa_id: str
    pos_x: float
    pos_y: float
    pos_z: float


@dataclass
class Edge:

    type: str
    weight: float


class ProtGraph:

    def __init__(self, pdb_struct: Structure):

        self.pdb_id = pdb_struct.get_id()
        self.graph = MultiGraph()
        self.res_df = self.add_residues(pdb_struct)
        print(self.res_df)
        self.edge_df = pd.DataFrame(columns=["u", "v", "type", "weight"])

        return

    @property
    def __len__(self):

        return len(self.res_df)

    def add_residues(self, pdb_struct: Structure):

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
                pos = c_alpha.get_coord()
                self.graph.add_node(
                    chain_id + str(chain_i),
                    **dict(
                        chain=chain_id,
                        chain_i=chain_i,
                        aa_id=res.get_resname(),
                        pos_x=pos[0],
                        pos_y=pos[1],
                        pos_z=pos[2]
                    )
                )
                i += 1

        print(f"Added {len(self.graph)} residues")

        return pd.DataFrame.from_dict(self.graph.nodes, orient="index")

    def add_sequence_edges(self):

        seq_edges = []
        for (res_u, res_v) in itertools.combinations(
            self.graph.nodes(data=True), 2
        ):
            if res_u[1]["chain"] == res_v[1]["chain"] and abs(
                res_u[1]["chain_i"] - res_v[1]["chain_i"]
               ) == 1:
                edge = dict(u=res_u[0], v=res_v[0], type="seq", weight=None)
                self.graph.add_edge(res_u[0], res_v[0], data=edge)
                seq_edges.append(edge)

        print(f"Added {len(seq_edges)} sequence edges")
        self.edge_df = pd.concat(
            [self.edge_df, pd.DataFrame(seq_edges)], ignore_index=True
        )

        return

    def add_radius_edges(self, r: float, seq_gap: int = 0):

        radius_edges = []
        for (res_u, res_v) in itertools.combinations(
            self.graph.nodes(data=True), 2
        ):
            if res_u[1]["chain"] == res_v[1]["chain"] and abs(
                res_u[1]["chain_i"] - res_v[1]["chain_i"]
               ) < seq_gap:
                continue
            elif euclidean(
                [res_u[1]["pos_x"], res_u[1]["pos_y"], res_u[1]["pos_z"]],
                [res_v[1]["pos_x"], res_v[1]["pos_y"], res_v[1]["pos_z"]]
            ) <= r:
                edge = dict(u=res_u[0], v=res_v[0], type="radius", weight=None)
                self.graph.add_edge(res_u[0], res_v[0], data=edge)
                radius_edges.append(edge)

        print(f"Added {len(radius_edges)} sequence edges")
        self.edge_df = pd.concat(
            [self.edge_df, pd.DataFrame(radius_edges)], ignore_index=True
        )

        return

    def add_knn_edges(self, k: int, seq_gap: int = 0):

        k_adj_mat = kneighbors_graph(
            [
                [res[1]["pos_x"], res[1]["pos_y"], res[1]["pos_z"]]
                for res in self.graph.nodes(data=True)
            ],
            k,
            mode="distance",
            metric="euclidean"
        )

        knn_edges = []
        for (res_u, res_v) in itertools.combinations(
            self.graph.nodes(data=True), 2
        ):
            if (
                res_u[1]["chain"] == res_v[1]["chain"] and
                abs(res_u[1]["chain_i"] - res_v[1]["chain_i"]) < seq_gap
            ):
                continue
            if (
                k_adj_mat[res_u[0], res_v[0]] > 0 or
                k_adj_mat[res_v[0], res_u[0]] > 0
            ):
                edge = dict(u=res_u[0], v=res_v[0], type="knn", weight=None)
                self.graph.add_edge(res_u[0], res_v[0], data=edge)
                knn_edges.append(edge)

        print(f"Added {len(knn_edges)} knn edges")
        self.edge_df = pd.concat(
            [self.edge_df, pd.DataFrame(knn_edges)], ignore_index=True
        )

        return

    def visualize(
        self, color_residue_by: str = "chain", color_edge_by: str = "type"
    ):

        fig = go.Figure()

        self._plot_residues(fig, color_residue_by)
        self._draw_edges(fig, color_edge_by)

        fig.show()

        return

    def _plot_residues(self, fig: go.Figure, color_field: str):

        field_vals = np.sort(self.res_df[color_field].unique())
        color_scale = sample_colorscale(
            "viridis", minmax_scale(range(len(field_vals)))
        )
        val_color_map = {
            val: color_scale[i] for i, val in enumerate(field_vals)
        }

        for val in field_vals:
            fig.add_trace(
                go.Scatter3d(
                    x=self.res_df[self.res_df[color_field] == val].pos_x,
                    y=self.res_df[self.res_df[color_field] == val].pos_y,
                    z=self.res_df[self.res_df[color_field] == val].pos_z,
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=3,
                        color=val_color_map[val]
                    ),
                    text=self.res_df.index,
                    hoverinfo="text",
                    name=f"{color_field}: {val}",
                    legend="legend1"
                )
            )

        return

    def _draw_edges(self, fig: go.Figure, color_field: str):

        field_vals = np.sort(self.edge_df[color_field].unique())
        val_color_map = {val: i for i, val in enumerate(field_vals)}

        for val in field_vals:
            fig.add_trace(
                go.Scatter3d(
                    x=[
                        x for _, edge in self.edge_df.iterrows()
                        for x in [
                            self.res_df.loc[edge.u].pos_x,
                            self.res_df.loc[edge.v].pos_x,
                            None
                        ]
                    ],
                    y=[
                        y for _, edge in self.edge_df.iterrows()
                        for y in [
                            self.res_df.loc[edge.u].pos_y,
                            self.res_df.loc[edge.v].pos_y,
                            None
                        ]
                    ],
                    z=[
                        z for _, edge in self.edge_df.iterrows()
                        for z in [
                            self.res_df.loc[edge.u].pos_z,
                            self.res_df.loc[edge.v].pos_z,
                            None
                        ]
                    ],
                    mode="lines",
                    line=dict(color=val_color_map[val]),
                    name=val,
                    legend="legend2",
                    opacity=0.5
                )
            )

        return
