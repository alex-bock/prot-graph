
import abc
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from plotly.colors import sample_colorscale
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import minmax_scale

from ..structures.structure import Structure


class ProtGraph(abc.ABC):

    def __init__(self, struct: Structure):

        self.struct = struct

        return

    @property
    def id(self) -> str:

        return self.struct.id

    @abc.abstractmethod
    def get_nodes(
        self, struct: Structure
    ) -> Tuple[pd.DataFrame, np.ndarray]:

        raise NotImplementedError

    @abc.abstractmethod
    def add_nodes(self, node_df: pd.DataFrame) -> nx.Graph:

        raise NotImplementedError

    @property
    def __len__(self):

        return len(self.node_df)

    def get_atom_pairs(
        self, dist: float, types: List[str] = None, res_types: List[str] = None,
        theta: float = None, vertex: str = None, dist_metric: str = "euclidean"
    ) -> List[Tuple[int, int]]:

        atom_pairs = self.struct.get_atom_pairs(
            dist, types=types, res_types=res_types, theta=theta, vertex=vertex,
            dist_metric=dist_metric
        )

        atom_us = self.node_df.loc[[x[0] for x in atom_pairs]]
        atom_vs = self.node_df.loc[[x[1] for x in atom_pairs]]

        return zip(atom_us.iterrows(), atom_vs.iterrows())

    @property
    def adj_matrix(self):

        n_nodes = len(self.node_pos_mat)
        n_edge_types = len(self.edge_types)
        adj_matrix = np.zeros((n_edge_types, n_nodes, n_nodes), dtype=int)

        for i in range(n_edge_types):
            edge_df = self.edge_df[self.edge_df.type == self.edge_types[i]]
            adj_matrix[
                i, edge_df.u.values.astype(int), edge_df.v.values.astype(int)
            ] = 1
            adj_matrix[
                i, edge_df.v.values.astype(int), edge_df.u.values.astype(int)
            ] = 1

        return adj_matrix

    def show_contact_map(self, edge_type: str):

        adj_mat = self.adj_matrix
        i = self.edge_types.index(edge_type)

        fig = px.imshow(adj_mat[i, :, :])
        fig.show()

        return

    def compare_edges(
        self, type_1: str, type_2: str, display: bool = True
    ) -> np.ndarray:

        adj_mat = self.adj_matrix
        i_1, i_2 = self.edge_types.index(type_1), self.edge_types.index(type_2)
        adj_mat_1, adj_mat_2 = (
            adj_mat[i_1, :, :],
            adj_mat[i_2, :, :]
        )
        conf_mat = confusion_matrix(adj_mat_1.flatten(), adj_mat_2.flatten())

        if display:
            fig = px.imshow(conf_mat)
            fig.show()

        return conf_mat

    def visualize(self, color_node_by: str = "type", hide_nodes: bool = False):

        fig = go.Figure()

        if not hide_nodes:
            self._plot_nodes(fig, color_by=color_node_by)
        self._draw_edges(fig)

        return fig

    def _plot_nodes(self, fig: go.Figure, color_by: str = "type"):

        field_vals = np.sort(self.node_df[color_by].unique())
        color_scale = sample_colorscale(
            "viridis", minmax_scale(range(len(field_vals)))
        )
        val_color_map = {
            val: color_scale[i] for i, val in enumerate(field_vals)
        }

        for val, val_df in self.node_df.groupby(color_by):
            fig.add_trace(
                go.Scatter3d(
                    x=[self.node_pos_mat[i][0] for i in val_df.index.values],
                    y=[self.node_pos_mat[i][1] for i in val_df.index.values],
                    z=[self.node_pos_mat[i][2] for i in val_df.index.values],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=3,
                        color=val_color_map[val]
                    ),
                    text=val_df.id,
                    hoverinfo="text",
                    name=f"{color_by}: {val}"
                )
            )

        return

    def _draw_edges(self, fig: go.Figure):

        adj_matrix = self.adj_matrix

        for i in range(adj_matrix.shape[0]):
            edge_adj_matrix = adj_matrix[i, :, :]
            edges = list(zip(*np.where(edge_adj_matrix)))
            fig.add_trace(
                go.Scatter3d(
                    x=[
                        x for node_i in edges for x in [
                            self.node_pos_mat[node_i[0]][0],
                            self.node_pos_mat[node_i[1]][0],
                            None
                        ]
                    ],
                    y=[
                        y for node_i in edges for y in [
                            self.node_pos_mat[node_i[0]][1],
                            self.node_pos_mat[node_i[1]][1],
                            None
                        ]
                    ],
                    z=[
                        z for node_i in edges for z in [
                            self.node_pos_mat[node_i[0]][2],
                            self.node_pos_mat[node_i[1]][2],
                            None
                        ]
                    ],
                    mode="lines",
                    name=self.edge_types[i],
                    opacity=0.5
                )
            )

        return
