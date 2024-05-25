
import torch

from torchdrug.data import Protein

from plotly.colors import sample_colorscale
import plotly.graph_objects as go
from sklearn.preprocessing import minmax_scale


class ProtGraph:

    def __init__(self, protein: Protein):

        self.protein = protein

        return

    def visualize(
        self, color_node_by: str = "residue_type", hide_nodes: bool = False
    ):

        fig = go.Figure()

        if not hide_nodes:
            self._plot_nodes(fig, color_by=color_node_by)

        self._draw_edges(fig)

        fig.show()

        return

    def _plot_nodes(self, fig: go.Figure, color_by: str = "residue_type"):

        if color_by == "residue_type":
            vals = self.protein.residue_type[self.protein.atom2residue]
            val_name_dict = self.protein.id2residue
        elif color_by == "atom_type":
            vals = self.protein.atom_name
            atom_symbols = [
                self.protein.id2atom_name[val.item()][0] for val in vals
            ]
            val_name_dict = {ord(x): x for x in atom_symbols}
            vals = torch.tensor([ord(x) for x in atom_symbols])

        val_set = vals.unique()
        color_scale = sample_colorscale(
            "viridis", minmax_scale(range(len(val_set)))
        )
        val_color_map = {
            val.item(): color_scale[i] for i, val in enumerate(val_set)
        }

        for val in val_set:
            val = val.item()
            val_name = val_name_dict[val]
            mask = vals == val
            fig.add_trace(
                go.Scatter3d(
                    x=self.protein.node_position[:, 0][mask],
                    y=self.protein.node_position[:, 1][mask],
                    z=self.protein.node_position[:, 2][mask],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=3,
                        color=val_color_map[val]
                    ),
                    text=val,
                    hoverinfo="text",
                    name=f"{color_by}: {val_name}"
                )
            )

        return

    def _draw_edges(self, fig: go.Figure):

        relations = self.protein.edge_list[:, 2]
        relation_set = relations.unique()

        for relation in relation_set:
            relation = relation.item()
            edges = self.protein.edge_list[relations == relation]
            fig.add_trace(
                go.Scatter3d(
                    x=torch.stack(
                        [
                            self.protein.node_position[edges[:, 0]][:, 0],
                            self.protein.node_position[edges[:, 1]][:, 0],
                            torch.tensor([float("nan")] * len(edges))
                        ]).t().flatten(),
                    y=torch.stack(
                        [
                            self.protein.node_position[edges[:, 0]][:, 1],
                            self.protein.node_position[edges[:, 1]][:, 1],
                            torch.tensor([float("nan")] * len(edges))
                        ]).t().flatten(),
                    z=torch.stack(
                        [
                            self.protein.node_position[edges[:, 0]][:, 2],
                            self.protein.node_position[edges[:, 1]][:, 2],
                            torch.tensor([float("nan")] * len(edges))
                        ]).t().flatten(),
                    mode="lines",
                    opacity=0.5
                )
            )

        return