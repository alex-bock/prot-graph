
from typing import Union

import pandas as pd
import torch
from torch import Tensor

import plotly.graph_objects as go
import plotly.express as px

from torchdrug.data import Protein

from .constants import ATOM_TYPE2ID, ID2ATOM_TYPE


def visualize(
    protein: Protein, color_node_by: str = "residue_type",
    separate_chains: bool = False, hide_nodes: bool = False
):

    fig = go.Figure()
    n_layers = protein.edge_list[:, 2].max() + 1

    if separate_chains:
        chain_ids = protein.chain_id.unique()
        for chain_id in chain_ids:
            if len(protein.chain_id) == len(protein.atom2residue):
                chain = protein.subgraph(protein.chain_id == chain_id)
            else:
                chain = protein.subgraph(
                    protein.chain_id[protein.atom2residue] == chain_id
                )
            if not hide_nodes:
                plot_nodes(
                    chain, fig, color_by=color_node_by, chain=chain_id.item()
                )
            draw_edges(chain, fig, chain=chain_id.item(), n_layers=n_layers)
        fig.update_layout(legend_groupclick="toggleitem")
    else:
        if not hide_nodes:
            plot_nodes(protein, fig, color_by=color_node_by)
        draw_edges(protein, fig)

    fig.show()

    return


def plot_nodes(
    protein: Protein, fig: go.Figure, color_by: str = "residue_type",
    chain: str = None
):

    if color_by == "residue_type":
        vals = protein.residue_type[protein.atom2residue]
        val_name_dict = protein.id2residue
        color_scale = px.colors.qualitative.Dark24
    elif color_by == "atom_type":
        atom_types = [
            protein.id2atom_name[s.item()][0]
            for s in protein.atom_name
        ]
        vals = Tensor([ATOM_TYPE2ID[x] for x in atom_types])
        val_name_dict = ID2ATOM_TYPE
        color_scale = px.colors.qualitative.Dark2
    elif color_by == "chain":
        vals = protein.chain_id[protein.atom2residue] - 1
        val_name_dict = {v.item(): int(v.item() + 1) for v in vals.unique()}
        color_scale = px.colors.qualitative.D3

    color_scale = color_scale[:len(val_name_dict.keys())]
    val_color_map = {i: color_scale[i] for i in range(len(color_scale))}
    val_set = vals.unique()

    for val in val_set:
        val = val.item()
        val_name = val_name_dict[val]
        mask = vals == val
        fig.add_trace(
            go.Scatter3d(
                x=protein.node_position[:, 0][mask],
                y=protein.node_position[:, 1][mask],
                z=protein.node_position[:, 2][mask],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=3,
                    color=val_color_map[val]
                ),
                hoverinfo="text",
                text=val_name,
                name=f"{color_by}: {val_name}",
                legendgroup=chain,
                legendgrouptitle=dict(text=f"chain {chain}")
            )
        )

    return


def draw_edges(
    protein: Protein, fig: go.Figure, chain: str = None,
    n_layers: Union[int, None] = None
):

    layer_ids = protein.edge_list[:, 2]
    if n_layers is None:
        n_layers = layer_ids.max() + 1

    for layer in range(n_layers):
        edges = protein.edge_list[layer_ids == layer]
        if len(edges) == 0:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=torch.stack(
                    [
                        protein.node_position[edges[:, 0]][:, 0],
                        protein.node_position[edges[:, 1]][:, 0],
                        Tensor([float("nan")] * len(edges))
                    ]).t().flatten(),
                y=torch.stack(
                    [
                        protein.node_position[edges[:, 0]][:, 1],
                        protein.node_position[edges[:, 1]][:, 1],
                        Tensor([float("nan")] * len(edges))
                    ]).t().flatten(),
                z=torch.stack(
                    [
                        protein.node_position[edges[:, 0]][:, 2],
                        protein.node_position[edges[:, 1]][:, 2],
                        Tensor([float("nan")] * len(edges))
                    ]).t().flatten(),
                mode="lines",
                line=dict(color=px.colors.qualitative.Vivid[layer]),
                opacity=0.5,
                legendgroup=chain,
                legendgrouptitle=dict(text=f"chain {chain}"),
                name=f"Layer {layer + 1}"
            )
        )

    return
