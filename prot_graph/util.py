
import pandas as pd
import torch
from torch import Tensor

import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from sklearn.preprocessing import minmax_scale

from torchdrug.data import Protein


CONTACT2ID = {"hb": 4, "sb": 5, "pc": 6, "ps": 7, "ts": 8, "hp": 9, "vdw": 10}


def load_contacts(protein: Protein, contacts_fp: str):

    contacts_df = pd.read_csv(
        contacts_fp, sep="\t", names=["frame", "type", "u", "v"],
        skiprows=[0, 1]
    )

    edge_list = protein.edge_list
    n_contacts = list()
    n_relations = list()
    for type in contacts_df.type.unique():
        df = contacts_df[contacts_df.type == type]
        us, vs = [], []
        for _, row in df.iterrows():
            [chain_u, res_type_u, chain_u_i, atom_name_u] = row.u.split(":")
            [chain_v, res_type_v, chain_v_i, atom_name_v] = row.v.split(":")
            res_u_i = int((protein.chain_id == protein.alphabet2id[chain_u]).nonzero()[int(chain_u_i) - 1])
            res_v_i = int((protein.chain_id == protein.alphabet2id[chain_v]).nonzero()[int(chain_v_i) - 1])
            assert(protein.id2residue[int(protein.residue_type[res_u_i])] == res_type_u)
            assert(protein.id2residue[int(protein.residue_type[res_v_i])] == res_type_v)
            res_u_atom_is = (protein.atom2residue == res_u_i).nonzero().squeeze()
            res_v_atom_is = (protein.atom2residue == res_v_i).nonzero().squeeze()
            res_u_atom_names = protein.atom_name[res_u_atom_is]
            res_v_atom_names = protein.atom_name[res_v_atom_is]
            atom_u_i = res_u_atom_is[res_u_atom_names == protein.atom_name2id[atom_name_u]][0]
            atom_v_i = res_v_atom_is[res_v_atom_names == protein.atom_name2id[atom_name_v]][0]
            assert(protein.id2atom_name[int(protein.atom_name[atom_u_i])] == atom_name_u)
            assert(protein.id2atom_name[int(protein.atom_name[atom_v_i])] == atom_name_v)
            us.append(atom_u_i)
            vs.append(atom_v_i)
        edge_list = torch.cat([edge_list, torch.stack([Tensor(us), Tensor(vs), torch.full((len(us), ), CONTACT2ID[type])]).t().int()])
        n_contacts.append(len(us))
        n_relations.append(CONTACT2ID[type])

    return Protein(
        edge_list, atom_type=protein.atom_type, bond_type=edge_list[:, 2],
        residue_type=protein.residue_type, view=protein.view,
        atom_name=protein.atom_name, atom2residue=protein.atom2residue,
        residue_feature=protein.residue_feature,
        is_hetero_atom=protein.is_hetero_atom, occupancy=protein.occupancy,
        b_factor=protein.b_factor, residue_number=protein.residue_number,
        insertion_code=protein.insertion_code, chain_id=protein.chain_id,
        num_relation=edge_list[:, 2].max() + 1,
        node_position=protein.node_position

    )


def visualize(
    protein: Protein, color_node_by: str = "residue_type",
    hide_nodes: bool = False
):

    fig = go.Figure()

    if not hide_nodes:
        plot_nodes(protein, fig, color_by=color_node_by)

    draw_edges(protein, fig)

    fig.show()

    return

def plot_nodes(protein: Protein, fig: go.Figure, color_by: str = "residue_type"):

    if color_by == "residue_type":
        vals = protein.residue_type[protein.atom2residue]
        val_name_dict = protein.id2residue
    elif color_by == "atom_name":
        vals = protein.atom_name
        atom_symbols = [
            protein.id2atom_name[val.item()][0] for val in vals
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
                x=protein.node_position[:, 0][mask],
                y=protein.node_position[:, 1][mask],
                z=protein.node_position[:, 2][mask],
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

def draw_edges(protein: Protein, fig: go.Figure):

    relations = protein.edge_list[:, 2]
    relation_set = relations.unique()
    import pdb; pdb.set_trace()

    for relation in relation_set:
        relation = relation.item()
        edges = protein.edge_list[relations == relation]
        fig.add_trace(
            go.Scatter3d(
                x=torch.stack(
                    [
                        protein.node_position[edges[:, 0]][:, 0],
                        protein.node_position[edges[:, 1]][:, 0],
                        torch.tensor([float("nan")] * len(edges))
                    ]).t().flatten(),
                y=torch.stack(
                    [
                        protein.node_position[edges[:, 0]][:, 1],
                        protein.node_position[edges[:, 1]][:, 1],
                        torch.tensor([float("nan")] * len(edges))
                    ]).t().flatten(),
                z=torch.stack(
                    [
                        protein.node_position[edges[:, 0]][:, 2],
                        protein.node_position[edges[:, 1]][:, 2],
                        torch.tensor([float("nan")] * len(edges))
                    ]).t().flatten(),
                mode="lines",
                opacity=0.5,
                name={v: k for k, v in (protein.bond2id | CONTACT2ID).items()}[relation]
            )
        )

    return