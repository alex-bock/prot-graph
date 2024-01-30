
from dataclasses import dataclass

from Bio.PDB.Structure import Structure
import networkx as nx
import numpy as np
import pandas as pd
from plotly.colors import sample_colorscale
import plotly.graph_objects as go
from scipy.spatial.distance import euclidean, squareform, pdist
from sklearn.preprocessing import minmax_scale


C_ALPHA = "CA"
HBOND_ATOMS = [
    "ND", "NE", "NH", "NZ", "OD1", "OD2", "OE", "OG", "OH", "SD", "N", "O"
]


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
        self.graph = nx.MultiGraph()
        self.res_df, self.atom_df = self.add_residues(pdb_struct)
        self.edge_df = pd.DataFrame(columns=["u", "v", "type", "weight"])

        return

    @property
    def __len__(self):

        return self.res_df.shape[0]

    def add_residues(self, pdb_struct: Structure):

        atoms = list()
        for chain in pdb_struct.get_chains():
            chain_id = chain.get_id().capitalize()
            for chain_i, res in enumerate(chain.get_residues()):
                res_id = chain_id + str(chain_i)
                c_alpha = None
                res_atoms = list()
                for atom in res.get_atoms():
                    atom_id = atom.get_name()
                    if atom_id == C_ALPHA and c_alpha is None:
                        c_alpha = atom
                    pos = atom.get_coord()
                    res_atoms.append(
                        dict(
                            id=f"{res_id}_{atom_id}",
                            atom_id=atom_id,
                            res_id=res_id,
                            pos_x=pos[0],
                            pos_y=pos[1],
                            pos_z=pos[2]
                        )
                    )
                if c_alpha is None:
                    continue
                else:
                    atoms.extend(res_atoms)
                pos = c_alpha.get_coord()
                self.graph.add_node(
                    res_id,
                    **dict(
                        chain=chain_id,
                        chain_i=chain_i,
                        aa_id=res.get_resname(),
                        pos_x=pos[0],
                        pos_y=pos[1],
                        pos_z=pos[2]
                    )
                )

        print(f"Added {len(self.graph)} residues")

        return (
            pd.DataFrame.from_dict(self.graph.nodes, orient="index"),
            pd.DataFrame(atoms).set_index("id", inplace=False)
        )

    def is_adjacent(
        self, res_u: pd.Series, res_v: pd.Series, seq_gap: int = 1
    ):

        return (
            res_u.chain == res_v.chain and res_u.chain_i != res_v.chain_i and
            abs(res_u.chain_i - res_v.chain_i) <= seq_gap
        )
    
    def get_residue_interactions(
        self, atom_df: pd.DataFrame, min_dist: int,
        dist_metric: str = "euclidean", symmetric: bool = True
    ) -> None:
        
        dist_mat = squareform(
            pdist(atom_df[["pos_x", "pos_y", "pos_z"]], metric=dist_metric)
        )
        dist_mask = dist_mat <= min_dist
        if symmetric:
            dist_mask = np.triu(dist_mask)
        prox_i_pairs = set(list(zip(*np.where(dist_mask))))
        import pdb; pdb.set_trace()

        res_us = self.res_df.loc[atom_df.iloc[[x[0] for x in prox_i_pairs]].res_id.values]
        res_vs = self.res_df.loc[atom_df.iloc[[x[1] for x in prox_i_pairs]].res_id.values]
        
        return zip(res_us.iterrows(), res_vs.iterrows())
    
    def add_sequence_edges(self):

        seq_edges = list()
        for chain, chain_res_df in self.res_df.groupby("chain"):
            chain_res_df.sort_values("chain_i", inplace=True)
            for i in range(len(chain_res_df) - 1):
                u, v = chain + str(i), chain + str(i + 1)
                edge = dict(u=u, v=v, type="seq", weight=None)
                self.graph.add_edge(u, v, data=edge)
                seq_edges.append(edge)
        
        print(f"Added {len(seq_edges)} sequence edges")
        self.edge_df = pd.concat(
            [self.edge_df, pd.DataFrame(seq_edges)], ignore_index=True
        )

        return

    def add_radius_edges(self, r: float, seq_gap: int = 1):

        atom_df = self.atom_df[self.atom_df.atom_id == C_ALPHA]
        res_pairs = self.get_residue_interactions(atom_df, r)

        radius_edges = list()
        for ((u, res_u), (v, res_v)) in res_pairs:
            if u == v or self.is_adjacent(res_u, res_v, seq_gap=seq_gap):
                continue
            edge = dict(u=u, v=v, type="radius", weight=None)
            self.graph.add_edge(u, v, data=edge)
            radius_edges.append(edge)

        print(f"Added {len(radius_edges)} radius edges")
        self.edge_df = pd.concat(
            [self.edge_df, pd.DataFrame(radius_edges)], ignore_index=True
        )

        return

    def add_hydrogen_bonds(self, threshold: float = 3.5, seq_gap: int = 3):

        atom_df = self.atom_df[self.atom_df.atom_id.isin(HBOND_ATOMS)]
        res_pairs = self.get_residue_interactions(atom_df, threshold)

        hbonds = list()
        for ((u, res_u), (v, res_v)) in res_pairs:
            if u == v or self.is_adjacent(res_u, res_v, seq_gap=seq_gap):
                continue
            edge = dict(u=u, v=v, type="hbond", weight=None)
            self.graph.add_edge(u, v, data=edge)
            hbonds.append(edge)

        print(f"Added {len(hbonds)} hydrogen bonds")
        self.edge_df = pd.concat(
            [self.edge_df, pd.DataFrame(hbonds)], ignore_index=True
        )

        return

    def add_disulfide_bridges(self, threshold: float = 2.2):

        cys_res_df = self.res_df[self.res_df.aa_id == "CYS"]
        cys_atom_df = self.atom_df[self.atom_df.res_id.isin(cys_res_df.index)]
        sulf_df = cys_atom_df[cys_atom_df.atom_id == "SG"]
        res_pairs = self.get_residue_interactions(sulf_df, threshold)

        disulf_bridges = list()
        for ((u, res_u), (v, res_v)) in res_pairs:
            if u == v or self.is_adjacent(res_u, res_v):
                continue
            edge = dict(u=u, v=v, type="disulf", weight=None)
            self.graph.add_edge(u, v, data=edge)
            disulf_bridges.append(edge)

        print(f"Added {len(disulf_bridges)} disulfide bridges")
        self.edge_df = pd.concat(
            [self.edge_df, pd.DataFrame(disulf_bridges)], ignore_index=True
        )

        return
    
    def find_louvain_communities(self):

        self.res_df["louvain"] = -1
        if nx.is_connected(self.graph):
            communities = nx.community.louvain_communities(self.graph)
        else:
            subgraph = nx.subgraph(
                self.graph,
                sorted(
                    nx.connected_components(self.graph),
                    key=len,
                    reverse=True
                )[0]
            )
            communities = nx.community.louvain_communities(subgraph)

        for i, res_ids in enumerate(communities):
            self.res_df.loc[self.res_df.index.isin(res_ids), "louvain"] = i
        
        print(self.res_df)

        return

    def visualize(
        self, color_residue_by: str = "chain", draw_rays: bool = False
    ):

        fig = go.Figure()

        self._plot_residues(fig, color_residue_by)
        self._draw_edges(fig)

        if draw_rays:
            self._draw_rays(fig)

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

        for val, val_res_df in self.res_df.groupby(color_field):
            fig.add_trace(
                go.Scatter3d(
                    x=val_res_df.pos_x,
                    y=val_res_df.pos_y,
                    z=val_res_df.pos_z,
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=3,
                        color=val_color_map[val]
                    ),
                    text=val_res_df.index,
                    hoverinfo="text",
                    name=f"{color_field}: {val}"
                )
            )

        return

    def _draw_edges(self, fig: go.Figure):

        for edge_type, edge_type_df in self.edge_df.groupby("type"):
            fig.add_trace(
                go.Scatter3d(
                    x=[
                        x for _, edge in edge_type_df.iterrows()
                        for x in [
                            self.res_df.loc[edge.u].pos_x,
                            self.res_df.loc[edge.v].pos_x,
                            None
                        ]
                    ],
                    y=[
                        y for _, edge in edge_type_df.iterrows()
                        for y in [
                            self.res_df.loc[edge.u].pos_y,
                            self.res_df.loc[edge.v].pos_y,
                            None
                        ]
                    ],
                    z=[
                        z for _, edge in edge_type_df.iterrows()
                        for z in [
                            self.res_df.loc[edge.u].pos_z,
                            self.res_df.loc[edge.v].pos_z,
                            None
                        ]
                    ],
                    mode="lines",
                    name=edge_type,
                    opacity=0.5
                )
            )

        return

    def _draw_rays(self, fig: go.Figure):

        xs = list()
        ys = list()
        zs = list()
        for _, res_atom_df in self.atom_df.groupby("res_id"):
            ca = res_atom_df[res_atom_df.atom_id == C_ALPHA].iloc[0]
            res_atom_df["ca_dist"] = res_atom_df.apply(
                lambda atom: euclidean(
                    [atom.pos_x, atom.pos_y, atom.pos_z],
                    [ca.pos_x, ca.pos_y, ca.pos_z]
                ),
                axis=1
            )
            max_dist_atom = res_atom_df.loc[res_atom_df.ca_dist.idxmax()]
            xs.extend([ca.pos_x, max_dist_atom.pos_x, None])
            ys.extend([ca.pos_y, max_dist_atom.pos_y, None])
            zs.extend([ca.pos_z, max_dist_atom.pos_z, None])

        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color="black"),
                name="rays"
            )
        )

        return
