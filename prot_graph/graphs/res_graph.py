
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .prot_graph import ProtGraph
from ..structures.structure import Structure

from .constants import PEPTIDE, HBOND, HBOND_ATOMS


class ResGraph(ProtGraph):

    def __init__(self, struct: Structure):

        super().__init__(struct=struct)

        self.node_df, self.node_pos_mat = self.get_nodes(self.struct)
        self.graph = self.add_nodes(self.node_df)

        self.edge_df = pd.DataFrame(columns=["u", "v", "type"])
        self.edge_types = list()

        return

    def get_nodes(
        self, struct: Structure
    ) -> Tuple[pd.DataFrame, np.ndarray]:

        resx = list()
        res_pos_lst = list()
        skipped = list()

        for res_i, res_atoms in struct.atom_df.groupby("res_i"):
            if "CA" in res_atoms.type.values:
                ca = res_atoms[res_atoms.type == "CA"].iloc[0]
                pos = ca[["x", "y", "z"]].tolist()
                resx.append(
                    dict(
                        i=res_i,
                        id=res_atoms.res_id.unique()[0],
                        type=res_atoms.res_type.unique()[0],
                        chain=res_atoms.chain.unique()[0],
                        chain_i=res_atoms.chain_i.unique()[0]
                    )
                )
            else:
                pos = [-1, -1, -1]
                skipped.append(res_i)
            res_pos_lst.append(pos)

        print(f"Added {len(resx)} residues ({len(skipped)} skipped)")

        return pd.DataFrame(resx).set_index("i"), np.array(res_pos_lst)

    def add_nodes(self, node_df: pd.DataFrame) -> nx.Graph:

        res_graph = nx.MultiGraph()

        for i, _ in node_df.iterrows():
            res_graph.add_node(i)

        return res_graph

    def is_adjacent(self, u: pd.Series, v: pd.Series, seq_gap: int = 0) -> bool:

        return u.chain == v.chain and abs(u.chain_i - v.chain_i) < seq_gap + 1

    def get_res_pairs(
        self, dist: float, types: List[str] = None,
        atom_types: List[str] = None, dist_metric: str = "euclidean"
    ) -> List[Tuple]:

        res_pairs = self.struct.get_res_pairs(
            dist, types=types, atom_types=atom_types, dist_metric=dist_metric
        )

        res_us = self.node_df.loc[[x[0] for x in res_pairs]]
        res_vs = self.node_df.loc[[x[1] for x in res_pairs]]

        return zip(res_us.iterrows(), res_vs.iterrows())

    def add_peptide_bonds(self):

        peptide_bonds = list()

        for _, chain_res_df in self.node_df.groupby("chain"):
            chain_res_df.sort_values("chain_i", inplace=True)
            for i in range(len(chain_res_df) - 1):
                u, v = chain_res_df.iloc[i].name, chain_res_df.iloc[i + 1].name
                self.graph.add_edge(u, v, type=PEPTIDE)
                peptide_bonds.append({"u": u, "v": v, "type": PEPTIDE})

        print(f"Added {len(peptide_bonds)} peptide bonds")
        self.edge_df = pd.concat(
            [self.edge_df, pd.DataFrame(peptide_bonds)], ignore_index=True
        )
        self.edge_types.append(PEPTIDE)

        return

    def add_hydrogen_bonds(self, dist: float = 3.5, seq_gap: int = 3):

        res_pairs = self.get_res_pairs(dist, atom_types=HBOND_ATOMS)
        hydrogen_bonds = list()

        for ((u, res_u), (v, res_v)) in res_pairs:
            if u == v or self.is_adjacent(res_u, res_v, seq_gap=seq_gap):
                continue
            self.graph.add_edge(u, v, type=HBOND)
            hydrogen_bonds.append({"u": u, "v": v, "type": HBOND})

        print(f"Added {len(hydrogen_bonds)} hydrogen bonds")
        self.edge_df = pd.concat(
            [self.edge_df, pd.DataFrame(hydrogen_bonds)], ignore_index=True
        )
        self.edge_types.append(HBOND)

        return
