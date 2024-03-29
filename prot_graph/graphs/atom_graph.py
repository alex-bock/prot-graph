
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .prot_graph import ProtGraph
from ..structures.structure import Structure


from .constants import PEP, PEP_ATOMS, HB, HB_ATOMS


class AtomGraph(ProtGraph):

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

        atom_df = struct.atom_df.drop(columns=["x", "y", "z"])
        atom_pos_mat = struct.atom_df[["x", "y", "z"]].to_numpy()

        return atom_df, atom_pos_mat

    def add_nodes(self, node_df: pd.DataFrame) -> nx.Graph:

        atom_graph = nx.Graph()

        for i, _ in node_df.iterrows():
            atom_graph.add_node(i)

        return atom_graph

    def is_adjacent(self, u: pd.Series, v: pd.Series, seq_gap: int = 0):

        return u.chain == v.chain and abs(u.chain_i - v.chain_i) < seq_gap + 1

    def add_peptide_bonds(self, dist: float = 1.5):

        atom_pairs = self.get_atom_pairs(dist, types=PEP_ATOMS)
        peptide_bonds = list()

        for ((u, _), (v, _)) in atom_pairs:
            if u == v:
                continue
            self.graph.add_edge(u, v, type=PEP)
            peptide_bonds.append({"u": u, "v": v, "type": PEP})

        print(f"Added {len(peptide_bonds)} peptide bonds")
        self.edge_df = pd.concat(
            [self.edge_df, pd.DataFrame(peptide_bonds)], ignore_index=True
        )
        self.edge_types.append(PEP)

        return

    def add_hydrogen_bonds(self, dist: float = 3.5, seq_gap: int = 3):

        atom_pairs = self.get_atom_pairs(
            dist, types=HB_ATOMS, theta=90.0
        )
        hydrogen_bonds = list()

        for ((u, atom_u), (v, atom_v)) in atom_pairs:
            if self.is_adjacent(atom_u, atom_v, seq_gap=seq_gap):
                continue
            self.graph.add_edge(u, v, type=HB)
            hydrogen_bonds.append({"u": u, "v": v, "type": HB})

        print(f"Added {len(hydrogen_bonds)} hydrogen bonds")
        self.edge_df = pd.concat(
            [self.edge_df, pd.DataFrame(hydrogen_bonds)], ignore_index=True
        )
        self.edge_types.append(HB)

        return
