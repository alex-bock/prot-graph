
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .prot_graph import ProtGraph
from ..structures.structure import Structure

from .constants import (
    PEP, PEP_ATOMS,
    HB, HB_ATOMS,
    HP, HP_RES,
    IB, IB_POS_RES, IB_NEG_RES,
    SB, SB_ANION_RES, SB_CATION_RES, SB_ANIONS, SB_CATIONS,
    DB, DB_RES, DB_ATOMS
)


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

        for res_i in range(struct.atom_df.res_i.max() + 1):
            res_atoms = struct.atom_df[struct.atom_df.res_i == res_i]
            res_id = res_atoms.res_id.unique()[0]
            if "CA" in res_atoms.type.values:
                ca = res_atoms[res_atoms.type == "CA"].iloc[0]
                pos = ca[["x", "y", "z"]].tolist()
                resx.append(
                    dict(
                        i=res_i,
                        id=res_id,
                        type=res_atoms.res_type.unique()[0],
                        chain=res_atoms.chain.unique()[0],
                        chain_i=res_atoms.chain_i.unique()[0]
                    )
                )
                res_pos_lst.append(pos)
            else:
                skipped.append(res_id)

        print(f"Added {len(resx)} residues ({len(skipped)} skipped)")

        return pd.DataFrame(resx).set_index("i"), np.array(res_pos_lst)

    def add_nodes(self, node_df: pd.DataFrame) -> nx.Graph:

        res_graph = nx.MultiGraph()

        for i, _ in node_df.iterrows():
            res_graph.add_node(i)

        return res_graph

    def is_adjacent(self, u: pd.Series, v: pd.Series, seq_gap: int = 0) -> bool:

        return u.chain == v.chain and abs(u.chain_i - v.chain_i) < seq_gap + 1

    def complementary_res(
        self, u: pd.Series, v: pd.Series, s1: List, s2: List
    ) -> bool:

        return (
            (u.type in s1 and v.type in s2) or (u.type in s2 and v.type in s1)
        )

    def get_res_pairs(self, atom_pairs: List[Tuple]) -> List[Tuple]:

        res_pairs = self.struct.get_res_pairs(atom_pairs)

        res_us = self.node_df.loc[[x[0] for x in res_pairs]]
        res_vs = self.node_df.loc[[x[1] for x in res_pairs]]

        return zip(res_us.iterrows(), res_vs.iterrows())

    def add_peptide_bonds(self, dist: float = 1.5):

        atom_pairs = self.struct.get_atom_pairs(dist, types=PEP_ATOMS)
        res_pairs = self.get_res_pairs(atom_pairs)
        peptide_bonds = list()

        for ((u, _), (v, _)) in res_pairs:
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

    def add_hydrogen_bonds(
        self, dist: float = 3.5, seq_gap: int = 3, theta: float = None
    ):

        atom_pairs = self.struct.get_atom_pairs(
            dist, types=HB_ATOMS, theta=theta
        )
        res_pairs = self.get_res_pairs(atom_pairs)
        hydrogen_bonds = list()

        for ((u, res_u), (v, res_v)) in res_pairs:
            if self.is_adjacent(res_u, res_v, seq_gap=seq_gap):
                continue
            self.graph.add_edge(u, v, type=HB)
            hydrogen_bonds.append({"u": u, "v": v, "type": HB})

        print(f"Added {len(hydrogen_bonds)} hydrogen bonds")
        self.edge_df = pd.concat(
            [self.edge_df, pd.DataFrame(hydrogen_bonds)], ignore_index=True
        )
        self.edge_types.append(HB)

        return

    def add_hydrophobic_interactions(self, dist: float = 5.0, seq_gap: int = 3):

        res_pairs = self.get_res_pairs(dist, types=HP_RES)
        hp_interactions = list()

        for ((u, res_u), (v, res_v)) in res_pairs:
            if self.is_adjacent(res_u, res_v, seq_gap=seq_gap):
                continue
            self.graph.add_edge(u, v, type=HP)
            hp_interactions.append({"u": u, "v": v, "type": HP})

        print(f"Added {len(hp_interactions)} hydrophobic interactions")
        self.edge_df = pd.concat(
            [self.edge_df, pd.DataFrame(hp_interactions)], ignore_index=True
        )
        self.edge_types.append(HP)

        return

    def add_ionic_bonds(self, dist: float = 6.0, seq_gap: int = 3):

        res_pairs = self.get_res_pairs(dist, types=IB_POS_RES + IB_NEG_RES)
        hp_interactions = list()

        for ((u, res_u), (v, res_v)) in res_pairs:
            if self.is_adjacent(res_u, res_v, seq_gap=seq_gap):
                continue
            elif not self.complementary_res(
                res_u, res_v, IB_POS_RES, IB_NEG_RES
            ):
                continue
            self.graph.add_edge(u, v, type=IB)
            hp_interactions.append({"u": u, "v": v, "type": IB})

        print(f"Added {len(hp_interactions)} ionic bonds")
        self.edge_df = pd.concat(
            [self.edge_df, pd.DataFrame(hp_interactions)], ignore_index=True
        )
        self.edge_types.append(IB)

        return

    def add_salt_bridges(self, dist: float = 4.0, seq_gap: int = 3):

        res_pairs = self.get_res_pairs(
            dist, types=SB_ANION_RES + SB_CATION_RES,
            atom_types=SB_ANIONS + SB_CATIONS
        )
        salt_bridges = list()

        for ((u, res_u), (v, res_v)) in res_pairs:
            if self.is_adjacent(res_u, res_v, seq_gap=seq_gap):
                continue
            elif not self.complementary_res(
                res_u, res_v, SB_ANION_RES, SB_CATION_RES
            ):
                continue
            self.graph.add_edge(u, v, type=SB)
            salt_bridges.append({"u": u, "v": v, "type": SB})

        print(f"Added {len(salt_bridges)} salt bridges")
        self.edge_df = pd.concat(
            [self.edge_df, pd.DataFrame(salt_bridges)], ignore_index=True
        )
        self.edge_types.append(SB)

        return

    def add_disulfide_bridges(self, dist: float = 2.2, seq_gap: int = 3):

        res_pairs = self.get_res_pairs(
            dist, types=DB_RES, atom_types=DB_ATOMS
        )
        disulfide_bridges = list()

        for ((u, res_u), (v, res_v)) in res_pairs:
            if self.is_adjacent(res_u, res_v, seq_gap=seq_gap):
                continue
            self.graph.add_edge(u, v, type=DB)
            disulfide_bridges.append({"u": u, "v": v, "type": DB})

        print(f"Added {len(disulfide_bridges)} disulfide bridges")
        self.edge_df = pd.concat(
            [self.edge_df, pd.DataFrame(disulfide_bridges)], ignore_index=True
        )
        self.edge_types.append(DB)

        return
