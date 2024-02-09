
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .prot_graph import ProtGraph
from ..structures.structure import Structure


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
