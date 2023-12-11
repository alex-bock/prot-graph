
from dataclasses import dataclass
from typing import List

from Bio.PDB.Structure import Structure
import numpy as np
from sklearn.neighbors import radius_neighbors_graph


C_ALPHA = "CA"
DEFAULT_NODE_RADIUS = 10.0


@dataclass
class Residue:

    id: int
    chain: str
    aa_id: str
    pos: np.ndarray


@dataclass
class Edge:

    u: str
    v: str


class ProtGraph:

    def __init__(
        self,
        pdb_struct: Structure,
        radius: float = DEFAULT_NODE_RADIUS
    ):

        self.pdb_id = pdb_struct.get_id()
        self.nodes, self.edges = ProtGraph.build_graph(pdb_struct, radius)
        self.n = len(self.nodes)

        return

    @staticmethod
    def build_graph(pdb_struct: Structure, radius: float):

        nodes = ProtGraph.populate_nodes(pdb_struct)
        edges = ProtGraph.create_edges(nodes, radius)

        return nodes, edges

    @staticmethod
    def populate_nodes(pdb_struct: Structure) -> List[Residue]:

        nodes = list()
        for chain in pdb_struct.get_chains():
            chain_id = chain.get_id()
            for i, residue in enumerate(chain.get_residues()):
                c_alpha = next(
                    (
                        atom for atom in residue.get_atoms()
                        if atom.get_name() == C_ALPHA
                    ),
                    None
                )
                if c_alpha is None:
                    continue
                nodes.append(
                    Residue(
                        id=chain_id + str(i),
                        chain=chain_id,
                        aa_id=residue.get_resname(),
                        pos=c_alpha.get_coord()
                    )
                )

        return nodes

    @staticmethod
    def create_edges(nodes: List[Residue], radius: float) -> List[Edge]:

        adj_mat = radius_neighbors_graph([node.pos for node in nodes], radius)

        edges = list()
        for i, n1 in enumerate(nodes):
            for j, n2 in enumerate(nodes[(i + 1):]):
                if adj_mat[i, i + j] > 0:
                    edges.append(Edge(u=n1.id, v=n2.id))

        return edges

    @property
    def __len__(self) -> int:

        return self.n

    def get_neighbors(self, res_id: str) -> List[str]:

        neighbors = list()
        for edge in self.edges:
            if edge.u == res_id:
                neighbors.append(edge.v)
            elif edge.v == res_id:
                neighbors.append(edge.u)

        return neighbors
