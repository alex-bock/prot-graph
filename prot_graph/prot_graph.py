
from dataclasses import dataclass
from typing import List

from Bio.PDB.Structure import Structure
import numpy as np
from scipy.spatial.distance import euclidean


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
        node_radius: float = DEFAULT_NODE_RADIUS
    ):

        self.pdb_id = pdb_struct.get_id()
        self._load_graph(pdb_struct, node_radius)

        return

    def _load_graph(self, pdb_struct: Structure, node_radius: float):

        self.nodes = list()
        for chain in pdb_struct.get_chains():
            chain_id = chain.get_id()
            for i, residue in enumerate(chain.get_residues()):
                try:
                    c_alpha = list(
                        filter(
                            lambda atom: atom.get_name() == C_ALPHA,
                            residue.get_atoms()
                        )
                    )[0]
                except IndexError:
                    continue
                self.nodes.append(
                    Residue(
                        id=chain_id + str(i),
                        chain=chain_id,
                        aa_id=residue.get_resname(),
                        pos=c_alpha.get_coord()
                    )
                )
        self.n = len(self.nodes)

        if node_radius is None:
            node_radius = 0.0

        self.edges = list()
        for i, u in enumerate(self.nodes):
            for v in self.nodes[(i + 1):]:
                if euclidean(u.pos, v.pos) <= node_radius:
                    self.edges.append(Edge(u=u.id, v=v.id))

        return

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
