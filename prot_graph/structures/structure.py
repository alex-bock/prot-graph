
import abc
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist, euclidean

from ..datasets.dataset import Dataset


class Structure(abc.ABC):

    def __init__(self, id: str, db: Dataset):

        self.id = id
        self.struct = db.load_record(id)

        self.atom_df = self.add_atoms(self.struct)

        return

    @abc.abstractmethod
    def add_atoms(self, struct: Any) -> Tuple[pd.DataFrame, Dict]:

        raise NotImplementedError

    def filter_atom_df(
        self, types: List[str] = None, res_types: List[str] = None
    ) -> pd.DataFrame:

        atom_df = self.atom_df
        if types is not None:
            atom_df = atom_df[atom_df.type.isin(types)]
        if res_types is not None:
            atom_df = atom_df[atom_df.res_type.isin(res_types)]

        return atom_df

    def get_atom_pairs(
        self, dist: float, types: List[str] = None, res_types: List[str] = None,
        theta: float = None, vertex: str = None, dist_metric: str = "euclidean"
    ) -> List[Tuple[int, int]]:

        atom_df = self.filter_atom_df(types=types, res_types=res_types)
        atom_dist_mat = squareform(
            pdist(atom_df[["x", "y", "z"]], metric=dist_metric)
        )
        atom_is = list(zip(*np.where(np.triu(atom_dist_mat <= dist))))

        atom_pairs = list(set(zip(
            atom_df.iloc[[x[0] for x in atom_is]].index.values,
            atom_df.iloc[[x[1] for x in atom_is]].index.values,
        )))

        # TODO: probably a faster way of doing this (squareform?)
        if theta is not None:
            atom_pairs = [
                (i, j) for (i, j) in atom_pairs
                if self.calculate_hbond_angle(i, j) >= theta
            ]

        return atom_pairs

    def get_res_pairs(self, atom_pairs: List[Tuple]) -> List[Tuple[int, int]]:

        res_pairs = list(set(zip(
            self.atom_df.loc[[x[0] for x in atom_pairs]].res_i.values,
            self.atom_df.loc[[x[1] for x in atom_pairs]].res_i.values
        )))

        return res_pairs

    def calculate_hbond_angle(self, atom_i: int, atom_j: int) -> float:

        h_i, dist_i = self.get_nearest_hydrogen(atom_i)
        h_j, dist_j = self.get_nearest_hydrogen(atom_j)
        h, _ = min([(h_i, dist_i), (h_j, dist_j)], key=lambda x: x[1])

        pos_i = self.atom_df.loc[atom_i][["x", "y", "z"]].to_numpy()
        pos_h = self.atom_df.loc[h][["x", "y", "z"]].to_numpy()
        pos_j = self.atom_df.loc[atom_j][["x", "y", "z"]].to_numpy()

        cos_theta = (
            np.dot(pos_i - pos_h, pos_j - pos_h) /
            (np.linalg.norm(pos_i - pos_h) * np.linalg.norm(pos_j - pos_h))
        )
        theta = np.arccos(min(cos_theta, 1.0))

        return np.degrees(theta)

    def get_nearest_hydrogen(self, atom_id: int) -> Tuple[int, float]:

        atom = self.atom_df.loc[atom_id]
        res_atom_df = self.atom_df[self.atom_df.res_id == atom.res_id]
        res_h_df = res_atom_df[res_atom_df.type.str.startswith("H")]
        h_dists = res_h_df.apply(
            lambda x: euclidean([x.x, x.y, x.z], [atom.x, atom.y, atom.z]),
            axis=1
        )

        return (h_dists.idxmin(), h_dists.min())
