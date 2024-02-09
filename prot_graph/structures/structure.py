
import abc
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist

from ..datasets.dataset import Dataset


class Structure(abc.ABC):

    def __init__(self, id: str, db: Dataset):

        self.id = id
        self.struct = db.load_record(id)

        self.atom_df, self.res_type_map = self.add_atoms(self.struct)

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
        dist_metric: str = "euclidean"
    ) -> List[Tuple[int, int]]:

        atom_df = self.filter_atom_df(types=types, res_types=res_types)
        atom_dist_mat = squareform(
            pdist(atom_df[["x", "y", "z"]], metric=dist_metric)
        )

        return list(zip(*np.where(np.triu(atom_dist_mat <= dist))))

    def get_res_pairs(
        self, dist: float, types: List[str] = None,
        atom_types: List[str] = None, dist_metric: str = "euclidean"
    ) -> List[Tuple[int, int]]:

        atom_df = self.filter_atom_df(types=atom_types, res_types=types)
        atom_pairs = self.get_atom_pairs(
            dist, types=atom_types, res_types=types, dist_metric=dist_metric
        )
        res_pairs = list(set(zip(
            atom_df.iloc[[x[0] for x in atom_pairs]].res_i.values,
            atom_df.iloc[[x[1] for x in atom_pairs]].res_i.values
        )))

        return res_pairs
