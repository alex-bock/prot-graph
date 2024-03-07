
import abc
import glob
import os
from typing import List, Tuple

import pandas as pd

from ..structures.structure import Structure


DATA_DIR = "./data"


class Dataset(abc.ABC):

    def __init__(self, data_dir: str = DATA_DIR):

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        self.data_dir = data_dir

        self.metadata = pd.DataFrame(columns=["id"])
        self.metadata.id = self.ids
        self.metadata.set_index("id", inplace=True)

        return

    @abc.abstractmethod
    def load_structure(self, id: str) -> Structure:

        raise NotImplementedError

    def _find_file(self, id: str) -> Tuple[str, str]:

        fn = id + self.ext
        fp = os.path.join(self.data_dir, fn)

        return fp

    @abc.abstractmethod
    def _download_record(self, url: str, dest_fp: str):

        raise NotImplementedError

    @property
    def fps(self) -> List[str]:

        return glob.glob(os.path.join(self.data_dir, f"*{self.ext}"))

    @property
    def ids(self) -> List[str]:

        return [os.path.basename(fp).split(".")[0] for fp in self.fps]

    def __getitem__(self, idx: int) -> str:

        return self.ids[idx]

    def load_metadata(self, df_fp: str):

        src_df = pd.read_csv(df_fp).set_index("id")
        self.metadata = self.metadata.join(src_df)

        return

    def filter_by_metadata(self, field: str, val: str):

        if field == "ec":
            df = self._filter_by_ec(val)
        else:
            df = self.metadata[self.metadata[field] == val]

        return df.index.values

    def _filter_by_ec(self, ec: str):

        return self.metadata[
            self.metadata.ec.apply(lambda x: ec in x)
        ]
