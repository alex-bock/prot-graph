
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
        self.ext = None
        self.metadata = pd.DataFrame()

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

    def __iter__(self) -> List[str]:

        return [os.path.basename(fp).split(".")[0] for fp in self.fps]

    def __getitem__(self, idx: int) -> str:

        return self.__iter__()[idx]
