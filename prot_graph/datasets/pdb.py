
import os
import urllib.request

from Bio.PDB import PDBParser

from .dataset import Dataset
from ..structures import PDBStructure


PDB_DIR = "./data/pdb/"
PDB_EXT = ".pdb"
PDB_URL = "http://files.rcsb.org/download/"


class PDB(Dataset):

    def __init__(self, pdb_dir: str = PDB_DIR):

        super().__init__(data_dir=pdb_dir)

        self.ext = PDB_EXT
        self._pdb_parser = PDBParser(QUIET=True)

        return

    def load_structure(self, id: str, replace: bool = False) -> PDBStructure:

        pdb_fp = self._find_file(id)

        if not os.path.exists(pdb_fp) or replace:
            self._download_record(
                os.path.join(PDB_URL, os.path.basename(pdb_fp)), pdb_fp
            )

        pdb_struct = self._pdb_parser.get_structure(id, pdb_fp)

        return PDBStructure(id, pdb_struct)

    def _download_record(self, url: str, dest_fp: str):

        urllib.request.urlretrieve(url, dest_fp)

        return
