
import os
import urllib.request

from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure

from .dataset import Dataset


PDB_DIR = "./data/pdb/"
PDB_EXT = ".pdb"
PDB_URL = "http://files.rcsb.org/download/"


class PDB(Dataset):

    def __init__(self, pdb_dir: str = PDB_DIR):

        super().__init__(data_dir=pdb_dir)

        self.ext = PDB_EXT
        self.parser = PDBParser(QUIET=True)

        return

    def download_record(self, id: str, replace: bool = False):

        pdb_fn = id + PDB_EXT
        pdb_fp = os.path.join(self.data_dir, pdb_fn)

        if os.path.exists(pdb_fp) and not replace:
            return

        urllib.request.urlretrieve(os.path.join(PDB_URL, pdb_fn), pdb_fp)

        return
    
    def find_file(self, id: str) -> str:

        fp = os.path.join(self.data_dir, id + self.ext)

        if not os.path.exists(fp):
            raise FileNotFoundError
        
        return fp

    def load_record(self, id: str) -> Structure:

        pdb_fp = os.path.join(self.data_dir, id + self.ext)

        if not os.path.exists(pdb_fp):
            raise FileNotFoundError

        return self.parser.get_structure(id, pdb_fp)
