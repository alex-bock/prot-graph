
import os
import urllib.request

from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure


DEFAULT_PDB_CACHE_DIR = "./data/pdb/"
PDB_EXT = ".pdb"
PDB_URL = "http://files.rcsb.org/download/"


class PDB:

    def __init__(self, cache_dir: str = DEFAULT_PDB_CACHE_DIR):

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.cache_dir = cache_dir
        self.parser = PDBParser(QUIET=True)

        return

    def download_pdb_record(self, pdb_id: str, replace: bool = False):

        pdb_fn = pdb_id + PDB_EXT
        pdb_fp = os.path.join(self.cache_dir, pdb_fn)

        if os.path.exists(pdb_fp) and not replace:
            return

        urllib.request.urlretrieve(os.path.join(PDB_URL, pdb_fn), pdb_fp)

        return

    def load_pdb_structure(self, pdb_id: str) -> Structure:

        pdb_fp = os.path.join(self.cache_dir, pdb_id + PDB_EXT)

        if not os.path.exists(pdb_fp):
            raise FileNotFoundError

        return self.parser.get_structure(pdb_id, pdb_fp)
