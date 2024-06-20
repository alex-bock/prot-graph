
import functools
import glob
import os
import subprocess
import sys
from tqdm.contrib.concurrent import process_map
from typing import List


def run_gc(gc_args: List[str], dest_dir: str, pdb_fp: str):

    subprocess.call(
        [
            "get_static_contacts.py", "--structure", pdb_fp, "--output",
            os.path.join(dest_dir, os.path.basename(pdb_fp)), "--itypes"
        ] + gc_args
    )

    return


if __name__ == "__main__":

    src_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    gc_args = sys.argv[3:]

    pdb_fps = glob.glob(os.path.join(src_dir, "*.pdb"))
    process_map(functools.partial(run_gc, gc_args, dest_dir), pdb_fps)
