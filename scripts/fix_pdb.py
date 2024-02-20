
import argparse
import glob
from itertools import repeat
import json
from multiprocessing import Pool
from pathlib import Path
import time

from pdbfixer import PDBFixer
from openmm.app import PDBFile


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("src_dir")
    parser.add_argument("--missing_res", action="store_true")
    parser.add_argument("--ns_res", action="store_true")
    parser.add_argument("--rm_heterogens", action="store_true")
    parser.add_argument("--missing_atoms", action="store_true")
    parser.add_argument("--missing_h", action="store_true")
    parser.add_argument("--dest_dir", type=str, default=None)

    return parser.parse_args()


def fix_pdb(pdb_fp: str, fix_args: argparse.Namespace, fix_fp: str):

    try:

        fixer = PDBFixer(filename=pdb_fp)

        if fix_args.missing_res:
            fixer.findMissingResidues()
        if fix_args.ns_res:
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
        if fix_args.rm_heterogens:
            fixer.removeHeterogens(keepWater=False)
        if fix_args.missing_atoms:
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
        if fix_args.missing_h:
            fixer.addMissingHydrogens(pH=7.0)

        with open(fix_fp, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)

        print(f"Fixed {pdb_fp}")

    except:

        print(f"Can't fix {pdb_fp}")

    return


if __name__ == "__main__":

    arguments = parse_args()
    src_dir = Path(arguments.src_dir)

    if arguments.dest_dir is None:
        dest_dir = src_dir.parent.joinpath("fixed")

    dataset_tag = str(int(time.time()))
    dest_dir = dest_dir.joinpath(dataset_tag)
    dest_dir.mkdir(parents=True)

    with open(dest_dir.joinpath("config.json"), "w") as f:
        json.dump(vars(arguments), f)

    pdb_fps = glob.glob(str(src_dir.joinpath("*.pdb")))
    Pool(processes=4).starmap(
        fix_pdb,
        zip(
            pdb_fps,
            repeat(arguments),
            [dest_dir.joinpath(Path(pdb_fp).name) for pdb_fp in pdb_fps]
        )
    )
