
import argparse
import functools
import glob
import json
from pathlib import Path
from subprocess import Popen, PIPE
import time
from tqdm.contrib.concurrent import process_map

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
    parser.add_argument("--prot", action="store_true")
    parser.add_argument("--dest_dir", type=str, default=None)
    parser.add_argument("--tag", action="store_true")

    return parser.parse_args()


def fix_pdb(fix_args: argparse.Namespace, dest_dir: Path, pdb_fp: str):

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

        fix_fp = dest_dir.joinpath(Path(pdb_fp).name)
        with open(fix_fp, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)

        print(f"Fixed {pdb_fp}")

        if fix_args.prot:
            protonate(fix_fp, fix_fp)

    except:

        print(f"Can't fix {pdb_fp}")

    return


def protonate(in_pdb_file: str, out_pdb_file: str):

    args = ["reduce", "-Trim", in_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, _ = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode("utf-8").rstrip())
    outfile.close()

    args = ["reduce", "-HIS", out_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, _ = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode("utf-8"))
    outfile.close()

    return


if __name__ == "__main__":

    arguments = parse_args()
    src_dir = Path(arguments.src_dir)

    if arguments.dest_dir is None:
        dest_dir = src_dir.parent.joinpath("fixed")
    else:
        dest_dir = Path(arguments.dest_dir)

    if arguments.tag:
        dataset_tag = str(int(time.time()))
        dest_dir = dest_dir.joinpath(dataset_tag)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with open(dest_dir.joinpath("config.json"), "w") as f:
        json.dump(vars(arguments), f)

    pdb_fps = glob.glob(str(src_dir.joinpath("*.pdb")))
    r = process_map(functools.partial(fix_pdb, arguments, dest_dir), pdb_fps)
