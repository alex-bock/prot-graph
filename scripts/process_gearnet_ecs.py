
import os
from pathlib import Path
import sys

import pandas as pd

sys.path.append(os.getcwd())

from prot_graph.datasets import PDB


if __name__ == "__main__":

    ec_fp, data_dir = Path(sys.argv[1]), Path(sys.argv[2])

    with open(ec_fp, "r") as f:
        lines = f.readlines()

    records = []
    for line in lines[3:]:
        pdb_id, ec_ids_str = line.strip().split("\t")
        ec_ids = ec_ids_str.split(",")
        for ec_id in filter(lambda ec_id: ec_id.endswith("-"), ec_ids):
            [ec_1, ec_2, ec_3] = ec_id.split(".")[:3]
            records.append(
                {"pdb_id": pdb_id, "ec_1": ec_1, "ec_2": ec_2, "ec_3": ec_3}
            )

    ec_df = pd.DataFrame(records)
    ec_df.to_csv(ec_fp.parent.joinpath("ec.csv"))
