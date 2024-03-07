
from collections import defaultdict
import os
from pathlib import Path
import sys

import pandas as pd

sys.path.append(os.getcwd())


if __name__ == "__main__":

    ec_fp = Path(sys.argv[1])

    with open(ec_fp, "r") as f:
        lines = f.readlines()

    records = defaultdict(list)
    for line in lines[3:]:
        pdb_id, ec_ids_str = line.strip().split("\t")
        ec_ids = ec_ids_str.split(",")
        for ec_id in filter(lambda ec_id: ec_id.endswith("-"), ec_ids):
            records[pdb_id].append(ec_id)

    max_num_ecs = max([len(ecs) for ecs in records.values()])
    init_ec_cols = [i for i in range(max_num_ecs)]
    ec_df = pd.DataFrame.from_dict(
        records, orient="index", columns=init_ec_cols
    )
    ec_df.index.name = "id"
    ec_df["ec"] = [
        [ec_id for ec_id in rec if ec_id is not None]
        for rec in ec_df.values.tolist()
    ]
    ec_df.drop(init_ec_cols, axis=1, inplace=True)
    ec_df.to_csv(ec_fp.parent.joinpath("ec.csv"))
