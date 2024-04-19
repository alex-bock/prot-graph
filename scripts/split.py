
from ast import literal_eval
import json
import os
import shutil
import time
from tqdm import tqdm
from typing import Sequence

import pandas as pd


def get_ecs(df: pd.DataFrame) -> Sequence[str]:

    ecs = []

    for _, row in df.iterrows():
        ecs.extend(literal_eval(row.ec))

    return set(ecs)


if __name__ == "__main__":

    with open("./data/gearnet/splits.json", "r") as f:
        splits = json.load(f)

    df = pd.read_csv("./data/gearnet/ec.csv")

    train_df = df[df.id.isin(splits["train"])]
    test_df = df[df.id.isin(splits["test"])]
    valid_df = df[df.id.isin(splits["valid"])]

    train_ecs = get_ecs(train_df)
    test_ecs = get_ecs(test_df)
    valid_ecs = get_ecs(valid_df)

    print(f"{len(train_df)} structures in train split covering {len(train_ecs)} EC labels")
    print(f"{len(test_df)} structures in test split covering {len(test_ecs)} EC labels ({len(test_ecs - train_ecs)} not in train split)")
    print(f"{len(valid_df)} structures in valid split covering {len(valid_ecs)} EC labels ({len(valid_ecs - train_ecs)} not in train split)")

    n_train = 500
    train_subset_df = train_df.sample(n_train)
    train_subset_ecs = get_ecs(train_subset_df)
    print(f"Sampled {n_train} structures for training subset covering {len(train_subset_ecs)} EC labels")

    complement_df = df[~df.id.isin(train_subset_df.id.values)]
    complement_df["seen"] = complement_df.ec.apply(
        lambda ecs: all([ec in train_subset_ecs for ec in literal_eval(ecs)])
    )
    seen_df = complement_df[complement_df.seen]
    print(f"{len(seen_df)} structures outside training subset with EC labels seen in training subset")

    test_seen_df = seen_df[seen_df.id.isin(test_df.id.values)]
    valid_seen_df = seen_df[seen_df.id.isin(valid_df.id.values)]
    print(f"{len(test_seen_df)} structures in test split with EC labels seen in training subset")
    print(f"{len(valid_seen_df)} structures in valid split with EC labels seen in training subset")

    n_test = 50
    n_valid = 50
    test_subset_df = test_seen_df.sample(n_test)
    valid_subset_df = valid_seen_df.sample(n_valid)

    subset_splits = {
        "train": train_subset_df.id.values,
        "test": test_subset_df.id.values,
        "valid": valid_subset_df.id.values
    }

    subset_id = str(int(time.time()))
    subset_dir = os.path.join("./data/gearnet/subsets/", subset_id)
    os.makedirs(subset_dir)
    for split in ("train", "test", "valid"):
        for pdb_id in tqdm(subset_splits[split]):
            src_fp = f"./data/gearnet/fixed/1711403862/{pdb_id}.pdb"
            shutil.copy(src_fp, subset_dir)
            shutil.copy(src_fp,
                os.path.join(
                    "/Users/moose/scratch/protein-datasets/small/EnzymeCommission/",
                    split
                )
            )