
import os
import shutil
import sys
from tqdm import tqdm


if __name__ == "__main__":

    dir = sys.argv[1]
    dest = sys.argv[2]

    train = os.listdir(os.path.join(dir, "train"))[:500]
    test = os.listdir(os.path.join(dir, "test"))[:50]
    valid = os.listdir(os.path.join(dir, "valid"))[:50]
    splits = {"train": train, "test": test, "valid": valid}

    if not os.path.exists(dest):
        os.makedirs(dest)

    for fn in os.listdir(dir):
        x = os.path.join(dir, fn)
        if os.path.isfile(x) and not x.endswith(".zip"):
            shutil.copy(x, os.path.join(dest, fn))

    for split in ["train", "test", "valid"]:
        split_path = os.path.join(dest, split)
        os.makedirs(split_path)
        for x in tqdm(splits[split]):
            shutil.copy(
                os.path.join(dir, split, x), os.path.join(split_path, x)
            )
