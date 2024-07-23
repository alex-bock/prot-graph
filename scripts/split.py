
import os
import shutil
import sys
from tqdm import tqdm


if __name__ == "__main__":

    dir = sys.argv[1]
    dest = sys.argv[2]

    train = [x for x in os.listdir(os.path.join(dir, "train"))[:500] if x in os.listdir(os.path.join(dir, "contacts"))]
    test = [x for x in os.listdir(os.path.join(dir, "test"))[:50] if x in os.listdir(os.path.join(dir, "contacts"))]
    valid = [x for x in os.listdir(os.path.join(dir, "valid"))[:50] if x in os.listdir(os.path.join(dir, "contacts"))]
    splits = {"train": train, "test": test, "valid": valid}

    if not os.path.exists(dest):
        os.makedirs(dest)

    for fn in os.listdir(dir):
        x = os.path.join(dir, fn)
        if os.path.isfile(x) and not x.endswith(".zip"):
            shutil.copy(x, os.path.join(dest, fn))

    pdb_ids = []
    for split in ["train", "test", "valid"]:
        split_path = os.path.join(dest, split)
        os.makedirs(split_path)
        for x in tqdm(splits[split]):
            shutil.copy(
                os.path.join(dir, split, x), os.path.join(split_path, x)
            )
            pdb_ids.append(x)
    
    dir_contacts_path = os.path.join(dir, "contacts")
    if os.path.exists(dir_contacts_path):
        dest_contacts_path = os.path.join(dest, "contacts")
        os.makedirs(dest_contacts_path)
        for pdb_id in pdb_ids:
            shutil.copy(
                os.path.join(dir_contacts_path, pdb_id), 
                os.path.join(dest_contacts_path, pdb_id)
            )
