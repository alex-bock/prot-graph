
import os
import glob

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.ProteinDataset")
@utils.copy_args(data.ProteinDataset.load_pdbs)
class ProteinDataset(data.ProteinDataset):

    """
    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    def __init__(self, path: str, verbose: bool = True, **kwargs):

        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        
        pkl_file = os.path.join(path, "protein_dataset.pkl.gz")
        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            pdb_files = []
            for split in ["train", "test", "valid"]:
                split_path = os.path.join(path, split)
                print(split_path)
                pdb_files += sorted(
                    glob.glob(os.path.join(split_path, "*.pdb"))
                )
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)

        return
    
    def get_item(self, index: int):

        if getattr(self, "lazy", False):
            protein = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
        else:
            protein = self.data[index].clone()

        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()

        item = {"graph": protein}
        if self.transform:
            item = self.transform(item)

        return item