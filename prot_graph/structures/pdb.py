
from Bio.PDB.Structure import Structure as PDBStructObj
import pandas as pd

from .structure import Structure


class PDBStructure(Structure):

    def add_atoms(self, struct: PDBStructObj) -> pd.DataFrame:

        atoms = list()
        i = 0
        res_i = 0

        for chain in struct.get_chains():
            chain_id = chain.get_id().capitalize()
            for _, res in enumerate(chain.get_residues()):
                chain_i = res.get_full_id()[3][1]
                res_type = res.get_resname()
                res_id = chain_id + str(chain_i)
                for atom in res.get_atoms():
                    atom_type = atom.get_name()
                    pos = atom.get_coord()
                    atoms.append(
                        dict(
                            i=i,
                            id=f"{res_id}_{atom_type}",
                            chain=chain_id,
                            chain_i=chain_i,
                            res_id=res_id,
                            res_type=res_type,
                            type=atom_type,
                            res_i=res_i,
                            x=pos[0],
                            y=pos[1],
                            z=pos[2]
                        )
                    )
                    i += 1
                res_i += 1

        return pd.DataFrame(atoms).set_index("i")
