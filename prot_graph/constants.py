
DIST = "dist"

PEP = "peptide"
PEP_ATOMS = ["C", "N"]

HB = "hbond"
HB_ATOMS = [
    "N", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ", "O", "OD1",
    "OD2", "OE1", "OE2", "OG", "OG1", "OH"
]

HP = "hp"
HP_RES = ["ALA", "ILE", "LEU", "MET", "PHE", "PRO", "TRP", "TYR", "VAL"]

IB = "ib"
IB_POS_RES = ["ARG", "LYS", "HIS"]
IB_NEG_RES = ["ASP", "GLU"]

SB = "sb"
SB_ANION_RES = ["ASP", "GLU"]
SB_CATION_RES = ["ARG", "LYS"]
SB_ANIONS = ["OD1", "OD2", "OE1", "OE2"]
SB_CATIONS = ["NH1", "NH2", "NZ"]

DB = "db"
DB_RES = ["CYS"]
DB_ATOMS = ["SG"]

ID2ATOM_TYPE = {0: "U", 1: "C", 2: "H", 3: "N", 4: "O", 5: "P", 6: "S"}
ATOM_TYPE2ID = {v: k for k, v in ID2ATOM_TYPE.items()}
