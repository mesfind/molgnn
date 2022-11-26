import psi4
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
psi4.core.set_output_file('output1.dat', True)


def mol2xyz(mol):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, useExpTorsionAnglePrefs=True,useBasicKnowledge=True)
    AllChem.UFFOptimizeMolecule(mol)
    atoms = mol.GetAtoms()
    string = string = "\n"
    for i, atom in enumerate(atoms):
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        string += "{} {} {} {}\n".format(atom.GetSymbol(), pos.x, pos.y, pos.z)
    string += "units angstrom\n"
    return string, mol

mol = Chem.MolFromSmiles('O=C1CC(=O)C=C1C(=O)O')
xyz, mol=mol2xyz(mol)
psi4.set_memory('4 GB')
psi4.set_num_threads(4)
benz = psi4.geometry(xyz)
scf_e, scf_wfn = psi4.energy('B3LYP/cc-pVDZ', return_wfn=True)
