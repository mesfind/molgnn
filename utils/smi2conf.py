from rdkit import Chem
from rdkit.Chem import AllChem
from MolTo3DView import MolTo3DView 
def smi2conf(smiles):
    '''Convert SMILES to rdkit.Mol with 3D coordinates'''
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        return mol
    else:
        return None

smi = 'COc3nc(OCc2ccc(C#N)c(c1ccc(C(=O)O)cc1)c2P(=O)(O)O)ccc3C[NH2+]CC(I)NC(=O)C(F)(Cl)Br'
conf = smi2conf(smi)
viewer = MolTo3DView(conf, size=(600, 300), style='sphere')
viewer.show()
