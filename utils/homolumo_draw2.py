import sys
sys.path.append('/Users/admin/devel/psikit/psikit')
from psikit import Psikit
from rdkit import Chem
pk = Psikit()
# acetic acid
pk.read_from_smiles("CC(=O)[O-]")
pk.optimize()
Chem.MolToMolFile(pk.mol, 'acetic_acid.mol')
pk.getMOview()
 
# tetrazole
pk.read_from_smiles("CC1=NN=N[N-]1")
pk.optimize()
pk.getMOview()