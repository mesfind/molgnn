from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from psikit import Psikit
smi = 'COc1cccnc1'
mol = Chem.MolFromSmiles(smi)
AllChem.Compute2DCoords(mol)
#mol


pk = Psikit()
pk.read_from_smiles(smi)
pk.optimize()

pk.getMOview()
