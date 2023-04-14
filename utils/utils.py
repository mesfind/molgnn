
from rdkit import Chem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score
from torch.optim.lr_scheduler import _LRScheduler
#from torch.optim import Adam, Optimizer
from typing import Dict, Iterator, List, Optional, Union, OrderedDict, Tuple
# optimizer
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from gpaw import GPAW

def rdkit_to_ase(mol):
    '''
    # Load the molecular structure from SMILES string
    mol_smiles = 'CCCCC'
    mol = Chem.MolFromSmiles(mol_smiles)
    '''
    AllChem.EmbedMolecule(mol)
    coords = mol.GetConformer().GetPositions()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    return Atoms(symbols=symbols, positions=coords)



def parse_free_energy(filename):
    """ Parse out the free energy from a Psi4 vibrational analysis output file in a.u. """
    for line in open(filename).readlines():
        if "Correction G" in line:
            return float(line.split()[-2])

def parse_total_energy(filename):
    """ Parse out the total energy from a Psi4 vibrational analysis output file in a.u"""
    for line in open(filename).readlines():
        if "Total Energy" in line:
            return float(line.split()[-1])

def mol2xyz(mol,charge=0, multiplicity=1):
    #charge = Chem.GetFormalCharge(mol)
    xyz_string = "\n{} {}\n".format(charge, multiplicity)
    for atom in mol.GetAtoms():
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        xyz_string += "{} {} {} {}\n".format(atom.GetSymbol(), pos.x, pos.y, pos.z)
    return xyz_string


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where :code:`warmup_steps = warmup_epochs * steps_per_epoch`).
    Then the learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr` over the
    course of the remaining :code:`total_steps - warmup_steps` (where :code:`total_steps =
    total_epochs * steps_per_epoch`). This is roughly based on the learning rate
    schedule from `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_, section 5.3.
    """
    def __init__(self,
                 optimizer: 'Optimizer',
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float]):

        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        return list(self.lr)

    def step(self, current_step: int = None):
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]

def optimal_cutoff(target, predicted):
    """ Find the optimal probability cutoff point for classification
    ----------
    target: true labels
    predicted: positive probability predicted by the model.
    i.e. model.prdict_proba(X_test)[:, 1], NOT 0/1 prediction array
    Returns
    -------     
    cut-off value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    
    return round(list(roc_t['threshold'])[0], 2)

def plot_confusion_matrix(y_true, y_pred):
    # confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    data = conf_matrix.transpose()  
    
    _, ax = plt.subplots()
    ax.matshow(data, cmap="Blues")
    # printing exact numbers
    for (i, j), z in np.ndenumerate(data):
        ax.text(j, i, '{}'.format(z), ha='center', va='center')
    # axis formatting 
    plt.xticks([])
    plt.yticks([])
    plt.title("True label\n 0  {}     1\n".format(" "*18), fontsize=14)
    plt.ylabel("Predicted label\n 1   {}     0".format(" "*18), fontsize=14)
    
def draw_roc_curve(y_true, y_proba):
    '''
    y_true: 0/1 true labels for test set
    y_proba: model.predict_proba[:, 1] or probabilities of predictions
    
    Return:
        ROC curve with appropriate labels and legend 
    
    '''
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    
    _, ax = plt.subplots()
    
    ax.plot(fpr, tpr, color='r');
    ax.plot([0, 1], [0, 1], color='y', linestyle='--')
    ax.fill_between(fpr, tpr, label=f"AUC: {round(roc_auc_score(y_true, y_proba), 3)}")
    ax.set_aspect(0.90)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim(-0.02, 1.02);
    ax.set_ylim(-0.02, 1.02);
    plt.legend()
    plt.show()


def summerize_results(y_true, y_pred):
    '''
     Takes the true labels and the predicted probabilities
     and prints some performance metrics.
    '''
    print("\n=========================")
    print("        RESULTS")
    print("=========================")

    print("Accuracy: ", accuracy_score(y_true, y_pred).round(2))
    conf_matrix = confusion_matrix(y_true, y_pred)
    sensitivity = round(conf_matrix[1, 1]/(conf_matrix[1, 1] + conf_matrix[1, 0]), 2)
    specificity = round(conf_matrix[0, 0]/(conf_matrix[0, 0] + conf_matrix[0, 1]), 2)
    
    ppv = round(conf_matrix[1, 1]/(conf_matrix[1, 1] + conf_matrix[0, 1]), 2)
    npv = round(conf_matrix[0, 0]/(conf_matrix[0, 0] + conf_matrix[1, 0]), 2)
    
    print("-------------------------")
    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)
    
    print("-------------------------")
    
    print("positive predictive value: ", ppv)
    print("negative predictive value: ", npv)
    
    print("-------------------------")
    print("precision: ", precision_score(y_true, y_pred).round(2))
    print("recall: ", recall_score(y_true, y_pred).round(2))




from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from gpaw import GPAW, FermiDirac
from ase.optimize import BFGS

# Define a SMILES string for the molecule
smiles = 'CCO'

# Generate an RDKit molecule object from the SMILES string
rdkit_mol = Chem.MolFromSmiles(smiles)

# Add explicit hydrogens to the molecule
mol = Chem.AddHs(rdkit_mol)

# Check the number of conformers in the molecule
num_conformers = mol.GetNumConformers()

if num_conformers == 0:
    # Generate a new conformer for the molecule
    # Embed the molecule
    AllChem.EmbedMolecule(mol)

# Create a new RDKit molecule object and copy over the atom positions and properties
new_rdkit_mol = Chem.RWMol(mol)
for atom in mol.GetAtoms():
    new_atom = new_rdkit_mol.GetAtomWithIdx(atom.GetIdx())
    new_atom.SetAtomicNum(atom.GetAtomicNum())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetIsAromatic(atom.GetIsAromatic())

for i in range(num_conformers):
    try:
        conf = mol.GetConformer(i)
        new_rdkit_mol.AddConformer(conf, assignId=True)
    except ValueError:
        continue

# Convert the RDKit molecule to an ASE Atoms object
symbols = [a.GetSymbol() for a in new_rdkit_mol.GetAtoms()]
positions = new_rdkit_mol.GetConformer().GetPositions()
cell = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]  # Example cubic cell with side length 10 angstroms
pbc = [True, True, True]  # Example periodic boundary conditions
ase_atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)

# Set up the GPAW calculator and optimize the structure
calc = GPAW(mode='pw', xc='PBE', kpts=(8, 8, 8), nbands=-50, convergence={'bands': -25}, occupations=FermiDirac(0.1), txt='output.txt')

# Attach the calculator to the atoms object
ase_atoms.set_calculator(calc)
dyn = BFGS(ase_atoms)
dyn.run(fmax=0.01)


# Optimize the structure
ase_atoms.get_potential_energy()

# Calculate the first and second redox potentials
# First Redox Potential
HOMO = calc.get_homo()
LUMO = calc.get_lumo()
Redox1 = (LUMO - HOMO) * 27.2114

# Second Redox Potential
HOMO2 = calc.get_homo(n=1)
LUMO2 = calc.get_lumo(n=1)
Redox2 = (LUMO2 - HOMO2) * 27.2114

# Print the results
print(f"First Redox Potential: {Redox1:.2f} eV")
print(f"Second Redox Potential: {Redox2:.2f} eV")
