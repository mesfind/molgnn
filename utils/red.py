import psi4
import numpy as np
from psikit import Psikit
from utils import parse_free_energy, mol2xyz
psi4.set_memory('2 GB')
pk = Psikit()
pk.read_from_smiles('C1=CC=C(C=C1)[N+](=O)[O-]') #ntrobez
charge=-1
multip=2
mol = pk.mol
xyz = mol2xyz(mol,charge=charge,multiplicity=multip)
nitrobenz = psi4.geometry(xyz)

# Part A
molecule_name = "nitrobenzene"
#nitrobenz = psi4.geometry(f"""
#-1 2
#C       -3.5048225421    0.0711805817    0.1456897967
#C       -2.1242069042    0.0676957680    0.1437250554
#C       -1.4565144627    1.2657898054    0.0112805274
#C       -2.1243502782    2.4616659201   -0.1394727314
#C       -3.5049153121    2.4578370923   -0.1457245349
#C       -4.1936081427    1.2645153194    0.0001955136
#H       -4.0381801262   -0.8505059514    0.2559173303
#H       -1.5620288767   -0.8346363876    0.2389155097
#H       -1.5619534389    3.3630228735   -0.2428628637
#H       -4.0382012347    3.3785626398   -0.2639829256
#H       -5.2650389640    1.2641688843   -0.0022762561
#N       -0.0085078655    1.2648596634   -0.0056641832
#O        0.5639468379    0.1670702678   -0.1297708787
#O        0.5668300231    2.3598431617    0.1306822195
#""")

basis = 'cc-pVDZ'# 'B3LYP/3-21G' 
reference =  'uks' # rhf for singlet and uks for reduction
psi4.core.clean()


psi4.set_output_file(F'{molecule_name}_initial_energy_red.dat', False)
psi4.set_options({'reference': 'uks'})
psi4.energy('B3LYP/3-21G')

def parse_total_energy(filename):
    """ Parse out the free energy from a Psi4 vibrational analysis output file in a.u. """
    for line in open(filename).readlines():
        if "Total Energy =" in line:
            return float(line.split()[-1])

# Part B

psi4.set_output_file(F'{molecule_name}_geometry_optimization_red.dat', False)
psi4.set_options({'g_convergence': 'gau_tight'}) # this forces the optimizer to get close to the minimum
psi4.optimize('B3LYP/3-21G', molecule=nitrobenz)

#Part C

psi4.set_output_file(F'{molecule_name}_frequency_analysis_red.dat', False)
b3lyp_321g_energy, b3lyp_321g_wfn = psi4.frequency('B3LYP/3-21G', molecule=nitrobenz, return_wfn=True, dertype='gradient')

def parse_free_energy(filename):
    """ Parse out the free energy from a Psi4 vibrational analysis output file in a.u. """
    for line in open(filename).readlines():
        if "Correction G" in line:
            return float(line.split()[-2])
parse_free_energy(F'{molecule_name}_frequency_analysis_red.dat')
red_e = parse_total_energy(F'{molecule_name}_frequency_analysis_red.dat')
print("Total Energy of reduction(-1):",red_e)
## Part D
psi4.set_options({
  'pcm': True,
  'pcm_scf_type': 'total',
})

psi4.pcm_helper("""
   Units = Angstrom
   Medium {
   SolverType = CPCM
   Solvent = Acetonitrile
   }
   Cavity {
   RadiiSet = UFF
   Type = GePol
   Scaling = False
   Area = 0.3
   Mode = Implicit
   }
""")

psi4.set_output_file(F'{molecule_name}_solvent_energy_red.dat', False)
psi4.energy('B3LYP/heavy-aug-cc-pVDZ', molecule=nitrobenz)
