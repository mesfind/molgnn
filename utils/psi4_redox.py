import psi4
import time
import argparse

def psi4_optimise(xyz, charge, mult, method):
    start_time = time.time()
    #psi4.core.set_output_file(f"opt_{str(charge)}.out") 
    psi4.core.clean()

    xyz[0] = f"{charge} {mult}\n"
    xyz = "".join(xyz)

    #print(xyz)

    mol = psi4.geometry(xyz)
    #mol.set_molecular_charge(charge)
    #mol.set_multiplicity(mult)

    energy = psi4.optimize(method,molecule=mol)
    print(f"Opt Runtime: {(time.time()-start_time)/60}")
    return mol,energy

def psi4_energy(mol, charge, mult, method):
    
    E_mol = mol.clone()
    start_time = time.time()
    psi4.core.set_output_file(f"energy_{str(charge)}.out")
    psi4.core.clean()
    
    E_mol.set_molecular_charge(charge)
    E_mol.set_multiplicity(mult)

    energy = psi4.energy(method,molecule=E_mol)
    print(f"Energy Runtime: {(time.time()-start_time)/60}")
    return energy

def main():
    import psi4
    parser = argparse.ArgumentParser(description="Run the Reorgs")

    parser.add_argument("xyz_files", type=str, help="XYZ file")

    parser.add_argument("-b","--basis_set", default="6-311G(D,P)", type=str, help="Basis Set")
    parser.add_argument("-f","--dft_functional", default="B3LYP", type=str, help="DFT Functional")
    parser.add_argument("-j","--cores", default=2, type=int, help="Number of Cores")

    args = parser.parse_args()

    #psi4.set_num_threads(args.cores)

    s2_time = time.time()

    with open(args.xyz_files,"r") as r:
        xyz = r.readlines()
        xyz.pop(0)

    basis = args.basis_set
    method = args.dft_functional

    #psi4.set_options({"basis":basis,
    #                    "geom_maxiter":1000,
    #                    "reference": "uks",
    #                    "MAXITER":1000})

    #psi4.set_memory(args.cores * 4000000000)

    #print("Starting Ground State Opt")
    #r_0 ,e_0_r_0 = psi4_optimise(xyz, 0, 1, method) # ground state opt

    print("Starting -1 State Opt")
    r_n, e_n_r_n = psi4_optimise(xyz, -1, 2, method) #optimised -1 state and energy

    #print("Starting Ground State Opt -1 Energy")
    #e_n_r_0 = psi4_energy(r_0, -1, 2, method) #optimised ground state energy at -1

    #print("Starting -1 State Opt 0 Energy")
    #e_0_r_n = psi4_energy(r_n, 0, 1, method) #optimised -1 state energy @ 0 charge


    print(f"Run Time: {(time.time()-s2_time)/60} minutes")
    #reorgE = (e_n_r_0 - e_0_r_0 + e_0_r_n - e_n_r_n) * 27.2113961
    #print(f"Reorg: {reorgE}")

    print(f"-1 opt energy: {e_n_r_n}")

    #print(r_0.save_string_xyz())
    print(r_n.save_string_xyz())
    #r_0.save_xyz_file('groundstate_opt.xyz',1)

if __name__ == "__main__":
    main()
