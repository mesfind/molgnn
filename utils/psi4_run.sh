#!/bin/bash
#SBATCH --job-name=PENCEN_psi4
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --time=1:00:00
#SBATCH --partition=scavenger

python psi4_redox.py PENCEN_uff.xyz -j 20
