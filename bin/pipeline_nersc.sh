#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --constraint=gpu
#SBATCH --output='/global/u2/m/mpinon/_sbatch/all_particles_density.log'

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

python /global/u2/m/mpinon/densitysplit/bin/all_particles_density.py