#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --time=00:30:00
#SBATCH --array=0-4
#SBATCH --output='/global/u2/m/mpinon/_sbatch/job_%j.log'

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

python /global/u2/m/mpinon/densitysplit/bin/density_split_corr.py --env 'nersc' --redshift 0.8 --nbar 0.0034 --cellsize2 10 --imock $SLURM_ARRAY_TASK_ID