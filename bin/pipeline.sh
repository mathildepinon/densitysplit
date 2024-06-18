#!/bin/bash
#SBATCH --job-name=densitysplit
#SBATCH -c 32
#SBATCH --mem 30GB
#SBATCH --array=1-24
#SBATCH --output='/feynman/home/dphp/mp270220/logs/densitysplit_%a.log'

source /feynman/work/dphp/adematti/cosmodesi_environment.sh main

python /feynman/home/dphp/mp270220/densitysplit/bin/density_split_corr.py --redshift 0.8 --nbar 0.0034 --cellsize2 10 --imock $SLURM_ARRAY_TASK_ID --rsd True