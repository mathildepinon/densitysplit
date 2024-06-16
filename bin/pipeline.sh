#!/bin/bash
#SBATCH --job-name=densitysplit
#SBATCH -c 32
#SBATCH --output='/feynman/home/dphp/mp270220/pipeline.log'

source /feynman/work/dphp/adematti/cosmodesi_environment.sh main

python /feynman/home/dphp/mp270220/densitysplit/bin/density_split_corr.py --redshift 0.8 --nbar 0.0034 --cellsize2 10