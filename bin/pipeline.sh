#!/bin/bash
#SBATCH --job-name=densitysplit
#SBATCH --mem=64GB
#SBATCH --output='/feynman/home/dphp/mp270220/logs/densitysplit.log'
#SBATCH -p htc

#source /feynman/work/dphp/adematti/cosmodesi_environment.sh main

#python /feynman/home/dphp/mp270220/densitysplit/bin/density_split_corr.py --redshift 0.8 --nbar 0.0034 --imock $SLURM_ARRAY_TASK_ID --rsd True
python /feynman/home/dphp/mp270220/densitysplit/bin/density_split_corr.py --redshift 0.8 --nbar 0.003443 --imock 0 --cellsize 5 --cellsize2 5 --resampler 'tophat' --smoothing_radius 10 --todo 'ds_data_corr'

#python /feynman/home/dphp/mp270220/densitysplit/bin/compute_LDTdensitysplits.py --redshift 0.8 --nbar 0.0034 --cellsize 2 --cellsize2 2 --resampler 'ngp' --smoothing_radius 10 --imock 0

#python /feynman/home/dphp/mp270220/densitysplit/bin/plot_densitysplits.py --redshift 0.8 --nbar 0.0034 --cellsize 2 --cellsize2 2 --resampler 'ngp' --smoothing_radius 10
#python /feynman/home/dphp/mp270220/densitysplit/bin/save_density.py

#python /feynman/home/dphp/mp270220/densitysplit/bin/get_abacus_particles.py

#python /feynman/home/dphp/mp270220/densitysplit/bin/density_pdf2D.py --redshift 0.8 --nbar 0.0034 --cellsize2 10 --tracer 'particles' --method 'flow' --size 10000000

#python /feynman/home/dphp/mp270220/densitysplit/bin/density_pdf2D.py --redshift 0.8 --nbar 0.0034 --cellsize2 10 --tracer 'particles' --method 'gram-charlier' --bins 1000 --exporder 6

#python /feynman/home/dphp/mp270220/densitysplit/bin/bias_function.py --redshift 0.8 --nbar 0.003443 --tracer 'particles' --cellsize 5 --cellsize2 5 --resampler 'tophat' --smoothing_radius 10 --bins 50 --sep 70

#python /feynman/home/dphp/mp270220/densitysplit/bin/plots.py --redshift 0.8 --nbar 0.003443 --tracer 'particles' --cellsize 5 --cellsize2 5 --resampler 'tophat' --smoothing_radius 10 --sep 70

#python /feynman/home/dphp/mp270220/densitysplit/bin/plots.py --redshift 0.8 --nbar 0.003443 --imock 0 --tracer 'particles' --cellsize 2 --cellsize2 2 --resampler 'ngp' --smoothing_radius 15 --sep 40