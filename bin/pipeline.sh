#!/bin/bash
#SBATCH --job-name=densitysplit
#SBATCH --mem=64GB
#SBATCH --array=0
#SBATCH --output='/feynman/home/dphp/mp270220/logs/densitysplits_ELG_compress.log'
#SBATCH -p htc

source /feynman/work/dphp/adematti/cosmodesi_environment.sh main
source activate $ANACONDA_DIR

module load cports/rhel-8.x86_64 # for texlive
module load texlive

######################################
# Compute density split measurements #
######################################

#python /feynman/home/dphp/mp270220/densitysplit/bin/density_split_corr.py --redshift 0.8 --nbar 0.0034 --imock $SLURM_ARRAY_TASK_ID --cellsize 5 --cellsize2 5 --resampler 'tophat' --nsplits 3 --smoothing_radius 10 --todo ds_data_corr --rsd True --nthreads 32
#python /feynman/home/dphp/mp270220/densitysplit/bin/density_split_corr.py --redshift 0.8 --nbar 0.002 --tracer ELG --imock $SLURM_ARRAY_TASK_ID --cellsize 5 --cellsize2 5 --resampler 'tophat' --smoothing_radius 10 --todo ds_data_corr --nthreads 64

########################################################
# Save density split measurements in compressed format #
########################################################

#python /feynman/home/dphp/mp270220/densitysplit/bin/compress_measurements.py --redshift 0.8 --nbar 0.0034 --tracer 'particles' --cellsize 5 --cellsize2 5 --resampler 'tophat' --smoothing_radius 10 --nsplits 3 --sep 40 --rsd True
#python /feynman/home/dphp/mp270220/densitysplit/bin/compress_measurements.py --redshift 0.8 --nbar 0.0005 --tracer 'particles' --cellsize 5 --cellsize2 5 --resampler 'tophat' --smoothing_radius 10 --nsplits 3 --sep {0..150..5}
python /feynman/home/dphp/mp270220/densitysplit/bin/compress_measurements.py --redshift 0.8 --nbar 0.002 --tracer 'ELG' --cellsize 5 --cellsize2 5 --resampler 'tophat' --smoothing_radius 10 --sep 20 40 #{0..150..10}

#####################################
# Plots (can be run in the console) #
#####################################

#python /feynman/home/dphp/mp270220/densitysplit/bin/plot.py --redshift 0.8 --nbar 0.0034 --tracer 'particles' --cellsize 5 --cellsize2 5 --resampler 'tophat' --smoothing_radius 10 --to_plot pdf1D