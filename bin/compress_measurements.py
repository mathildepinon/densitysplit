import os
import sys
import argparse
import pickle
import logging
import numpy as np
from matplotlib import pyplot as plt

from pycorr import setup_logging
from densitysplit import CountInCellsDensitySplitMeasurement


if __name__ == '__main__':

    setup_logging()
    
    parser = argparse.ArgumentParser(description='density_split_measurements')
    parser.add_argument('--imock', type=int, required=False, default=None)
    parser.add_argument('--redshift', type=float, required=False, default=None)
    parser.add_argument('--tracer', type=str, required=False, default='particles', choices=['particles', 'halos'])
    parser.add_argument('--nbar', type=float, required=False, default=0.0034)
    parser.add_argument('--cellsize', type=int, required=False, default=10)
    parser.add_argument('--cellsize2', type=int, required=False, default=None)
    parser.add_argument('--resampler', type=str, required=False, default='tsc')
    parser.add_argument('--smoothing_radius', type=int, required=False, default=None)
    parser.add_argument('--use_weights', type=bool, required=False, default=False)
    parser.add_argument('--rsd', type=bool, required=False, default=False)
    parser.add_argument('--los', type=str, required=False, default='x')
    parser.add_argument('--nsplits', type=int, required=False, default=3)
    parser.add_argument('--randoms_size', type=int, required=False, default=4)
    parser.add_argument('--sep', type=float, nargs='+', required=False, default=None)
    
    args = parser.parse_args()
    z = args.redshift
    ells = [0, 2, 4] if args.rsd else [0]
    nells = len(ells)
    interpolate = False
    sep = list(args.sep)

    # Directories
    data_dir = '/feynman/scratch/dphp/mp270220/abacus/'
    ds_dir = '/feynman/work/dphp/mp270220/outputs/densitysplit/'
    mesh_dir = '/feynman/scratch/dphp/mp270220/outputs'
    plots_dir = '/feynman/home/dphp/mp270220/plots/densitysplit'
    
    # Filenames
    if args.tracer == 'halos':
        sim_name = 'AbacusSummit_2Gpc_z{:.3f}_{{}}'.format(z)
    elif args.tracer == 'particles':
        sim_name = 'AbacusSummit_2Gpc_z{:.3f}_{{}}_downsampled_particles_nbar{:.4f}'.format(z, args.nbar)
    base_name = sim_name + '_cellsize{:d}{}_resampler{}{}'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler, '_smoothingR{:d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '')
    ds_name = base_name + '_3splits_randoms_size4_RH_CCF{}'.format('_RSD' if args.rsd else '')

    # Load measured density splits
    nmocks = 25 if args.nbar < 0.01 else 8
    mocks_fns = [os.path.join(ds_dir, ds_name.format('ph0{:02d}'.format(i))+'.npy') for i in range(nmocks)]

    measurements = CountInCellsDensitySplitMeasurement(mocks_fns)
    measurements.set_pdf1D()

    for s in sep:
        measurements.set_bias_function(s)
        measurements.set_pdf2D(s, swidth=0.1)

    measurements.set_densitysplits()

    outputname = ds_name.format('{}mocks'.format(nmocks)) + '_compressed'
    measurements.save(os.path.join(ds_dir, outputname))
    
    
