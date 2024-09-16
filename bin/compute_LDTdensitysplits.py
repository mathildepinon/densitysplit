import os
import sys
import copy
import argparse
import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm

from densitysplit import catalog_data, density_split
from densitysplit.density_split import DensitySplit
from densitysplit.lssfast import LDT, LDTDensitySplitModel, setup_logging

from density_split_corr import compute_lognormal_split_bins


if __name__ == '__main__':
    
    setup_logging()
        
    parser = argparse.ArgumentParser(description='density_splits_ldt')
    parser.add_argument('--env', type=str, required=False, default='feynman', choices=['feynman', 'nersc'])
    parser.add_argument('--imock', type=int, required=False, default=None)
    parser.add_argument('--redshift', type=float, required=False, default=None)
    parser.add_argument('--simulation', type=str, required=False, default='abacus', choices=['abacus', 'gaussian', 'lognormal'])
    parser.add_argument('--tracer', type=str, required=False, default='particles', choices=['particles', 'halos'])
    parser.add_argument('--nbar', type=float, required=False, default=0.0034)
    parser.add_argument('--cellsize', type=int, required=False, default=10)
    parser.add_argument('--cellsize2', type=int, required=False, default=None)
    parser.add_argument('--resampler', type=str, required=False, default='tsc')
    parser.add_argument('--smoothing_radius', type=int, required=False, default=None)
    parser.add_argument('--use_weights', type=bool, required=False, default=False)
    parser.add_argument('--rsd', type=bool, required=False, default=False)
    parser.add_argument('--los', type=str, required=False, default='x')
    parser.add_argument('--mu', type=float, required=False, default=None)
    parser.add_argument('--nsplits', type=int, required=False, default=3)
    
    args = parser.parse_args()

    z = args.redshift
    sep = np.linspace(0, 150, 51)

    if args.rsd and (args.mu is None):
        mu = np.linspace(-1, 1, 201)
    else:
        mu = args.mu
    
    if args.env == 'feynman':
        datadir = '/feynman/scratch/dphp/mp270220/abacus/'
    elif args.env == 'nersc':
        datadir = '/pscratch/sd/m/mpinon/abacus/'
    if args.tracer == 'halos':
        simname = 'AbacusSummit_2Gpc_z{:.3f}_{{}}'.format(z)
    elif args.tracer == 'particles':
        simname = 'AbacusSummit_2Gpc_z{:.3f}_{{}}_downsampled_particles_nbar{:.4f}'.format(z, args.nbar)

    # get nbar
    catalog = catalog_data.Data.load(os.path.join(datadir, simname.format('ph000')+'.npy'))
    nbar = catalog.size / catalog.boxsize**3

    ds_dir = '/feynman/work/dphp/mp270220/outputs/densitysplit/'
    ds_fn = simname+'_cellsize{:d}{}_resampler{}{}_3splits_randoms_size4_RH_CCF{}'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler, '_smoothingR{:02d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '', '_RSD' if args.rsd else '')
    
    # get sigma
    density_dir = '/feynman/work/dphp/mp270220/outputs/density/'
    density_fn = simname+'_cellsize{}_resampler{}{}_delta_R.npy'.format(args.cellsize, args.resampler, '_smoothingR{:02d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '')
    densities = [np.load(os.path.join(density_dir, density_fn.format('ph0{:02d}'.format(i)))) for i in range(25)]
    deltaR = np.concatenate(densities)
    sigma = np.std(deltaR)
    sigma_noshotnoise = np.sqrt(sigma**2 - 1 / (nbar * 4/3 * np.pi * args.smoothing_radius**3))

    # density bins defined from first mock
    f = os.path.join(density_dir, density_fn.format('ph000'))
    if os.path.isfile(f):
        delta_R_0 = np.load(f)
        bins = compute_lognormal_split_bins(delta_R_0, args.nsplits)
    else:
        raise FileNotFoundError('No file {}. Density of mock 0 needs to be computed to define bins!'.format(f))

    # xi
    if args.imock is not None:
        dsplit = DensitySplit.load(os.path.join(ds_dir, ds_fn.format('ph0{:02d}'.format(args.imock))+'.npy'))
        xiR = np.mean(dsplit.smoothed_corr(sep), axis=1)
    else:
        xiR = list()
        for i in range(25):
            dsplit = DensitySplit.load(os.path.join(ds_dir, ds_fn.format('ph0{:02d}'.format(i))+'.npy'))
            xiR.append(np.mean(dsplit.smoothed_corr(sep), axis=1))
        xiR = np.mean(np.array(xiR), axis=0)
    print('xi: ', xiR)

    from matplotlib import pyplot as plt

    plots_dir = '/feynman/home/dphp/mp270220/plots/densitysplit'
    plt.plot(sep, sep**2 * xiR)
    plt.savefig(os.path.join(plots_dir, 'xi_ldtmodel.pdf'), dpi=500)
    plt.close()

    # compute LDT model
    ldtmodel = LDT(redshift=z, smoothing_scale=args.smoothing_radius, nbar=nbar)
    ldtmodel.interpolate_sigma()
    ldtmodel.compute_ldt(sigma_noshotnoise)
    
    rhovals = np.linspace(0.1, 4, 100)
    density_pdf = ldtmodel.density_pdf(rhovals)
    plots_dir = '/feynman/home/dphp/mp270220/plots/densitysplit'
    plt.plot(rhovals, density_pdf)
    plt.savefig(os.path.join(plots_dir, 'density_ldtmodel.pdf'), dpi=500)
    plt.close()
    
    ldtdsplitmodel = LDTDensitySplitModel(ldtmodel, density_bins=bins)
    dsplits = ldtdsplitmodel.compute_dsplits(xiR)

    res = {'sep': sep, 'corr': dsplits}

    # save result
    outputdir = '/feynman/work/dphp/mp270220/outputs/densitysplit/'
    outputname = simname.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '25mocks') + '_cellsize{:d}{}_resampler{}{}_{:d}splits'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler, '_smoothingR{:02d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '', args.nsplits) + '_RH_CCF{}_LDT_model'.format('_RSD' if args.rsd else '')
    print('Saving result at {}'.format(os.path.join(outputdir, outputname)))
    np.save(os.path.join(outputdir, outputname), res)

