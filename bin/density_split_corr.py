import os
import sys
import copy
import argparse
import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm

from cosmoprimo import *
from pycorr import TwoPointCorrelationFunction, setup_logging

from densitysplit import catalog_data, density_split


def compute_lognormal_split_bins(delta_R, nsplits):
    m2 = np.mean(delta_R**2)
    m3 = np.mean(delta_R**3)
    def tomin(delta0):
        return (m3 - 3/delta0 * m2**2 - 1/delta0**3 * m2**3)**2
    res = minimize(tomin, x0=1.)
    delta0 = res.x[0]
    sigma = np.sqrt(np.log(1 + m2/res.x[0]**2))
    splits = np.linspace(0, 1, nsplits+1)
    bins = lognorm.ppf(splits, sigma, -delta0, delta0*np.exp(-sigma**2/2.))
    bins[0] = -1
    return bins


if __name__ == '__main__':
    
    setup_logging()
        
    parser = argparse.ArgumentParser(description='density_split_corr')
    parser.add_argument('--todo', type=str, nargs='+', required=False, default=['density', 'ds_data_corr'], choices=['density', 'ds_data_corr'])
    parser.add_argument('--env', type=str, required=False, default='feynman', choices=['feynman', 'nersc'])
    parser.add_argument('--imock', type=int, required=False, default=0)
    parser.add_argument('--redshift', type=float, required=False, default=None)
    parser.add_argument('--simulation', type=str, required=False, default='abacus', choices=['abacus', 'gaussian', 'lognormal'])
    parser.add_argument('--tracer', type=str, required=False, default='particles', choices=['particles', 'halos'])
    parser.add_argument('--nbar', type=float, required=False, default=0.0034)
    parser.add_argument('--cellsize', type=int, required=False, default=10)
    parser.add_argument('--cellsize2', type=int, required=False, default=None)
    parser.add_argument('--resampler', type=str, required=False, default='tsc')
    parser.add_argument('--smoothing_radius', type=int, required=False, default=None)
    parser.add_argument('--smoothing_radius2', type=int, required=False, default=None)
    parser.add_argument('--use_weights', type=bool, required=False, default=False)
    parser.add_argument('--rsd', type=bool, required=False, default=False)
    parser.add_argument('--los', type=str, required=False, default='x')
    parser.add_argument('--nsplits', type=int, required=False, default=3)
    parser.add_argument('--randoms_size', type=int, required=False, default=4)
    
    args = parser.parse_args()

    z = args.redshift
    
    # Edges (s, mu) to compute correlation function at
    mode = 'smu'
    edges = (np.linspace(0., 150., 51), np.linspace(-1, 1, 201))
    #mode = 's'
    #edges = np.linspace(0., 150., 51)

    if args.simulation == 'abacus':
        cosmology=fiducial.AbacusSummitBase()
        if args.env == 'feynman':
            datadir = '/feynman/scratch/dphp/mp270220/abacus/'
        elif args.env == 'nersc':
            datadir = '/pscratch/sd/m/mpinon/abacus/'
        if args.tracer == 'halos':
            simname0 = 'AbacusSummit_2Gpc_z{:.3f}_ph0{{:02d}}'.format(z)
            simname = simname0.format(args.imock)
        elif args.tracer == 'particles':
            simname0 = 'AbacusSummit_2Gpc_z{:.3f}_ph0{{:02d}}_downsampled_particles_nbar{:.4f}'.format(z, args.nbar)
            simname = simname0.format(args.imock)

        cosmology = fiducial.AbacusSummitBase()
        bg = cosmology.get_background()
        hz = 100 * bg.efunc(z)

        # compute density contrast
        mock = catalog_data.Data.load(os.path.join(datadir, simname+'.npy'))
        mock_density = density_split.DensitySplit(mock)
        mock_density.compute_density(data=mock, cellsize=args.cellsize, resampler=args.resampler, smoothing_radius=args.smoothing_radius, cellsize2=args.cellsize2, smoothing_radius2=args.smoothing_radius2, use_rsd=args.rsd, los=args.los, hz=hz, use_weights=args.use_weights, return_counts=True)

        if args.resampler!='tophat':
            density_fn = os.path.join('/feynman/scratch/dphp/mp270220/outputs', simname + '_cellsize{:d}{}_resampler{}{}'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler, '_smoothingR{:d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '') + '_density_mesh{}'.format('_RSD' if args.rsd else ''))
            mock_density.save_mesh(density_fn)

        if 'density' in args.todo:
            if args.resampler=='tophat':
                from densitysplit.cic_density import compute_cic_density
                delta_R = mock_density.readout_density(positions='mesh', resampler=args.resampler)
                outputname = simname + '_cellsize{:d}_resampler{}{}_N{}'.format(args.cellsize, args.resampler, '_smoothingR{:02d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '', '_RSD' if args.rsd else '')
            else:
                delta_R = mock_density.readout_density(positions='randoms', resampler=args.resampler, seed=args.imock)
                outputname = simname + '_cellsize{:d}_resampler{}{}_delta_R{}'.format(args.cellsize, args.resampler, '_smoothingR{:02d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '', '_RSD' if args.rsd else '')
            print('Save density at: {}'.format(os.path.join('/feynman/work/dphp/mp270220/outputs/density', outputname)))
            np.save(os.path.join('/feynman/work/dphp/mp270220/outputs/density', outputname), delta_R)

        if 'ds_data_corr' in args.todo:
            # compute density splits
            # define split bins
            if args.resampler=='tophat':
                delta0name = simname0.format(0) + '_cellsize{:d}_resampler{}{}_N{}.npy'.format(args.cellsize, args.resampler, '_smoothingR{:02d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '', '_RSD' if args.rsd else '')
                nbar = args.nbar
                norm = nbar * 4/3 * np.pi * args.smoothing_radius**3
            else:
                delta0name = simname0.format(0) + '_cellsize{:d}_resampler{}{}_delta_R{}.npy'.format(args.cellsize, args.resampler, '_smoothingR{:02d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '', '_RSD' if args.rsd else '')
                norm = None
            f = os.path.join('/feynman/work/dphp/mp270220/outputs/density', delta0name)
            if os.path.isfile(f):
                if args.resampler=='tophat':
                    delta_R_0 = (np.load(f) / norm - 1)
                else:
                    delta_R_0 = np.load(f)
                bins = compute_lognormal_split_bins(delta_R_0, args.nsplits)
            else:
                raise FileNotFoundError('No file {}. Density of mock 0 needs to be computed to define bins!'.format(f))
            print('Compute density splits in bins: ', bins)
            mock_density.split_density(args.nsplits, bins=bins)
            
            if args.rsd:
                if mock.positions_rsd is None:
                    mock.set_rsd(hz=hz, los=los)
                positions = mock.positions_rsd
            else:
                positions = mock.positions
    
            if args.use_weights:
                weights = mocks.weights
            else:
                weights = None
    
            mock_density.compute_smoothed_corr(edges, positions2=positions, weights2=weights, seed=args.imock, los=args.los, nthreads=32, norm=norm, mode=mode)
            mock_density.compute_ds_data_corr(edges, positions2=positions, weights2=weights, seed=args.imock, randoms_size=args.randoms_size, los=args.los, nthreads=32, norm=norm, mode=mode)

            # save result
            if args.env == 'feynman':
                outputdir = '/feynman/work/dphp/mp270220/outputs/densitysplit/'
            elif args.env == 'nersc':
                outputdir = '/pscratch/sd/m/mpinon/abacus/densitysplit'
            outputname = simname + '_cellsize{:d}{}_resampler{}{}_{:d}splits_randoms_size{:d}'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler, '_smoothingR{:d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '', args.nsplits, args.randoms_size) + '_RH_CCF{}'.format('_RSD' if args.rsd else '')
            mock_density.save(os.path.join(outputdir, outputname), save_density_mesh=(args.resampler=='tophat'))
            print('DensitySplit saved at: {}'.format(os.path.join(outputdir, outputname)))
        