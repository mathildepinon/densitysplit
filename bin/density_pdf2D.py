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
from densitysplit.numerical_model import DensitySplitModel
from densitysplit.density_split import DensitySplit

if __name__ == '__main__':
    
    setup_logging()
        
    parser = argparse.ArgumentParser(description='density_split_corr')
    parser.add_argument('--env', type=str, required=False, default='feynman', choices=['feynman', 'nersc'])
    parser.add_argument('--imock', type=int, required=False, default=0)
    parser.add_argument('--redshift', type=float, required=False, default=None)
    parser.add_argument('--simulation', type=str, required=False, default='abacus', choices=['abacus', 'gaussian', 'lognormal'])
    parser.add_argument('--tracer', type=str, required=False, default='particles', choices=['particles', 'halos'])
    parser.add_argument('--nbar', type=float, required=False, default=0.0034)
    parser.add_argument('--cellsize', type=int, required=False, default=10)
    parser.add_argument('--cellsize2', type=int, required=False, default=None)
    parser.add_argument('--resampler', type=str, required=False, default='tsc')
    parser.add_argument('--use_weights', type=bool, required=False, default=False)
    parser.add_argument('--rsd', type=bool, required=False, default=False)
    parser.add_argument('--los', type=str, required=False, default='x')
    parser.add_argument('--size', type=int, required=False, default=None)
    parser.add_argument('--bins', type=int, required=False, default=100)
    #parser.add_argument('--sep', type=float, required=False, default=None)
    parser.add_argument('--mu', type=float, required=False, default=None)
    parser.add_argument('--nsplits', type=int, required=False, default=3)
    parser.add_argument('--exporder', type=int, required=False, default=3)
    
    args = parser.parse_args()

    z = args.redshift
    sep = np.linspace(30, 150, 13)
    
    if args.simulation == 'abacus':
        cosmology=fiducial.AbacusSummitBase()
        if args.env == 'feynman':
            datadir = '/feynman/scratch/dphp/mp270220/abacus/'
        elif args.env == 'nersc':
            datadir = '/pscratch/sd/m/mpinon/abacus/'
        if args.tracer == 'halos':
            simname = 'AbacusSummit_2Gpc_z{:.3f}_ph0{:02d}'.format(z, args.imock)
        elif args.tracer == 'particles':
            simname = 'AbacusSummit_2Gpc_z{:.3f}_ph0{:02d}_downsampled_particles_nbar{:.4f}'.format(z, args.imock, args.nbar)

        cosmology = fiducial.AbacusSummitBase()
        bg = cosmology.get_background()
        hz = 100 * bg.efunc(z)

        # compute density contrast
        #mock = catalog_data.Data.load(os.path.join(datadir, simname+'.npy'))
        #mock_density = density_split.DensitySplit(mock)
        #mock_density.compute_density(data=mock, cellsize=args.cellsize, resampler=args.resampler, cellsize2=args.cellsize2, use_rsd=args.rsd, los=args.los, hz=hz, use_weights=args.use_weights)
        
        ds_dir = '/feynman/work/dphp/mp270220/outputs/densitysplit/'
        ds_fn = simname+'_cellsize{:d}{}_resampler{}_3splits_randoms_size4_RH_CCF{}'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler, '_RSD' if args.rsd else '')
        mock_density = DensitySplit.load(os.path.join(ds_dir, ds_fn.format(args.imock)+'.npy'))
        
        if args.tracer == 'particles':
            if args.rsd:
                bins = np.array([-1., -0.23633639, 0.10123832, np.inf])
            else:
                bins = np.array([-1., -0.19435888, 0.09070214, np.inf])
        else:
            if not args.rsd:
                bins = np.array([-1., -0.27505452, 0.15097056, np.inf])

        plots_dir = '/feynman/home/dphp/mp270220/plots/density'
        plt_fn = 'density_PDF_r{}{}_model.png'.format(args.cellsize, '_RSD' if args.rsd else '')
        
        deltaR1, deltaR2 = mock_density.compute_jointpdf_delta_R1_R2(s=100, query_positions='mesh', sample_size=args.size, mu=args.mu, los=args.los)
        model = DensitySplitModel(nsplits=args.nsplits, density_bins=bins, nbar=args.nbar)
        norm = model.compute_ds_nbar(deltaR1, plot_fn=os.path.join(plots_dir, plt_fn))
        print(norm)
        
        split_xi = list()
        for s in sep:
            print('Computing correlation function at separation {} Mpc/h.'.format(s))
            deltaR1, deltaR2 = mock_density.compute_jointpdf_delta_R1_R2(s=s, query_positions='mesh', sample_size=args.size, mu=args.mu, los=args.los)
            plt_fn = 'joint_density_PDF_r{}_r{}_s{}_mu{}{}_model.png'.format(args.cellsize, args.cellsize2, s, args.mu, '_RSD' if args.rsd else '')
            legend=(r'$s = {} \; \mathrm{{Mpc}}/h, \; \mu = {}$'.format(s, args.mu) if args.rsd else r'$s = {} \; \mathrm{{Mpc}}/h$'.format(s))
            xiRds = model.compute_gram_charlier_dsplits(n=args.exporder, delta1=deltaR1, delta2=deltaR2, bins=args.bins, norm=norm, plot_fn=os.path.join(plots_dir, plt_fn), legend=legend)
            split_xi.append(xiRds)

        split_xi = np.array(split_xi).T

        res = {'sep': sep, 'corr': split_xi}

        # save result
        outputdir = '/feynman/work/dphp/mp270220/outputs/densitysplit/'
        outputname = simname + '_cellsize{:d}{}_resampler{}_{:d}splits'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler, args.nsplits) + '_RH_CCF{}_nummodel'.format('_RSD' if args.rsd else '')
        print('Saving result at {}'.format(os.path.join(outputdir, outputname)))
        np.save(os.path.join(outputdir, outputname), res)

