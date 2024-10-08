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
        
    parser = argparse.ArgumentParser(description='density_pdf')
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
    parser.add_argument('--size', type=int, required=False, default=None)
    parser.add_argument('--bins', type=int, required=False, default=100)
    #parser.add_argument('--sep', type=float, required=False, default=None)
    parser.add_argument('--mu', type=float, required=False, default=None)
    parser.add_argument('--nsplits', type=int, required=False, default=3)
    parser.add_argument('--method', type=str, required=False, default='measurement', choices=['measurement', 'gram-charlier', 'flow', 'ldt'])
    parser.add_argument('--exporder', type=int, required=False, default=3)
    
    args = parser.parse_args()

    z = args.redshift
    method = args.method
    sep = np.linspace(10, 150, 15) if method=='gram-charlier' else np.linspace(0, 150, 16)

    if args.rsd and (args.mu is None):
        mu = np.linspace(-1, 1, 201)
    else:
        mu = args.mu
    
    if args.simulation == 'abacus':
        cosmology=fiducial.AbacusSummitBase()
        if args.env == 'feynman':
            datadir = '/feynman/scratch/dphp/mp270220/abacus/'
        elif args.env == 'nersc':
            datadir = '/pscratch/sd/m/mpinon/abacus/'
        if args.tracer == 'halos':
            simname = 'AbacusSummit_2Gpc_z{:.3f}_{{}}'.format(z)
        elif args.tracer == 'particles':
            simname = 'AbacusSummit_2Gpc_z{:.3f}_{{}}_downsampled_particles_nbar{:.4f}'.format(z, args.nbar)

        cosmology = fiducial.AbacusSummitBase()
        bg = cosmology.get_background()
        hz = 100 * bg.efunc(z)

        # compute density contrast
        #mock = catalog_data.Data.load(os.path.join(datadir, simname+'.npy'))
        #mock_density = density_split.DensitySplit(mock)
        #mock_density.compute_density(data=mock, cellsize=args.cellsize, resampler=args.resampler, cellsize2=args.cellsize2, use_rsd=args.rsd, los=args.los, hz=hz, use_weights=args.use_weights)
        
        ds_dir = '/feynman/work/dphp/mp270220/outputs/densitysplit/'
        ds_fn = simname+'_cellsize{:d}{}_resampler{}{}_3splits_randoms_size4_RH_CCF{}'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler, '_smoothingR{:02d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '', '_RSD' if args.rsd else '')
        if args.imock is not None:
            mock_density = DensitySplit.load(os.path.join(ds_dir, ds_fn.format('ph0{:02d}'.format(args.imock))+'.npy'))
        else:
            mock_density = [DensitySplit.load(os.path.join(ds_dir, ds_fn.format('ph0{:02d}'.format(i))+'.npy')) for i in range(25)]
        
        delta0name = simname.format('ph000') + '_cellsize{:d}_resampler{}{}_delta_R{}.npy'.format(args.cellsize, args.resampler, '_smoothingR{:02d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '', '_RSD' if args.rsd else '')
        f = os.path.join('/feynman/work/dphp/mp270220/outputs/density', delta0name)
        if os.path.isfile(f):
            delta_R_0 = np.load(f)
            bins = compute_lognormal_split_bins(delta_R_0, args.nsplits)
        else:
            raise FileNotFoundError('No file {}. Density of mock 0 needs to be computed to define bins!'.format(f))

        plots_dir = '/feynman/home/dphp/mp270220/plots/density'
        plt_fn = 'density_PDF_r{}{}_{}_model.png'.format(args.smoothing_radius, '_RSD' if args.rsd else '', method)

        model = DensitySplitModel(nsplits=args.nsplits, density_bins=bins, nbar=args.nbar)
        
        #if method=='gram-charlier':
        #    deltaR1, deltaR2 = mock_density.compute_jointpdf_delta_R1_R2(s=100, query_positions='mesh', sample_size=args.size, mu=mu, los=args.los)
        #    norm = model.compute_ds_nbar(deltaR1, plot_fn=os.path.join(plots_dir, plt_fn))
        #    print(norm)

        split_xi = list()
        for s in sep:
            print('Computing correlation function at separation {} Mpc/h.'.format(s))
            
            if args.imock is not None:
                deltaR1, deltaR2 = mock_density.compute_jointpdf_delta_R1_R2(s=s, query_positions='mesh', sample_size=args.size, mu=mu, los=args.los)
            else:
                print('Concatenate densities from 25 mocks')
                deltaR = [mock_density[i].compute_jointpdf_delta_R1_R2(s=s, query_positions='mesh', sample_size=args.size, mu=mu, los=args.los) for i in range(25)]
                deltaR1 = np.concatenate([deltaR[i][0] for i in range(25)])
                deltaR2 = np.concatenate([deltaR[i][1] for i in range(25)])
            
            plt_fn = 'joint_density_PDF_r{}_r{}_s{}_mu{}{}_{}.png'.format(args.cellsize, args.cellsize2, s, mu, '_RSD' if args.rsd else '', method)
            legend=(r'$s = {} \; \mathrm{{Mpc}}/h, \; \mu = {}$'.format(s, args.mu) if args.rsd else r'$s = {} \; \mathrm{{Mpc}}/h$'.format(s))
            
            if method=='gram-charlier':
                xiRds = model.compute_gram_charlier_dsplits(n=args.exporder, delta1=deltaR1, delta2=deltaR2, bins=args.bins, norm=None, plot_fn=os.path.join(plots_dir, plt_fn), legend=legend)
            
            elif method=='measurement':
                xiRds = model.compute_dsplits(delta1=deltaR1, delta2=deltaR2, mu=mu if args.rsd else None)
                if mu is not None and len(list(mu)):
                    xiRds = xiRds.T

            elif method=='flow':
                #from .lognormal_model import LognormalDensityModel
                #model = LognormalDensityModel()
                #sigma1, delta01 = model.get_params_from_moments(sample=deltaR1)
                #sigma2, delta02 = model.get_params_from_moments(sample=deltaR2)
                #X1 = np.log(1 + deltaR1/delta01) + sigma1**2/2.
                #X2 = np.log(1 + deltaR2/delta02) + sigma2**2/2.
                flow_dir = '/feynman/work/dphp/mp270220/outputs/harvest'
                flow_fn = simname.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '25mocks')+'_cellsize{:d}{}_resampler{}{}'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler, '_RSD' if args.rsd else '')
                
                from densitysplit.flows import NormalizingFlows
                
                deltaR = np.array([deltaR1, deltaR2]).T
                flow = NormalizingFlows(deltaR, n_flows=1, output=os.path.join(flow_dir, flow_fn))
                flow.train()
                #flow.save()
                sample = flow.sample(size=args.size, seed=0)
                xiRds = model.compute_dsplits(delta1=sample[:, 0], delta2=sample[:, 1], mu=mu if args.rsd else None)

                from matplotlib import pyplot as plt

                import getdist.plots as gdplt
                from getdist import MCSamples
                data_sample =  MCSamples(samples=deltaR)
                model_sample =  MCSamples(samples=sample)
                
                g = gdplt.get_subplot_plotter()
                g.settings.num_plot_contours = 4
                g.triangle_plot([data_sample, model_sample], filled=False, legend_labels=['measured', 'flow'])
                plt.suptitle(legend)
                plt.savefig(os.path.join(plots_dir, plt_fn), dpi=500)
                plt.close()

            split_xi.append(xiRds)

        split_xi = np.array(split_xi).T

        res = {'sep': sep, 'corr': split_xi}

        # save result
        outputdir = '/feynman/work/dphp/mp270220/outputs/densitysplit/'
        outputname = simname.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '25mocks') + '_cellsize{:d}{}_resampler{}{}_{:d}splits'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler, '_smoothingR{:02d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '', args.nsplits) + '_RH_CCF{}_{}_model'.format('_RSD' if args.rsd else '', method)
        print('Saving result at {}'.format(os.path.join(outputdir, outputname)))
        np.save(os.path.join(outputdir, outputname), res)

