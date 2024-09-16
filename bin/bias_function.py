import os
import sys
import copy
import argparse
import numpy as np
from matplotlib import pyplot as plt

from pycorr import setup_logging

from densitysplit.catalog_data import Data
from densitysplit.numerical_model import DensitySplitModel
from densitysplit.density_split import DensitySplit

if __name__ == '__main__':

    plt.style.use(os.path.join('/feynman/home/dphp/mp270220/densitysplit/nb', 'densitysplit.mplstyle'))
    
    setup_logging()
        
    parser = argparse.ArgumentParser(description='bias_function')
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
    parser.add_argument('--size', type=int, required=False, default=None)
    parser.add_argument('--bins', type=int, required=False, default=100)
    parser.add_argument('--sep', type=float, required=False, default=None)
    parser.add_argument('--plot', type=bool, required=False, default=True)
    
    args = parser.parse_args()

    z = args.redshift
    s = args.sep #[50]#np.linspace(0, 150, 16)
    edges = np.linspace(-1, 4, args.bins)

    ds_dir = '/feynman/work/dphp/mp270220/outputs/densitysplit/'
    
    if args.env == 'feynman':
        datadir = '/feynman/scratch/dphp/mp270220/abacus/'
    elif args.env == 'nersc':
        datadir = '/pscratch/sd/m/mpinon/abacus/'
    if args.tracer == 'halos':
        simname = 'AbacusSummit_2Gpc_z{:.3f}_{{}}'.format(z)
    elif args.tracer == 'particles':
        simname = 'AbacusSummit_2Gpc_z{:.3f}_{{}}_downsampled_particles_nbar{:.4f}'.format(z, args.nbar)
    
    ds_fn = simname + '_cellsize{:d}{}_resampler{}{}_3splits_randoms_size4'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler, '_smoothingR{:d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '') + '_RH_CCF{}'.format('_RSD' if args.rsd else '')

    if args.imock is not None:
        f = os.path.join(ds_dir, ds_fn.format('ph0{:02d}'.format(args.imock)))
        if os.path.isfile(f+'.npy'):
            # if DensitySplit file already exists
            mock_density = DensitySplit.load(f+'.npy')
        else:
            # if not, create DensitySplit instance
            mock = Data.load(os.path.join(datadir, simname.format('ph0{:02d}'.format(args.imock))+'.npy'))
            mock_density = DensitySplit(mock)
            mock_density.compute_density(data=mock, cellsize=args.cellsize, resampler=args.resampler, smoothing_radius=args.smoothing_radius, cellsize2=args.cellsize2)
            mock_density.save(f)
    
    else:
        nmocks = 25
        mock_density = list()
        for i in range(nmocks):
            f = os.path.join(ds_dir, ds_fn.format('ph0{:02d}'.format(i)))
            if os.path.isfile(f+'.npy'):
                mocki_density = DensitySplit.load(f+'.npy')
            else:
                mocki = Data.load(os.path.join(datadir, simname.format('ph0{:02d}'.format(i))+'.npy'))
                mocki_density = DensitySplit(mocki)
                mocki_density.compute_density(data=mocki, cellsize=args.cellsize, resampler=args.resampler, smoothing_radius=args.smoothing_radius, cellsize2=args.cellsize2)
                mocki_density.save(f)
            mock_density.append(mocki_density)
                
    model = DensitySplitModel(nbar=args.nbar)
    
    bias_func = list()
    
    print('Computing 2D density PDF at separation {} Mpc/h.'.format(s))

    nbar = args.nbar
    v = 4/3 * np.pi * args.smoothing_radius**3
    norm = nbar*v

    if args.imock is not None:
        deltaR1, deltaR2 = mock_density.compute_jointpdf_delta_R1_R2(s=s, sbin=(s-0.5, s+0.5), query_positions='mesh', sample_size=args.size)
        if args.resampler == 'tophat':
            deltaR1, deltaR2 = deltaR1/norm - 1, deltaR2/norm - 1
        corr = np.mean(deltaR1*deltaR2)
        bias_func = model.compute_bias_function(delta1=deltaR1, delta2=deltaR2, edges=edges) / corr
    else:
        print('Concatenate densities from all mocks')
        deltaR = [mock_density[i].compute_jointpdf_delta_R1_R2(s=s, sbin=(s-0.5, s+0.5), query_positions='mesh', sample_size=args.size) for i in range(nmocks)]
        deltaR1 = [deltaR[i][0]/norm - 1 for i in range(nmocks)]
        deltaR2 = [deltaR[i][1]/norm - 1 for i in range(nmocks)]
        corr = [np.mean(deltaR1[i]*deltaR2[i]) for i in range(nmocks)]
        bias_func = [model.compute_bias_function(delta1=deltaR1[i], delta2=deltaR2[i], edges=edges) / corr[i] for i in range(nmocks)]
        mean = np.mean(np.array(bias_func), axis=0)
        error = np.std(np.array(bias_func), axis=0)

    res = {'sep': s, 'bins': edges, 'bias': bias_func, 'corr': corr}

    # save result
    outputdir = '/feynman/work/dphp/mp270220/outputs/density/'
    outputname = simname.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_cellsize{:d}{}_resampler{}{}{}_biasfunction_sep{}'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler, '_smoothingR{:d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '', '_RSD' if args.rsd else '', s)
    print('Saving result at {}'.format(os.path.join(outputdir, outputname)))
    np.save(os.path.join(outputdir, outputname), res)

    if args.plot:
        x = (edges[1:]+edges[:-1])/2
        plots_dir = '/feynman/home/dphp/mp270220/plots/density'
        plt_fn = 'biasfunction_cellsize{}_cellsize{}_resampler{}{}_s{}_{}.png'.format(args.cellsize, args.cellsize2, args.resampler, '_smoothingR{:d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '', s, 'ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks))
        plt.figure(figsize=(6, 4))
        if args.imock is not None:
            plt.plot(x, bias_func)
        else:
            plt.plot(x, mean, color='C0')
            plt.fill_between(x, mean-error, mean+error, facecolor='C0', alpha=0.3)
        plt.xlabel(r'$\delta_R$')
        plt.ylabel(r'$b(\delta_R)$')
        plt.savefig(os.path.join(plots_dir, plt_fn), dpi=500)
        plt.close()
        print('Plot saved at {}'.format(os.path.join(plots_dir, plt_fn)))


    

