import os
import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, moment
from scipy.optimize import minimize

from cosmoprimo import *
from pycorr import TwoPointCorrelationFunction

from densitysplit.corr_func_utils import get_poles, get_split_poles
from densitysplit.lognormal_model import *
from densitysplit import DensitySplit

plt.style.use(os.path.join(os.path.abspath('/feynman/home/dphp/mp270220/densitysplit/nb'), 'densitysplit.mplstyle'))

    
if __name__ == '__main__':
    
    setup_logging()
        
    parser = argparse.ArgumentParser(description='density_split_plot')
    parser.add_argument('--imock', type=int, required=False, default=0)
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
    
    args = parser.parse_args()

    plots_dir = '/feynman/home/dphp/mp270220/plots/densitysplit'

    z = args.redshift
    
    if args.rsd:
        ells = [0, 2, 4]    
    else:
        ells = [0]
    nells = len(ells)

    # Load measured density splits
    if args.tracer == 'halos':
        simname = 'AbacusSummit_2Gpc_z{:.3f}_{{}}'.format(z)
    elif args.tracer == 'particles':
        simname = 'AbacusSummit_2Gpc_z{:.3f}_{{}}_downsampled_particles_nbar{:.4f}'.format(z, args.nbar)

    ds_dir = '/feynman/work/dphp/mp270220/outputs/densitysplit/'
    ds_fn = simname + '_cellsize{:d}{}_resampler{}{}_3splits_randoms_size4'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler, '_smoothingR{:d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '') + '_RH_CCF{}'.format('_RSD' if args.rsd else '')
    ds_corr = list()
    for i in range(1):
        f = os.path.join(ds_dir, ds_fn.format('ph0{:02d}'.format(i)))
        if os.path.isfile(f+'.npy'):
            mocki_density = DensitySplit.load(f+'.npy')
            ds_corr.append(mocki_density.ds_data_corr)
    split_xi, cov = get_split_poles(ds_corr, ells=ells, nsplits=args.nsplits)
    std = np.zeros_like(split_xi) if cov.size == 1 else np.array_split(np.array(np.array_split(np.diag(cov)**0.5, nells)), args.nsplits, axis=1)
    s, _, _ = ds_corr[0][0].get_corr(return_sep=True)

    # Load LDT model
    model_fn = simname + '_cellsize{:d}{}_resampler{}{}_3splits'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler, '_smoothingR{:d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '') + '_RH_CCF{}'.format('_RSD' if args.rsd else '')
    ldtmodelname = model_fn.format('ph000')+'_LDT_model.npy'
    ldtmodel = np.load(os.path.join(ds_dir, ldtmodelname), allow_pickle=True).item()
    sep = ldtmodel['sep']
    ldtcorr = ldtmodel['corr']
    #print('sep:', sep)
    #print('ldt model:', ldtcorr)
        
    figsize = (8, 4) if args.rsd else (4, 4)
    fig, axes = plt.subplots(2, nells, figsize=figsize, sharex=True, sharey='row', gridspec_kw={'height_ratios': [3, 1]})
    colors = ['firebrick', 'violet', 'olivedrab']
    
    for ill, ell in enumerate(ells):
        ax0 = axes[0][ill] if nells > 1 else axes[0]
        ax1 = axes[1][ill] if nells > 1 else axes[1]
    
        for ds in range(args.nsplits):
            ax0.plot(s, split_xi[ds][ill], color=colors[ds], label=r'DS{} $\times$ all'.format(ds))
            #ax0.fill_between(s, (split_xi[ds][ill] - std[ds][ill]), s**2 * (split_xi[ds][ill] + std[ds][ill]), facecolor=colors[ds], alpha=0.3)
            ax0.plot(sep, ldtcorr[ds], color=colors[ds], ls='', marker='.')
                
            split_xi_interp = np.interp(sep, s, split_xi[ds][ill])
            std_interp = np.interp(sep, s, std[ds][ill])
            ax1.plot(sep, (ldtcorr[ds] - split_xi_interp)/1, ls='', marker='.', markersize=4, color=colors[ds])
            #ax1.set_ylim(-5, 5)
        if args.rsd:
            ax0.set_title(r'$\ell = {}$'.format(ell))
        ax1.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
    
    ax0 = axes[0][0] if nells > 1 else axes[0]
    ax0.set_ylabel(r'$s^2 \xi_{R}^{DS}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
    ax1 = axes[1][0] if nells > 1 else axes[1]
    ax1.set_ylabel(r'$\Delta \xi_{R}^{DS}(s) / \sigma$')
    fig.align_ylabels()
    ax0.legend()
    plt.savefig(os.path.join(plots_dir, ds_fn.format('ph000')+'_ldtmodel.pdf'), dpi=500)
    plt.show()

        


