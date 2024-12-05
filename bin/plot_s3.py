import os
import sys
import argparse
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.optimize import minimize, curve_fit

from densitysplit.corr_func_utils import get_split_poles
from densitysplit import CountInCellsDensitySplitMeasurement
from densitysplit.lognormal_model import LognormalDensityModel, LognormalDensitySplitModel
from densitysplit.lssfast import LDT, LDTDensitySplitModel, setup_logging
from density_split_corr import compute_lognormal_split_bins

plt.style.use(os.path.join(os.path.abspath('/feynman/home/dphp/mp270220/densitysplit/nb'), 'densitysplit.mplstyle'))


if __name__ == '__main__':
    
    setup_logging()
        
    parser = argparse.ArgumentParser(description='density_split_plots')
    parser.add_argument('--imock', type=int, required=False, default=None)
    parser.add_argument('--redshift', type=float, required=False, default=None)
    parser.add_argument('--tracer', type=str, required=False, default='particles', choices=['particles', 'halos'])
    parser.add_argument('--nbar', type=float, required=False, default=0.0034)
    parser.add_argument('--cellsize', type=int, required=False, default=10)
    parser.add_argument('--cellsize2', type=int, required=False, default=None)
    parser.add_argument('--resampler', type=str, required=False, default='tsc')
    parser.add_argument('--smoothing_radius', type=int, nargs='+', required=False, default=None)
    parser.add_argument('--use_weights', type=bool, required=False, default=False)
    parser.add_argument('--rsd', type=bool, required=False, default=False)
    parser.add_argument('--los', type=str, required=False, default='x')
    parser.add_argument('--nsplits', type=int, required=False, default=3)
    parser.add_argument('--randoms_size', type=int, required=False, default=4)
    parser.add_argument('--size', type=int, required=False, default=None)
    parser.add_argument('--rescale_var', type=float, required=False, default=1)
    
    args = parser.parse_args()
    z = args.redshift
    interpolate = False

    # Plotting
    if args.nbar < 0.001:
        data_style = dict(color='C0', ls='', marker="o", ms=3, mec='C0', mfc='C0')
        model_styles = {'ldt': dict(color='C1', ls='', marker="o", ms=2, mec='C1', mfc='C1'),
                         'lognormal': dict(color='C3', ls='', marker="o", ms=2, mec='C3', mfc='white')}
    else:
        data_style = dict(ls='-', color='C0')
        model_styles = {'ldt': dict(ls='-', color='C1'),
                         'lognormal': dict(ls=':', color='C3'),
                         'lognormal_approx': dict(ls='--', color='C3')}        

    data_label = 'AbacusSummit'
    model_labels = {'ldt': 'LDT',
                    'lognormal': 'lognormal',
                    'lognormal_approx': r'lognormal ($s \rightarrow + \infty$)'}

    # Directories
    ds_dir = '/feynman/work/dphp/mp270220/outputs/densitysplit/'
    plots_dir = '/feynman/home/dphp/mp270220/plots/densitysplit'
    
    # Filenames
    if args.tracer == 'halos':
        sim_name = 'AbacusSummit_2Gpc_z{:.3f}_{{}}'.format(z)
    elif args.tracer == 'particles':
        sim_name = 'AbacusSummit_2Gpc_z{:.3f}_{{}}_downsampled_particles_nbar{:.4f}'.format(z, args.nbar)

    s3_mean_list = list()
    s3_std_list = list()
    s3_pred_list = list()

    for smoothing_radius in args.smoothing_radius:       
        base_name = sim_name + '_cellsize{:d}{}_resampler{}{}'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler, '_smoothingR{:d}'.format(smoothing_radius) if smoothing_radius is not None else '')
        ds_name = base_name + '_3splits_randoms_size4_RH_CCF{}'.format('_RSD' if args.rsd else '')
    
        # Load measured density split measurements
        nmocks = 25 if args.nbar < 0.01 else 8
        fn = os.path.join(ds_dir, ds_name.format('{}mocks'.format(nmocks)) + '_compressed.npy')
        print('Loading density split measurements: {}'.format(fn))
        result = CountInCellsDensitySplitMeasurement.load(fn)
    
        sigma = result.sigma_all
        delta3 = result.delta3_all
        s3 = delta3/sigma**4
        print('s3: ', s3)
        s3_mean_list.append(np.mean(s3))
        s3_std_list.append(np.std(s3))

        nbar = result.nbar
        norm = result.norm
        smoothing_radius = result.smoothing_radius
    
        # 1D PDF
        mean_pdf1D = np.mean(result.pdf1D, axis=0)
        std_pdf1D = np.std(result.pdf1D, axis=0)
        sigma = np.sqrt(np.sum(result.pdf1D_x**2 * mean_pdf1D)/norm)
        mask0 = std_pdf1D > 0
        
        # LDT model
        ldtmodel = LDT(redshift=z, smoothing_scale=smoothing_radius, smoothing_kernel=1, nbar=nbar)
        sigma_noshotnoise = np.sqrt(sigma**2 - 1 / (nbar * 4/3 * np.pi * smoothing_radius**3))
        print('sigma no shotnoise:', sigma_noshotnoise)
        ldtmodel.interpolate_sigma()
    
        def fit_sigma():
            def to_fit(x, sig):
                ldtmodel.compute_ldt(sig, k=(1 + x)*norm)
                ldtpdf1D = ldtmodel.density_pdf()
                return ldtpdf1D
                
            fit = curve_fit(to_fit, result.pdf1D_x[mask0], mean_pdf1D[mask0], p0=sigma_noshotnoise, sigma=std_pdf1D[mask0])
            print(fit)
            return fit[0]
    
            def compute_ldt(sig):
                ldtmodel.compute_ldt(sig, k=(1 + result.pdf1D_x)*norm)
                ldtpdf1D = ldtmodel.density_pdf()
                residuals = (ldtpdf1D[mask0] - mean_pdf1D[mask0])/std_pdf1D[mask0]
                return np.sum(residuals**2)        
            mini = minimize(compute_ldt, sigma_noshotnoise)
            print(mini)
            return mini.x
    
        sigma_noshotnoise = fit_sigma()
        print('sigma no shotnoise test:', sigma_noshotnoise)
    
        if args.resampler=='tophat':
            if interpolate:
                ldtmodel.compute_ldt(sigma_noshotnoise)
                ldtpdf1D = ldtmodel.density_pdf(1+result.pdf1D_x)
            else:
                ldtmodel.compute_ldt(sigma_noshotnoise, k=(1 + result.pdf1D_x)*norm)
                ldtpdf1D = ldtmodel.density_pdf()            
        else:
            ldtmodel.compute_ldt(sigma_noshotnoise)
            ldtpdf1D = ldtmodel.density_pdf(1 + result.pdf1D_x)

        predicted_s3 = np.sum(result.pdf1D_x**3 * ldtpdf1D)/(np.sum(result.pdf1D_x**2 * ldtpdf1D))**2 * norm
        s3_pred_list.append(predicted_s3)

    s3_mean = np.array(s3_mean_list)
    s3_std = np.array(s3_std_list)/np.sqrt(nmocks)
    s3_pred = np.array(s3_pred_list)

    base_name = sim_name + '_cellsize{:d}{}_resampler{}'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler)
    plot_name = base_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_s3.pdf'
        
    fig, axes = plt.subplots(2, 1, figsize = (3.5, 3.5), sharex=True, sharey='row', gridspec_kw={'height_ratios': [3, 1]})
    axes[0].errorbar(args.smoothing_radius, s3_mean, s3_std, marker="o", mec=None, mfc='C0', ms=3, capsize=2, ls='', label='AbacusSummit')
    axes[0].plot(args.smoothing_radius, s3_pred, marker="o", mec=None, mfc='C1', ms=2, ls='', label='LDT')
    axes[1].errorbar(args.smoothing_radius, s3_pred - s3_mean, s3_std, marker="o", mec=None, mfc='C0', ms=3, capsize=2, ls='')
    axes[1].axhline(0, color='black', lw=0.7, ls='')
    axes[1].set_ylabel(r'$\Delta S_3$')         
    axes[1].set_xlabel(r'$R \; [\mathrm{Mpc}/h]$')
    axes[0].set_ylabel(r'$S_3$')
    axes[1].ticklabel_format(style='sci', scilimits=(-3, 3))
    axes[0].legend()
    fig.align_ylabels()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, plot_name), dpi=500)
    plt.show()
    
