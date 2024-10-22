import os
import sys
import argparse
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import minimize, curve_fit

from densitysplit.corr_func_utils import get_split_poles
from densitysplit import CountInCellsDensitySplitMeasurement
from densitysplit.lognormal_model import LognormalDensityModel, LognormalDensitySplitModel
from densitysplit.lssfast import LDT, LDTDensitySplitModel, setup_logging
from density_split_corr import compute_lognormal_split_bins

plt.style.use(os.path.join(os.path.abspath('/feynman/home/dphp/mp270220/densitysplit/nb'), 'densitysplit.mplstyle'))


def plot_pdf1D(x, mean_pdf1D, std_pdf1D, xlim=None, rebin=None, data_style=None, data_label=None, models=None, model_labels=None, model_styles=None, rtitle=False, fn=None):

    if rebin is not None:
        x = x[::rebin]
        mean_pdf1D = mean_pdf1D[::rebin]
        std_pdf1D = std_pdf1D[::rebin]

    if xlim is not None:
        mask = (x >= xlim[0]) & (x <= xlim[1])
        x = x[mask]
        mean_pdf1D = mean_pdf1D[mask]
        std_pdf1D = std_pdf1D[mask]
    
    fig, axes = plt.subplots(2, 1, figsize = (3.5, 3.5), sharex=True, sharey='row', gridspec_kw={'height_ratios': [3, 1]})

    axes[0].plot(x, mean_pdf1D, label=data_label, **data_style)
    axes[0].fill_between(x, mean_pdf1D - std_pdf1D, mean_pdf1D + std_pdf1D, facecolor=data_style['color'], alpha=0.3)
    axes[1].fill_between(x, -std_pdf1D, std_pdf1D, facecolor=data_style['color'], alpha=0.3)
    
    for m in models.keys():
        if rebin is not None:
            models[m] = models[m][::rebin]
        if xlim is not None:
            models[m] = models[m][mask]
            
        axes[0].plot(x, models[m], label=model_labels[m], **model_styles[m])
        axes[1].plot(x, (models[m] - mean_pdf1D), **model_styles[m])

    axes[1].ticklabel_format(style='sci', scilimits=(-3, 3))
    axes[1].set_xlabel(r'$\delta_R$')
    axes[0].set_ylabel(r'$\mathcal{P}(\delta_R)$')
    axes[1].set_ylabel(r'$\Delta \mathcal{P}(\delta_R)$')  
    if rtitle:
        axes[0].set_title(r'$R = {} \; \mathrm{{Mpc}}/h$'.format(smoothing_radius))
    axes[0].legend(loc='upper right')
    fig.align_ylabels()
    plt.savefig(fn, dpi=500)
    plt.close()
    print('Saved 1D PDF plot: {}.'.format(fn))


def plot_bias_function(x, mean_bias, std_bias, xlim=None, rebin=None, data_style=None, data_label=None, models=None, model_labels=None, model_styles=None, sep=None, fn=None):

    if rebin is not None:
        x = x[::rebin]
        mean_bias = mean_bias[::rebin]
        std_bias = std_bias[::rebin]

    if xlim is not None:
        mask = (x >= xlim[0]) & (x <= xlim[1])
        x = x[mask]
        mean_bias = mean_bias[mask]
        std_bias = std_bias[mask]
    
    fig, axes = plt.subplots(2, 1, figsize = (3.5, 3.5), sharex=True, sharey='row', gridspec_kw={'height_ratios': [3, 1]})
 
    axes[0].plot(x, mean_bias, label=data_label, **data_style)
    axes[0].fill_between(x, mean_bias - std_bias, mean_bias + std_bias, facecolor=data_style['color'], alpha=0.3)
    #axes[1].fill_between(x, -std_bias, std_bias, facecolor=data_style['color'], alpha=0.3)        

    for m in models.keys():
        if rebin is not None:
            models[m] = models[m][::rebin]
        if xlim is not None:
            models[m] = models[m][mask]

        axes[0].plot(x, models[m], label=model_labels[m], **model_styles[m])
        axes[1].plot(x,  (models[m] - mean_bias)/std_bias, **model_styles[m])

    axes[0].set_ylim(-15, 18)
    axes[0].legend(loc='lower right')
    axes[1].set_xlabel(r'$\delta_R$')
    axes[0].set_ylabel(r'$b(\delta_R)$')
    axes[1].set_ylabel(r'$\Delta b(\delta_R) / \sigma$')
    #axes[0].set_title(r'$R = {} \; \mathrm{{Mpc}}/h$'.format(smoothing_radius))

    if sep is not None:
        axes[0].text(0.1, 0.9, r'$s = {:.0f} \; \mathrm{{Mpc}}/h$'.format(sep), ha='left', va='top', transform = axes[0].transAxes, fontsize=12)
    
    fig.align_ylabels()
    plt.savefig(fn, dpi=500)
    plt.close()
    print('Plot saved at {}'.format(fn))


def plot_density_splits(x, mean_ds, std_ds, std_ds_ref=None, data_style=None, data_label=None, models=None, model_labels=None, model_styles=None, fn=None):

    ells = [0, 2, 4]
    nells = len(list(mean_ds[0]))
    ells = ells[:nells]
    nsplits = len(mean_ds)

    figsize = (3.5, 3.5)
    fig, axes = plt.subplots(2, nells, figsize=figsize, sharex=True, sharey='row', gridspec_kw={'height_ratios': [3, 1]})
    
    #colors = ['firebrick', 'violet', 'olivedrab']
    base_colors = ['cornflowerblue', 'red']
    cmap = LinearSegmentedColormap.from_list("mycmap", base_colors, N=nsplits)
    colors = [cmap(i) for i in range(nsplits)]

    for ill, ell in enumerate(ells):
        ax0 = axes[0][ill] if nells > 1 else axes[0]
        ax1 = axes[1][ill] if nells > 1 else axes[1]

        for ds in range(nsplits):
            ax0.plot(x, x**2 * mean_ds[ds][ill], color=colors[ds], label=r'DS = {}'.format(ds + 1))
            if not np.isnan(std_ds).all():
                ax0.fill_between(x, x**2 * (mean_ds[ds][ill] - std_ds[ds][ill]), x**2 * (mean_ds[ds][ill] + std_ds[ds][ill]), facecolor=colors[ds], alpha=0.3)
                if std_ds_ref is not None:
                    ax1.fill_between(x, -std_ds_ref[ds][ill]/std_ds[ds][ill], std_ds_ref[ds][ill]/std_ds[ds][ill], facecolor=colors[ds], alpha=0.3)
            
            for m in models.keys():
                ax0.plot(x, x**2 * models[m][ds], color=colors[ds], ls=':')
                if not np.isnan(std_ds).all():
                    ax1.plot(x, (models[m][ds] - mean_ds[ds][ill])/std_ds[ds][ill], color=colors[ds], ls=':')
                    #ax1.plot(x, (models[m][ds] - mean_ds[ds][ill]), color=colors[ds], ls=':')
                    ax1.set_ylabel(r'$\Delta \xi_{R}^{\rm DS}(s) / \sigma$')
                else:
                    ax1.plot(x, (models[m][ds] - mean_ds[ds][ill]), color=colors[ds], ls=':')
                    ax1.set_ylabel(r'$\Delta \xi_{R}^{\rm DS}(s)$')

        #ax0.set_ylim(-45, 50)
        if nells > 1:
            ax0.set_title(r'$\ell = {}$'.format(ell))
        ax1.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
        ax1.set_ylim(-5, 5)

    for m in models.keys():
        l, = ax0.plot([], ls=':', color='black')
        legend = ax0.legend([l], [model_labels[m]], loc='lower right')
    ax0.legend(loc='upper right')
    ax0.add_artist(legend)

    ax0 = axes[0][0] if nells > 1 else axes[0]
    ax0.set_ylabel(r'$s^2 \xi_{R}^{\rm DS}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
    ax1 = axes[1][0] if nells > 1 else axes[1]
    fig.align_ylabels()
    plt.savefig(fn, dpi=500)
    print('Plot saved at {}'.format(fn))


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
    parser.add_argument('--smoothing_radius', type=int, required=False, default=None)
    parser.add_argument('--use_weights', type=bool, required=False, default=False)
    parser.add_argument('--rsd', type=bool, required=False, default=False)
    parser.add_argument('--los', type=str, required=False, default='x')
    parser.add_argument('--nsplits', type=int, required=False, default=3)
    parser.add_argument('--randoms_size', type=int, required=False, default=4)
    parser.add_argument('--size', type=int, required=False, default=None)
    parser.add_argument('--show_lognormal', type=bool, required=False, default=True)
    parser.add_argument('--show_ldt', type=bool, required=False, default=True)
    parser.add_argument('--rescale_var', type=float, required=False, default=1)
    parser.add_argument('--to_plot', type=str, nargs='+', required=False, default=['pdf1D', 'bias', 'densitysplits'], choices=['pdf1D', 'bias', 'densitysplits'])
    
    args = parser.parse_args()
    z = args.redshift
    ells = [0, 2, 4] if args.rsd else [0]
    nells = len(ells)

    swidth = 0.01
    interpolate = False

    # Plotting
    if args.nbar < 0.001:
        data_style = dict(color='C0', ls='', marker="o", ms=3, mec='C0', mfc='C0')
        model_styles = {'ldt': dict(color='C1', ls='', marker="o", ms=2, mec='C1', mfc='C1'),
                         'lognormal': dict(color='C3', ls='', marker="o", ms=2, mec='C3', mfc='white')}
    else:
        data_style = dict(ls='-', color='C0')
        model_styles = {'ldt': dict(ls='-', color='C1'),
                         'lognormal': dict(ls=':', color='C3')}        

    data_label = 'AbacusSummit'
    model_labels = {'ldt': 'LDT',
                    'lognormal': 'lognormal'}

    # Directories
    ds_dir = '/feynman/work/dphp/mp270220/outputs/densitysplit/'
    plots_dir = '/feynman/home/dphp/mp270220/plots/densitysplit'
    
    # Filenames
    if args.tracer == 'halos':
        sim_name = 'AbacusSummit_2Gpc_z{:.3f}_{{}}'.format(z)
    elif args.tracer == 'particles':
        sim_name = 'AbacusSummit_2Gpc_z{:.3f}_{{}}_downsampled_particles_nbar{:.4f}'.format(z, args.nbar)
    base_name = sim_name + '_cellsize{:d}{}_resampler{}{}'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler, '_smoothingR{:d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '')
    ds_name = base_name + '_3splits_randoms_size4_RH_CCF{}'.format('_RSD' if args.rsd else '')

    # Load measured density split measurements
    nmocks = 25 if args.nbar < 0.01 else 8
    fn = os.path.join(ds_dir, ds_name.format('{}mocks'.format(nmocks)) + '_compressed.npy')
    print('Loading density split measurements: {}'.format(fn))
    result = CountInCellsDensitySplitMeasurement.load(fn)

    sigma = result.sigma
    print('sigma:', sigma)
    print('sigma test:', np.sqrt(np.mean(result.sigma_all**2)))
    delta3 = result.delta3
    print('delta3:', delta3)
    nbar = result.nbar
    norm = result.norm
    smoothing_radius = result.smoothing_radius

    # 1D PDF
    mean_pdf1D = np.mean(result.pdf1D, axis=0)
    std_pdf1D = np.std(result.pdf1D, axis=0)
    sigma = np.sqrt(np.sum(result.pdf1D_x**2 * mean_pdf1D)/norm)
    delta3 = np.sum(result.pdf1D_x**3 * mean_pdf1D)/norm
    print('sigma test 2:', sigma)
    print('delta3 test:', delta3)
    
    # Lognormal model
    lognormalmodel = LognormalDensityModel()
    #lognormalmodel.get_params_from_moments(m2=sigma**2, m3=delta3)
    #lognormalmodel.get_params_from_moments(m2=sigma**2, m3=delta3)
    mask0 = std_pdf1D > 0
    lognormalmodel.fit_params_from_pdf(delta=result.pdf1D_x[mask0], density_pdf=mean_pdf1D[mask0], sigma=std_pdf1D[mask0])
    lognormalpdf1D = lognormalmodel.density(result.pdf1D_x)

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

    models_pdf1D = {'ldt': ldtpdf1D, 'lognormal': lognormalpdf1D} if args.nbar > 0.001 else {'ldt': ldtpdf1D}

    # plot settings
    plotting = {'data_style': data_style, 'data_label': data_label, 'model_labels': model_labels, 'model_styles': model_styles}

    if 'pdf1D' in args.to_plot:
        # Plot 1D PDF
        density_name = sim_name + '_cellsize{:d}_resampler{}{}{}'.format(args.cellsize, args.resampler, '_smoothingR{:d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '', '_rescaledvar{}'.format(args.rescale_var) if args.rescale_var!=1 else '')
        plot_name = density_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_1DPDF.pdf'

        if args.nbar > 0.01:
            rebin = 4
        else:
            rebin = None

        if args.nbar > 0.001:
            xlim=(-1, 3)
        else:
            xlim=(-1, 4)
        
        plot_pdf1D(result.pdf1D_x, mean_pdf1D, std_pdf1D, xlim=xlim, rebin=rebin, models=models_pdf1D, rtitle=args.nbar<0.001, fn=os.path.join(plots_dir, plot_name), **plotting)

    if 'bias' in args.to_plot:
        if args.nbar > 0.01:
            rebin = 8
        else:
            rebin = None

        # Bias function
        for sep in result.bias_function.keys():
            # Bias function
            mean_bias = np.mean(result.bias_function[sep], axis=0)
            std_bias = np.std(result.bias_function[sep], axis=0)
    
            # LDT model
            ldtbiasmodel = np.array([ldtmodel.bias(1+x) for x in result.bias_function_x[sep]])
    
             # Lognormal model
            lognormalbiasmodel = lognormalmodel.compute_bias_function(result.bias_function_x[sep], xiR=np.mean(result.bias_corr[sep]))
    
            models_bias = {'ldt': ldtbiasmodel, 'lognormal': lognormalbiasmodel}
    
            # Plot bias function
            plot_name = base_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_s{:.0f}_biasfunction.pdf'.format(float(sep))
    
            plot_bias_function(result.bias_function_x[sep], mean_bias, std_bias, rebin=rebin, xlim=(-1, 4), sep=float(sep), models=models_bias, fn=os.path.join(plots_dir, plot_name), **plotting)

    if 'densitysplits' in args.to_plot:
        # Density splits
        mean_xiR = np.mean(result.smoothed_corr, axis=0)
        mean_ds, cov = get_split_poles(result.ds_corr, ells=None if args.resampler=='tophat' else ells)
        std_ds = np.nan if cov.size == 1 else np.array_split(np.array(np.array_split(np.diag(cov)**0.5, nells)), args.nsplits, axis=1)
        sep = result.sep

        if args.nbar < 0.001:
            # save errors
            fname = base_name.format('{}mocks'.format(nmocks)) + '_densitysplits_std'
            np.save(os.path.join(ds_dir, fname), std_ds)

        # Load errors from mocks with nbar = 0.0005
        fname = 'AbacusSummit_2Gpc_z0.800_25mocks_downsampled_particles_nbar0.0005_cellsize5_cellsize5_resamplertophat_smoothingR10_densitysplits_std.npy'
        std_ds_desi = np.load(os.path.join(ds_dir, fname))
        print(std_ds)
        print(std_ds_desi)
            
        # LDT model
        ldtdsplitmodel = LDTDensitySplitModel(ldtmodel, density_bins=result.bins)
        ldtdsplits = ldtdsplitmodel.compute_dsplits(mean_xiR)
    
        # Lognormal model
        lognormaldsplitmodel = LognormalDensitySplitModel(density_bins=result.bins)
        lognormaldsplitmodel.set_params(sigma=lognormalmodel.sigma, delta0=lognormalmodel.delta0, delta02=lognormalmodel.delta0, sigma2=lognormalmodel.sigma)
        lognormaldsplits = lognormaldsplitmodel.compute_dsplits(smoothing=1, sep=sep, xiR=mean_xiR, rsd=False, ells=ells)
    
        models_ds = {'ldt': ldtdsplits, 'lognormal': lognormaldsplits}
    
        for m in ['lognormal', 'ldt']:
            plot_name = base_name.format('{}mocks'.format(nmocks)) + '_densitysplits_{}model.pdf'.format(m)
            models = {m: models_ds[m]}
            plot_density_splits(sep, mean_ds, std_ds, std_ds_ref=std_ds_desi, models=models, fn=os.path.join(plots_dir, plot_name), **plotting)
        