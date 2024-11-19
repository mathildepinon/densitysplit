import os
import sys
import argparse
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d

from densitysplit.corr_func_utils import get_split_poles
from densitysplit import CountInCellsDensitySplitMeasurement
from densitysplit.lognormal_model import LognormalDensityModel, BiasedLognormalDensityModel, LognormalDensitySplitModel
from densitysplit.ldt_model import LDT, LDTDensitySplitModel, setup_logging
from density_split_corr import compute_lognormal_split_bins

plt.style.use(os.path.join(os.path.abspath('/feynman/home/dphp/mp270220/densitysplit/nb'), 'densitysplit.mplstyle'))


def plot_pdf1D(x, mean_pdf1D, std_pdf1D, xlim=None, rebin=None, residuals='absolute', data_style=None, data_label=None, models=None, model_labels=None, model_styles=None, rtitle=False, fn=None):

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
    #axes[0].fill_between(x, mean_pdf1D - std_pdf1D, mean_pdf1D + std_pdf1D, facecolor=data_style['color'], alpha=0.3)
    if residuals=='absolute':
        axes[1].fill_between(x, -std_pdf1D, std_pdf1D, facecolor=data_style['color'], alpha=0.3)
    elif residuals=='sigmas':
        axes[1].fill_between(x, -1, 1, facecolor=data_style['color'], alpha=0.3)
    elif residuals=='percent':
        axes[1].fill_between(x, -std_pdf1D/mean_pdf1D, std_pdf1D/mean_pdf1D, facecolor=data_style['color'], alpha=0.3)
 
    if models is not None:
        for m in models.keys():
            if rebin is not None:
                models[m] = models[m][::rebin]
            if xlim is not None:
                models[m] = models[m][mask]
                
            axes[0].plot(x, models[m], label=model_labels[m], **model_styles[m])
            if residuals=='absolute':
                axes[1].plot(x, (models[m] - mean_pdf1D), **model_styles[m])
            elif residuals=='sigmas':
                axes[1].plot(x, (models[m] - mean_pdf1D)/std_pdf1D, **model_styles[m])
            elif residuals=='percent':
                axes[1].plot(x, (models[m] - mean_pdf1D)/mean_pdf1D, **model_styles[m])

    axes[1].ticklabel_format(style='sci', scilimits=(-3, 3))
    axes[1].set_xlabel(r'$\delta_R$')
    axes[0].set_ylabel(r'$\mathcal{P}(\delta_R)$')

    if residuals=='absolute':
        axes[1].set_ylabel(r'$\Delta \mathcal{P}(\delta_R)$')  
    elif residuals=='sigmas':
        axes[1].set_ylabel(r'$\Delta \mathcal{P}(\delta_R) / \sigma$') 
    elif residuals=='percent':
        axes[1].set_ylabel(r'$\Delta \mathcal{P}(\delta_R) / \mathcal{P}(\delta_R)$')
    
    if rtitle:
        axes[0].set_title(r'$R = {} \; \mathrm{{Mpc}}/h$'.format(smoothing_radius))
    axes[0].legend(loc='upper right')
    fig.align_ylabels()
    plt.savefig(fn, dpi=500)
    plt.close()
    print('Saved 1D PDF plot: {}.'.format(fn))


def plot_pdf1D_cov(x, pdf1D, xlim=None, rebin=None, rtitle=False, fn=None):
    if rebin is not None:
        x = x[::rebin]
        pdf1D = np.array([pdf1D[i][::rebin] for i in range(len(pdf1D))])

    if xlim is not None:
        mask = (x >= xlim[0]) & (x <= xlim[1])
        x = x[mask]
        pdf1D = np.array([pdf1D[i][mask] for i in range(len(pdf1D))])

    pdf1D_cov = np.cov(pdf1D, rowvar=False)
    stddev = np.sqrt(np.diag(pdf1D_cov).real)
    corrcoef = pdf1D_cov / stddev[:, None] / stddev[None, :]

    fig = plt.figure(figsize = (3, 3))

    norm = Normalize(vmin=-1, vmax=1)
    image = plt.pcolor(x, x, corrcoef.T, norm=norm, cmap=plt.get_cmap('coolwarm'))
    plt.xlabel(r'$\delta_R$')
    plt.ylabel(r'$\delta_R$')
    fig = plt.gcf()
    fig.subplots_adjust(left=0.1, right=1, bottom=0.1)
    cax = plt.axes((1., 0.195, 0.03, 0.74))
    cbar = fig.colorbar(image, cax=cax)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    if rtitle:
        plt.title(r'$R = {} \; \mathrm{{Mpc}}/h$'.format(smoothing_radius))
    plt.tight_layout()
    plt.savefig(fn, bbox_inches='tight', dpi=500)
    plt.close()
    print('Saved 1D PDF covariance: {}.'.format(fn))


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

    #axes[0].set_ylim(-15, 18)
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


def plot_pdf2D(x, mean_hist, std_hist, plot='mean_hist', cbar_label=None, xlim=None, vmax=None, show_contours=True, data_style=None, data_label=None, models=None, model_labels=None, model_styles=None, sep=None, fn=None):
    to_plot = {'mean_hist': mean_hist, 'std_hist': std_hist}
    if models is not None:
        for model in models.keys():
            to_plot[model] = models[model]

    levels = [0.011, 0.135, 0.607]
    xmid = (x[1:]+x[:-1])/2

    plt.figure(figsize=(3, 3))

    if plot in ['mean_hist', 'std_hist']:
        cmap = LinearSegmentedColormap.from_list("mycmap", ['white', 'red'])
        if vmax is not None:
            norm = Normalize(vmin=0, vmax=vmax)
        else:
            norm = None
        image = plt.imshow(to_plot[plot], origin='lower', extent=(np.min(x), np.max(x), np.min(x), np.max(x)), cmap=cmap, norm=norm)
    else:
        cmap = LinearSegmentedColormap.from_list("mycmap", ['cornflowerblue', 'white', 'red'])
        if vmax is None:
            vmax = np.nanmax(np.abs(to_plot[plot] - mean_hist))
        img = to_plot[plot] - mean_hist
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0., vmax=vmax)
        image = plt.imshow(img, origin='lower', extent=(np.min(x), np.max(x), np.min(x), np.max(x)), cmap=cmap, norm=norm)
        if show_contours:
            plt.contour(xmid, xmid, to_plot[plot], levels=levels, colors='lightgrey', alpha=0.5, linestyles={'dotted'})
    
    if show_contours:
        plt.contour(xmid, xmid, to_plot['mean_hist'], levels=levels, colors='lightgrey', alpha=0.5)

    if sep is not None:
        plt.plot([], [], label=r'$s = {:.0f} \; \mathrm{{Mpc}}/h$'.format(sep), alpha=0) # for legend
        #plt.legend(labelcolor='white')
        plt.legend()

    if model_labels is not None:
        plt.plot([], [], label=model_labels[plot], alpha=0) # for legend
        plt.legend()
            
    plt.xlabel(r'$\delta_{R}(\mathbf{r})$')
    plt.ylabel(r'$\delta_{R}(\mathbf{r + s})$')
    plt.grid(False)
    plt.xlim(xlim)
    plt.ylim(xlim)
    
    fig = plt.gcf()
    fig.subplots_adjust(left=0.1, right=1, bottom=0.1)
    cax = plt.axes((1., 0.195, 0.03, 0.74))
    cbar = fig.colorbar(image, cax=cax)
    #cax.ticklabel_format(style='sci', scilimits=(-2, 2))
    cbar.set_label(cbar_label, rotation=90)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()
    plt.savefig(fn, bbox_inches='tight', dpi=500)
    plt.close()
    print('Saved 2D PDF map plot: {}.'.format(fn))


def plot_density_splits(x, mean_ds, std_ds, std_ds_ref=None, data_style=None, data_label=None, models=None, xmodel=None, model_labels=None, model_styles=None, fn=None):

    ells = [0, 2, 4]
    nells = len(list(mean_ds[0]))
    ells = ells[:nells]
    nsplits = len(mean_ds)

    if xmodel is None:
        xmodel = x

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
            if (nsplits > 4) & (ds > 0) & (ds < nsplits - 1):
                ds_label = None
            else:
                ds_label = r'DS = {}'.format(ds + 1)
            ax0.plot(x, x**2 * mean_ds[ds][ill], color=colors[ds], label=ds_label)
            if not np.isnan(std_ds).all():
                ax0.fill_between(x, x**2 * (mean_ds[ds][ill] - std_ds[ds][ill]), x**2 * (mean_ds[ds][ill] + std_ds[ds][ill]), facecolor=colors[ds], alpha=0.3)
                if std_ds_ref is not None:
                    ax1.fill_between(x, -std_ds_ref[ds][ill]/std_ds[ds][ill], std_ds_ref[ds][ill]/std_ds[ds][ill], facecolor=colors[ds], alpha=0.3)
            
            for m in models.keys():
                ax0.plot(xmodel, xmodel**2 * models[m][ds], color=colors[ds], ls=':')

                mean_ds_interp = interp1d(x, mean_ds[ds][ill], kind=1, bounds_error=False)(xmodel)

                if not np.isnan(std_ds).all():
                    std_ds_interp =  interp1d(x, std_ds[ds][ill], kind=1, bounds_error=False)(xmodel)
                    ax1.plot(xmodel, (models[m][ds] - mean_ds_interp)/std_ds_interp, color=colors[ds], ls=':')
                    #ax1.plot(x, (models[m][ds] - mean_ds[ds][ill]), color=colors[ds], ls=':')
                    ax1.set_ylabel(r'$\Delta \xi_{R}^{\rm DS}(s) / \sigma$')
                else:
                    ax1.plot(xmodel, (models[m][ds] - mean_ds_interp), color=colors[ds], ls=':')
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
    parser.add_argument('--tracer', type=str, required=False, default='particles', choices=['particles', 'halos', 'ELG', 'LRG'])
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
    parser.add_argument('--lognormal_shotnoise', type=bool, required=False, default=True)
    parser.add_argument('--to_plot', type=str, nargs='+', required=False, default=['pdf1D', 'bias', 'pdf2D', 'densitysplits'], choices=['pdf1D', 'pdf1D_cov', 'bias', 'pdf2D', 'densitysplits'])
    parser.add_argument('--residuals', type=str, required=False, default='absolute', choices=['absolute', 'sigmas', 'percent'])
    
    args = parser.parse_args()
    z = args.redshift
    ells = [0, 2, 4] if args.rsd else [0]
    nells = len(ells)

    interpolate = False

    # Plotting
    if args.nbar <= 0.001:
        data_style = dict(color='C0', ls='', marker="o", ms=3, mec='C0', mfc='lightskyblue')
        model_styles = {'ldt': dict(color='C1', ls='', marker="o", ms=3, mec='C1', mfc='coral'),
                         'lognormal': dict(color='C3', ls='', marker="o", ms=3, mec='C3', mfc='white')}
    else:
        data_style = dict(ls='-', color='C0')
        model_styles = {'ldt': dict(ls='-', color='C1'),
                         'lognormal': dict(ls=':', color='C3'),
                         'lognormal_approx': dict(ls='--', color='C3')}        

    data_label = 'AbacusSummit'
    model_labels = {'ldt': 'LDT',
                    'lognormal': 'lognormal',
                    'lognormal_approx': r'lognormal ($s \rightarrow + \infty$)',
                    'gaussian': 'Gaussian',
                    'test': 'measured 2D PDF'}

    # Directories
    ds_dir = '/feynman/work/dphp/mp270220/outputs/densitysplit/'
    plots_dir = '/feynman/home/dphp/mp270220/plots/densitysplit'
    
    # Filenames
    if args.tracer == 'halos':
        sim_name = 'AbacusSummit_2Gpc_z{:.3f}_{{}}'.format(z)
    elif args.tracer == 'particles':
        sim_name = 'AbacusSummit_2Gpc_z{:.3f}_{{}}_downsampled_particles_nbar{:.4f}'.format(z, args.nbar)
    elif args.tracer in ['ELG', 'LRG']:
        sim_name = 'AbacusSummit_1Gpc_z0.8-1.1_{}'.format(args.tracer)

    fit_bias = args.tracer in ['ELG', 'LRG']

    for smoothing_radius in args.smoothing_radius:       
        base_name = sim_name + '_cellsize{:d}{}_resampler{}{}'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler, '_smoothingR{:d}'.format(smoothing_radius) if smoothing_radius is not None else '')
        ds_name = base_name + '_{:d}splits_randoms_size4_RH_CCF{}'.format(args.nsplits, '_RSD' if args.rsd else '')
    
        # Load measured density split measurements
        if args.tracer in ['ELG', 'LRG']:
            nmocks = 1
        else:
            nmocks = 25 if (args.nbar < 0.01) & (args.nsplits == 3) else 8
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
       
        sigma_noshotnoise = np.sqrt(sigma**2 - 1 / (nbar * 4/3 * np.pi * smoothing_radius**3))
        print('sigma no shotnoise:', sigma_noshotnoise)
         
        # Lognormal model
        lognormalmodel = LognormalDensityModel()
        mask0 = (std_pdf1D > 0) if (nmocks > 1) else np.full_like(std_pdf1D, True, dtype=bool)
        if fit_bias:
            lognormalmodel = BiasedLognormalDensityModel(lognormalmodel)
        lognormalmodel.fit_params_from_pdf(delta=result.pdf1D_x[mask0], density_pdf=mean_pdf1D[mask0],
                                            sigma=std_pdf1D[mask0] if nmocks > 1 else None, shotnoise=args.lognormal_shotnoise, norm=norm)
        if args.lognormal_shotnoise:
            lognormalpdf1D = lognormalmodel.density_shotnoise(delta=result.pdf1D_x, norm=norm)
        else:
            lognormalpdf1D = lognormalmodel.density(result.pdf1D_x)
        #sigmaYR_test = np.log(1+sigma_noshotnoise**2/lognormalmodel.delta0**2)
        #print('sigmaYR squared:', sigmaYR_test)
    
        # LDT model
        ldtmodel = LDT(redshift=z, smoothing_scale=smoothing_radius, smoothing_kernel=1, nbar=nbar)
        ldtmodel.interpolate_sigma()
    
        def fit_pdf(fit_bias=False):
            def to_fit(x, *params):
                ldtmodel.compute_ldt(params[0], k=(1 + x)*norm)
                if fit_bias:
                    ldtpdf1D = ldtmodel.density_pdf(b1=params[1])
                else:
                    ldtpdf1D = ldtmodel.density_pdf()
                return ldtpdf1D
                
            p0 = [sigma_noshotnoise, 1.] if fit_bias else [sigma_noshotnoise]
            fit = curve_fit(to_fit, result.pdf1D_x[mask0], mean_pdf1D[mask0], p0=p0, sigma=std_pdf1D[mask0] if nmocks > 1 else None)
            print(fit)
            return fit[0]
    
            # def compute_ldt(sig):
            #     ldtmodel.compute_ldt(sig, k=(1 + result.pdf1D_x)*norm)
            #     ldtpdf1D = ldtmodel.density_pdf()
            #     residuals = (ldtpdf1D[mask0] - mean_pdf1D[mask0])/std_pdf1D[mask0]
            #     return np.sum(residuals**2)        
            # mini = minimize(compute_ldt, sigma_noshotnoise)
            # print(mini)
            # return mini.x
    
        if fit_bias:
            sigma_noshotnoise, b1 = fit_pdf(fit_bias=True)
            #b1 = 1.1
        else:
            sigma_noshotnoise = fit_pdf(fit_bias=False)
            b1 = 1
        print('fitted sigma no shotnoise:', sigma_noshotnoise)
        print('fitted b1:', b1)
    
        if args.resampler=='tophat':
            if interpolate:
                ldtmodel.compute_ldt(sigma_noshotnoise)
                ldtpdf1D = ldtmodel.density_pdf(1+result.pdf1D_x, b1=b1)
            else:
                ldtmodel.compute_ldt(sigma_noshotnoise, k=(1 + result.pdf1D_x)*norm)
                ldtpdf1D = ldtmodel.density_pdf(b1=b1)            
        else:
            ldtmodel.compute_ldt(sigma_noshotnoise)
            ldtpdf1D = ldtmodel.density_pdf(1 + result.pdf1D_x, b1=b1)
    
        models_pdf1D = {'ldt': ldtpdf1D, 'lognormal': lognormalpdf1D}
    
        # plot settings
        plotting = {'data_style': data_style, 'data_label': data_label, 'model_labels': model_labels, 'model_styles': model_styles}
    
        if 'pdf1D' in args.to_plot:
            # Plot 1D PDF
            density_name = sim_name + '_cellsize{:d}_resampler{}{}{}'.format(args.cellsize, args.resampler, '_smoothingR{:d}'.format(smoothing_radius) if smoothing_radius is not None else '', '_rescaledvar{}'.format(args.rescale_var) if args.rescale_var!=1 else '')
            plot_name = density_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_1DPDF.pdf'
    
            if args.nbar > 0.01:
                rebin = 4
            else:
                rebin = None
    
            if args.nbar > 0.001:
                xlim=(-1, 3)
            else:
                xlim=(-1, 4)
            
            plot_pdf1D(result.pdf1D_x, mean_pdf1D, std_pdf1D, xlim=xlim, rebin=rebin, models=models_pdf1D, residuals=args.residuals, rtitle=args.nbar<0.001, fn=os.path.join(plots_dir, plot_name), **plotting)

        if 'pdf1D_cov' in args.to_plot:
            # Plot 1D PDF
            density_name = sim_name + '_cellsize{:d}_resampler{}{}{}'.format(args.cellsize, args.resampler, '_smoothingR{:d}'.format(smoothing_radius) if smoothing_radius is not None else '', '_rescaledvar{}'.format(args.rescale_var) if args.rescale_var!=1 else '')
            plot_name = density_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_1DPDF_cov.pdf'
            
            if args.nbar > 0.01:
                rebin = 4
            else:
                rebin = None
    
            if args.nbar > 0.001:
                xlim=(-1, 3)
            else:
                xlim=(-1, 4)
   
            plot_pdf1D_cov(result.pdf1D_x, result.pdf1D, xlim=xlim, rebin=rebin, rtitle=args.nbar<0.001, fn=os.path.join(plots_dir, plot_name))

        if 'bias' in args.to_plot:
            if args.nbar > 0.01:
                rebin = 8
            else:
                rebin = None
    
            # Bias function
            for sep in result.bias_function.keys():
                if float(sep) in [20, 40, 70]:
                    # Mocks
                    mean_bias = np.mean(result.bias_function[sep], axis=0)
                    std_bias = np.std(result.bias_function[sep], axis=0)
            
                    # LDT model
                    ldtbiasmodel = ldtmodel.bias(rho=1+result.bias_function_x[sep], b1=b1)
            
                    # Lognormal model
                    if args.lognormal_shotnoise:
                        lognormalbiasmodel = lognormalmodel.compute_bias_function_shotnoise(delta=result.bias_function_x[sep], xiR=np.mean(result.bias_corr[sep]), norm=norm)
                    else:
                        lognormalbiasmodel = lognormalmodel.compute_bias_function(delta=result.bias_function_x[sep], xiR=np.mean(result.bias_corr[sep]))
                    lognormalbiasmodel_approx = lognormalmodel.compute_bias_function_approx(result.bias_function_x[sep])
            
                    #models_bias = {'ldt': ldtbiasmodel, 'lognormal': lognormalbiasmodel, 'lognormal_approx': lognormalbiasmodel_approx}
                    models_bias = {'ldt': ldtbiasmodel, 'lognormal': lognormalbiasmodel}
            
                    # Plot bias function
                    plot_name = base_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_s{:.0f}_biasfunction.pdf'.format(float(sep))
            
                    plot_bias_function(result.bias_function_x[sep], mean_bias, std_bias, rebin=rebin, xlim=(-1, 4), sep=float(sep), models=models_bias, fn=os.path.join(plots_dir, plot_name), **plotting)
    
        # LDTmodel
        ldtdsplitmodel = LDTDensitySplitModel(ldtmodel, density_bins=result.bins)
    
        # Lognormal model
        lognormaldsplitmodel = LognormalDensitySplitModel(density_bins=result.bins)
        lognormaldsplitmodel.set_params(sigma=lognormalmodel.sigma, delta0=lognormalmodel.delta0, delta02=lognormalmodel.delta0, sigma2=lognormalmodel.sigma)
    
        if 'pdf2D' in args.to_plot:
            # 2D PDF
            xlim=(-1, 2.5) if args.nbar > 0.001 else (-1, 4)

            # Plot either probability or probability multiplied by delta
            delta_mul = False
            show_contours = True

            for sep in result.hist2D.keys():
                if float(sep) in [20, 40, 70]:
                    mean_hist = np.mean(result.hist2D[sep], axis=0)
                    std_hist = np.std(result.hist2D[sep], axis=0)
                    
                    # LDT model
                    ldtmodelpdf2D = ldtdsplitmodel.joint_density_pdf(np.mean(result.bias_corr[sep]), b1=b1)

                    # Lognormal model
                    xiYR = np.log(1 + np.mean(result.bias_corr[sep])/lognormaldsplitmodel.delta0**2)
                    cov = np.array([[lognormaldsplitmodel.sigma**2, xiYR],
                                    [xiYR, lognormaldsplitmodel.sigma**2]])
                    x = (result.hist2D_x[sep][1:]+result.hist2D_x[sep][:-1])/2
                    xmesh = np.meshgrid(x, x, indexing='ij')
                    if args.lognormal_shotnoise:
                        lognormalmodelpdf2D = lognormaldsplitmodel.density2D_shotnoise(norm=norm, delta=x, cov=cov)
                    else:
                        delta = np.dstack(np.meshgrid(x, x, indexing='ij'))
                        lognormalmodelpdf2D = lognormaldsplitmodel.density2D(delta=delta, cov=cov)

                    prefac = xmesh[1] if delta_mul else 1
                    prelabel = r'$\delta_R(\mathbf{r + s})$' if delta_mul else ''
                    
                    # Plot 2D PDF
                    plot_name = base_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_s{:.0f}_hist2D_mean.pdf'.format(float(sep))
                    plot_pdf2D(result.hist2D_x[sep], prefac*mean_hist, np.abs(prefac)*std_hist, plot='mean_hist', cbar_label=prelabel+r'$\mathcal{P}$', xlim=xlim, sep=float(sep), fn=os.path.join(plots_dir, plot_name))
                        
                    vmax = np.max([np.nanmax(np.abs(prefac)*std_hist), np.nanmax(np.abs(prefac*(ldtmodelpdf2D-mean_hist))), np.nanmax(np.abs(prefac*(lognormalmodelpdf2D-mean_hist)))])
       
                    # Plot 2D PDF error
                    plot_name = base_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_s{:.0f}_hist2D_std.pdf'.format(float(sep))
                    plot_pdf2D(result.hist2D_x[sep], prefac*mean_hist, np.abs(prefac)*std_hist, plot='std_hist', vmax=vmax, cbar_label=prelabel+r'$\sigma_{\mathcal{P}}$', xlim=xlim, sep=float(sep), fn=os.path.join(plots_dir, plot_name))
                         
                    # Plot LDT model
                    models = {'ldt': prefac*ldtmodelpdf2D}
                    plot_name = base_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_s{:.0f}_hist2D_ldt.pdf'.format(float(sep))
                    plot_pdf2D(result.hist2D_x[sep], prefac*mean_hist, np.abs(prefac)*std_hist, plot='ldt', vmax=vmax, show_contours=show_contours, cbar_label=prelabel+r'$\Delta \mathcal{P}$', xlim=xlim, sep=float(sep), models=models, model_labels=model_labels, fn=os.path.join(plots_dir, plot_name))

                    # Plot lognormal model
                    models = {'lognormal': prefac*lognormalmodelpdf2D}
                    plot_name = base_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_s{:.0f}_hist2D_lognormal.pdf'.format(float(sep))
                    plot_pdf2D(result.hist2D_x[sep], prefac*mean_hist, np.abs(prefac)*std_hist, plot='lognormal', vmax=vmax, show_contours=show_contours, cbar_label=prelabel+r'$\Delta \mathcal{P}$', xlim=xlim, sep=float(sep), models=models, model_labels=model_labels, fn=os.path.join(plots_dir, plot_name))
                     
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
            if args.nsplits==3:
                std_ds_desi = np.load(os.path.join(ds_dir, fname))
            else:
                std_ds_desi = None
                
            # LDT model
            ldtdsplits = ldtdsplitmodel.compute_dsplits(mean_xiR, b1=b1)
        
            # Lognormal model
            if args.lognormal_shotnoise:
                lognormaldsplits = lognormaldsplitmodel.compute_dsplits_shotnoise(xi=mean_xiR, norm=norm, delta=result.pdf1D_x)
            else:
                lognormaldsplits = lognormaldsplitmodel.compute_dsplits(smoothing=1, sep=sep, xiR=mean_xiR, rsd=False, ells=ells)
            #lognormaldsplits = None

            # Test with measured 2D PDF
            #density_pdf_2D_list = [result.hist2D[sep] for sep in result.hist2D.keys()]
            testdsplits_list = list()
            for i in range(nmocks):
                tmp = [result.hist2D[sep][i] for sep in result.hist2D.keys()]
                testdsplits = lognormaldsplitmodel.compute_dsplits_test(density_pdf_2D=tmp, norm=result.norm_all[i], delta=result.pdf1D_x)
                testdsplits_list.append(testdsplits)
            testdsplits = np.mean(testdsplits_list, axis=0)

            # Gaussian model
            ds_delta_tilde = list()
            for i in range(len(result.bins)-1):
                dlt1, dlt2 = max(result.bins[i], -1), result.bins[i+1]
                ds_mask = (result.pdf1D_x >= dlt1) & (result.pdf1D_x < dlt2)
                delta_tilde = np.sum(result.pdf1D_x[ds_mask] * mean_pdf1D[ds_mask])/np.sum(mean_pdf1D[ds_mask])
                ds_delta_tilde.append(delta_tilde)
            gaussiandsplits = np.array(ds_delta_tilde)[:, None] * mean_xiR[None, :] / sigma**2
        
            models_ds = {'ldt': ldtdsplits, 'lognormal': lognormaldsplits, 'gaussian': gaussiandsplits, 'test': testdsplits}
        
            seps = np.array([float(s) for s in result.hist2D.keys()])

            for m in ['ldt']:
                plot_name = base_name.format('{}mocks'.format(nmocks)) + '_{:d}densitysplits_{}model.pdf'.format(args.nsplits, m)
                models = {m: models_ds[m]}

                #print('sep: ', sep)
                #print('seps: ', seps)

                if m=='test':
                    plot_density_splits(sep, mean_ds, std_ds, std_ds_ref=std_ds_desi, models=models, xmodel=seps, fn=os.path.join(plots_dir, plot_name), **plotting)
                else:
                    plot_density_splits(sep, mean_ds, std_ds, std_ds_ref=std_ds_desi, models=models, fn=os.path.join(plots_dir, plot_name), **plotting)

            
            