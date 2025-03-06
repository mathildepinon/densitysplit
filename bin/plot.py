import os
import time
import sys
import argparse
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import matplotlib.ticker as ticker
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d
from iminuit import Minuit

from densitysplit.corr_func_utils import get_split_poles
from densitysplit import CountInCellsDensitySplitMeasurement
from densitysplit.lognormal_model import LognormalDensityModel, BiasedLognormalDensityModel, LognormalDensitySplitModel
from densitysplit.ldt_model import LDT, LDTDensitySplitModel, setup_logging
from density_split_corr import compute_lognormal_split_bins

plt.style.use(os.path.join(os.path.abspath('/feynman/home/dphp/mp270220/densitysplit/nb'), 'densitysplit.mplstyle'))


def plot_pdf1D(x, mean_pdf1D, std_pdf1D, xlim=None, rebin=None, residuals='absolute', data_style=None, data_label=None, models=None, model_labels=None, model_styles=None, rtitle=False, galaxies=False, fn=None):

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

    #axes[0].plot(x, mean_pdf1D, label=data_label, **data_style)
    axes[0].errorbar(x, mean_pdf1D, std_pdf1D, label=data_label, **data_style)
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
    dlabel = '\delta_{R,g}' if galaxies else '\delta_{R}'
    plabel = ''
    axes[1].set_xlabel(r'${}$'.format(dlabel))
    axes[0].set_ylabel(r'$\mathcal{{P}}{}({})$'.format(plabel, dlabel))

    if residuals=='absolute':
        axes[1].set_ylabel(r'$\Delta \mathcal{{P}}{}({})$'.format(plabel, dlabel))
    elif residuals=='sigmas':
        axes[1].set_ylabel(r'$\Delta \mathcal{{P}}{}({}) / \sigma$'.format(plabel, dlabel)) 
    elif residuals=='percent':
        axes[1].set_ylabel(r'$\Delta \mathcal{{P}}{}({}) / \mathcal{{P}}{}({})$'.format(plabel, dlabel, plabel, dlabel))

    axes[0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    axes[0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axes[1].yaxis.set_minor_locator(ticker.AutoMinorLocator())

    if rtitle:
        axes[0].set_title(r'$R = {} \; \mathrm{{Mpc}}/h$'.format(smoothing_radius))
    axes[0].legend(loc=(0.34, 0.58))
    #axes[0].legend(loc='upper right')
    plt.rcParams["figure.autolayout"] = False
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)
    fig.align_ylabels()
    plt.savefig(fn, dpi=500)
    plt.close()
    print('Saved 1D PDF plot: {}.'.format(fn))


def get_cumulants(x, pdf_list, reduced=False):
    c1_all = np.sum(x * pdf_list, axis=-1)/norm
    if pdf_list.ndim ==  1:
        x = x - c1_all
    else:
        x = x[None, :] - c1_all[:, None]
    c2_all = np.sum(x**2 * pdf_list, axis=-1)/norm
    c3_all = np.sum(x**3 * pdf_list, axis=-1)/norm
    c4_all = np.sum(x**4 * pdf_list, axis=-1)/norm - 3 * c2_all**2
    c5_all = np.sum(x**5 * pdf_list, axis=-1)/norm - 10 * c3_all * c2_all
    cumulants_all = np.array([c1_all, c2_all, c3_all, c4_all, c5_all])
    if reduced:
        scaling = np.array([c2_all**i for i in range(5)])
        cumulants_all /= scaling
    cumulants  = dict()
    for i, c in enumerate(cumulants_all):
        cumulants[i+1] = np.mean(c), np.std(c)
    return cumulants


def print_table_latex(rownames, col1, col2=None, col3=None, col_labels=None):
    import tabulate

    data = []
    for i, rowname in enumerate(rownames):
        row = []
        row.append(rowname)
        row.append(col1[i])
        if col2 is not None:
            row.append(col2[i])
        if col3 is not None:
            row.append(col3[i])
        data.append(row)
    tab = tabulate.tabulate(data, headers=col_labels, tablefmt='latex_raw')
    print(tab)


def plot_pdf1D_shotnoise(x, xlim=None, rebin=None, data_style=None, data_label=None, models=None, model_labels=None, model_styles=None, rtitle=False, fn=None):

    if rebin is not None:
        x = x[::rebin]

    if xlim is not None:
        mask = (x >= xlim[0]) & (x <= xlim[1])
        x = x[mask]
    
    fig = plt.figure(figsize = (3.5, 3))
    ax = plt.gca()

    if models is not None:
        for m in models.keys():
            if rebin is not None:
                models[m] = models[m][::rebin]
            if xlim is not None:
                models[m] = models[m][mask]
                
            ax.plot(x, models[m], label=model_labels[m], **model_styles[m])

    ax.ticklabel_format(style='sci', scilimits=(-3, 3))
    ax.set_xlabel(r'$\delta_R$')
    ax.set_ylabel(r'$\mathcal{P}(\delta_R)$')

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    if rtitle:
        ax.set_title(r'$R = {} \; \mathrm{{Mpc}}/h$'.format(smoothing_radius))
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(fn, dpi=500)
    plt.close()
    print('Saved 1D PDF shot noise plot: {}.'.format(fn))


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


def plot_bias_function(x, mean_bias, std_bias, xlim=None, rebin=None, data_style=None, data_label=None, models=None, model_labels=None, model_styles=None, sep=None, rescale_errorbars=1, galaxies=False, fn=None):

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
 
    #axes[0].plot(x, mean_bias, label=data_label, **data_style)
    #axes[0].fill_between(x, mean_bias - std_bias, mean_bias + std_bias, facecolor=data_style['color'], alpha=0.3)
    axes[0].errorbar(x, mean_bias, std_bias/rescale_errorbars, label=data_label, **data_style)

    for m in models.keys():
        if rebin is not None:
            models[m] = models[m][::rebin]
        if xlim is not None:
            models[m] = models[m][mask]

        axes[0].plot(x, models[m], label=model_labels[m], **model_styles[m], zorder=20)
        axes[1].plot(x, (models[m] - mean_bias)/std_bias, **model_styles[m])

    #axes[0].set_ylim(ymax=12)
    axes[1].set_ylim(-2, 2)
    axes[0].legend(loc='lower right')
    dlabel = '\delta_{R,g}' if galaxies else '\delta_{R}'
    plabel = ''
    axes[1].set_xlabel(r'${}$'.format(dlabel))
    axes[0].set_ylabel(r'$b{}({})$'.format(plabel, dlabel))
    axes[1].set_ylabel(r'$\Delta b{}({}) / \sigma$'.format(plabel, dlabel))
    #axes[0].set_title(r'$R = {} \; \mathrm{{Mpc}}/h$'.format(smoothing_radius))
    axes[0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    axes[0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axes[1].yaxis.set_minor_locator(ticker.AutoMinorLocator())

    if sep is not None:
        axes[0].text(0.1, 0.9, r'$s = {:.0f} \; \mathrm{{Mpc}}/h$'.format(sep), ha='left', va='top', transform = axes[0].transAxes, fontsize=12)
    
    fig.align_ylabels()
    plt.rcParams["figure.autolayout"] = False
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(fn, dpi=500)
    plt.close()
    print('Plot saved at {}'.format(fn))


def plot_bias_function_shotnoise(x, xlim=None, rebin=None, data_style=None, data_label=None, models=None, model_labels=None, model_styles=None, sep=None, rescale_errorbars=1, fn=None):

    if rebin is not None:
        x = x[::rebin]

    if xlim is not None:
        mask = (x >= xlim[0]) & (x <= xlim[1])
        x = x[mask]
    
    fig = plt.figure(figsize = (3.5, 3))
    ax = plt.gca()

    for m in models.keys():
        if rebin is not None:
            models[m] = models[m][::rebin]
        if xlim is not None:
            models[m] = models[m][mask]

        ax.plot(x, models[m], label=model_labels[m], **model_styles[m], zorder=20)

    #ax.set_ylim(-7, 16)
    ax.legend(loc='lower right')
    ax.set_xlabel(r'$\delta_R$')
    ax.set_ylabel(r'$b(\delta_R)$')
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    if sep is not None:
        ax.text(0.1, 0.9, r'$s = {:.0f} \; \mathrm{{Mpc}}/h$'.format(sep), ha='left', va='top', transform = ax.transAxes, fontsize=12)

    plt.tight_layout()
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


def plot_density_splits(x, mean_ds, std_ds, std_ds_ref=None, data_style=None, data_label=None, models=None, xmodel=None, model_labels=None, model_styles=None, rescale_errorbars=1, fn=None):

    ells = [0, 2, 4]
    nells = len(list(mean_ds[0]))
    ells = ells[:nells]
    nsplits = len(mean_ds)

    if xmodel is None:
        xmodel = x

    figsize = (2.5*nells+1, 3.5)
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
            #ax0.plot(x, x**2 * mean_ds[ds][ill], color=colors[ds], label=ds_label)
            if np.isnan(std_ds).all():
                ax0.plot(x, x**2 * mean_ds[ds][ill], ls='', marker="o", ms=2.6, markeredgewidth=0.9, color=colors[ds], mfc='white', mec=colors[ds], label=ds_label)
            else:
                ax0.errorbar(x, x**2 * mean_ds[ds][ill], x**2 * std_ds[ds][ill]/rescale_errorbars, ls='', marker="o", ms=2.6, elinewidth=0.9, markeredgewidth=0.9, color=colors[ds], mfc='white', mec=colors[ds], label=ds_label)
                #ax0.fill_between(x, x**2 * (mean_ds[ds][ill] - std_ds[ds][ill]), x**2 * (mean_ds[ds][ill] + std_ds[ds][ill]), facecolor=colors[ds], alpha=0.3)
                if std_ds_ref is not None:
                    ax1.fill_between(x, -std_ds_ref[ds][ill]/std_ds[ds][ill], std_ds_ref[ds][ill]/std_ds[ds][ill], facecolor=colors[ds], alpha=0.3)
            
            if models is not None:
                for m in models.keys():
                    if nells > 1:
                        model_ds = models[m][ds][ill]
                    else:
                        model_ds = models[m][ds]
                    ax0.plot(xmodel, xmodel**2 * model_ds, color=colors[ds], ls='-')

                    mean_ds_interp = interp1d(x, mean_ds[ds][ill], kind=1, bounds_error=False)(xmodel)

                    if not np.isnan(std_ds).all():
                        std_ds_interp =  interp1d(x, std_ds[ds][ill], kind=1, bounds_error=False)(xmodel)
                        ax1.plot(xmodel, (model_ds - mean_ds_interp)/std_ds_interp, color=colors[ds], ls='-')
                        #ax1.plot(x, (models[m][ds] - mean_ds[ds][ill]), color=colors[ds], ls=':')
                        ax1.set_ylim(-5, 5)
                    else:
                        ax1.plot(xmodel, (model_ds - mean_ds_interp), color=colors[ds], ls='-')
                        ax1.set_ylim(-0.001, 0.001)

        #ax0.set_ylim(-45, 50)
        if nells > 1:
            ax0.set_title(r'$\ell = {}$'.format(ell))
        ax1.set_xlabel(r'$s$ [$h^{-1}\mathrm{Mpc}$]')
        
        ax0.set_xlim(0, 151)
        ax0.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax0.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        ax0.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax0.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        if not np.isnan(std_ds).all():
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
            ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1))
 
    if models is not None:
        for m in models.keys():
            l, = ax0.plot([], ls='-', color='black')
            legend = ax0.legend([l], [model_labels[m]], loc='lower right')
        ax0.add_artist(legend)
    ax0.legend(loc='upper right')

    ax0 = axes[0][0] if nells > 1 else axes[0]
    ax1 = axes[1][0] if nells > 1 else axes[1]
    ax0.set_ylabel(r'$s^2 \xi_{R}^{\rm DS}(s)$ [$(h^{-1}\mathrm{Mpc})^{2}$]')
    if np.isnan(std_ds).all():
        ax1.set_ylabel(r'$\Delta \xi_{R}^{\rm DS}(s)$')
    else:
        ax1.set_ylabel(r'$\Delta \xi_{R}^{\rm DS}(s) / \sigma$')
    #ax0.set_title(r'$R = {} \; \mathrm{{Mpc}}/h$'.format(smoothing_radius))

    plt.rcParams["figure.autolayout"] = False
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
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
    parser.add_argument('--smoothing_radius', type=int, nargs='+', required=False, default=None)
    parser.add_argument('--use_weights', type=bool, required=False, default=False)
    parser.add_argument('--rsd', type=bool, required=False, default=False)
    parser.add_argument('--los', type=str, required=False, default='x')
    parser.add_argument('--nsplits', type=int, required=False, default=3)
    parser.add_argument('--randoms_size', type=int, required=False, default=4)
    parser.add_argument('--size', type=int, required=False, default=None)
    parser.add_argument('--rescale_var', type=float, required=False, default=1)
    parser.add_argument('--lognormal_shotnoise', type=bool, required=False, default=True)
    parser.add_argument('--to_plot', type=str, nargs='+', required=False, default=['pdf1D', 'bias', 'pdf2D', 'densitysplits'], choices=['pdf1D', 'shotnoise', 'pdf1D_cov', 'bias', 'pdf2D', 'densitysplits'])
    parser.add_argument('--residuals', type=str, required=False, default='absolute', choices=['absolute', 'sigmas', 'percent'])
    
    args = parser.parse_args()
    z = args.redshift
    ells = [0, 2, 4] if args.rsd else [0]
    nells = len(ells)

    interpolate = False

    # Plotting
    if (args.nbar <= 0.0001) &  (np.min(args.smoothing_radius) < 20):
        data_style = dict(marker="o", ls='', ms=2.6, elinewidth=0.9, markeredgewidth=0.9, color='C0', mfc='white', mec='C0')
        model_styles = {'ldt': dict(color='C1', ls='', marker="o", ms=2.6, mec='C1', markeredgewidth=0.9, mfc='C1'),
                        'lognormal': dict(color='C3', ls='', marker="o", ms=2.6, mec='C3', markeredgewidth=0.9, mfc='white')}
    else:
        data_style = dict(marker="o", ls='', ms=2.6, elinewidth=0.9, markeredgewidth=0.9, color='C0', mfc='white', mec='C0')
        model_styles = {'ldt': dict(ls='-', color='C1'),
                        'ldt_noshotnoise': dict(ls=':', color='C1'),
                        'lognormal': dict(ls=':', color='C3'),
                        'lognormal_approx': dict(ls='--', color='C3')}        

    data_label = 'AbacusSummit'
    model_labels = {'ldt': 'LDT',
                    'ldt_noshotnoise': 'no shot noise',
                    'lognormal': 'log-normal',
                    'lognormal_approx': r'lognormal ($s \rightarrow + \infty$)',
                    'gaussian': 'Gaussian',
                    'test': 'measured 2D PDF',
                    'fit': 'bias fit'}

    # Directories
    ds_dir = '/feynman/work/dphp/mp270220/outputs/densitysplit/' #if ((args.nsplits == 3) and (args.tracer != 'ELG')) else '/feynman/scratch/dphp/mp270220/outputs/densitysplit/'
    plots_dir = '/feynman/home/dphp/mp270220/plots/densitysplit'
    
    # Filenames
    if args.tracer == 'halos':
        sim_name = 'AbacusSummit_2Gpc_z{:.3f}_{{}}'.format(z)
    elif args.tracer == 'particles':
        sim_name = 'AbacusSummit_2Gpc_z{:.3f}_{{}}_downsampled_particles_nbar{:.4f}'.format(z, args.nbar)

    for smoothing_radius in args.smoothing_radius:       
        base_name = sim_name + '_cellsize{:d}{}_resampler{}{}'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler, '_smoothingR{:d}'.format(smoothing_radius) if smoothing_radius is not None else '')
        ds_name = base_name + '_{:d}splits_randoms_size4_RH_CCF{}'.format(args.nsplits, '_rsd' if args.rsd else '')
    
        # Load measured density split measurements
        nmocks = 25 if (args.nbar < 0.01) else 8
        if args.rsd: 
            nmocks = 8
        fn = os.path.join(ds_dir, ds_name.format('{}mocks'.format(nmocks)) + '_compressed.npy')
        print('Loading density split measurements: {}'.format(fn))
        result = CountInCellsDensitySplitMeasurement.load(fn)

        # Load density vectors
        #from densitysplit import DensitySplit
        #t0 = time.time()
        #print('reading...')
        #fn = [os.path.join('/feynman/scratch/dphp/mp270220/outputs/densitysplit/', ds_name.format('ph0{:02d}'.format(i)) + '.npy') for i in range(nmocks)]
        #ds = [DensitySplit.load(f) for f in fn]
        #N_sample = np.array([ds[i].density_mesh.value.flatten() for i in range(nmocks)])

        #deltaname = sim_name + '_cellsize{:d}_resampler{}{}_N{}.npy'.format(args.cellsize, args.resampler, '_smoothingR{:02d}'.format(smoothing_radius) if args.smoothing_radius is not None else '', '_rsd' if args.rsd else '')
        #fn = [os.path.join('/feynman/work/dphp/mp270220/outputs/density/', deltaname.format('ph0{:02d}'.format(i))) for i in range(nmocks)]
        #Nvector = np.array([np.load(fn[i]) for i in range(nmocks)])
        #print('elapsed time to read data: {}'.format(time.time()-t0))
    
        sigma = result.sigma
        delta3 = result.delta3
        print('sigma direct:', sigma)
        print('sigma test:', np.sqrt(np.mean(result.sigma_all**2)))
        print('delta3 direct:', delta3)
        nbar = result.nbar
        norm = result.norm
        smoothing_radius = result.smoothing_radius
    
        # 1D PDF
        mean_pdf1D = np.mean(result.pdf1D, axis=0)
        std_pdf1D = np.std(result.pdf1D, axis=0)

        # Cumulants
        cumulants = get_cumulants(result.pdf1D_x, result.pdf1D)
        print("Measured cumulants: ", cumulants)
       
        sigma = sigma#np.sqrt(np.sum(result.pdf1D_x**2 * mean_pdf1D)/norm)
        print('sigma integral:', sigma)
        sigma_noshotnoise = np.sqrt(sigma**2 - 1 / (nbar * 4/3 * np.pi * smoothing_radius**3))
        print('sigma no shotnoise:', sigma_noshotnoise)
 
        delta3 = delta3#cumulants[3][0]
        print('delta3 integral:', delta3)
        nV = nbar * 4/3 * np.pi * smoothing_radius**3
        delta3_noshotnoise = delta3 - 1/nV**2 - 3/nV * sigma_noshotnoise**2
        print('delta3 no shotnoise:', delta3_noshotnoise)
         
        # Lognormal model
        lognormalmodel = LognormalDensityModel()
        mask0 = (std_pdf1D > 0) if (nmocks > 1) else np.full_like(std_pdf1D, True, dtype=bool)
        lognormalmodel.fit_params_from_pdf(delta=result.pdf1D_x[mask0], density_pdf=mean_pdf1D[mask0],
                                            sigma=std_pdf1D[mask0] if nmocks > 1 else None, shotnoise=args.lognormal_shotnoise, norm=norm)
        #lognormalmodel.fit_params_from_sample(N_sample/norm-1, params_init=np.array([sigma_noshotnoise, 1.]), shotnoise=args.lognormal_shotnoise, norm=norm)
        #lognormalmodel.get_params_from_moments(m2=sigma_noshotnoise**2, m3=delta3_noshotnoise, delta0_init=1.)
        if args.lognormal_shotnoise:
            lognormalpdf1D = lognormalmodel.density_shotnoise(delta=result.pdf1D_x, norm=norm)
        else:
            lognormalpdf1D = lognormalmodel.density(result.pdf1D_x)
        #sigmaYR_test = np.log(1+sigma_noshotnoise**2/lognormalmodel.delta0**2)
        #print('sigmaYR squared:', sigmaYR_test)
    
        # LDT model
        ldtmodel = LDT(redshift=z, smoothing_scale=smoothing_radius, smoothing_kernel=1, nbar=nbar)
        ldtmodel.interpolate_sigma()

        # Fit variance
        sigma_noshotnoise  = ldtmodel.fit_from_pdf(result.pdf1D_x, mean_pdf1D, err=std_pdf1D if nmocks > 1 else None, fix_sigma=False, sigma_init=sigma_noshotnoise, norm=norm)               
        print('fitted sigma no shotnoise:', sigma_noshotnoise)
        #np.save('/feynman/scratch/dphp/mp270220/outputs/ldt_sigma_fit', sigma_noshotnoise)
    
        if args.resampler=='tophat':
            if interpolate:
                ldtmodel.compute_ldt(sigma_noshotnoise)
                ldtpdf1D = ldtmodel.density_pdf(1+result.pdf1D_x)
            else:
                ldtmodel.compute_ldt(sigma_noshotnoise, k=(1 + result.pdf1D_x)*norm)
                ldtpdf1D = ldtmodel.density_pdf()
            if 'shotnoise' in args.to_plot:
                x = ldtmodel.yvals[ldtmodel.yvals < ldtmodel.ymax]
                ldtpdf1D_noshotnoise = ldtmodel.density_pdf_noshotnoise(rho=x)   
                ldtpdf1D_noshotnoise = interp1d(x, ldtpdf1D_noshotnoise, bounds_error=False, fill_value=0)(1 + result.pdf1D_x)           
        else:
            ldtmodel.compute_ldt(sigma_noshotnoise)
            ldtpdf1D = ldtmodel.density_pdf(1 + result.pdf1D_x)

        ldt_cumulants = get_cumulants(result.pdf1D_x, ldtpdf1D)
        print("LDT cumulants: ", ldt_cumulants)

        lognormal_cumulants = get_cumulants(result.pdf1D_x, lognormalpdf1D)
        print("lognormal cumulants: ", lognormal_cumulants)

        rownames = [r'$\kappa_{}$'.format(i) for i in range(1, 6)]
        #rownames = [r'$S_{}$'.format(i) for i in range(1, 6)]
        col_labels = ['', 'measured', 'LDT', 'log-normal']
        col1 = [r'${:.4g} \pm {:.2g}$'.format(c[0], c[1]) for c in cumulants.values()]
        col2 = [r'${:.4g}$'.format(c[0]) for c in ldt_cumulants.values()]
        col3 = [r'${:.4g}$'.format(c[0]) for c in lognormal_cumulants.values()]
        print_table_latex(rownames, col1, col2, col3, col_labels)
     
        models_pdf1D = {'ldt': ldtpdf1D, 'lognormal': lognormalpdf1D}
    
        # plot settings
        plotting = {'data_style': data_style, 'data_label': data_label, 'model_labels': model_labels, 'model_styles': model_styles}
    
        if 'pdf1D' in args.to_plot:
            # Plot 1D PDF
            density_name = sim_name + '_cellsize{:d}_resampler{}{}{}{}'.format(args.cellsize, args.resampler, '_smoothingR{:d}'.format(smoothing_radius) if smoothing_radius is not None else '', '_rescaledvar{}'.format(args.rescale_var) if args.rescale_var!=1 else '', '_rsd' if args.rsd else '')
            plot_name = density_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_1DPDF.pdf'
    
            if args.nbar > 0.01:
                rebin = 4
            else:
                rebin = None
    
            #if args.nbar > 0.003:
            #    xlim=(-1, 3)
            #else:
            #    xlim=(-1, 4)

            maxpdf = result.pdf1D_x[np.argmax(mean_pdf1D)]
            print('maxpdf', maxpdf)
            xlim=(maxpdf-5*sigma, maxpdf+5*sigma)
            print('xlim', xlim)
            
            plot_pdf1D(result.pdf1D_x, mean_pdf1D, std_pdf1D, xlim=xlim, rebin=rebin, models=models_pdf1D, residuals=args.residuals, rtitle=args.nbar<0.001, fn=os.path.join(plots_dir, plot_name), **plotting)

            if 'shotnoise' in args.to_plot:
                models_pdf1D = {'ldt': ldtpdf1D, 'ldt_noshotnoise': ldtpdf1D_noshotnoise}
                plot_name = density_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_1DPDF_LDT_shotnoise.pdf'
                plot_pdf1D_shotnoise(result.pdf1D_x, xlim=xlim, rebin=rebin, models=models_pdf1D, rtitle=args.nbar<0.001, fn=os.path.join(plots_dir, plot_name), **plotting)

        if 'pdf1D_cov' in args.to_plot:
            # Plot 1D PDF
            density_name = sim_name + '_cellsize{:d}_resampler{}{}{}{}'.format(args.cellsize, args.resampler, '_smoothingR{:d}'.format(smoothing_radius) if smoothing_radius is not None else '', '_rescaledvar{}'.format(args.rescale_var) if args.rescale_var!=1 else '', '_rsd' if args.rsd else '')
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
                if float(sep) in [5, 10, 20, 40, 70, 110]:
                    # Mocks
                    mean_bias = np.mean(result.bias_function[sep], axis=0)
                    std_bias = np.std(result.bias_function[sep], axis=0)

                    if args.tracer=='particles':
                        xiR = np.mean(result.bias_corr[sep])
                        np.save('/feynman/scratch/dphp/mp270220/outputs/'+sim_name.format('{}mocks'.format(nmocks))+'_xiR_sep{}.npy'.format(sep), xiR)
                    else:
                        xiR = np.load('/feynman/scratch/dphp/mp270220/outputs/'+'AbacusSummit_2Gpc_z{:.3f}_{}_downsampled_particles_nbar0.0034'.format(0.8, '{}mocks'.format(25))+'_xiR_sep{}.npy'.format(sep))
                
                    print('xiR:')
                    print(xiR)
                    print(np.mean(result.bias_corr[sep]))
                    print('ratio:', xiR/np.mean(result.bias_corr[sep]))
                    print('sqrt:', np.sqrt(xiR/np.mean(result.bias_corr[sep])))

                    # LDT model
                    ldtbiasmodel = ldtmodel.bias(rho=1+result.bias_function_x[sep])
            
                    # Lognormal model
                    if args.lognormal_shotnoise:
                        lognormalbiasmodel = lognormalmodel.compute_bias_function_shotnoise(delta=result.bias_function_x[sep], xiR=xiR, norm=norm)
                    else:
                        lognormalbiasmodel = lognormalmodel.compute_bias_function(delta=result.bias_function_x[sep], xiR=xiR)
                    lognormalbiasmodel_approx = lognormalmodel.compute_bias_function_approx(result.bias_function_x[sep])
            
                    #models_bias = {'ldt': ldtbiasmodel, 'lognormal': lognormalbiasmodel, 'lognormal_approx': lognormalbiasmodel_approx}
                    models_bias = {'ldt': ldtbiasmodel, 'lognormal': lognormalbiasmodel}
            
                    # Plot bias function
                    plot_name = base_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + ('_rsd' if args.rsd else '') + '_s{:.0f}_biasfunction.pdf'.format(float(sep))
            
                    plot_bias_function(result.bias_function_x[sep], mean_bias, std_bias, rebin=rebin, xlim=(maxpdf-8*sigma, maxpdf+8*sigma), sep=float(sep), models=models_bias, fn=os.path.join(plots_dir, plot_name), rescale_errorbars=np.sqrt(nmocks), **plotting)
    
                    if 'shotnoise' in args.to_plot:
                        ldtbiasmodel_noshotnoise = ldtmodel.bias_noshotnoise(rho=1+result.bias_function_x[sep])
                        models_pdf1D = {'ldt': ldtbiasmodel, 'ldt_noshotnoise': ldtbiasmodel_noshotnoise}
                        plot_name = base_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + ('_rsd' if args.rsd else '') + '_s{:.0f}_biasfunction_LDT_shotnoise.pdf'.format(float(sep))
                        plot_bias_function_shotnoise(result.bias_function_x[sep], xlim=(-1, 4), rebin=rebin, sep=float(sep), models=models_pdf1D, fn=os.path.join(plots_dir, plot_name), rescale_errorbars=np.sqrt(nmocks), **plotting)
        
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
                    ldtmodelpdf2D = ldtdsplitmodel.joint_density_pdf(np.mean(result.bias_corr[sep]))

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
                    plot_name = base_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_rsd' if args.rsd else '' + '_s{:.0f}_hist2D_mean.pdf'.format(float(sep))
                    plot_pdf2D(result.hist2D_x[sep], prefac*mean_hist, np.abs(prefac)*std_hist, plot='mean_hist', cbar_label=prelabel+r'$\mathcal{P}$', xlim=xlim, sep=float(sep), fn=os.path.join(plots_dir, plot_name))
                        
                    vmax = np.max([np.nanmax(np.abs(prefac)*std_hist), np.nanmax(np.abs(prefac*(ldtmodelpdf2D-mean_hist))), np.nanmax(np.abs(prefac*(lognormalmodelpdf2D-mean_hist)))])
       
                    # Plot 2D PDF error
                    plot_name = base_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_rsd' if args.rsd else '' + '_s{:.0f}_hist2D_std.pdf'.format(float(sep))
                    plot_pdf2D(result.hist2D_x[sep], prefac*mean_hist, np.abs(prefac)*std_hist, plot='std_hist', vmax=vmax, cbar_label=prelabel+r'$\sigma_{\mathcal{P}}$', xlim=xlim, sep=float(sep), fn=os.path.join(plots_dir, plot_name))
                         
                    # Plot LDT model
                    models = {'ldt': prefac*ldtmodelpdf2D}
                    plot_name = base_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_rsd' if args.rsd else '' + '_s{:.0f}_hist2D_ldt.pdf'.format(float(sep))
                    plot_pdf2D(result.hist2D_x[sep], prefac*mean_hist, np.abs(prefac)*std_hist, plot='ldt', vmax=vmax, show_contours=show_contours, cbar_label=prelabel+r'$\Delta \mathcal{P}$', xlim=xlim, sep=float(sep), models=models, model_labels=model_labels, fn=os.path.join(plots_dir, plot_name))

                    # Plot lognormal model
                    models = {'lognormal': prefac*lognormalmodelpdf2D}
                    plot_name = base_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_rsd' if args.rsd else '' + '_s{:.0f}_hist2D_lognormal.pdf'.format(float(sep))
                    plot_pdf2D(result.hist2D_x[sep], prefac*mean_hist, np.abs(prefac)*std_hist, plot='lognormal', vmax=vmax, show_contours=show_contours, cbar_label=prelabel+r'$\Delta \mathcal{P}$', xlim=xlim, sep=float(sep), models=models, model_labels=model_labels, fn=os.path.join(plots_dir, plot_name))
                     
        if 'densitysplits' in args.to_plot:
            # Density splits
            try:
                mean_xiR = np.mean([res(ells=ells if args.rsd else None, ignore_nan=True) for res in result.smoothed_corr], axis=0)
            except:
                mean_xiR = np.mean(result.smoothed_corr, axis=0)
            mean_ds, cov = get_split_poles(result.ds_corr, ells=None if (args.resampler=='tophat') & (not args.rsd) else ells)
            std_ds = np.nan if cov.size == 1 else np.array_split(np.array(np.array_split(np.diag(cov)**0.5, nells)), args.nsplits, axis=1)
            sep = result.sep
    
            if args.nbar < 0.001:
                # save errors
                fname = base_name.format('{}mocks'.format(nmocks)) + '_densitysplits_std'
                np.save(os.path.join(ds_dir, fname), std_ds)
    
            # Load errors from mocks with nbar = 0.0005
            ds_dir = '/feynman/work/dphp/mp270220/outputs/densitysplit/'
            fname = 'AbacusSummit_2Gpc_z0.800_25mocks_downsampled_particles_nbar0.0005_cellsize5_cellsize5_resamplertophat_smoothingR10_densitysplits_std.npy'
            if (args.nsplits==3) & (args.nbar > 0.001) & (args.smoothing_radius[0]==10):
                std_ds_desi = np.load(os.path.join(ds_dir, fname))
            else:
                std_ds_desi = None
            if args.rsd:
                std_ds_desi = None

            # LDT model
            #seps = np.array([float(s) for s in result.hist2D.keys()])
            #mean_bias_allseps = np.mean(np.array([result.bias_function[sep] for sep in result.bias_function.keys()]), axis=1)
            #mean_bias_interp = interp1d(seps, mean_bias_allseps, axis=0)(sep)
            #print('bias shape: ', mean_bias_interp.T.shape)
            #ldtdsplits = ldtdsplitmodel.compute_dsplits(mean_xiR, bias=mean_bias_allseps.T[:, seps==20], density_pdf=None)
            ldtdsplits = ldtdsplitmodel.compute_dsplits(mean_xiR)
        
            # Lognormal model
            if args.rsd:
                lognormaldsplits = None
            else:
                if args.lognormal_shotnoise:
                    lognormaldsplits = lognormaldsplitmodel.compute_dsplits_shotnoise(xi=mean_xiR, norm=norm, delta=result.pdf1D_x)
                else:
                    lognormaldsplits = lognormaldsplitmodel.compute_dsplits(smoothing=1, sep=sep, xiR=mean_xiR, rsd=False, ells=ells)

            # Test with measured 2D PDF
            #density_pdf_2D_list = [result.hist2D[sep] for sep in result.hist2D.keys()]
            # testdsplits_list = list()
            # for i in range(nmocks):
            #     tmp = [result.hist2D[sep][i] for sep in result.hist2D.keys()]
            #     testdsplits = lognormaldsplitmodel.compute_dsplits_test(density_pdf_2D=tmp, norm=result.norm_all[i], delta=result.pdf1D_x)
            #     testdsplits_list.append(testdsplits)
            # testdsplits = np.mean(testdsplits_list, axis=0)

            # Gaussian model
            ds_delta_tilde = list()
            for i in range(len(result.bins)-1):
                dlt1, dlt2 = max(result.bins[i], -1), result.bins[i+1]
                ds_mask = (result.pdf1D_x >= dlt1) & (result.pdf1D_x < dlt2)
                delta_tilde = np.sum(result.pdf1D_x[ds_mask] * mean_pdf1D[ds_mask])/np.sum(mean_pdf1D[ds_mask])
                ds_delta_tilde.append(delta_tilde)
            gaussiandsplits = np.array(ds_delta_tilde)[:, None] * mean_xiR[None, :] / sigma**2

            # Fitted model
            def fit_model():
                def to_fit(b, ds):
                    model_ds = b * mean_xiR
                    #mask0 = (1 - np.isnan(mean_xiR)) & (1 - np.isnan(mean_ds[ds][0]))
                    mask = sep > 30
                    residuals = (mean_ds[ds][0][mask] - model_ds[mask])/std_ds[ds][0][mask]
                    return np.nansum(residuals**2)
                
                blist = list()
                for ds in range(args.nsplits):
                    mini = minimize(to_fit, 1, args=(ds))
                    print('b{}: '.format(ds+1), mini.x)  
                    blist.append(mini.x)
                bfit = np.array(blist)  
                return bfit

            if args.rsd:
                fitted_model = None
            else:
                bfit = fit_model()
                fitted_model = bfit * mean_xiR[None, :]
                
            
            models_ds = {'ldt': ldtdsplits, 'lognormal': lognormaldsplits, 'gaussian': gaussiandsplits, 'fit': fitted_model}
        
            seps = np.array([float(s) for s in result.hist2D.keys()])

            for m in ['ldt', 'lognormal']:
                plot_name = base_name.format('{}mocks'.format(nmocks)) + ('_rsd' if args.rsd else '') + '_{:d}densitysplits_{}model.pdf'.format(args.nsplits, m)
                models = {m: models_ds[m]}

                #print('sep: ', sep)
                #print('seps: ', seps)

                if m=='test':
                    plot_density_splits(sep, mean_ds, std_ds, std_ds_ref=std_ds_desi, models=models, xmodel=seps, fn=os.path.join(plots_dir, plot_name), rescale_errorbars=np.sqrt(nmocks), **plotting)
                else:
                    plot_density_splits(sep, mean_ds, std_ds, std_ds_ref=std_ds_desi, models=models, fn=os.path.join(plots_dir, plot_name), rescale_errorbars=np.sqrt(nmocks), **plotting)

            
            