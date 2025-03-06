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
from scipy.special import factorial, loggamma
from iminuit import Minuit

from densitysplit import DensitySplit
from densitysplit.corr_func_utils import get_split_poles
from densitysplit import CountInCellsDensitySplitMeasurement
from densitysplit.lognormal_model import LognormalDensityModel, BiasedLognormalDensityModel, LognormalDensitySplitModel
from densitysplit.ldt_model import LDT, LDTDensitySplitModel, setup_logging
from density_split_corr import compute_lognormal_split_bins

from plot import plot_pdf1D, plot_bias_function, plot_density_splits

plt.style.use(os.path.join(os.path.abspath('/feynman/home/dphp/mp270220/densitysplit/nb'), 'densitysplit.mplstyle'))


def compute_bias(x, matter_sample, tracer_sample, save_fn=None):
    tracer_bias = list()
    tracer_shotnoise = list()

    if tracer_sample.ndim <= 1:
        matter_sample = matter_sample[None, :]
        tracer_sample = tracer_sample[None, :]
    n = len(tracer_sample)

    for p in range(n):
        tracer_bias_p = list()
        tracer_shotnoise_p = list()
        for i in range(len(x)-1):
            mask = (matter_sample[p] >= x[i]) & (matter_sample[p] < x[i+1])
            tracer_masked = tracer_sample[p][mask]
            tracer_bias_p.append(np.mean(tracer_masked))
            tracer_shotnoise_p.append(np.var(tracer_masked))
        tracer_bias.append(tracer_bias_p)
        tracer_shotnoise.append(tracer_shotnoise_p)

    tracer_bias = np.array(tracer_bias)
    tracer_shotnoise = np.array(tracer_shotnoise)

    if save_fn is not None:
        print('Saving bias relation to file {}'.format(save_fn))
        tosave = {'delta_matter': x, 'delta_tracer': tracer_bias, 'delta_tracer_scatter': tracer_shotnoise}
        np.save(save_fn, tosave)

    return tracer_bias


def poisson(x, N, norm):
    log_poisson_pdf = N * np.log(norm * x[:, None]) - (norm * x[:, None]) - loggamma(N+1) # log to avoid overflow
    poisson_pdf = np.exp(log_poisson_pdf)
    return poisson_pdf


def compute_joint_pdf(matter_edges, tracer_edges, matter_sample, tracer_sample, save_fn=None):
    if tracer_sample.ndim <= 1:
        matter_sample = matter_sample[None, :]
        tracer_sample = tracer_sample[None, :]
    n = len(tracer_sample)

    hist2D = list()
    hist_matter = list()
    for i in range(n):
        hist2D.append(np.histogram2d(matter_sample[i], tracer_sample[i], bins=(matter_edges, tracer_edges), density=True)[0])
        hist_matter.append(np.histogram(matter_sample[i], bins=matter_edges, density=True)[0])
    
    if save_fn is not None:
        print('Saving 2D PDF to file {}'.format(save_fn))
        tosave = {'edges': (matter_edges, tracer_edges), 'pdf2D': np.array(hist2D), 'pdf_matter': np.array(hist_matter)}
        np.save(save_fn, tosave)

    return np.array(hist2D)


def plot_bias_relation(x, y, err=None, xlim=None, data='average', rescale_errorbars=1, data_style=None, data_label=None, models=None, model_labels=None, model_styles=None, fn=None):

    if xlim is not None:
        mask = (x >= xlim[0]) & (x <= xlim[1])
        x = x[mask]
        y = y[mask]
        if err is not None:
            err = err[mask]
    
    fig, axes = plt.subplots(2, 1, figsize = (3.5, 3.5), sharex=True, sharey='row', gridspec_kw={'height_ratios': [3, 1]})

    axes[0].errorbar(x, y, err/rescale_errorbars, label=data_label, **data_style)

    if models is not None:
        for m in models.keys():
            if xlim is not None:
                models[m] = models[m][mask]
                
            axes[0].plot(x, models[m], label=model_labels[m], **model_styles[m], zorder=10 if data=='average' else 0)
            axes[1].plot(x, (models[m] - y)/err, **model_styles[m])

    axes[1].ticklabel_format(style='sci', scilimits=(-3, 3))
    axes[1].set_ylim((-8, 8)) if data=='average' else axes[1].set_ylim((-3, 3))
    axes[1].set_xlabel(r'$\delta_{R, m}$')
    if data=='average':
        axes[0].set_ylabel(r'$\langle \delta_{R, g} | \delta_{R, m} \rangle$')
        axes[1].set_ylabel(r'$\Delta \langle \delta_{R, g} | \delta_{R, m} \rangle / \sigma$')
    elif data=='scatter':
        #axes[0].set_ylabel(r'$\bar{N_g} \frac{ \langle \delta_{R, g}^2 | \delta_{R, m} \rangle } {1 + \langle \delta_{R, g} | \delta_{R, m} \rangle}$')
        axes[0].set_ylabel(r'$\alpha(\delta_{R, m})$')
        axes[1].set_ylabel(r'$\Delta \alpha(\delta_{R, m}) / \sigma$')
    axes[0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    axes[0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axes[1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    if data=='average':
        axes[0].legend(loc='lower right')
    elif data=='scatter':
        axes[0].legend(loc='upper left')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    fig.align_ylabels()
    plt.savefig(fn, dpi=500)
    plt.close()
    print('Saved tracer bias plot: {}.'.format(fn))


def plot_pdf2D(x, y, mean_hist, cbar_label=None, xlim=None, vmax=None, fn=None):

    plt.figure(figsize=(3, 3))

    cmap = LinearSegmentedColormap.from_list("mycmap", ['white', 'red'])
    if vmax is not None:
        norm = Normalize(vmin=0, vmax=vmax)
    else:
        norm = None
    image = plt.imshow(mean_hist, origin='lower', extent=(np.min(x), np.max(x), np.min(y), np.max(y)), cmap=cmap, norm=norm)

    plt.xlabel(r'$\delta_{R, m}$')
    plt.ylabel(r'$\delta_{R, g}$')
    plt.grid(False)
    plt.xlim(xlim)
    plt.ylim(xlim)
    
    fig = plt.gcf()
    fig.subplots_adjust(left=0.1, right=1, bottom=0.1)
    cax = plt.axes((1., 0.19, 0.03, 0.74))
    cbar = fig.colorbar(image, cax=cax)
    #cax.ticklabel_format(style='sci', scilimits=(-2, 2))
    cbar.set_label(cbar_label, rotation=90)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()
    plt.savefig(fn, bbox_inches='tight', dpi=500)
    plt.close()
    print('Saved 2D PDF map plot: {}.'.format(fn))


if __name__ == '__main__':
    
    setup_logging()
        
    parser = argparse.ArgumentParser(description='galaxy_bias_model')
    parser.add_argument('--imock', type=int, required=False, default=None)
    parser.add_argument('--redshift', type=float, required=False, default=None)
    parser.add_argument('--tracer', type=str, required=False, default='ELG', choices=['ELG', 'LRG'])
    parser.add_argument('--nbar_matter', type=float, required=False, default=0.0371)
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
    ells = [0, 2, 4] if args.rsd else [0]
    nells = len(ells)
    smoothing_radius = args.smoothing_radius[0]

    # Plotting
    data_style = dict(marker="o", ls='', ms=2.6, elinewidth=0.9, markeredgewidth=0.9, color='C0', mfc='white', mec='C0')
    model_styles = {'ldt': dict(ls='-', color='C1'),
                    'ldt_noshotnoise': dict(ls=':', color='C1'),
                    'gaussian': dict(ls='-', color='C1'),
                    'linear': dict(ls=':', color='C3'),
                    'eulerian': dict(ls='--', color='C3'),
                    'poisson': dict(ls=':', color='C1'),
                    'extended': dict(ls='-', color='C1'),
 }        

    data_label = 'AbacusSummit'
    model_labels = {'ldt': 'LDT',
                    'poisson': 'Poisson',
                    'extended': 'fit',
                    'ldt_noshotnoise': 'LDT (no SN)',
                    'gaussian': 'Gaussian bias',
                    'linear': 'linear bias',
                    'eulerian': 'Eulerian  bias'}

    # plot settings
    plotting = {'data_style': data_style, 'data_label': data_label, 'model_labels': model_labels, 'model_styles': model_styles}

    # Directories
    ds_dir = '/feynman/work/dphp/mp270220/outputs/densitysplit/' #if ((args.nsplits == 3) and (args.tracer != 'ELG')) else '/feynman/scratch/dphp/mp270220/outputs/densitysplit/'
    plots_dir = '/feynman/home/dphp/mp270220/plots/densitysplit'
    
    # Filenames
    matter_sim_name = 'AbacusSummit_2Gpc_z{:.3f}_{{}}_downsampled_particles_nbar{:.4f}'.format(z, args.nbar_matter)
    tracer_sim_name = 'AbacusSummit_2Gpc_{}_z{:.3f}_{{}}'.format(args.tracer, z)
     
    # Load density split measurements for matter and galaxies
    matter_base_name = matter_sim_name + '_cellsize{:d}{}_resampler{}{}'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler, '_smoothingR{:d}'.format(smoothing_radius) if smoothing_radius is not None else '')
    matter_ds_name = matter_base_name + '_{:d}splits_randoms_size4_RH_CCF{}'.format(args.nsplits, '_rsd' if args.rsd else '')
    matter_fn = os.path.join(ds_dir, matter_ds_name.format('{}mocks'.format(25 if args.nbar_matter < 0.01 else 8)) + '_compressed.npy')
    print('Loading density split measurements: {}'.format(matter_fn))
    matter_result = CountInCellsDensitySplitMeasurement.load(matter_fn)
    matter_norm = matter_result.norm
    
    tracer_base_name = tracer_sim_name + '_cellsize{:d}{}_resampler{}{}'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler, '_smoothingR{:d}'.format(smoothing_radius) if smoothing_radius is not None else '')
    tracer_ds_name = tracer_base_name + '_{:d}splits_randoms_size4_RH_CCF{}'.format(args.nsplits, '_rsd' if args.rsd else '')
    tracer_fn = os.path.join(ds_dir, tracer_ds_name.format('{}mocks'.format(25)) + '_compressed.npy')
    print('Loading density split measurements: {}'.format(tracer_fn))
    tracer_result = CountInCellsDensitySplitMeasurement.load(tracer_fn)
    tracer_norm = tracer_result.norm

    nmocks = 8#25 if args.nbar_matter < 0.01 else 8
    base_fname = tracer_sim_name + '_cellsize{:d}_resampler{}{}{}'.format(args.cellsize, args.resampler, '_smoothingR{:d}'.format(smoothing_radius) if smoothing_radius is not None else '', '_rsd' if args.rsd else '')
    fname = base_fname.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_bias_relation_matter_nbar1.2384'
    fname2 = base_fname.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_joint_pdf_matter_tracer_matter_nbar1.2384'
    print(os.path.join(ds_dir, fname)+'.npy')
    if False:#os.path.isfile(os.path.join(ds_dir, fname)+'.npy') & os.path.isfile(os.path.join(ds_dir, fname2)+'.npy'):
        bias_relation = np.load(os.path.join(ds_dir, fname)+'.npy', allow_pickle=True).item()['delta_tracer']
        bias_scatter = np.load(os.path.join(ds_dir, fname)+'.npy', allow_pickle=True).item()['delta_tracer_scatter']/(1+bias_relation)*tracer_norm
        pdf2D = np.load(os.path.join(ds_dir, fname2)+'.npy', allow_pickle=True).item()['pdf2D']
        pdf_matter = np.load(os.path.join(ds_dir, fname2)+'.npy', allow_pickle=True).item()['pdf_matter']
    else:
        # Load density vectors
        print('reading...')
        # matter
        #fn = [os.path.join('/feynman/scratch/dphp/mp270220/outputs/densitysplit/', matter_ds_name.format('ph0{:02d}'.format(i)) + '.npy') for i in range(nmocks)]
        #ds = [DensitySplit.load(f) for f in fn]
        #matter_sample = np.array([ds[i].density_mesh.value.flatten() for i in range(nmocks)]) / matter_norm - 1
        fn = [os.path.join('/feynman/work/dphp/mp270220/outputs/density/', 'AbacusSummit_2Gpc_z0.800_ph0{:02d}_particles_nbar1.2384_cellsize20_resamplertophat_smoothingR10_delta_R_jax.npy'.format(i)) for i in range(nmocks)]
        matter_sample = np.array([np.load(f) for f in fn])
        sorted_matter_sample = np.sort(matter_sample.flatten())
        vr = 4/3*np.pi*smoothing_radius**3
        hd_norm = 1/np.min(np.abs(np.diff(np.unique(sorted_matter_sample))))
        hd_nbar = hd_norm/vr
        rebin = 200
        hd_norm = hd_norm/rebin
        print('nbar', hd_nbar)
        nbar_test = 9906847883/2000**3
        print('nbar compare', nbar_test)
        print('are equal', nbar_test==hd_nbar)
        hd_nbar = nbar_test
        kvals = np.arange(0, np.round(10*hd_norm))
        dedges = 1/hd_norm
        edges = kvals/hd_norm - 1 - dedges/2
        x = (edges[1:] + edges[:-1])/2

        # galaxies
        #fn = [os.path.join('/feynman/scratch/dphp/mp270220/outputs/densitysplit/', tracer_ds_name.format('ph0{:02d}'.format(i)) + '.npy') for i in range(nmocks)]
        #ds = [DensitySplit.load(f) for f in fn]
        #tracer_sample = np.array([ds[i].density_mesh.value.flatten() for i in range(nmocks)]) / tracer_norm - 1
        fn = ['/feynman/scratch/dphp/mp270220/outputs/densitysplit/AbacusSummit_2Gpc_ELG_z0.800_ph0{:02d}_cellsize20_cellsize20_resamplertophat_smoothingR10_3splits_randoms_size4_RH_CCF.npy'.format(i) for i in range(nmocks)]
        ds = [DensitySplit.load(f) for f in fn]
        tracer_sample = np.array([ds[i].density_mesh.value.flatten() for i in range(nmocks)]) / tracer_norm - 1

        bias_relation = compute_bias(edges, matter_sample, tracer_sample, save_fn=os.path.join(ds_dir, fname))
        pdf2D = compute_joint_pdf(edges, tracer_result.edges, matter_sample, tracer_sample, save_fn=os.path.join(ds_dir, fname2))
        bias_scatter = np.load(os.path.join(ds_dir, fname)+'.npy', allow_pickle=True).item()['delta_tracer_scatter']/(1+bias_relation)*tracer_norm
        pdf_matter = np.load(os.path.join(ds_dir, fname2)+'.npy', allow_pickle=True).item()['pdf_matter']

    mean_bias_relation = np.mean(bias_relation, axis=0)
    std_bias_relation = np.std(bias_relation, axis=0)
    mean_bias_scatter= np.mean(bias_scatter, axis=0)
    std_bias_scatter = np.std(bias_scatter, axis=0)
   
    if (args.tracer=='ELG') & (z==0.8):
        sigma_noshotnoise = np.load('/feynman/scratch/dphp/mp270220/outputs/ldt_sigma_fit.npy')
    else:
        #to do
        pass

    #xlim = (-0.8, 1)
    xlim = (-0.6, 0.8)
 
    # 1D PDF
    # matter
    #mean_pdf1D = np.mean(matter_result.pdf1D, axis=0)
    matter_pdf1D = np.array([np.histogram(matter_sample[i], bins=edges, density=True)[0] for i in range(4)])
    mean_pdf1D = np.mean(matter_pdf1D, axis=0)
    std_pdf1D = np.std(matter_pdf1D, axis=0)

    # LDT model
    ldtmodel = LDT(redshift=z, smoothing_scale=smoothing_radius, smoothing_kernel=1, nbar=hd_nbar)
    ldtmodel.interpolate_sigma()
    #ldtmodel.compute_ldt(sigma_noshotnoise, k=(1 + matter_result.pdf1D_x)*matter_result.norm)
    ldtmodel.compute_ldt(sigma_noshotnoise)
    ldtpdf1D = ldtmodel.density_pdf(rho=1 + x) 
    ldtpdf1D_noshotnoise = ldtmodel.density_pdf_noshotnoise(rho=1 + x) 
    
    models = {'ldt_noshotnoise': ldtpdf1D_noshotnoise, 'ldt': ldtpdf1D}
    density_name = matter_sim_name + '_cellsize{:d}_resampler{}{}{}{}'.format(args.cellsize, args.resampler, '_smoothingR{:d}'.format(smoothing_radius) if smoothing_radius is not None else '', '_rescaledvar{}'.format(args.rescale_var) if args.rescale_var!=1 else '', '_rsd' if args.rsd else '')
    plot_name = density_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_1DPDF.pdf'
    #plot_pdf1D(matter_result.pdf1D_x, mean_pdf1D, std_pdf1D, xlim=(-1, 3), models=models, fn=os.path.join(plots_dir, plot_name), **plotting)
    plot_pdf1D(x, mean_pdf1D, std_pdf1D, xlim=(-1, 3), models=models, fn=os.path.join(plots_dir, plot_name), **plotting)

    # Plot P(delta_m, delta_t)
    mean_pdf2D = np.mean(pdf2D, axis=0)
    vmax = 1.5#max(np.max(mean_pdf2D), np.max(toplot))
    print('vmax:', vmax)

    plot_name = density_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_{}_condPDF.pdf'.format(args.tracer)
    plot_pdf2D(edges, tracer_result.edges, mean_pdf2D/np.mean(pdf_matter, axis=0)[:, None], cbar_label=r'$\mathcal{P}(\delta_{R, m} | \delta_{R, g})$', xlim=(-1, 3), vmax=vmax, fn=os.path.join(plots_dir, plot_name))
        
    # tracer
    # LDT model
    ldtmodel = LDT(redshift=z, smoothing_scale=smoothing_radius, smoothing_kernel=1, nbar=tracer_result.nbar)
    ldtmodel.interpolate_sigma()

    mean_pdf1D = np.mean(tracer_result.pdf1D, axis=0)
    print('test norm pdf tracer:', np.sum(tracer_result.pdf1D_x * mean_pdf1D)/tracer_norm)
    std_pdf1D = np.std(tracer_result.pdf1D, axis=0)

    # b = ldtmodel.fit_from_pdf(tracer_result.pdf1D_x, mean_pdf1D, err=std_pdf1D, xlim=xlim, fix_sigma=True, sigma_init=sigma_noshotnoise, bias='linear', norm=tracer_result.norm, matter_norm=matter_result.norm, super_poisson=False)
    # b1, b2 = ldtmodel.fit_from_pdf(tracer_result.pdf1D_x, mean_pdf1D, err=std_pdf1D, fix_sigma=True, sigma_init=sigma_noshotnoise, bias='gaussian', norm=tracer_result.norm, matter_norm=matter_result.norm, super_poisson=False)
    # b1E, b2E = ldtmodel.fit_from_pdf(tracer_result.pdf1D_x, mean_pdf1D, err=std_pdf1D, fix_sigma=True, sigma_init=sigma_noshotnoise, bias='eulerian', norm=tracer_result.norm, matter_norm=matter_result.norm, super_poisson=False)
    # print('linear bias:', b)
    # print('b1, b2:', b1, b2)
    # print('b1E, b2E:', b1E, b2E)

    print('Using pre-computed value of sigma_m (no shot noise):', sigma_noshotnoise)
    ldtmodel.compute_ldt(sigma_noshotnoise, k=(1 + tracer_result.pdf1D_x)*tracer_result.norm)
    
    def fit_bias(delta_m, delta_t, err=1, model='linear'):
        if np.all(np.isnan(err)) or np.all(err==0):
            err = 1
        def to_min(b):
            if model=='linear':
                y = b[0] * delta_m
            else:
                y = ldtmodel.delta_t_expect(rho=1+delta_m, bG1=b[0], bG2=b[1], model=model)
            toret = np.sum((y - delta_t)**2/err**2)
            return toret
        if model=='linear':
            p0 = np.array([1])
            bounds = np.array([(0, 2)])
        else:
            p0 = np.array([1, 1])
            bounds = np.array([(0, 2), (-2, 2)])
        mini = minimize(to_min, x0=p0, bounds=bounds)
        print(mini)
        return mini.x

    mask = (x >= xlim[0]) & (x <= xlim[1])
    b = fit_bias(x[mask], mean_bias_relation[mask], std_bias_relation[mask], model='linear')
    b1, b2 = fit_bias(x[mask], mean_bias_relation[mask], std_bias_relation[mask], model='gaussian')
    b1E, b2E = fit_bias(x[mask], mean_bias_relation[mask], std_bias_relation[mask], model='eulerian')
    print('linear bias:', b)
    print('b1, b2:', b1, b2)
    print('b1E, b2E:', b1E, b2E)

    ldtmodel.compute_ldt(sigma_noshotnoise, k=(1 + tracer_result.pdf1D_x)*tracer_result.norm)
    ldtbiasmodel = ldtmodel.delta_t_expect(rho=1+x, bG1=b1, bG2=b2)
    renorm = 1 + np.sum(ldtbiasmodel*ldtmodel.density_pdf_noshotnoise(rho=1+x))/hd_norm
    print('renorm', renorm)
    #ldtbiasmodel = (1 + ldtbiasmodel)/renorm - 1
    ldteulerianbiasmodel = ldtmodel.delta_t_expect(rho=1+x, bG1=b1E, bG2=b2E, model='eulerian')
    linearbiasmodel = b * x
    models = {'gaussian': ldtbiasmodel, 'eulerian': ldteulerianbiasmodel}

    plot_fname = base_fname.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_bias_relation.pdf'
    plot_bias_relation(x, mean_bias_relation, err=std_bias_relation, rescale_errorbars=np.sqrt(nmocks), xlim=xlim, models=models, fn=os.path.join(plots_dir, plot_fname), **plotting)

    def fit_shotnoise(delta_m, scatter, err):
        if np.all(np.isnan(err)) or np.all(err==0):
            err = 1
        def to_min(a):
            y = ldtmodel.alpha(rho=1+delta_m, alpha0=a[0], alpha1=a[1], alpha2=a[2])
            toret = np.nansum((y - scatter)**2/err**2)
            return toret
        p0 = np.array([1, 0, 0])
        mini = minimize(to_min, x0=p0)
        print(mini)
        return mini.x

    a0, a1, a2 = fit_shotnoise(x[mask], mean_bias_scatter[mask], std_bias_scatter[mask])
    shotnoise_params = {'alpha0': a0, 'alpha1': a1, 'alpha2': a2}

    shotnoise_model = ldtmodel.alpha(rho=1+x, alpha0=a0, alpha1=a1, alpha2=a2)
    models = {'extended': shotnoise_model}

    plot_fname = base_fname.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_bias_relation_scatter.pdf'
    plot_bias_relation(x, mean_bias_scatter, err=std_bias_scatter, data='scatter', rescale_errorbars=np.sqrt(nmocks), xlim=xlim, models=models, fn=os.path.join(plots_dir, plot_fname), **plotting)

    ldtpdf1D = ldtmodel.tracer_density_pdf(bG1=b1, bG2=b2, **shotnoise_params, matter_norm=hd_norm) 
    ldtpdf1D_eulerianbias = ldtmodel.tracer_density_pdf(bG1=b1E, bG2=b2E, **shotnoise_params, model='eulerian', matter_norm=hd_norm)
    ldtpdf1D_linearbias = ldtmodel.density_pdf(b1=b) 
    models_pdf1D = {'gaussian': ldtpdf1D, 'eulerian': ldtpdf1D_eulerianbias}

    # Plot 1D PDF
    density_name = tracer_sim_name + '_cellsize{:d}_resampler{}{}{}{}'.format(args.cellsize, args.resampler, '_smoothingR{:d}'.format(smoothing_radius) if smoothing_radius is not None else '', '_rescaledvar{}'.format(args.rescale_var) if args.rescale_var!=1 else '', '_rsd' if args.rsd else '')
    plot_name = density_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_1DPDF.pdf'
    
    plot_pdf1D(tracer_result.pdf1D_x, mean_pdf1D, std_pdf1D, xlim=(-1, 4), models=models_pdf1D, rtitle=args.nbar_matter<0.001, fn=os.path.join(plots_dir, plot_name), galaxies=True, **plotting)

    # Bias function
    for sep in tracer_result.bias_function.keys():
        if float(sep) in [5, 10, 20, 40, 70, 110]:
            # Mocks
            mean_bias = np.mean(tracer_result.bias_function[sep], axis=0)
            std_bias = np.std(tracer_result.bias_function[sep], axis=0)

            xiR_matter = np.mean(matter_result.bias_corr[sep])
            xiR_tracer = np.mean(tracer_result.bias_corr[sep])
            print('xi ratio:', xiR_matter/xiR_tracer)
       
            # LDT model
            ldtbiasmodel = ldtmodel.tracer_bias(rho=1+tracer_result.bias_function_x[sep], bG1=b1, bG2=b2, **shotnoise_params, model='gaussian')*np.sqrt(xiR_matter/xiR_tracer)
            ldtbiasmodel_eulerianbias = ldtmodel.tracer_bias(rho=1+tracer_result.bias_function_x[sep], bG1=b1E, bG2=b2E, **shotnoise_params, model='eulerian')*np.sqrt(xiR_matter/xiR_tracer)
            models_bias = {'gaussian': ldtbiasmodel, 'eulerian': ldtbiasmodel_eulerianbias}
    
            # Plot bias function
            plot_name = tracer_base_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + ('_rsd' if args.rsd else '') + '_s{:.0f}_biasfunction.pdf'.format(float(sep))
            plot_bias_function(tracer_result.bias_function_x[sep], mean_bias, std_bias, xlim=(-1, 4), sep=float(sep), models=models_bias, fn=os.path.join(plots_dir, plot_name), rescale_errorbars=np.sqrt(nmocks), galaxies=True, **plotting)

    # Density splits
    try:
        matter_xiR = np.mean([res(ells=ells, ignore_nan=True) for res in matter_result.smoothed_corr], axis=0)
    except:
        matter_xiR = np.mean(matter_result.smoothed_corr, axis=0)
    tracer_xiR = np.mean([res(ignore_nan=True) for res in tracer_result.smoothed_corr], axis=0)
    mean_ds, cov = get_split_poles(tracer_result.ds_corr, ells=None if (not args.rsd) else ells)
    std_ds = np.nan if cov.size == 1 else np.array_split(np.array(np.array_split(np.diag(cov)**0.5, nells)), args.nsplits, axis=1)
    sep = tracer_result.sep

    std_ds_desi = None

    # LDT model
    ldtdsplitmodel = LDTDensitySplitModel(ldtmodel, density_bins=tracer_result.bins)
    ldtdsplits_gaussian = ldtdsplitmodel.compute_dsplits(matter_xiR, bias_model='gaussian', bG1=b1, bG2=b2, **shotnoise_params)
    ldtdsplits_eulerian = ldtdsplitmodel.compute_dsplits(matter_xiR, bias_model='eulerian', bG1=b1E, bG2=b2E, **shotnoise_params)
    
    models_ds = {'gaussian': ldtdsplits_gaussian, 'eulerian': ldtdsplits_eulerian}

    seps = np.array([float(s) for s in tracer_result.hist2D.keys()])

    for m in ['gaussian', 'eulerian']:
        plot_name = tracer_base_name.format('{}mocks'.format(nmocks)) + ('_rsd' if args.rsd else '') + '_{:d}densitysplits_LDT_{}biasmodel.pdf'.format(args.nsplits, m)
        models = {m: models_ds[m]}

        plot_density_splits(sep, mean_ds, std_ds, std_ds_ref=std_ds_desi, models=models, fn=os.path.join(plots_dir, plot_name), rescale_errorbars=np.sqrt(nmocks), **plotting)

    
    