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
    parser.add_argument('--smoothing_radius', type=int, required=False, default=None)
    parser.add_argument('--use_weights', type=bool, required=False, default=False)
    parser.add_argument('--rsd', type=bool, required=False, default=False)
    parser.add_argument('--los', type=str, required=False, default='x')
    parser.add_argument('--nsplits', type=int, required=False, default=3)
    parser.add_argument('--randoms_size', type=int, required=False, default=4)
    parser.add_argument('--size', type=int, required=False, default=None)
    parser.add_argument('--sep', type=float, required=False, default=None)
    parser.add_argument('--show_lognormal', type=bool, required=False, default=True)
    parser.add_argument('--show_ldt', type=bool, required=False, default=True)
    
    args = parser.parse_args()
    z = args.redshift
    ells = [0, 2, 4] if args.rsd else [0]
    nells = len(ells)

    # Directories
    data_dir = '/feynman/scratch/dphp/mp270220/abacus/'
    ds_dir = '/feynman/work/dphp/mp270220/outputs/densitysplit/'
    mesh_dir = '/feynman/scratch/dphp/mp270220/outputs'
    plots_dir = '/feynman/home/dphp/mp270220/plots/densitysplit'
    
    # Filenames
    if args.tracer == 'halos':
        sim_name = 'AbacusSummit_2Gpc_z{:.3f}_{{}}'.format(z)
    elif args.tracer == 'particles':
        sim_name = 'AbacusSummit_2Gpc_z{:.3f}_{{}}_downsampled_particles_nbar{:.4f}'.format(z, args.nbar)
    base_name = sim_name + '_cellsize{:d}{}_resampler{}{}'.format(args.cellsize, '_cellsize{:d}'.format(args.cellsize2) if args.cellsize2 is not None else '', args.resampler, '_smoothingR{:d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '')
    ds_name = base_name + '_3splits_randoms_size4_RH_CCF{}'.format('_RSD' if args.rsd else '')

    # Load measured density splits (1 or all mocks)
    if args.imock is not None:
        f = os.path.join(ds_dir, ds_name.format('ph0{:02d}'.format(args.imock)))
        if os.path.isfile(f+'.npy'):
            # if DensitySplit file already exists
            if args.resampler == 'ngp':
                 # density mesh is stored in scratch because it's very large
                mesh_filename = os.path.join(mesh_dir, base_name.format('ph0{:02d}'.format(args.imock)) + '_density_mesh{}.npy'.format('_RSD' if args.rsd else ''))
            else:
                mesh_filename = None
            mock_density = DensitySplit.load(f+'.npy', mesh_filename=mesh_filename)
        else:
            # if not, create DensitySplit instance
            mock = Data.load(os.path.join(data_dir, sim_name.format('ph0{:02d}'.format(args.imock))+'.npy'))
            mock_density = DensitySplit(mock)
            mock_density.compute_density(data=mock, cellsize=args.cellsize, resampler=args.resampler, smoothing_radius=args.smoothing_radius, cellsize2=args.cellsize2)
            #mock_density.save(f)
    else:
        nmocks = 20
        mock_density = list()
        for i in range(nmocks):
            f = os.path.join(ds_dir, ds_name.format('ph0{:02d}'.format(i)))
            if os.path.isfile(f+'.npy'):
                print("Loading DensitySplit for mock {}.".format(i))
                if args.resampler == 'ngp':
                    # density mesh is stored in scratch because it's very large
                    mesh_filename = os.path.join(mesh_dir, base_name.format('ph0{:02d}'.format(i)) + '_density_mesh{}.npy'.format('_RSD' if args.rsd else ''))
                else:
                    mesh_filename = None
                mocki_density = DensitySplit.load(f+'.npy', mesh_filename=mesh_filename)
            else:
                print("DensitySplit for mock {} does not exist, loading mock {}.".format(i, i))
                mocki = Data.load(os.path.join(data_dir, sim_name.format('ph0{:02d}'.format(i))+'.npy'))
                mocki_density = DensitySplit(mocki)
                mocki_density.compute_density(data=mocki, cellsize=args.cellsize, resampler=args.resampler, smoothing_radius=args.smoothing_radius, cellsize2=args.cellsize2)
                #mocki_density.save(f)
            mock_density.append(mocki_density)

    # Get density samples
    # For tophat smoothing, we actually have number counts of particles in spheres: need to normalize and retrieve 1 to get density contrast
    if args.imock is not None:
        nbar = mock_density.size/mock_density.boxsize**3
    else:
        nbar = np.mean([mock_density[i].size/mock_density[i].boxsize**3 for i in range(nmocks)])
    print(nbar)
    if args.smoothing_radius is not None:
        v = 4/3 * np.pi * args.smoothing_radius**3
        norm = nbar*v

    if args.imock is not None:
        print('Get density for mock {}.'.format(args.imock))
        deltaR1, deltaR2 = mock_density.compute_jointpdf_delta_R1_R2(s=args.sep, sbin=(args.sep-0.5, args.sep+0.5), query_positions='mesh', sample_size=args.size)
    else:
        print('Concatenate densities from all mocks')
        deltaR = [mock_density[i].compute_jointpdf_delta_R1_R2(s=args.sep, sbin=(args.sep-0.5, args.sep+0.5), query_positions='mesh', sample_size=args.size) for i in range(nmocks)]
        deltaR1 = np.array([deltaR[i][0] for i in range(nmocks)])
        deltaR2 = np.array([deltaR[i][1] for i in range(nmocks)])  
        
    if args.resampler == 'tophat':
        N1, N2 = deltaR1, deltaR2
        k1vals = np.arange(0, np.max(N1))
        k2vals = np.arange(0, np.max(N2))
        deltaR1, deltaR2 = N1/norm - 1, N2/norm - 1 # density constrast from number counts
        dedges = 1/norm
        edges = k1vals/norm - 1 - dedges/2 # bin centers
    else:
        edges, step = np.arange(-1, 10, 0.01), 0.01
    deltavals = (edges[1:] + edges[:-1])/2

    # Compute 1D PDF measured from mocks
    if args.imock is not None:
        pdf1D = np.histogram(deltaR1, bins=edges, density=True)[0]
        mean_pdf1D = pdf1D
        std_pdf1D = np.nan
    else:  
        pdf1D = np.array([np.histogram(deltaR1[i], bins=edges, density=True)[0] for i in range(nmocks)])
        mean_pdf1D = np.mean(pdf1D, axis=0)
        std_pdf1D = np.std(pdf1D, axis=0)

    if args.imock is None:
        if args.resampler == 'tophat':
            N10, N20 = N1[0], N2[0]
        deltaR10, deltaR20 = deltaR1[0], deltaR2[0]
    else:
        if args.resampler == 'tophat':
            N10, N20 = N1, N2
        deltaR10, deltaR20 = deltaR1, deltaR2      

    # Lognormal model
    lognormalmodel = LognormalDensityModel()
    sigma1, delta01 = lognormalmodel.get_params_from_moments(sample=deltaR1.flatten())
    sigma2, delta02 = lognormalmodel.get_params_from_moments(sample=deltaR2.flatten())
    lognormalpdf1D = lognormalmodel.density(deltavals)

    # LDT model
    smoothing_radius = args.smoothing_radius if args.smoothing_radius is not None else args.cellsize
    ldtmodel = LDT(redshift=z, smoothing_scale=smoothing_radius, smoothing_kernel=1, nbar=nbar)
    sigma = np.std(deltaR1.flatten())
    sigma_noshotnoise = np.sqrt(sigma**2 - 1 / (nbar * 4/3 * np.pi * smoothing_radius**3))
    ldtmodel.interpolate_sigma()
    if args.resampler=='tophat':
        ldtmodel.compute_ldt(sigma_noshotnoise, k=(1 + deltavals)*norm)
        ldtpdf1D = ldtmodel.density_pdf()*norm
    else:
        ldtmodel.compute_ldt(sigma_noshotnoise)
        ldtpdf1D = ldtmodel.density_pdf(1+deltavals)

    # Plot 1D PDF
    if True:
        density_name = sim_name + '_cellsize{:d}_resampler{}{}'.format(args.cellsize, args.resampler, '_smoothingR{:d}'.format(args.smoothing_radius) if args.smoothing_radius is not None else '')
        mask = deltavals < 3
        std_pdf1D = std_pdf1D[mask] if args.imock is None else std_pdf1D
        plot_name = density_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_1DPDF.pdf'
        fig, axes = plt.subplots(2, 1, figsize = (4, 4), sharex=True, sharey='row', gridspec_kw={'height_ratios': [3, 1]})
        axes[0].errorbar(deltavals[mask], mean_pdf1D[mask], std_pdf1D, marker='.', ls='', label='AbacusSummit')
        if args.show_lognormal:
            axes[0].plot(deltavals[mask], lognormalpdf1D[mask], color='C1', alpha=0.5, label='lognormal', zorder=9)
        if args.show_ldt:
            axes[0].plot(deltavals[mask], ldtpdf1D[mask], color='C1', label='LDT', zorder=10)
            #ldtpdf1D_noshotnoise = ldtmodel.density_pdf_noshotnoise(1 + deltavals[mask][1:])
            #axes[0].plot(deltavals[mask][1:], ldtpdf1D_noshotnoise, color='C1', label='LDT (no SN)', zorder=11, ls='--')
        if args.imock is not None:
            if args.show_lognormal:
                axes[1].plot(deltavals[mask], (lognormalpdf1D[mask]-mean_pdf1D[mask]), alpha=0.5, color='C1')
            if args.show_ldt:
                axes[1].plot(deltavals[mask], (ldtpdf1D[mask]-mean_pdf1D[mask]), color='C1')
            axes[1].set_ylabel(r'$\Delta \mathcal{P}(\delta_R)$')
        else:
            if args.show_lognormal:
                axes[1].plot(deltavals[mask], (lognormalpdf1D[mask]-mean_pdf1D[mask])/std_pdf1D, alpha=0.5, color='C1')
            if args.show_ldt:
                axes[1].plot(deltavals[mask], (ldtpdf1D[mask]-mean_pdf1D[mask])/std_pdf1D, color='C1')
            axes[1].set_ylabel(r'$\Delta \mathcal{P}(\delta_R) / \sigma$')            
        axes[1].set_xlabel(r'$\delta_R$')
        axes[0].set_ylabel(r'$\mathcal{P}(\delta_R)$')
        #axes[0].set_xlim(-1, 3)
        axes[0].set_title(r'$R = {} \; \mathrm{{Mpc}}/h$'.format(smoothing_radius))
        axes[0].legend()
        fig.align_ylabels()
        plt.savefig(os.path.join(plots_dir, plot_name), dpi=500)
        plt.close()
        print('Saved 1D PDF plot: {}.'.format(os.path.join(plots_dir, plot_name)))

    # Plot bias function
    if True:
        edges = edges[edges < 4]
        bias_func = list()
        for i in range(len(edges)-1):
            mask = (deltaR1 >= edges[i]) & (deltaR1 < edges[i+1])
            bias = np.nanmean(np.where(mask, deltaR2, np.nan), axis=-1)
            bias_func.append(bias)
        corr = np.mean(deltaR1*deltaR2, axis=-1)
        print(corr)
        #corr = np.mean(deltaR1*deltaR2, axis=-1) - np.mean(deltaR1, axis=-1)*np.mean(deltaR2, axis=-1)
        #print(corr)
        bias_func = np.array(bias_func) / corr
        #bias_func = np.array(bias2) / corr
        if args.imock is None:
            mean_bias_func = np.mean(bias_func, axis=1)
            std_bias_func = np.std(bias_func, axis=1)

        if args.show_ldt:
            plot_name = base_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_s{:.0f}_biasfunction.pdf'.format(args.sep)
            fig, axes = plt.subplots(2, 1, figsize = (4, 4), sharex=True, sharey='row', gridspec_kw={'height_ratios': [3, 1]})
            x = (edges[1:]+edges[:-1])/2
            #x = deltavals[1:]
            #bias_func = bias_func[1:]
    
            # LDT model
            ldtbiasmodel = [ldtmodel.bias(1+xi) for xi in x]
            
            if args.imock is not None:
                axes[0].plot(x, bias_func, color='C0', label='AbacusSummit')
                axes[1].plot(x, ldtbiasmodel-bias_func, color='C1')
                axes[1].set_ylabel(r'$\Delta b(\delta_R)$')
            else:
                axes[0].plot(x, mean_bias_func, color='C0', label='AbacusSummit')
                axes[0].fill_between(x, mean_bias_func-std_bias_func, mean_bias_func+std_bias_func, facecolor='C0', alpha=0.3)
                axes[1].plot(x, (ldtbiasmodel-mean_bias_func)/std_bias_func, color='C1')
                axes[1].set_ylabel(r'$\Delta b(\delta_R) / \sigma$')
            axes[0].plot(x, ldtbiasmodel, color='C1', label='LDT')
            axes[0].legend()
            axes[1].set_xlabel(r'$\delta_R$')
            axes[0].set_ylabel(r'$b(\delta_R)$')
            axes[0].set_title(r'$R = {} \; \mathrm{{Mpc}}/h$'.format(smoothing_radius))
            fig.align_ylabels()
            plt.savefig(os.path.join(plots_dir, plot_name), dpi=500)
            plt.close()
            print('Plot saved at {}'.format(os.path.join(plots_dir, plot_name)))

    if True:
        # Plot 2D PDF (onyl 1st mock)
        plot_name = base_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_s{:.0f}_2DPDF_contours.pdf'.format(args.sep)
        if args.resampler == 'tophat':
            d1edges = np.linspace(2, np.max(N10)+1, np.max(N10).astype('i4'))
            print(d1edges)
            d1edges = d1edges/norm - 1 - dedges/2
            d2edges = np.linspace(2, np.max(N20)+1, np.max(N20).astype('i4'))/norm - 1 - dedges/2
            d1edges = d1edges[d1edges < 4]
            d2edges = d2edges[d2edges < 4]
        else:
            d1edges, d2edges = (np.arange(-1, 5, 0.03), np.arange(-1, 5, 0.03))
        d1, d2 = np.meshgrid((d1edges[1:] + d1edges[:-1])/2, (d2edges[1:] + d2edges[:-1])/2)
        pos = np.dstack(np.meshgrid((d1edges[1:]+d1edges[:-1])/2, (d2edges[1:]+d2edges[:-1])/2))
        hist2D, d1edges, d2edges = np.histogram2d(deltaR10, deltaR20, bins=(d1edges, d2edges), density=True)

        bins = mock_density.split_bins if args.imock is not None else mock_density[0].split_bins
        print('Density split bins: {}'.format(bins))

        # Lognormal model
        lognormaldsplitmodel = LognormalDensitySplitModel(density_bins=bins)
        lognormaldsplitmodel.get_params_from_moments(sample=deltaR1.flatten())
        X, Y = np.log(1 + deltaR1/delta01) + sigma1**2/2, np.log(1 + deltaR2/delta02) + sigma2**2/2
        cov = np.cov(np.array([X.flatten(), Y.flatten()]))
        lognormalpdf2D = lognormaldsplitmodel.density2D(pos, sigma=sigma1, delta0=delta01, delta02=delta02, sigma2=sigma2, cov=cov)
            
        # LDT model
        ldtdsplitmodel = LDTDensitySplitModel(ldtmodel, density_bins=bins)
        ldtmodelpdf2D = ldtdsplitmodel.joint_density_pdf(1 + pos[..., 0], 1 + pos[..., 1], corr)

        # Lognormal space
        levels = [0.011, 0.135, 0.607]
        plt.figure(figsize=(4, 4))
        plt.plot([], [], label=r'$s = {:.0f} \; \mathrm{{Mpc}}/h$'.format(args.sep), alpha=0) # for legend
        plt.contour(d1, d2, hist2D.T, levels=levels, colors='C0')
        if args.show_lognormal:
            plt.plot([], [], label='lognormal', color='C1', alpha=0.5) # for legend
            plt.contour(d1, d2, lognormalpdf2D, levels=levels, colors='C1', alpha=0.5)
        if args.show_ldt:
            plt.plot([], [], label='LDT', color='C1', alpha=1) # for legend
            plt.contour(d1, d2, ldtmodelpdf2D, levels=levels, colors='C1', alpha=1)
        plt.xlabel(r'$\delta_{R}(r)$')
        plt.ylabel(r'$\delta_{R}(r + s)$')
        plt.xlim(-1.05, 2.2)
        plt.ylim(-1.05, 2.2)
        plt.legend()
        plt.savefig(os.path.join(plots_dir, plot_name), dpi=500)
        plt.close()
        print('Saved 2D PDF contour plot: {}.'.format(os.path.join(plots_dir, plot_name)))

        # Plot 2D histogram
        data_to_plot = {'mock': hist2D.T, 'lognormal': lognormalpdf2D, 'ldt': ldtmodelpdf2D}
        vmax = max(np.max(np.abs(lognormalpdf2D - hist2D.T)), np.max(np.abs(ldtmodelpdf2D - hist2D.T)))
        from matplotlib import colors
        norm = colors.Normalize(vmin=0, vmax=vmax)
        for to_plot in ['mock', 'lognormal', 'ldt']:
            plot_name = base_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_s{:.0f}_2DPDF_{}_histogram.pdf'.format(args.sep, to_plot)
            plt.figure(figsize=(4, 4))
            if to_plot=='mock':
                image = plt.imshow(data_to_plot[to_plot], origin='lower', extent=(np.min(d1edges), np.max(d1edges), np.min(d2edges), np.max(d2edges)))
            else:
                histdiff = np.abs(data_to_plot[to_plot] - hist2D.T)
                image = plt.imshow(histdiff, origin='lower', extent=(np.min(d1edges), np.max(d1edges), np.min(d2edges), np.max(d2edges)), norm=norm)
            plt.plot([], [], label=r'$s = {:.0f} \; \mathrm{{Mpc}}/h$'.format(args.sep), alpha=0) # for legend
            plt.xlabel(r'$\delta_{R}(r)$')
            plt.ylabel(r'$\delta_{R}(r + s)$')
            plt.xlim(np.min(d1edges), 2.2)
            plt.ylim(np.min(d2edges), 2.2)
            plt.grid(False)
            plt.legend(labelcolor='white')
            fig = plt.gcf()
            fig.subplots_adjust(left=0.1, right=1, bottom=0.1)
            cax = plt.axes((1., 0.15, 0.03, 0.8))
            cbar = fig.colorbar(image, cax=cax)
            cbar.set_label(r'$\mathcal{P}$' if to_plot=='mock' else r'$\Delta \mathcal{P}$', rotation=90)
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, plot_name), bbox_inches='tight', dpi=500)
            plt.close()
            print('Saved 2D PDF map plot: {}.'.format(os.path.join(plots_dir, plot_name)))

        # Change of variable -> Gaussian space
        x = np.log(1 + d1/delta01) + sigma1**2/2
        y = np.log(1 + d2/delta02) + sigma2**2/2
        
        plot_name = base_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_s{:.0f}_2DPDF_logtransform_contours.pdf'.format(args.sep)
        plt.figure(figsize=(4, 4))
        plt.plot([], [], label=r'$s = {:.0f} \; \mathrm{{Mpc}}/h$'.format(args.sep), alpha=0) # for legend
        plt.contour(x, y, hist2D*(delta01+d1)*(delta02+d2), levels=levels, colors='C0')
        if args.show_lognormal:
            plt.plot([], [], label='lognormal', color='C1', alpha=0.5) # for legend
            plt.contour(x, y, lognormalpdf2D*(delta01+d1)*(delta02+d2), levels=levels, colors='C1', alpha=0.5)
        if args.show_ldt:
            plt.plot([], [], label='LDT', color='C1', alpha=1) # for legend
            plt.contour(x, y, ldtmodelpdf2D*(delta01+d1)*(delta02+d2), levels=levels, colors='C1')
        ax = plt.gca()
        plt.xlabel(r'$\ln [1 + \delta_{R}(r)/\delta_{0}]$')
        plt.ylabel(r'$\ln [1 + \delta_{R}(r + s)/\delta_{0}]$')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.legend()
        plt.savefig(os.path.join(plots_dir, plot_name), dpi=500)
        print('Saved 2D PDF contour plot (lognormal transform): {}.'.format(os.path.join(plots_dir, plot_name)))

    if True:
        # Plot density splits
        sep = np.linspace(0, 150, 51)
        if args.imock is not None:
            ds_corr = [mock_density.ds_data_corr]
            if args.resampler=='tophat':
                xiR = mock_density.smoothed_corr(sep)
                ds_poles = [np.ravel(res.get_corr(return_sep=False)) for res in ds_corr[0]]
                std = np.nan
                s, _ = ds_corr[0][0].get_corr(return_sep=True)
            else:
                xiR = np.mean(mock_density.smoothed_corr(sep), axis=1)
                ds_poles, cov = get_split_poles(ds_corr, ells=ells, nsplits=args.nsplits)
                std = np.nan if cov.size == 1 else np.array_split(np.array(np.array_split(np.diag(cov)**0.5, nells)), args.nsplits, axis=1)
                s, _, _ = ds_corr[0][0].get_corr(return_sep=True)
        else:
            ds_corr = np.array([mock_density[i].ds_data_corr for i in range(nmocks)])
            if args.resampler=='tophat':
                xiR = np.mean(np.array([np.mean(mock_density[i].smoothed_corr(sep), axis=1) for i in range(nmocks)]), axis=0)
            else:
                xiR = np.mean(np.array([mock_density[i].smoothed_corr(sep) for i in range(nmocks)]), axis=0)
        #ds_poles, cov = get_split_poles(ds_corr, ells=ells, nsplits=args.nsplits)
        #std = np.nan if cov.size == 1 else np.array_split(np.array(np.array_split(np.diag(cov)**0.5, nells)), args.nsplits, axis=1)

        # Gaussian model
        print('Computing Gaussian density split model.')
        deltaRds = list()
        for i in range(len(bins)-1):
            del1 = max(bins[i], -1)
            del2 = bins[i+1]
            ds_mask = (deltaR1 >= del1) & (deltaR1 < del2)
            deltaRds.append(np.mean(deltaR1[ds_mask]))
        gaussiandsplits = np.array(deltaRds)[:, None] * xiR[None, :] / np.var(deltaRds)

        # Lognormal model
        print('Computing lognormal density split model.')
        lognormaldsplits = lognormaldsplitmodel.compute_dsplits(delta02=delta02 if args.cellsize2 != args.cellsize else delta01,
                                                          smoothing=2 if args.cellsize2 is not None else 1,
                                                          sep=sep, xiR=xiR,
                                                          rsd=False, ells=ells)

        # LDT model
        #ldtdsplits = ldtdsplitmodel.compute_dsplits_test(density_pdf=mean_pdf1D, joint_density_pdf=hist2D.T, x1=d1, x2=d2)
        ldtdsplits = ldtdsplitmodel.compute_dsplits(xiR)
        
        for model in ['lognormal', 'ldt']:
            plot_name = base_name.format('ph0{:02d}'.format(args.imock) if args.imock is not None else '{}mocks'.format(nmocks)) + '_densitysplits_{}model.pdf'.format(model)
            figsize = (8, 4) if args.rsd else (4, 4)
            fig, axes = plt.subplots(2, nells, figsize=figsize, sharex=True, sharey='row', gridspec_kw={'height_ratios': [3, 1]})
            colors = ['firebrick', 'violet', 'olivedrab']

            for ill, ell in enumerate(ells):
                ax0 = axes[0][ill] if nells > 1 else axes[0]
                ax1 = axes[1][ill] if nells > 1 else axes[1]
        
                for ds in range(args.nsplits):
                    if args.resampler=='tophat':
                        ax0.plot(s, s**2 * ds_poles[ds], color=colors[ds], label=r'DS{} $\times$ all'.format(ds))
                    else:
                        ax0.plot(s, s**2 * ds_poles[ds][ill], color=colors[ds], label=r'DS{} $\times$ all'.format(ds))                        
                    #ax0.plot(sep, sep**2 * gaussiandsplits[ds], color=colors[ds], ls='-', alpha=0.4)
                    if args.imock is None:
                        ax0.fill_between(s, s**2 * (ds_poles[ds][ill] - std[ds][ill]), s**2 * (ds_poles[ds][ill] + std[ds][ill]), facecolor=colors[ds], alpha=0.3)
                    if model == 'lognormal':
                        ax0.plot(sep, sep**2 * lognormaldsplits[ds], color=colors[ds], ls=':')
                    if model == 'ldt':
                        ax0.plot(sep, sep**2 * ldtdsplits[ds], color=colors[ds], ls=':')
                        #ax0.scatter(args.sep, args.sep**2 * ldtdsplits[ds], color=colors[ds])
                        
                    #split_xi_interp = np.interp(sep, s, split_xi[ds][ill])
                    #std_interp = np.interp(sep, s, std[ds][ill])
                    #ax1.plot(sep, (ldtcorr[ds] - split_xi_interp)/1, ls='', marker='.', markersize=4, color=colors[ds])
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
            plt.savefig(os.path.join(plots_dir, plot_name), dpi=500)
            print('Plot saved at {}'.format(os.path.join(plots_dir, plot_name)))


        


