import os
import numpy as np
import math
import scipy
import logging 
import time
from matplotlib import pyplot as plt

from scipy import integrate
from pycorr import setup_logging
from .utils import BaseClass
from .lognormal_model import LognormalDensityModel
from .gram_charlier import GramCharlier2D
from densitysplit.edgeworth_development import cumulant_from_moments, ExpandedNormal

plt.style.use(os.path.join('/feynman/home/dphp/mp270220/densitysplit/nb', 'densitysplit.mplstyle'))


class DensitySplitModel(BaseClass):
    """
    Class computing density-split statistics from input 2D PDF of the smoothed density constrast at given separation.
    """

    def __init__(self, *args, nsplits=3, density_bins=None, nbar=None, **kwargs):
        self.logger = logging.getLogger('DensitySplitsModel')
        self.logger.info('Initializing DensitySplitsModel with {} density splits'.format(nsplits))
        self.nsplits = nsplits
        self._fixed_density_bins = True
        self.density_bins = density_bins
        if density_bins is None:
            self._fixed_density_bins = False
        self.nbar = nbar
        

    def compute_ds_nbar(self, delta, edgeworth=4, bins=100, plot_fn=None):
        #pdf1D, edges = np.histogram(delta, bins=bins, density=True)
        #midpoints = (edges[1:] + edges[:-1]) / 2

        p = edgeworth
        model = LognormalDensityModel()
        sigma, delta0 = model.get_params_from_moments(sample=delta)
        # transform variables to get Gaussian distributions
        x = np.log(1 + delta/delta0) + sigma**2/2.
        moments = [np.mean(x**(i+1)) for i in range(p)]
        cumulants = [float(cumulant_from_moments(moments, i+1)) for i in range(p)]
        edgew = ExpandedNormal(cum=cumulants)
        lognormal_bins = np.log(1 + self.density_bins/delta0) + sigma**2/2.
        lognormal_bins[0] = -np.inf

        plt.figure(figsize=(4, 3))
        plt.hist(x, bins=1000, density=True, alpha=0.7)
        dd = np.linspace(-2, 2, 100)
        plt.plot(dd, edgew.pdf(dd), color='magenta', label='Edgeworth')
        plt.plot(dd, scipy.stats.norm.pdf(dd, np.mean(x), np.std(x)), color='C1', label='Gaussian')
        plt.yscale('log')
        plt.xlabel(r'$\ln ( 1 + \delta_R / \delta_{0} )$')
        plt.legend()
        plt.savefig(plot_fn, dpi=500)
        plt.close()

        res = list()

        for i in range(self.nsplits):
            split_min = lognormal_bins[i]
            split_max = lognormal_bins[i+1]
            #split_mask = (midpoints > split_min) & (midpoints <= split_max)

            #ds_midpoints = midpoints[split_mask]
            #ds_pdf1D = pdf1D[split_mask]
            #dx = (ds_midpoints[1]-ds_midpoints[0])
            #ds_nbar = np.sum(ds_pdf1D) * dx
            ds_nbar = integrate.quad(edgew.pdf, split_min,  split_max)[0]
            res.append(ds_nbar)

        return np.array(res)
            

    def compute_dsplits(self, delta1=None, delta2=None, norm=None):
        res = list()
        norm = list()

        for i in range(self.nsplits):
            split_min = self.density_bins[i]
            split_max = self.density_bins[i+1]
 
            split_mask = (delta1 > split_min) & (delta1 <= split_max)
            nrm = np.sum(split_mask)/len(delta1)
            dsplit = np.mean(1 + delta2[split_mask])*nrm
            res.append(dsplit) 
            norm.append(nrm) 

        #if norm is None:
        #    norm = self.compute_ds_nbar(delta1, bins=bins)
        print('norm: ', norm)
        return np.array(res)/norm - 1


    def compute_gram_charlier_dsplits(self, n=3, delta1=None, delta2=None, bins=100, norm=None, plot_fn=None, legend=None):        
        # lognormal transform to get near-Gaussian distribution
        model = LognormalDensityModel()
        sigma1, delta01 = model.get_params_from_moments(sample=delta1)
        sigma2, delta02 = model.get_params_from_moments(sample=delta2)
        delta0 = np.array([delta01, delta02])
        sigma = np.array([sigma1, sigma2])
        delta = np.array([delta1, delta2])
        x = np.log(1 + delta/delta0[..., None]) + sigma[..., None]**2/2.

        edges = np.linspace(-2, 2, bins+1)
        midpoints = (edges[1:]+edges[:-1])/2

        # Gram-Charlier 2D PDF
        gcharlier = GramCharlier2D(sample=x, n=n)
        posgrid = np.dstack(np.meshgrid(midpoints, midpoints))
        gc2D = gcharlier.pdf(posgrid)

        # plot
        xedges = (np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
        hist2D, xedges, yedges = np.histogram2d(x[0], x[1], bins=xedges, density=True)
        X, Y = np.meshgrid((xedges[1:]+xedges[:-1])/2, (yedges[1:]+yedges[:-1])/2)
        levels = [0.000335, 0.011, 0.135, 0.607]
        plt.figure(figsize=(4, 4))
        plt.contour(X, Y, hist2D, levels=levels, colors='C0')
        gausspdf = GramCharlier2D(sample=x, n=2).pdf(posgrid)
        plt.plot([], [], label=legend, alpha=0)
        plt.contour(midpoints, midpoints, gausspdf, levels=levels, colors='red', alpha=0.7)
        plt.contour(midpoints, midpoints, gc2D, levels=levels, colors='magenta', alpha=0.7)
        plt.xlabel(r'$\ln \left[ 1 + \delta_1 / \delta_{{0, 1}} \right]$')
        plt.ylabel(r'$\ln \left[ 1 + \delta_2 / \delta_{{0, 2}} \right]$')
        plt.legend()
        plt.savefig(plot_fn, dpi=500)
        plt.close()

        integ = np.sum(gc2D) * (posgrid[0, :, 0][1]-posgrid[0, :, 0][0])*(posgrid[:, 0, 1][1]-posgrid[:, 0, 1][0])
        print('PDF integrating to {}'.format(integ))

        res = list()
        norm = list()

        lognormal_bins = np.log(1 + self.density_bins/delta01) + sigma1**2/2.
        lognormal_bins[0] = -np.inf
        
        for i in range(self.nsplits):
            split_min = lognormal_bins[i]
            split_max = lognormal_bins[i+1]
            split_mask = (posgrid[..., 0] > split_min) & (posgrid[..., 0] <= split_max)

            split_delta = posgrid[split_mask]
            split_pdf2D = gc2D[split_mask]
            dx = midpoints[1]-midpoints[0]
            dsplit = np.sum(np.exp(split_delta[..., 1] - sigma2**2/2) * split_pdf2D) * dx**2

            split_mask = (x[0] > split_min) & (x[0] <= split_max)
            #split_x = x[:, split_mask]
            #test = np.mean(np.exp(split_x[1] - sigma2**2/2))
            #print('integ with hist: {}'.format(dsplit))
            #print('integ with mean: {}'.format(test))
            norm.append(np.sum(split_pdf2D) * dx**2)
            res.append(dsplit) 
        print(res)
        return np.array(res)/norm - 1

        