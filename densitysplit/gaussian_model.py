import numpy as np
import math
import scipy
import logging 
import time

from pycorr import setup_logging
from cosmoprimo import *
from .corr_func_utils import *
from .utils import BaseClass, integrate_pmesh_field
from pmesh import ParticleMesh
from pypower.fft_power import project_to_basis
from .edgeworth_development import cumulant_from_moments, ExpandedNormal

from .base_model import BaseTwoPointCorrelationFunctionModel, SmoothedTwoPointCorrelationFunctionModel


class SmoothedGaussianDensityModel(SmoothedTwoPointCorrelationFunctionModel):
    """
    Class implementing Gaussian model for the smoothed density contrast.
    """

    def __init__(self, *args, **kwargs):
        if len(args) and type(args[0]) is SmoothedTwoPointCorrelationFunctionModel:
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(**kwargs)
        self.logger.info('Initializing SmoothedGaussianDensityModel')

    def smoothed_density_moments(self, nbar=None):
        if nbar is not None and nbar != self.nbar:
            self.set_shotnoise(nbar)
        if not self.nbar:
            self.logger.info('nbar is 0, density is Gaussian.')
            res = [1, 1+self.double_smoothed_sigma**2, 0, 0]
        else:
            self.logger.info('Computing moments of the smoothed density contrast.')
            t0 = time.time()
            fourier_kernel = self.smoothing_kernel_3D
            norm_fourier_kernel = fourier_kernel / self.boxsize**3
            real_space_kernel = norm_fourier_kernel.c2r()
            real_space_kernel.value = np.real(real_space_kernel.value)
            xi_R_field = self.smoothed_pk_3D.c2r()
            xi_R_field.value = np.real(xi_R_field.value)
    
            ## p=1
            ## this is the moment of the variable (1 + delta)*p/nbar
            ## where delta is Gaussian and p follows a Poisson distribution of parameter nbar
            w1 = integrate_pmesh_field(real_space_kernel) ## should be 1
            res1 = w1
            self.logger.info('1st order moment: {}.'.format(float(res1)))
    
            ## p=2
            w2 = integrate_pmesh_field(real_space_kernel**2)
            ## second order moment of delta*p/nbar
            m2 = self.sigma**2 / self.nbar * w2 + self.double_smoothed_sigma**2
            ## second order moment of (1 + delta)*p/nbar
            res2 = m2 + w1**2 + w2 / self.nbar
            self.logger.info('2nd order moment: {}.'.format(float(res2)))
    
            ## p=3
            w3 = integrate_pmesh_field(real_space_kernel**3)
            w2_xiR = integrate_pmesh_field(real_space_kernel**2 * xi_R_field)
            ## third order moment of (1 + delta)*p/nbar
            res3 = (1 + 3 * self.sigma**2) * w3 / self.nbar**2 \
                + 3 * (1 + self.sigma**2) * w2 * w1 / self.nbar \
                + 6 * w2_xiR / self.nbar \
                + w1**3 \
                + 3 * w1 * self.double_smoothed_sigma**2
            self.logger.info('3rd order moment: {}.'.format(float(res3)))
    
            ## p=4
            w4 = integrate_pmesh_field(real_space_kernel**4)
            w3_xiR = integrate_pmesh_field(real_space_kernel**3 * xi_R_field)
            squared_real_kernel = real_space_kernel**2
            squared_real_xi = self.pk_3D.c2r()**2
            fourier_squared_kernel = squared_real_kernel.r2c()
            fourier_squared_xi_R = fourier_squared_kernel * squared_real_xi.r2c()
            term4 = integrate_pmesh_field(squared_real_kernel * fourier_squared_xi_R.c2r()) * self.boxsize**3
            #term5 = integrate_pmesh_field(real_space_kernel**2) * integrate_pmesh_field(real_space_kernel * xi_R_field)
            w2_xiR2 = integrate_pmesh_field(real_space_kernel**2 * xi_R_field**2)
            ## fourth order moment of delta*p/nbar
            m4 = 3 * self.sigma**4 * w4 / self.nbar**3 \
                + 12 * self.sigma**2 * w3_xiR / self.nbar**2 \
                + 3 * self.sigma**4 * w2**2 / self.nbar**2 \
                + 6 * term4 / self.nbar**2 \
                + 6 * self.sigma**2 * w2 * self.double_smoothed_sigma**2 / self.nbar \
                + 12 * w2_xiR2 / self.nbar \
                + 3 * self.double_smoothed_sigma**4
    
            ## fourth order moment of (1 + delta)*p/nbar
            fourier_aux = fourier_squared_kernel * self.pk_3D
            aux = integrate_pmesh_field(squared_real_kernel * fourier_aux.c2r()) * self.boxsize**3
            term4_bis = 4 * aux + 2 * term4
            res4 = m4 \
               + (1 + 6. * self.sigma**2) * w4 / self.nbar**3 \
               + 4 * (w3 * (1 + 3 * self.sigma**2) + 3 * w3_xiR) / self.nbar**2 \
               + 3. * (w2**2 + 2 * self.sigma**2 * w2**2 + 4. * aux) / self.nbar**2 \
               + 6. * ((1. + self.sigma**2) * w2 * w1**2 + 4. * w2_xiR * w1 + self.double_smoothed_sigma**2 * w2) / self.nbar \
               + w1**4 + 6. * self.double_smoothed_sigma**2 * w1**2
            self.logger.info('4th order moment: {}.'.format(float(res4)))
    
            res = [res1, res2, res3, res4]
            #res = [0, m2, 0, m4]
            self.logger.info('Moments of the smoothed density contrast computed in elapsed time {} seconds.'.format(time.time() - t0))
        
        self.density_moments = res
        return res

    def smoothed_density_cumulant(self, p=4, nbar=None):
        if nbar is not None and nbar != self.nbar:
            moments = self.smoothed_density_moments(nbar=nbar)
        else:
            if hasattr(self, 'density_moments'):
                moments = self.density_moments
            else:
                moments = self.smoothed_density_moments(nbar=nbar)

        if p==1:
            res = float(cumulant_from_moments(moments[0:1], 1)) - 1.
        else:
            res = float(cumulant_from_moments(moments[0:p], p))
        return res

    def density(self, delta, p=4, nbar=None):
        """Density PDF inlcuding shot noise using Edgeworth development from cumulants"""
        cumulants = [self.smoothed_density_cumulant(i+1, nbar) for i in range(p)]
        edgew = ExpandedNormal(cum=cumulants)
        return edgew.pdf(delta)


class GaussianDensitySplitsModel(SmoothedGaussianDensityModel):
    """
    Class implementing Gaussian model for density-split statistics.
    """

    def __init__(self, *args, nsplits=3, randoms_size=4, density_bins=None, **kwargs):
        if len(args) and type(args[0]) is SmoothedGaussianDensityModel:
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(**kwargs)
        self.logger.info('Initializing GaussianDensitySplitsModel with {} density splits'.format(nsplits))
        self.nsplits = nsplits
        self.randoms_size = randoms_size
        self.density_bins = density_bins
        if self.density_bins is None:
            splits = np.linspace(0, 1, self.nsplits+1)
            self.density_bins = scipy.stats.norm.ppf(splits, 0, self.double_smoothed_sigma)
            self.logger.info('No density bins provided, computing density bins from a Gaussian density: {}.'.format(density_bins))

    def compute_delta_tilde(self, density_bins=None, nbar=None, p=4):
        if density_bins is not None:
            self.density_bins = density_bins
        self.nsplits = len(self.density_bins)-1
        if nbar is not None and nbar != self.nbar:
            self.set_shotnoise(nbar)
        res = list()
        self.logger.info('Computing delta tilde.')
        if self.nbar:
            for i in range(len(self.density_bins) - 1):
                d1 = self.density_bins[i]
                d2 = self.density_bins[i+1]
                if not math.isfinite(d1):
                    d1 = -100
                if not math.isfinite(d2):
                    d2 = 100
                delta = np.linspace(d1, d2, 1000)
                pdf = self.density(delta, p=p, nbar=self.nbar)
                integrand1 = pdf
                integrand2 = delta * pdf
                integral1 = np.trapz(integrand1, delta)
                integral2 = np.trapz(integrand2, delta)
                res.append(integral2 / integral1)
        else:
            prefactor = -np.sqrt(2/np.pi)*self.double_smoothed_sigma
            for i in range(len(self.density_bins) - 1):
                d1 = self.density_bins[i]
                d2 = self.density_bins[i+1]
                num = np.exp(- d2**2 / (2 * self.double_smoothed_sigma**2)) - np.exp(- d1**2 / (2 * self.double_smoothed_sigma**2))
                denom = math.erf(d2 / (np.sqrt(2) * self.double_smoothed_sigma)) - math.erf(d1 / (np.sqrt(2) * self.double_smoothed_sigma))
                res.append(prefactor * num/denom)
        self.delta_tilde = np.array(res)
        return np.array(res)

    def ccf_randoms_tracers(self, density_bins=None, nbar=None, p=4):
        if density_bins is not None:
            self.density_bins = density_bins
        self.nsplits = len(self.density_bins)-1
        if nbar is not None and nbar != self.nbar:
            self.set_shotnoise(nbar)
            self.smoothed_density_moments(nbar=self.nbar)
        else:
            if self.nbar and not hasattr(self, 'density_moments'):
                self.smoothed_density_moments(nbar=self.nbar)
        self.logger.info('Computing cross-correlation of density-split randoms and all tracers with {} density splits, nbar = {}, and Edgeworth expansion of order {}.'.format(self.nsplits, self.nbar, p))
        res = list()
        if self.nbar:
            xiR = self.smoothed_xi
            sigmaRR = np.sqrt(self.density_moments[1]-1)
            prefactor = - np.sqrt(2/np.pi) * xiR/sigmaRR
            split_delta_tilde = self.compute_delta_tilde(self.density_bins, nbar=self.nbar, p=p)
            for i in range(len(self.density_bins) - 1):
                res.append(xiR / sigmaRR**2 * split_delta_tilde[i])
        else:
            xiR = self.smoothed_xi
            sigmaRR = self.double_smoothed_sigma
            prefactor = - np.sqrt(2/np.pi) * xiR/sigmaRR
            for i in range(len(self.density_bins) - 1):
                d1 = self.density_bins[i]
                d2 = self.density_bins[i+1]
                num = np.exp(- d2**2 / (2 * sigmaRR**2)) - np.exp(- d1**2 / (2 * sigmaRR**2))
                denom = math.erf(d2 / (np.sqrt(2) * sigmaRR)) - math.erf(d1 / (np.sqrt(2) * sigmaRR))
                res.append(prefactor * num/denom)
        return np.array(res)

    def acf_randoms(self, density_bins=None, nbar=None):
        if density_bins is not None:
            self.density_bins = density_bins
        self.nsplits = len(self.density_bins)-1
        if nbar is not None and nbar != self.nbar:
            self.set_shotnoise(nbar)
            self.smoothed_density_moments(nbar=self.nbar)
        else:
            if self.nbar and not hasattr(self, 'density_moments'):
                self.smoothed_density_moments(nbar=self.nbar)
        xiRR = self.double_smoothed_xi
        sigmaRR = np.sqrt(self.density_moments[1]-1)
        self.logger.info('Computing auto-correlation of density-split randoms with {} density splits, nbar = {}.'.format(self.nsplits, self.nbar))
        for i in range(len(self.density_bins) - 1):
            res = list()
            ## NB: starting from idx=1 because sigma_RR**2 == xi_RR[0]
            for idx in range(1, len(self.sep)):
                cov = np.array([[float(sigmaRR**2), float(xiRR[idx])],
                                [float(xiRR[idx]), float(sigmaRR**2)]])
                d1 = self.density_bins[i]
                d2 = self.density_bins[i+1]
                # cdf is not defined at -inf, but the pdf is symmetrical
                if not math.isfinite(d1):
                    d1, d2 = -d2, -d1
                cdf = mvn.cdf(np.array([d2, np.inf]), cov=cov) - mvn.cdf(np.array([d1, d2]), cov=cov) - mvn.cdf(np.array([d2, d1]), cov=cov) + mvn.cdf(np.array([d1, d1]), cov=cov)
                denom = scipy.special.erf(d2 / (np.sqrt(2) * sigma_RR)) - scipy.special.erf(d1 / (np.sqrt(2) * sigma_RR))
                res.append(4 * cdf/denom**2 - 1)
            cf.append(np.array(res))
        return np.array(cf)
    

    