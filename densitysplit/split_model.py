import numpy as np
import math
import scipy

from pycorr import TwoPointCorrelationFunction, TwoPointEstimator, NaturalTwoPointEstimator, project_to_multipoles, project_to_wp, utils, setup_logging
from cosmoprimo import *
from .corr_func_utils import *

class SplitCCFModel:
    """
    Class implementing analytical model for density split cross-correlation functions.
    """

    def __init__(self, k, redshift, cosmology, bias=1, nsplits=1, smoothing_scale=10, nbar=None):
        self.k = k
        self.redshift = redshift
        self.cosmology = cosmology
        self.bias = bias
        self.nbar = nbar
        if self.nbar is not None:
            self.shot_noise = 1. / nbar
        else:
            self.shot_noise = 0
        self.set_pk_model()
        self.set_xi_model()
        self.set_smoothing_scale(smoothing_scale)
        self.compute_sigma_R()
        self.compute_xi_R()
        self.compute_sigma_RR()

    def set_pk_model(self, extrap_kmin=1e-10, extrap_kmax=1e6):
        fo = Fourier(self.cosmology, engine='camb')
        pk = fo.pk_interpolator(nonlinear=False, extrap_kmin=extrap_kmin, extrap_kmax=extrap_kmax).to_1d(z=self.redshift)
        self.pk = pk

    def set_xi_model(self):
        powerToCorrelation = PowerToCorrelation(self.k)
        sep, xi = powerToCorrelation(self.bias**2 * (self.pk(self.k)))
        self.sep = sep
        self.xi = CorrelationFunctionInterpolator1D(sep, xi=xi)

    def set_smoothing_scale(self, smoothing_scale):
        self.smoothing_scale = smoothing_scale
        self.compute_sigma_R()
        self.compute_xi_R()
        self.compute_sigma_RR()

    def smoothing_kernel(self, k):
        res = np.sinc(self.smoothing_scale * k / 2. / np.pi)**6
        return res

    def compute_sigma_R(self):
        u = np.linspace(-5., 3., 100000)
        k = 10 ** u
        dk = (u[1] - u[0]) * k
        integrand = k**2 / (2 * np.pi**2) * self.bias**2 * (self.pk(k) + self.shot_noise) * self.smoothing_kernel(k)
        val = np.trapz(integrand, k, dk)
        self.sigma_R = np.sqrt(val)
        return self.sigma_R

    def compute_sigma_RR(self):
        u = np.linspace(-5., 3., 100000)
        k = 10 ** u
        dk = (u[1] - u[0]) * k
        integrand = k**2 / (2 * np.pi**2) * self.bias**2 * (self.pk(k) + self.shot_noise) * self.smoothing_kernel(k)**2
        val = np.trapz(integrand, k, dk)
        self.sigma_RR = np.sqrt(val)
        return self.sigma_RR

    def compute_xi_R(self, sep=np.linspace(1., 150., 150)):
        self.sep = sep
        powerToCorrelation = PowerToCorrelation(self.k)
        sep, xi = powerToCorrelation(self.bias**2 * (self.pk(self.k) + self.shot_noise) * self.smoothing_kernel(self.k))
        self.xi_R = CorrelationFunctionInterpolator1D(sep, xi=xi)
        return self.xi_R(sep)

    def ccf_randoms_tracers(self, density_bins):
        prefactor = - np.sqrt(2/np.pi) * self.xi_R(self.sep)/self.sigma_RR
        res = list()
        self.density_bins = density_bins
        self.nsplits = len(density_bins)-1
        for i in range(len(density_bins) - 1):
            d1 = density_bins[i]
            d2 = density_bins[i+1]
            num = np.exp(- d2**2 / (2 * self.sigma_RR**2)) - np.exp(- d1**2 / (2 * self.sigma_RR**2))
            denom = math.erf(d2 / (np.sqrt(2) * self.sigma_RR)) - math.erf(d1 / (np.sqrt(2) * self.sigma_RR))
            res.append(prefactor * num/denom)
        return np.array(res)

    def ccf_tracers(self, density_bins):
        prefactor_num = np.sqrt(2/np.pi) * self.xi_R(self.sep) / (self.sigma_RR * self.xi(self.sep))
        prefactor_denom = np.sqrt(2/np.pi) * self.sigma_R**2 / self.sigma_RR
        res = list()
        self.density_bins = density_bins
        self.nsplits = len(density_bins)-1
        for i in range(len(density_bins) - 1):
            d1 = density_bins[i]
            d2 = density_bins[i+1]
            delta1 = np.exp(- d2**2 / (2 * self.sigma_RR**2)) - np.exp(- d1**2 / (2 * self.sigma_RR**2))
            delta2 = self.sigma_R**2 / self.sigma_RR**2 * (d2 * np.exp(- d2**2 / (2 * self.sigma_RR**2)) - d1 * np.exp(- d1**2 / (2 * self.sigma_RR**2)))
            delta3 = math.erf(d2 / (np.sqrt(2) * self.sigma_RR)) - math.erf(d1 / (np.sqrt(2) * self.sigma_RR))
            res.append(self.xi(self.sep) * (1 - prefactor_num * (delta1 + delta2) / delta3) / (1 - prefactor_denom * delta1 / delta3))
        return np.array(res)
