import numpy as np
import math
import scipy

from pycorr import TwoPointCorrelationFunction, TwoPointEstimator, NaturalTwoPointEstimator, project_to_multipoles, project_to_wp, utils, setup_logging
from cosmoprimo import *
from .corr_func_utils import *
from .utils import integrate_pmesh_field
from pmesh import ParticleMesh
from functools import reduce # Valid in Python 2.6+, required in Python 3
from operator import mul
from pypower.fft_power import project_to_basis
from .edgeworth_development import cumulant_from_moments

class SplitCCFModel:
    """
    Class implementing analytical model for density split cross-correlation functions.
    """

    def __init__(self, k, redshift, cosmology, pk=None, bias=1, nsplits=1, smoothing_scale=10, shot_noise=0, nbar=0.01):
        self.k = k
        self.redshift = redshift
        self.cosmology = cosmology
        self.bias = bias
        self.shot_noise = shot_noise
        self.nbar = nbar
        self.s = np.linspace(0., 150., 151)
        self.boxsize = 1000
        self.nmesh = 512
        self.pm = ParticleMesh(BoxSize=[self.boxsize] * 3, Nmesh=[self.nmesh] * 3, dtype='c16')
        if pk is None:
            fo = Fourier(self.cosmology, engine='camb')
            pk = fo.pk_interpolator(nonlinear=False, extrap_kmin=1e-10, extrap_kmax=1e6).to_1d(z=self.redshift)
        self.pk = pk
        self.set_pk_3D()
        self.set_smoothing_scale(smoothing_scale)
        #self.set_smoothed_pk_3D()
        self.set_xi_model()
        self.compute_sigma()
        #self.compute_xi_R()
        #self.compute_sigma_R()
        #self.compute_sigma_RR()

    # def set_pk_model(self, extrap_kmin=1e-10, extrap_kmax=1e6, pk=None):
    #     if pk is None:
    #         fo = Fourier(self.cosmology, engine='camb')
    #         pk = fo.pk_interpolator(nonlinear=False, extrap_kmin=extrap_kmin, extrap_kmax=extrap_kmax).to_1d(z=self.redshift)
    #     self.pk = pk
    #     self.set_pk_3D()

    def set_pk_3D(self):
        # Generate P(k) on the mesh
        cfield = self.pm.create('complex')
        norm = 1 / self.pm.BoxSize.prod()
        for kslab, delta_slab in zip(cfield.slabs.x, cfield.slabs):
            # The square of the norm of k on the mesh
            k2 = sum(kk**2 for kk in kslab)
            k = (k2**0.5).ravel()
            mask_nonzero = k != 0.
            pk = np.zeros_like(k)
            pk[mask_nonzero] = self.pk(k[mask_nonzero])
            delta_slab[...].flat = (self.bias**2 * pk + self.shot_noise) * norm
        self.pk_3D = cfield

    def smoothing_kernel(self, k):
        res = np.sinc(self.smoothing_scale * k / 2. / np.pi)**6
        return res

    def isotropic_smoothing_kernel_3D(self):
        cfield = self.pm.create('complex')
        for kslab, w_slab in zip(cfield.slabs.x, cfield.slabs):
            k2 = sum(kk**2 for kk in kslab)
            k = (k2**0.5).ravel()
            w = np.sinc(self.smoothing_scale * k / 2. / np.pi)**6
            w_slab[...].flat = w
        return cfield

    def smoothing_kernel_3D(self):
        cfield = self.pm.create('complex')
        for kslab, w_slab in zip(cfield.slabs.x, cfield.slabs):
            # k2 = sum(kk**2 for kk in kslab)
            # k = (k2**0.5).ravel()
            w = reduce(mul, (np.sinc(self.smoothing_scale * kk / 2. / np.pi)**6 for kk in kslab), 1)
            # w = np.sinc(self.smoothing_scale * k / 2. / np.pi)**6
            w_slab[...].flat = w
        return cfield

    def set_smoothed_pk_3D(self):
        self.smoothed_pk_3D = self.pk_3D * self.smoothing_kernel_3D()
        self.double_smoothed_pk_3D = self.pk_3D * self.smoothing_kernel_3D()**2

    def smoothed_shot_noise_cumulant(self, p):
        fourier_kernel = self.smoothing_kernel_3D()
        norm_fourier_kernel = fourier_kernel / fourier_kernel.BoxSize.prod()
        real_space_kernel = norm_fourier_kernel.c2r()
        real_space_kernel.value = np.real(real_space_kernel.value)
        intg = integrate_pmesh_field((real_space_kernel)**p)
        return self.nbar * intg

    def smoothed_density_moment(self, p):
        fourier_kernel = self.smoothing_kernel_3D()
        norm_fourier_kernel = fourier_kernel / self.boxsize**3
        real_space_kernel = norm_fourier_kernel.c2r()
        real_space_kernel.value = np.real(real_space_kernel.value)
        xi_R_field = self.smoothed_pk_3D.c2r()
        xi_R_field.value = np.real(xi_R_field.value)
        if p % 2 != 0:
            res = 0
        if p==2:
            first_term = integrate_pmesh_field(real_space_kernel**2)
            second_term = integrate_pmesh_field(real_space_kernel * xi_R_field)  # NB: actually equal to self.simga_RR**2
            res = self.sigma**2 * self.nbar * first_term + self.nbar**2 * second_term
        if p==4:
            term1 = integrate_pmesh_field(real_space_kernel**4)
            term2 = integrate_pmesh_field(real_space_kernel**3 * xi_R_field)
            term3 = integrate_pmesh_field(real_space_kernel**2)**2
            squared_real_kernel = real_space_kernel**2
            squared_real_xi_R = self.pk_3D.c2r()**2
            fourier_squared_xi_R = squared_real_kernel.r2c() * squared_real_xi_R.r2c()
            term4 = integrate_pmesh_field(squared_real_kernel * fourier_squared_xi_R.c2r())
            term5 = integrate_pmesh_field(real_space_kernel) * integrate_pmesh_field(real_space_kernel**2 * xi_R_field)
            term6 = integrate_pmesh_field(real_space_kernel**2 * xi_R_field**2)
            term7 = integrate_pmesh_field(real_space_kernel * xi_R_field)**2 # NB: actually equal to self.simga_RR**4
            res = 3 * self.sigma**4 * self.nbar * term1.real / self.nmesh**3 \
                + 12 * self.sigma**2 * self.nbar**2 * term2.real / self.nmesh**3 \
                + 3 * self.sigma**4 * self.nbar**2 * term3.real \
                + 6 * self.nbar**2 * term4.real \
                + 6 * self.sigma**2 * self.nbar**3 * term5.real \
                + 12 * self.nbar**3 * term6.real \
                + 3 * self.nbar**4 * term7.real
        print(p, res / self.nbar**p)
        return res / self.nbar**p

    def smoothed_density_cumulant(self, p):
        moments = [self.smoothed_density_moment(i+1) for i in range(p)]
        return float(cumulant_from_moments(moments, p))

    def set_xi_model(self):
        # powerToCorrelation = PowerToCorrelation(self.k)
        # sep, xi = powerToCorrelation(self.bias**2 * (self.pk(self.k)))
        # self.xi = CorrelationFunctionInterpolator1D(sep, xi=xi)
        xifield = self.pk_3D.c2r()
        #xifield.value = np.real(xifield.value)
        sep, mu, xi = project_to_basis(xifield, edges=(self.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
        self.sep = sep
        self.xi = xi

    def compute_sigma(self):
        ## 1st method: integrating 1D pk
        # u = np.linspace(-5., 3., 100000)
        # k = 10 ** u
        # dk = (u[1] - u[0]) * k
        # integrand = k**2 / (2 * np.pi**2) * (self.bias**2 * self.pk(k) + self.shot_noise)
        # val = np.trapz(integrand, k, dk)

        ## 2nd method: xi(0)
        # val = self.xi[0]

        ## 3rd method: integrating 3D pk
        val = np.real(np.sum(self.pk_3D))

        self.sigma = np.sqrt(val)

        return self.sigma

    def set_smoothing_scale(self, smoothing_scale):
        self.smoothing_scale = smoothing_scale
        self.set_smoothed_pk_3D()
        self.compute_xi_R()
        self.compute_sigma_R()
        self.compute_sigma_RR()

    def compute_xi_R(self):
        # self.sep = sep
        # powerToCorrelation = PowerToCorrelation(self.k)
        # sep, xi = powerToCorrelation((self.bias**2 * self.pk(self.k) + self.shot_noise) * self.smoothing_kernel(self.k))
        # self.xi_R = CorrelationFunctionInterpolator1D(sep, xi=xi)
        xifield = self.smoothed_pk_3D.c2r()
        xifield.value = np.real(xifield.value)
        sep, mu, xi = project_to_basis(xifield, edges=(self.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
        self.xi_R = np.real(xi)
        return self.xi_R

    def compute_sigma_R(self):
        # u = np.linspace(-5., 3., 100000)
        # k = 10 ** u
        # dk = (u[1] - u[0]) * k
        # integrand = k**2 / (2 * np.pi**2) * (self.bias**2 * self.pk(k) + self.shot_noise) * self.smoothing_kernel(k)
        # val = np.trapz(integrand, k, dk)
        val = self.xi_R[0]
        self.sigma_R = np.sqrt(val)
        return self.sigma_R

    def compute_sigma_RR(self):
        # u = np.linspace(-5., 3., 100000)
        # k = 10 ** u
        # dk = (u[1] - u[0]) * k
        # integrand = k**2 / (2 * np.pi**2) * (self.bias**2 * self.pk(k) + self.shot_noise) * self.smoothing_kernel(k)**2
        # val = np.trapz(integrand, k, dk)
        xifield = self.double_smoothed_pk_3D.c2r()
        xifield.value = np.real(xifield.value)
        sep, mu, xi = project_to_basis(xifield, edges=(self.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
        val = xi[0].real
        self.sigma_RR = np.sqrt(val)
        return self.sigma_RR

    def compute_delta_tilde(self, density_bins):
        prefactor = -np.sqrt(2/np.pi)*self.sigma_RR
        self.density_bins = density_bins
        self.nsplits = len(density_bins)-1
        res = list()
        for i in range(len(density_bins) - 1):
            d1 = density_bins[i]
            d2 = density_bins[i+1]
            num = np.exp(- d2**2 / (2 * self.sigma_RR**2)) - np.exp(- d1**2 / (2 * self.sigma_RR**2))
            denom = math.erf(d2 / (np.sqrt(2) * self.sigma_RR)) - math.erf(d1 / (np.sqrt(2) * self.sigma_RR))
            res.append(prefactor * num/denom)
        return np.array(res)

    def ccf_randoms_tracers(self, density_bins):
        prefactor = - np.sqrt(2/np.pi) * self.xi_R/self.sigma_RR
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
        prefactor_num = np.sqrt(2/np.pi) * self.xi_R / (self.sigma_RR * self.xi)
        prefactor_denom = np.sqrt(2/np.pi) * self.sigma_R**2 / self.sigma_RR
        res = list()
        self.density_bins = density_bins
        self.nsplits = len(density_bins)-1
        for i in range(len(density_bins) - 1):
            d1 = density_bins[i]
            d2 = density_bins[i+1]
            delta1 = np.exp(- d2**2 / (2 * self.sigma_RR**2)) - np.exp(- d1**2 / (2 * self.sigma_RR**2))
            ## Handle the case where d1 or d2 is infinite
            if math.isfinite(d1):
                a = d1 * np.exp(- d1**2 / (2 * self.sigma_RR**2))
            else:
                a = 0
            if math.isfinite(d2):
                b = d2 * np.exp(- d2**2 / (2 * self.sigma_RR**2))
            else:
                b = 0
            delta2 = self.sigma_R**2 / self.sigma_RR**2 * (b - a)
            delta3 = math.erf(d2 / (np.sqrt(2) * self.sigma_RR)) - math.erf(d1 / (np.sqrt(2) * self.sigma_RR))
            res.append(self.xi * (1 - prefactor_num * (delta1 + delta2) / delta3) / (1 - prefactor_denom * delta1 / delta3))
        return np.array(res)
