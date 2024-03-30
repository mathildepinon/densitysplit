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
from .edgeworth_development import cumulant_from_moments, ExpandedNormal


def damping_function(k, k_lambda, sigma_lambda):
    if k < k_lambda:
        return 1
    else:
        return np.exp(-(k-k_lambda)**2/(2*sigma_lambda**2))


class SplitCCFModel:
    """
    Class implementing analytical model for density split cross-correlation functions.
    """

    def __init__(self, redshift, cosmology, k=np.logspace(-5, 3, 100000), pk=None, rsd=False, mu=np.linspace(-1, 1, 400), non_linear=False, damping=False, bias=1, nsplits=1, smoothing_kernel=6, smoothing_scale=10, shot_noise=0, nbar=0.01, boxsize=1000, nmesh=512):
        self.k = k
        self.redshift = redshift
        self.cosmology = cosmology
        self.bias = bias
        self.shot_noise = shot_noise
        self.nbar = nbar
        self.s = np.linspace(0., 150., 151)
        self.boxsize = boxsize
        self.nmesh = nmesh
        self.pm = ParticleMesh(BoxSize=[self.boxsize] * 3, Nmesh=[self.nmesh] * 3, dtype='c16')
        if pk is None:
            fo = Fourier(self.cosmology, engine='camb')
            pklin = fo.pk_interpolator(non_linear=non_linear, extrap_kmin=1e-10, extrap_kmax=1e6).to_1d(z=self.redshift)
            pklin_array = pklin(k)
            kN = np.pi*nmesh/boxsize
            if damping:
                pkdamped_func = lambda k: pklin(k) * np.array([damping_function(kk, 0.8*kN, 0.05*kN) for kk in k])
                pk = PowerSpectrumInterpolator1D.from_callable(k, pkdamped_func)
            else:
                pk = pklin
        self.pk = pk
        self.rsd = rsd
        self.mu = mu
        if rsd:
            f = cosmology.get_background().growth_rate(redshift)
            pk_rsd = (1 + f * mu**2)**2 * pklin_array[:, None]
            self.pk_rsd = pk_rsd
        self.set_pk_3D()
        self.set_xi_model()
        self.smoothing_kernel = smoothing_kernel
        self.compute_sigma()
        self.set_smoothing_scale(smoothing_scale)
        #self.set_smoothed_pk_3D()
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

    def set_pk_3D_RSD(self, mu, los='x'):
        f = self.cosmology.get_background().growth_rate(self.redshift)
        vlos = np.array([1. * (los == axis) for axis in 'xyz'])
        # Generate P(k) on the mesh
        cfield = self.pm.create('complex')
        norm = 1 / self.pm.BoxSize.prod()
        for kslab, delta_slab in zip(cfield.slabs.x, cfield.slabs):
            # The square of the norm of k on the mesh
            k2 = sum(kk**2 for kk in kslab)
            k = (k2**0.5).ravel()
            mask_nonzero = k != 0.
            mu = kslab * vlos / k
            pk = np.zeros_like(k)
            pk[mask_nonzero] = self.pk(k[mask_nonzero])
            delta_slab[...].flat = (self.bias**2 * (1 + f * mu**2)**2 * pk + self.shot_noise) * norm
        self.pk_3D_RSD = cfield

    # def smoothing_kernel(self, k):
    #     res = np.sinc(self.smoothing_scale * k / 2. / np.pi)**6
    #     return res
    #
    # def isotropic_smoothing_kernel_3D(self):
    #     cfield = self.pm.create('complex')
    #     for kslab, w_slab in zip(cfield.slabs.x, cfield.slabs):
    #         k2 = sum(kk**2 for kk in kslab)
    #         k = (k2**0.5).ravel()
    #         w = np.sinc(self.smoothing_scale * k / 2. / np.pi)**6
    #         w_slab[...].flat = w
    #     return cfield

    def set_smoothing_scale(self, smoothing_scale):
        self.smoothing_scale = smoothing_scale
        self.set_smoothing_kernel_3D()
        self.set_smoothed_pk_3D()
        self.compute_xi_R()
        self.compute_sigma_R()
        self.compute_sigma_RR()

    def set_smoothing_kernel_3D(self):
        # order of the smoothing kernel
        p = self.smoothing_kernel
        cfield = self.pm.create('complex')
        for kslab, w_slab in zip(cfield.slabs.x, cfield.slabs):
            # k2 = sum(kk**2 for kk in kslab)
            # k = (k2**0.5).ravel()
            w = reduce(mul, (np.sinc(self.smoothing_scale * kk / 2. / np.pi)**p for kk in kslab), 1)
            # w = np.sinc(self.smoothing_scale * k / 2. / np.pi)**6
            w_slab[...].flat = w
        self.smoothing_kernel_3D = cfield

    def set_smoothed_pk_3D(self):
        self.smoothed_pk_3D = self.pk_3D * self.smoothing_kernel_3D
        self.double_smoothed_pk_3D = self.pk_3D * self.smoothing_kernel_3D**2

    def smoothed_shot_noise_cumulant(self, p):
        fourier_kernel = self.smoothing_kernel_3D
        norm_fourier_kernel = fourier_kernel / fourier_kernel.BoxSize.prod()
        real_space_kernel = norm_fourier_kernel.c2r()
        real_space_kernel.value = np.real(real_space_kernel.value)
        intg = integrate_pmesh_field((real_space_kernel)**p)
        return self.nbar * intg

    def smoothed_density_moments(self):
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
        #print(res1)

        ## p=2
        w2 = integrate_pmesh_field(real_space_kernel**2)
        ## second order moment of delta*p/nbar
        m2 = self.sigma**2 / self.nbar * w2 + self.sigma_RR**2
        ## second order moment of (1 + delta)*p/nbar
        res2 = m2 + w1**2 + w2 / self.nbar
        #print(res2)

        ## p=3
        w3 = integrate_pmesh_field(real_space_kernel**3)
        w2_xiR = integrate_pmesh_field(real_space_kernel**2 * xi_R_field)
        ## third order moment of (1 + delta)*p/nbar
        res3 = (1 + 3 * self.sigma**2) * w3 / self.nbar**2 \
            + 3 * (1 + self.sigma**2) * w2 * w1 / self.nbar \
            + 6 * w2_xiR / self.nbar \
            + w1**3 \
            + 3 * w1 * self.sigma_RR**2
        #print(res3)

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
            + 6 * self.sigma**2 * w2 * self.sigma_RR**2 / self.nbar \
            + 12 * w2_xiR2 / self.nbar \
            + 3 * self.sigma_RR**4

        ## fourth order moment of (1 + delta)*p/nbar
        fourier_aux = fourier_squared_kernel * self.pk_3D
        aux = integrate_pmesh_field(squared_real_kernel * fourier_aux.c2r()) * self.boxsize**3
        term4_bis = 4 * aux + 2 * term4
        # res4 = 3 * (1 + 2*self.sigma**2 + self.sigma**4) * w4 / self.nbar**3 \
        #     + 12 * (1 + self.sigma**2) * (w3 * w1 + w3_xiR) / self.nbar**2 \
        #     + 3 * ((3 + 2 * self.sigma**2 + self.sigma**4) * w2**2 + term4_bis) / self.nbar**2 \
        #     + 6 * ((3 + self.sigma**2) * w2 * w1**2 + (1 + self.sigma**2) * w2 * self.sigma_RR**2 + 4 * w1 * w2_xiR \
        #     + 2 * w2_xiR2) / self.nbar \
        #     + 3 * (w1**4 + 2 * w1**2 * self.sigma_RR**2 + self.sigma_RR**4)

        res4 = m4 \
           + (1 + 6. * self.sigma**2) * w4 / self.nbar**3 \
           + 4 * (w3 * (1 + 3 * self.sigma**2) + 3 * w3_xiR) / self.nbar**2 \
           + 3. * (w2**2 + 2 * self.sigma**2 * w2**2 + 4. * aux) / self.nbar**2 \
           + 6. * ((1. + self.sigma**2) * w2 * w1**2 + 4. * w2_xiR * w1 + self.sigma_RR**2 * w2) / self.nbar \
           + w1**4 + 6. * self.sigma_RR**2 * w1**2

        #print(res4)

        res = [res1, res2, res3, res4]
        #res = [0, m2, 0, m4]
        self.density_moments = res

        return res


    def smoothed_density_cumulant(self, p=4):
        if hasattr(self, 'density_moments') and len(self.density_moments) >= p:
            moments = self.density_moments
        else:
            moments = self.smoothed_density_moments()
        if p==1:
            res = float(cumulant_from_moments(moments[0:1], 1)) - 1.
        else:
            res = float(cumulant_from_moments(moments[0:p], p))
        return res

    def density_with_shot_noise(self, delta, p=4):
        """Density PDF inlcuding shot noise using Edgeworth development from cumulants"""
        cumulants = [self.smoothed_density_cumulant(i+1) for i in range(p)]
        edgew = ExpandedNormal(cum=cumulants)
        return edgew.pdf(delta)

    def set_xi_model(self):
        # powerToCorrelation = PowerToCorrelation(self.k)
        # sep, xi = powerToCorrelation(self.bias**2 * (self.pk(self.k)))
        # self.xi = CorrelationFunctionInterpolator1D(sep, xi=xi)
        xifield = self.pk_3D.c2r()
        #xifield.value = np.real(xifield.value)
        sep, mu, xi = project_to_basis(xifield, edges=(self.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
        self.sep = sep
        self.xi = xi

    def set_xi_model_rsd(self, ells):
        powerToCorrelation = PowerToCorrelation(self.k, ell=ells)
        sep, xi = powerToCorrelation(self.bias**2 * (self.pk_rsd(self.k)))
        self.xi_rsd = CorrelationFunctionInterpolator1D(sep, xi=xi)
    
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


    def compute_xi_R(self):
        # self.sep = sep
        # powerToCorrelation = PowerToCorrelation(self.k)
        # sep, xi = powerToCorrelation((self.bias**2 * self.pk(self.k) + self.shot_noise) * self.smoothing_kernel(self.k))
        # self.xi_R = CorrelationFunctionInterpolator1D(sep, xi=xi)
        xiRfield = self.smoothed_pk_3D.c2r()
        xiRfield.value = np.real(xiRfield.value)
        sep, mu, xiR = project_to_basis(xiRfield, edges=(self.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
        xi_R = np.real(xiR)
        self.xi_R = xi_R

        xiRRfield = self.double_smoothed_pk_3D.c2r()
        xiRRfield.value = np.real(xiRRfield.value)
        sep, mu, xiRR = project_to_basis(xiRRfield, edges=(self.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
        xi_RR = np.real(xiRR)
        self.xi_RR = xi_RR

        ## Shot noise correction for xi_R
        wfield = self.smoothing_kernel_3D.c2r() / self.boxsize**3
        sep, mu, w = project_to_basis(wfield, edges=(self.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
        shot_noise_correction = (1 + self.sigma**2) * w / self.nbar
        self.xi_R_with_shot_noise = xi_R + np.real(shot_noise_correction)

        ## Shot noise correction for xi_RR
        sep, mu, w = project_to_basis(wfield**2, edges=(self.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
        shot_noise_correction = (1 + self.sigma**2) * w / self.nbar
        self.xi_RR_with_shot_noise = xi_RR + np.real(shot_noise_correction)

        return self.xi_R_with_shot_noise


    def compute_sigma_R(self):
        # u = np.linspace(-5., 3., 100000)
        # k = 10 ** u
        # dk = (u[1] - u[0]) * k
        # integrand = k**2 / (2 * np.pi**2) * (self.bias**2 * self.pk(k) + self.shot_noise) * self.smoothing_kernel(k)
        # val = np.trapz(integrand, k, dk)
        val1 = self.xi_R[0]
        self.sigma_R = np.sqrt(val1)
        val2 = self.xi_R_with_shot_noise[0]
        self.sigma_R_with_shot_noise = np.sqrt(val2)
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


    def compute_delta_tilde(self, density_bins, shot_noise=False, p=4):
        self.density_bins = density_bins
        self.nsplits = len(density_bins)-1
        res = list()
        if shot_noise:
            for i in range(len(density_bins) - 1):
                d1 = density_bins[i]
                d2 = density_bins[i+1]
                if not math.isfinite(d1):
                    d1 = -100
                if not math.isfinite(d2):
                    d2 = 100
                delta = np.linspace(d1, d2, 1000)
                pdf = self.density_with_shot_noise(delta, p)
                integrand1 = pdf
                integrand2 = delta * pdf
                integral1 = np.trapz(integrand1, delta)
                integral2 = np.trapz(integrand2, delta)
                res.append(integral2 / integral1)
        else:
            prefactor = -np.sqrt(2/np.pi)*self.sigma_RR
            for i in range(len(density_bins) - 1):
                d1 = density_bins[i]
                d2 = density_bins[i+1]
                num = np.exp(- d2**2 / (2 * self.sigma_RR**2)) - np.exp(- d1**2 / (2 * self.sigma_RR**2))
                denom = math.erf(d2 / (np.sqrt(2) * self.sigma_RR)) - math.erf(d1 / (np.sqrt(2) * self.sigma_RR))
                res.append(prefactor * num/denom)
        self.delta_tilde = np.array(res)
        return np.array(res)


    def compute_delta2_tilde(self, density_bins, shot_noise=False, p=4):
        self.density_bins = density_bins
        self.nsplits = len(density_bins)-1
        res = list()
        for i in range(len(density_bins) - 1):
            d1 = density_bins[i]
            d2 = density_bins[i+1]
            if not math.isfinite(d1):
                d1 = -100
            if not math.isfinite(d2):
                d2 = 100
            delta = np.linspace(d1, d2, 10000)
            if shot_noise:
                pdf = self.density_with_shot_noise(delta, p)
            else:
                pdf = self.stats.norm.pdf(delta, 0, self.sigma_RR)
            integrand1 = pdf
            integrand2 = delta**2 * pdf
            integral1 = np.trapz(integrand1, delta)
            integral2 = np.trapz(integrand2, delta)
            res.append(integral2 / integral1)
        self.delta2_tilde = np.array(res)
        return np.array(res)


    def ccf_randoms_tracers(self, density_bins, shot_noise=False, p=4):
        res = list()
        self.density_bins = density_bins
        self.nsplits = len(density_bins)-1
        if shot_noise:
            if not hasattr(self, 'density_moments'):
                self.smoothed_density_moments()

            xi_R = self.xi_R_with_shot_noise
            sigma_RR = np.sqrt(self.density_moments[1]-1)
            prefactor = - np.sqrt(2/np.pi) * xi_R/sigma_RR
            split_delta_tilde = self.compute_delta_tilde(density_bins, shot_noise, p)
            for i in range(len(density_bins) - 1):
                res.append(xi_R / sigma_RR**2 * split_delta_tilde[i])
        else:
            xi_R = self.xi_R
            sigma_RR = self.sigma_RR
            prefactor = - np.sqrt(2/np.pi) * xi_R/sigma_RR
            for i in range(len(density_bins) - 1):
                d1 = density_bins[i]
                d2 = density_bins[i+1]
                num = np.exp(- d2**2 / (2 * sigma_RR**2)) - np.exp(- d1**2 / (2 * sigma_RR**2))
                denom = math.erf(d2 / (np.sqrt(2) * sigma_RR)) - math.erf(d1 / (np.sqrt(2) * sigma_RR))
                res.append(prefactor * num/denom)
        return np.array(res)


    def acf_randoms(self, density_bins, shot_noise=False):
        mvn = scipy.stats.multivariate_normal
        cf = list()
        self.density_bins = density_bins
        self.nsplits = len(density_bins)-1
        if shot_noise:
            if not hasattr(self, 'density_moments'):
                self.smoothed_density_moments()
            xi_RR = self.xi_RR_with_shot_noise
            sigma_RR = np.sqrt(self.density_moments[1]-1)
        else:
            xi_RR = self.xi_RR
            sigma_RR = self.sigma_RR
        for i in range(len(density_bins) - 1):
            res = list()
            ## NB: starting from idx=1 because sigma_RR**2 == xi_RR[0]
            for idx in range(1, len(self.sep)):
                cov = np.array([[float(sigma_RR**2), float(xi_RR[idx])],
                                [float(xi_RR[idx]), float(sigma_RR**2)]])
                d1 = density_bins[i]
                d2 = density_bins[i+1]
                # cdf is not defined at -inf, but the pdf is symmetrical
                if not math.isfinite(d1):
                    d1, d2 = -d2, -d1
                cdf = mvn.cdf(np.array([d2, np.inf]), cov=cov) - mvn.cdf(np.array([d1, d2]), cov=cov) - mvn.cdf(np.array([d2, d1]), cov=cov) + mvn.cdf(np.array([d1, d1]), cov=cov)
                denom = scipy.special.erf(d2 / (np.sqrt(2) * sigma_RR)) - scipy.special.erf(d1 / (np.sqrt(2) * sigma_RR))
                res.append(4 * cdf/denom**2 - 1)
            cf.append(np.array(res))
        return np.array(cf)


    def nbar_R_DS(self, density_bins, shot_noise=False, p=4):
        res = list()
        self.density_bins = density_bins
        self.nsplits = len(density_bins)-1

        if not hasattr(self, 'density_moments'):
            self.smoothed_density_moments()

        split_delta_tilde = self.compute_delta_tilde(density_bins, shot_noise, p)

        if shot_noise:
            xi_R = self.xi_R_with_shot_noise
            sigma_R = self.sigma_R_with_shot_noise
            sigma_RR = np.sqrt(self.density_moments[1]-1)
        else:
            xi_R = self.xi_R
            sigma_R = self.sigma_R
            sigma_RR = self.sigma_RR
        for i in range(len(density_bins) - 1):
            d1 = density_bins[i]
            d2 = density_bins[i+1]
            if not math.isfinite(d1):
                d1 = -100
            if not math.isfinite(d2):
                d2 = 100
            delta = np.linspace(d1, d2, 1000)
            pdf = self.density_with_shot_noise(delta, p)
            integral = np.trapz(pdf, delta)
            res.append(float(integral * (1 + sigma_R**2 / sigma_RR**2 * split_delta_tilde[i])))
        return np.array(res)


    def ccf_tracers(self, density_bins, shot_noise=False, p=4):
        prefactor_num = np.sqrt(2/np.pi) * self.xi_R / (self.sigma_RR * self.xi)
        prefactor_denom = np.sqrt(2/np.pi) * self.sigma_R**2 / self.sigma_RR
        res = list()
        self.density_bins = density_bins
        self.nsplits = len(density_bins)-1

        if not hasattr(self, 'density_moments'):
            self.smoothed_density_moments()

        if shot_noise:
            xi_R = self.xi_R_with_shot_noise
            sigma_R = self.sigma_R_with_shot_noise
            sigma_RR = np.sqrt(self.density_moments[1]-1)
            split_delta_tilde = self.compute_delta_tilde(density_bins, shot_noise, p)
            split_delta2_tilde = self.compute_delta2_tilde(density_bins, shot_noise, p)

            for i in range(len(density_bins) - 1):
                denom = 1 + sigma_R**2 * split_delta_tilde[i] / sigma_RR**2
                num = self.xi + xi_R / sigma_RR**2 * (split_delta_tilde[i] - sigma_R**2 + sigma_R**2 * split_delta2_tilde[i] / sigma_RR**2)
                res.append(num/denom)

        else:
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
    
    
    def compute_xi_Y_R(self):
        self.xi_Y_R = np.log(1 + self.xi_R) - 1/4. * np.log(1 + self.sigma**2) * np.log(1 + self.sigma_RR**2) 
        
        return self.xi_Y_R
