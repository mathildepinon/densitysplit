import numpy as np
import math
import scipy
import logging 
import time

from pycorr import TwoPointCorrelationFunction, TwoPointEstimator, NaturalTwoPointEstimator, project_to_multipoles, project_to_wp, utils, setup_logging
from cosmoprimo import *
from .corr_func_utils import *
from .utils import BaseClass, integrate_pmesh_field
from pmesh import ParticleMesh
from functools import reduce # Valid in Python 2.6+, required in Python 3
from operator import mul
from pypower.fft_power import project_to_basis
from .edgeworth_development import cumulant_from_moments, ExpandedNormal


class BaseTwoPointCorrelationFunctionModel(BaseClass):
    """
    Class implementing two-point correlation function model.
    """
    _defaults = dict(redshift=None, cosmology=None, k=np.logspace(-5, 3, 100000), mu=np.linspace(-1, 1, 400), pk=None, 
                     damping=False, non_linear=False, b1=1., rsd=False, shotnoise=0,
                     boxsize=1000, nmesh=512)

    def __init__(self, **kwargs):
        self.logger = logging.getLogger('BaseTwoPointCorrelationFunctionModel')
        self.logger.info('Initializing BaseTwoPointCorrelationFunctionModel')
        super().__init__(**kwargs)
        if self.pk is None:
            self.logger.info('Initializing theoretical 1D power spectrum')
            fo = Fourier(self.cosmology, engine='class')
            pk_callable = fo.pk_interpolator(non_linear=self.non_linear, extrap_kmin=1e-10, extrap_kmax=1e6).to_1d(z=self.redshift)
            pk_array = pk_callable(self.k)
            if self.damping:
                self.logger.info('Applying damping from 80% of Nyquist frequency')
                def damping_function(k, k_lambda, sigma_lambda):
                    if k < k_lambda:
                        return 1
                    else:
                        return np.exp(-(k-k_lambda)**2/(2*sigma_lambda**2))
                kN = np.pi*self.nmesh/self.boxsize
                pkdamped_func = lambda k: pk_callable(k) * np.array([damping_function(kk, 0.8*kN, 0.05*kN) for kk in k])
                pk = PowerSpectrumInterpolator1D.from_callable(self.k, pkdamped_func)
            else:
                pk = pk_callable
            self.pk = pk
        if self.rsd:
            f = cosmology.get_background().growth_rate(self.redshift)
            pk_rsd = (1 + f * self.mu**2)**2 * pk_array[:, None]
            self.pk_rsd = pk_rsd
        self.logger.info('Initializing ParticleMesh with boxisze {}, nmesh {}'.format(self.boxsize, self.nmesh))
        self.pm = ParticleMesh(BoxSize=[self.boxsize] * 3, Nmesh=[self.nmesh] * 3, dtype='c16')
        self.set_pk_3D()
        self.set_xi()
        self.compute_sigma()

    def set_pk_3D(self):
        """Generate P(k) on the mesh."""
        cfield = self.pm.create('complex')
        norm = 1 / self.pm.BoxSize.prod()
        self.logger.info('Painting 1D power spectrum on 3D mesh')
        t0 = time.time()
        for kslab, delta_slab in zip(cfield.slabs.x, cfield.slabs):
            # The square of the norm of k on the mesh
            k2 = sum(kk**2 for kk in kslab)
            k = (k2**0.5).ravel()
            mask_nonzero = k != 0.
            pk = np.zeros_like(k)
            pk[mask_nonzero] = self.pk(k[mask_nonzero])
            delta_slab[...].flat = (self.b1**2 * pk + self.shotnoise) * norm
        self.pk_3D = cfield
        self.logger.info("3D power spectrum calculated in {:.2f} seconds.".format(time.time() - t0))

    def set_xi(self, s=np.linspace(0., 150., 151)):
        self.s = s
        xifield = self.pk_3D.c2r()
        sep, mu, xi = project_to_basis(xifield, edges=(self.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
        self.sep = sep
        self.xi = xi

    def compute_sigma(self):
        val = np.real(np.sum(self.pk_3D))   
        self.sigma = np.sqrt(val)
        return self.sigma
 

class SmoothedTwoPointCorrelationFunctionModel(BaseTwoPointCorrelationFunctionModel):
    """
    Class implementing two-point correlation function model with a smoothing kernel.
    """

    def __init__(self, *args, smoothing_kernel=6, smoothing_scale=10, nbar=None, **kwargs):
        if type(args[0]) is BaseTwoPointCorrelationFunctionModel:
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(**kwargs)
        self.logger.info('Initializing SmoothedTwoPointCorrelationFunctionModel')
        self.smoothing_kernel = smoothing_kernel
        self.nbar = nbar
        self.set_smoothing_scale(smoothing_scale)
        self.set_shotnoise(nbar)

    def set_shotnoise(self, nbar=None):
        self.logger.info('Setting nbar to {}'.format(nbar))
        if nbar is None:
            nbar = 0
        self.set_smoothed_xi(nbar)

    def set_smoothing_scale(self, smoothing_scale):
        self.logger.info('Setting smoothing scale to {}'.format(smoothing_scale))
        self.smoothing_scale = smoothing_scale
        self.set_smoothing_kernel_3D(p=self.smoothing_kernel)
        self.set_smoothed_pk_3D()
        self.set_smoothed_xi(nbar=self.nbar)
        self.compute_double_smoothed_sigma()

    def set_smoothing_kernel_3D(self, p=6):
        self.logger.info('Setting 3D smoothign kernel of order {}'.format(p))
        self.smoothing_kernel = p # order of the smoothing kernel
        cfield = self.pm.create('complex')
        for kslab, w_slab in zip(cfield.slabs.x, cfield.slabs):
            w = reduce(mul, (np.sinc(self.smoothing_scale * kk / 2. / np.pi)**p for kk in kslab), 1)
            w_slab[...].flat = w
        self.smoothing_kernel_3D = cfield

    def set_smoothed_pk_3D(self):
        self.smoothed_pk_3D = self.pk_3D * self.smoothing_kernel_3D
        self.double_smoothed_pk_3D = self.pk_3D * self.smoothing_kernel_3D**2

    def set_smoothed_xi(self, nbar=None):      
        xiRfield = self.smoothed_pk_3D.c2r()
        xiRfield.value = np.real(xiRfield.value)
        sep, mu, xiR = project_to_basis(xiRfield, edges=(self.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
        self.smoothed_xi = np.real(xiR)
        self.smoothed_sigma = np.sqrt(np.real(xiR))

        xiRRfield = self.double_smoothed_pk_3D.c2r()
        xiRRfield.value = np.real(xiRRfield.value)
        sep, mu, xiRR = project_to_basis(xiRRfield, edges=(self.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
        self.double_smoothed_xi = np.real(xiRR)

        ## shotnoise correction
        self.nbar = nbar if nbar is not None else 0
        if self.nbar:
            wfield = self.smoothing_kernel_3D.c2r() / self.boxsize**3
            sep, mu, w = project_to_basis(wfield, edges=(self.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
            sep, mu, w2 = project_to_basis(wfield**2, edges=(self.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
            shotnoise_corr = (1 + self.sigma**2) * w / self.nbar
            self.smoothed_xi  = self.smoothed_xi + np.real(shotnoise_corr)
            shotnoise_corr2 = (1 + self.sigma**2) * w2 / self.nbar
            self.double_smoothed_xi  = self.double_smoothed_xi + np.real(shotnoise_corr2)

        return self.smoothed_xi

    def compute_double_smoothed_sigma(self):
        xifield = self.double_smoothed_pk_3D.c2r()
        xifield.value = np.real(xifield.value)
        sep, mu, xi = project_to_basis(xifield, edges=(self.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
        self.double_smoothed_sigma = np.sqrt(xi[0].real)
        return self.double_smoothed_sigma


class SmoothedGaussianDensityModel(SmoothedTwoPointCorrelationFunctionModel):
    """
    Class implementing Gaussian model for the smoothed density contrast.
    """

    def __init__(self, *args, **kwargs):
        if type(args[0]) is SmoothedTwoPointCorrelationFunctionModel:
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(**kwargs)
        self.logger.info('Initializing SmoothedGaussianDensityModel')

    def smoothed_density_moments(self, nbar=None):
        if nbar is not None and nbar != self.nbar:
            self.set_shotnoise(nbar)
        if not self.nbar:
            self.logger.info('nbar is 0, density is Gaussian.')
            res = [0, self.double_smoothed_sigma**2, 0, 0]
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
            self.logger.info('1st order moment: {}.'.format(float(res1)-1))
    
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

    
class LognormalDensityModel(BaseClass):
    """
    Class implementing shifted lognormal model.
    """

    def __init__(self, sigma=None, delta0=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.delta0 = delta0

    def density(self, delta):
        pdf_model = np.zeros_like(delta)
        pdf_model[delta > -self.delta0] = scipy.stats.lognorm.pdf(delta[delta > -self.delta0], self.sigma, -self.delta0, self.delta0 * np.exp(-self.sigma**2 / 2))
        return pdf_model

    def get_params_from_moments(delta=None, m2=None, m3=None, delta0_init=1.):
        """Get parameters of the lognormal distribution (sigma, delta0) from the second and third order moments of the sample."""
        from scipy.optimize import minimize
        if delta is not None:
            m2 = np.mean(delta**2)
            m3 = np.mean(delta**3)
        def tomin(delta0):
            return (m3 - 3/delta0 * m2**2 - 1/delta0**3 * m2**3)**2
        res = minimize(tomin, x0=delta0_init)
        sigma = np.sqrt(np.log(1 + m2/res.x[0]**2))
        delta0 = res.x[0]
        self.sigma = sigma
        self.delta0 = delta0
        return sigma, delta0


class GaussianDensitySplitsModel(SmoothedGaussianDensityModel):
    """
    Class implementing Gaussian model for density-split statistics.
    """

    def __init__(self, *args, nsplits=3, randoms_size=4, density_bins=None, **kwargs):
        if type(args[0]) is SmoothedGaussianDensityModel:
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
        self.logger.info('Computing cross-correlation of density-split randoms and all tracers.')
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



    

    