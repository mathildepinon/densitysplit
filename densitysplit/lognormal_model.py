import numpy as np
import math
import scipy
import logging 
import time

from pycorr import setup_logging
from pypower.fft_power import project_to_basis
from .utils import BaseClass, integrate_pmesh_field
from .base_model import SmoothedTwoPointCorrelationFunctionModel


class LognormalDensityModel(BaseClass):
    """
    Class implementing shifted lognormal model.
    """
    _defaults = dict(sigma=None, delta0=None)

    def __init__(self, sigma=None, delta0=None, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger('LognormalDensityModel')
        self.logger.info('Initializing LognormalDensityModel')
        self.sigma = sigma
        self.delta0 = delta0

    def density(self, delta, sigma=None, delta0=None):
        if sigma is not None:
            self.sigma = sigma
        if delta0 is not None:
            self.delta0 = delta0
        #pdf_model = np.zeros_like(delta)
        #pdf_model[delta > -self.delta0] = scipy.stats.lognorm.pdf(delta[delta > -self.delta0], self.sigma, -self.delta0, self.delta0 * np.exp(-self.sigma**2 / 2))
        pdf_model = scipy.stats.lognorm.pdf(delta, self.sigma, -self.delta0, self.delta0 * np.exp(-self.sigma**2 / 2))
        return pdf_model

    def get_params_from_moments(self, sample=None, m2=None, m3=None, delta0_init=1.):
        """Get parameters of the lognormal distribution (sigma, delta0) from the second and third order moments of the sample."""
        self.logger.info('Computing sigma, delta0 from the second and third order moments of the density sample.')
        from scipy.optimize import minimize
        if sample is not None:
            m2 = np.mean(sample**2)
            m3 = np.mean(sample**3)
        def tomin(delta0):
            return (m3 - 3/delta0 * m2**2 - 1/delta0**3 * m2**3)**2
        res = minimize(tomin, x0=delta0_init)
        sigma = np.sqrt(np.log(1 + m2/res.x[0]**2))
        delta0 = res.x[0]
        self.sigma = sigma
        self.delta0 = delta0
        self.logger.info('Seting sigma to {:.3f}, delta0 to {:.3f}.'.format(self.sigma, self.delta0))        
        return sigma, delta0

    def get_sigma_from_theory(self, delta0=None, **kwargs):
        """Compute sigma from the theoretical variance of the smoothed density field."""
        if delta0 is not None:
            self.delta0 = delta0
        self.logger.info('Computing sigma from theory, assuming delta0 = {:.3f}.'.format(self.delta0))
        model = SmoothedTwoPointCorrelationFunctionModel(**kwargs)
        sigmaRR = float(model.double_smoothed_sigma)
        if model.nbar is not None:
            fourier_kernel = model.smoothing_kernel_3D
            norm_fourier_kernel = fourier_kernel / model.boxsize**3
            real_space_kernel = norm_fourier_kernel.c2r()
            real_space_kernel.value = np.real(real_space_kernel.value)
            w2 = integrate_pmesh_field(real_space_kernel**2)
            shotnoise = w2 / model.nbar
            sigmaRR = np.sqrt(sigmaRR**2 + shotnoise)
        self.logger.info('Square root of the theoretical variance of the smoothed density field computed: {:.3f}.'.format(float(sigmaRR)))        
        self.theory_double_smoothed_sigma = sigmaRR
        self.sigma = float(np.sqrt(np.log(1 + sigmaRR**2/self.delta0**2)))
        self.logger.info('Theoretical value for sigma (parameter of the lognormal distribution): {:.3f}.'.format(self.sigma))        
        return self.sigma

    def fit_params_from_pdf(self, delta=None, density_pdf=None, params_init=np.array([1., 1.]), sigma=None):
        """Fit parameters of the lognormal distribution (sigma, delta0) to match the input pdf."""
        self.logger.info('Fitting sigma, delta0 to match input PDF.')
        def to_fit(delta, *params):
            y = self.density(delta, params[0], params[1])
            return y
        fit = scipy.optimize.curve_fit(to_fit, delta, density_pdf, p0=params_init, sigma=sigma)
        bestfit_params = fit[0]
        self.sigma = bestfit_params[0]
        self.delta0 = bestfit_params[1]
        self.logger.info('Seting sigma to {:.3f}, delta0 to {:.3f}.'.format(self.sigma, self.delta0))        
        return bestfit_params


class BiasedLognormalDensityModel(LognormalDensityModel):
    """
    Class implementing shifted lognormal model with bias.
    """
    _defaults = dict(sigma=None, delta0=None)

    def __init__(self, *args, sigma=None, delta0=None, **kwargs):
        if len(args) and type(args[0]) is LognormalDensityModel:
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(**kwargs)
        self.logger = logging.getLogger('BiasedLognormalDensityModel')
        self.logger.info('Initializing BiasedLognormalDensityModel')

    def bias_function(self, delta, model='quadratic', sigma=None, delta0=None, b1=1., b2=0., b3=0., return_deriv=True):
        if model=='linear':
            biased_delta = b1 * delta
            deriv = b1
        delta2 = delta0**2*(np.exp(sigma**2)-1)
        if model=='quadratic':
            biased_delta = b1 * delta + b2 * (delta**2 - delta2)
            deriv = b1 + 2 * b2 * delta
        if model=='cubic':
            delta3 = delta2**3 / delta0**3 + 3 * delta2**2 / delta0
            biased_delta = b1 * delta + b2 * (delta**2 - sigma**2) + b3 * (delta**3 - delta3)
            deriv = b1 + 2 * b2 * delta + 3 * b3 * delta**2
        if model=='gaussian':
            b2 = b2 - b1**2
            mub = - b1 / b2
            Nb = np.exp(-b1**2/(2*b2)) / np.sqrt(b2*delta2 +1)
            sigmab = np.sqrt(-1/b2 - delta2)
            biased_delta = Nb * np.exp(-(delta - mub)**2 / (2*sigmab**2))
            print(biased_delta)
            deriv = Nb * (delta - mub) / sigmab**2 * np.exp(-(delta - mub)**2 / (2*sigmab**2))
        if return_deriv:
            return biased_delta, deriv
        else:
            return biased_deltas
        
    def density(self, delta, model='quadratic', sigma=None, delta0=None, b1=1., b2=0., b3=0.):
        if model=='test':
            delta2 = delta0**2*(np.exp(sigma**2)-1)
            delta3 = 3/delta0**2*delta2 + 1/delta0**3*delta2**(3./2)
            biaseddelta = b1*delta + b2/2.*(delta**2 - delta2) + b3*(delta**3 - delta3)
            pdf_model = scipy.stats.lognorm.pdf(biaseddelta, sigma, -delta0, delta0 * np.exp(-sigma**2 / 2))
            norm = b1 + 2*b2*delta + 2*b3*delta**2
            return pdf_model/np.abs(norm)
        else:
            pdf_model = scipy.stats.lognorm.pdf(delta, sigma, -delta0, delta0 * np.exp(-sigma**2 / 2))
            biaseddelta, deriv = self.bias_function(delta, model, sigma, delta0, b1, b2, b3, return_deriv=True)
            pdf_model /= np.abs(deriv)
            biased_pdf_model = np.interp(delta, biaseddelta, pdf_model, left=0, right=0)
            return biased_pdf_model
            
    def fit_params_from_pdf(self, delta=None, density_pdf=None, model='quadratic', params_init=None, sigma=None):
        """Fit parameters of the biased lognormal distribution to match the input pdf."""
        if model=='linear':
            params_init = np.array([1.])
        if model=='quadratic':
            params_init = np.array([1., 0.])
        if model=='cubic':
            params_init = np.array([1., 0., 0.])
        if model=='gaussian':
            params_init = np.array([0.37, 0.955, 1.52, 0.007])
        if model=='test':
            params_init = np.array([1., 1., 1., 0., 0.])
        self.logger.info('Fitting {} params'.format(len(params_init)))
        
        def to_fit(delta, *params):
            if model=='test' or model=='gaussian':
                y = self.density(delta, model, *params)
            else:
                y = self.density(delta, model, self.sigma, self.delta0, *params)
            return y

        fit = scipy.optimize.curve_fit(to_fit, delta, density_pdf, p0=params_init, sigma=sigma)
        bestfit_params = fit[0]
        self.bias_params = bestfit_params
        self.logger.info('Best fit parameters: {}'.format(bestfit_params))      
        return bestfit_params


class LognormalDensitySplitModel(LognormalDensityModel):
    """
    Class implementing lognormal model for density-split statistics.
    """

    def __init__(self, *args, nsplits=3, density_bins=None, **kwargs):
        if len(args) and type(args[0]) is LognormalDensityModel:
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(**kwargs)
        self.logger.info('Initializing LognormalDensitySplitsModel with {} density splits'.format(nsplits))
        self.nsplits = nsplits
        self._fixed_density_bins = True
        self.density_bins = density_bins
        if density_bins is None:
            self._fixed_density_bins = False
            self.set_params()

    def set_params(self, sigma=None, delta0=None):
        if sigma is not None:
            self.logger.info('Setting sigma to {:.3f}.'.format(sigma))
            self.sigma = sigma
        if delta0 is not None:
            self.logger.info('Setting delta0 to {:.3f}.'.format(delta0))
            self.delta0 = delta0
        if not self._fixed_density_bins:
            splits = np.linspace(0, 1, self.nsplits+1)
            self.density_bins = scipy.stats.lognorm.ppf(splits, self.sigma, -self.delta0, self.delta0*np.exp(-self.sigma**2/2.))
            self.logger.info('No density bins provided, computing density bins from a lognormal density with sigma = {:.3f}, delta0 = {:.3f}: {}.'.format(self.sigma, self.delta0, self.density_bins))

    def set_smoothed_xi_model(self, **kwargs):
        if 'nbar' in kwargs.keys():
            self.nbar = kwargs.pop('nbar')
        else:
            self.nbar = 0
        model = SmoothedTwoPointCorrelationFunctionModel(nbar=0, **kwargs)
        self.smoothed_xi = model.smoothed_xi.ravel()
        self.sep = model.sep.ravel()
        if self.nbar is not None and self.nbar:
            self.logger.info('Adding shotnoise correction to the smoothed 2PCF.')
            wfield = model.smoothing_kernel_3D.c2r() / model.boxsize**3
            sep, mu, w = project_to_basis(wfield, edges=(model.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
            shotnoise = np.real(w / self.nbar)
            self.smoothed_xi += shotnoise.ravel()

    def _compute_main_term(self, delta, sigma=None, delta0=None, delta01=1., xiR=None, sep=None, **kwargs):
        self.set_params(sigma=sigma, delta0=delta0)
        self.delta01 = delta01
        if sep is not None:
            self.sep = sep
        if xiR is None:
            if not hasattr(self, 'smoothed_xi'):
                self.set_smoothed_xi_model(**kwargs)
            xiR = self.smoothed_xi
        if math.isfinite(delta):
            a = scipy.special.erf((np.log(1 + delta/self.delta0) + self.sigma**2/2. - np.log(1 + xiR/(self.delta0*self.delta01))) / (np.sqrt(2) * self.sigma))
            b = scipy.special.erf((np.log(1 + delta/self.delta0) + self.sigma**2/2.) / (np.sqrt(2) * self.sigma))
        else:
            if delta > 0:
                a = np.full_like(xiR, 1)
                b = np.full_like(xiR, 1)
            if delta < 0:
                a = np.full_like(xiR, -1)
                b = np.full_like(xiR, -1)
        return a, b
    
    def compute_dsplits(self, delta0=None, **kwargs):
        self.logger.info('Computing lognormal density split model.')
        if delta0 is None:
            delta0 = self.delta0
        dsplits = list()
        for i in range(len(self.density_bins)-1):
            d1 = max(self.density_bins[i], -delta0)
            d2 = self.density_bins[i+1]
            a1, b1 = self._compute_main_term(d1, delta0=delta0, **kwargs)
            a2, b2 = self._compute_main_term(d2, delta0=delta0, **kwargs)
            main_term = (a2 - a1) / (b2 - b1)
            dsplits.append(self.delta01*(main_term - 1))
        return dsplits
        
    
        