import numpy as np
import math
import scipy
import logging 
import time
from scipy.interpolate import interp1d
from scipy.special import factorial, loggamma

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

    def density_shotnoise(self, norm, k=None, delta=None, sigma=None, delta0=None):
        if sigma is not None:
            self.sigma = sigma
        if delta0 is not None:
            self.delta0 = delta0
        x = np.arange(max(0, 1-self.delta0)+0.001, 11, 0.01)
        density_pdfvals = self.density(x-1, sigma, delta0)
        def func(N):
            log_poisson_pdf = N * np.log(norm * x[:, None]) - (norm * x[:, None]) - loggamma(N+1) # log to avoid overflow
            poisson_pdf = np.exp(log_poisson_pdf)
            res = np.trapz(poisson_pdf * density_pdfvals[:, None], x=x, axis=0)
            return res
        if delta is not None:
            k = np.round(norm * (1 + delta))
            k = np.append(k, np.max(k)+1)
            return interp1d(k, func(k), bounds_error=False, fill_value=0)(norm * (1+delta)) * norm
        else:
            if k is None:
                k = np.arange(0, 100)
            return func(k) * norm # k may not be flat
 
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
        self.logger.info('Setting sigma to {:.3f}, delta0 to {:.3f}.'.format(self.sigma, self.delta0))        
        return sigma, delta0

    def get_sigma_from_theory(self, delta0=None, rsd=False, **kwargs):
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
        if rsd:
            self.logger.info('Computing variance of the smoothed density field in redshift space.')        
            bg = cosmology.get_background(engine='camb')
            f = bg.growth_rate(z)
            bias = 1.
            beta = f / bias
            if 'non_linear' in kwargs.keys():
                kwargs.pop('non_linear')
            model = SmoothedTwoPointCorrelationFunctionModel(non_linear=False, **kwargs)
            sigmaRR = np.sqrt((1 + 2 * beta /3 + beta**2 /5.)*model.sigma_RR**2 + shotnoise)
        self.logger.info('Square root of the theoretical variance of the smoothed density field computed: {:.3f}.'.format(float(sigmaRR)))        
        self.theory_double_smoothed_sigma = sigmaRR
        self.sigma = float(np.sqrt(np.log(1 + sigmaRR**2/self.delta0**2)))
        self.logger.info('Theoretical value for sigma (parameter of the lognormal distribution): {:.3f}.'.format(self.sigma))        
        return self.sigma

    def fit_params_from_pdf(self, delta=None, density_pdf=None, params_init=np.array([1., 1.]), sigma=None, shotnoise=False, norm=None):
        """Fit parameters of the lognormal distribution (sigma, delta0) to match the input pdf."""
        self.logger.info('Fitting sigma, delta0 to match input PDF.')
        def to_fit(delta, *params):
            if shotnoise:
                y = self.density_shotnoise(norm=norm, delta=delta, sigma=params[0], delta0=params[1])
            else:
                y = self.density(delta, params[0], params[1])
            return y
        fit = scipy.optimize.curve_fit(to_fit, delta, density_pdf, p0=params_init, sigma=sigma)
        bestfit_params = fit[0]
        self.sigma = bestfit_params[0]
        self.delta0 = bestfit_params[1]
        self.logger.info('Seting sigma to {:.3f}, delta0 to {:.3f}.'.format(self.sigma, self.delta0))    
        return bestfit_params

    def compute_bias_function(self, delta, xiR, **kwargs):
        logxiR = np.log(1 + xiR/(self.delta0**2))
        r = logxiR/self.sigma**2
        logdelta = np.log(1 + delta/self.delta0) + self.sigma**2/2
        res = self.delta0 * (-1 + np.exp(-r**2 * self.sigma**2 /2 + r*logdelta))
        return res/xiR

    # bias function convolved with Poisson shot noise
    def compute_bias_function_shotnoise(self, xiR, norm, delta=None, k=None, **kwargs):
        x = np.arange(max(0, 1-self.delta0)+0.001, 11, 0.01)
        bias_func = self.compute_bias_function(x-1, xiR, **kwargs)
        density_pdfvals_noshotnoise = self.density(x-1)
        prod = bias_func*density_pdfvals_noshotnoise
        density_pdfvals = self.density_shotnoise(norm=norm, delta=delta, k=k)        
        def func(N):
            log_poisson_pdf = N * np.log(norm * x[:, None]) - (norm * x[:, None]) - loggamma(N+1) # log to avoid overflow
            poisson_pdf = np.exp(log_poisson_pdf)
            res = np.trapz(poisson_pdf * prod[:, None], x=x, axis=0)
            return res
        if (k is None) and (delta is not None):
            k = np.round(norm * (1 + delta))
            k = np.append(k, np.max(k)+1)
            return interp1d(k, func(k), bounds_error=False, fill_value=0)(norm * (1 + delta)) * norm / density_pdfvals
        else:
            k = np.arange(0, 100)
            return func(k) * norm / density_pdfvals

    def compute_bias_function_approx(self, delta, **kwargs):
        # large separation limit
        y = np.log(1 + delta/self.delta0) + self.sigma**2/2
        return y/(self.delta0*self.sigma**2)


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

    def set_params(self, sigma=1, delta0=1, delta02=1, sigma2=1):
        if sigma is not None:
            self.logger.info('Setting sigma to {:.3f}.'.format(sigma))
            self.sigma = sigma
        if delta0 is not None:
            self.logger.info('Setting delta0 to {:.3f}.'.format(delta0))
            self.delta0 = delta0
        if delta02 is not None:
            self.logger.info('Setting delta02 to {:.3f}.'.format(delta02))
            self.delta02 = delta02
        if sigma2 is not None:
            self.logger.info('Setting sigma2 to {:.3f}.'.format(sigma2))
            self.sigma2 = sigma2
        if not self._fixed_density_bins:
            splits = np.linspace(0, 1, self.nsplits+1)
            self.density_bins = scipy.stats.lognorm.ppf(splits, self.sigma, -self.delta0, self.delta0*np.exp(-self.sigma**2/2.))
            self.logger.info('No density bins provided, computing density bins from a lognormal density with sigma = {:.3f}, delta0 = {:.3f}: {}.'.format(self.sigma, self.delta0, self.density_bins))

    def density2D(self, delta, sigma=None, delta0=None, delta02=None, sigma2=None, cov=None):
        self.set_params(sigma, delta0, delta02, sigma2)
        # lognormal transform
        X = np.log(1 + delta[..., 0]/self.delta0) + self.sigma**2/2
        Y = np.log(1 + delta[..., 1]/self.delta02) + self.sigma2**2/2
        pdf_model = scipy.stats.multivariate_normal(mean=[0, 0], cov=cov).pdf(np.array([X, Y]).T).T
        pdf_model = pdf_model / ((self.delta0 + delta[..., 0])*(self.delta02 + delta[..., 1]))
        return pdf_model

    def density2D_shotnoise(self, norm, delta=None, k=None, **kwargs):
        x = np.arange(max(0, 1-self.delta0)+0.001, 11, 0.01)
        xx = np.dstack(np.meshgrid(x, x, indexing='ij'))
        x1 = xx[..., 0]
        x2 = xx[..., 1]
        density2D_noshotnoise = self.density2D(delta=xx-1, **kwargs)
        def func(N):
            log_poisson_pdf1 = N[None, None, ...] * np.log(norm * x1[..., None]) - (norm * x1[..., None]) - loggamma(N[None, None, ...]+1) # log to avoid overflow
            log_poisson_pdf2 = N[None, None, ...] * np.log(norm * x2[..., None]) - (norm * x2[..., None]) - loggamma(N[None, None, ...]+1)
            prod1 = np.exp(log_poisson_pdf1) * density2D_noshotnoise[..., None]
            integ1 = np.trapz(prod1, x=x, axis=0)
            prod2 = np.exp(log_poisson_pdf2[0][:, None, :]) * integ1[..., None]
            res = np.trapz(prod2, x=x, axis=0)
            return res
        if (k is None) and (delta is not None):
            k = np.round(norm * (1 + delta))
            #return interp1d(k, func(k), bounds_error=False, fill_value=0)(norm * (1 + delta)) * norm / density2D_noshotnoise
            test = func(k) * norm**2
            return test
        else:
            if k is None:
                k = np.arange(0, 100)
            return func(k) * norm**2

    def set_smoothed_xi_model(self, **kwargs):
        if 'nbar' in kwargs.keys():
            self.nbar = kwargs.pop('nbar')
        else:
            self.nbar = 0
        model = SmoothedTwoPointCorrelationFunctionModel(nbar=0, **kwargs)
        self.smoothed_xi = model.smoothed_xi.ravel()
        self.double_smoothed_xi = model.double_smoothed_xi.ravel()
        self.sep = model.sep.ravel()
        if self.nbar is not None and self.nbar:
            self.logger.info('Adding shotnoise correction to the smoothed 2PCF.')
            wfield = model.smoothing_kernel_3D.c2r() / model.boxsize**3
            sep, mu, w = project_to_basis(wfield, edges=(model.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
            shotnoise = np.real(w / self.nbar)
            self.smoothed_xi += shotnoise.ravel()

            if model.smoothing_scale2 != model.smoothing_scale:
                square_cfield = model.smoothing_kernel_3D * model.smoothing_kernel2_3D
            else:
                square_cfield = model.smoothing_kernel_3D**2
            wfield = square_cfield.c2r() / model.boxsize**3
            sep, mu, w2 = project_to_basis(wfield, edges=(model.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
            shotnoise = np.real(w2 / self.nbar)
            self.double_smoothed_xi  += shotnoise.ravel()
        
    def compute_bias_function(self, delta, sigma=None, delta0=None, delta02=1., xiR=None, sep=None, **kwargs):
        self.set_params(sigma=sigma, delta0=delta0, delta02=delta02)
        if sep is not None:
            self.sep = sep
        if xiR is None:
            if not hasattr(self, 'smoothed_xi'):
                self.set_smoothed_xi_model(**kwargs)
            if smoothing == 1:
                xiR = self.smoothed_xi
            elif smoothing > 1:
                xiR = self.double_smoothed_xi

        # Nb: here we assume delta0 == delta02
        logxiR = np.log(1 + xiR/(self.delta0**2))
        r = logxiR/self.sigma**2
        logdelta = np.log(1 + delta/self.delta0) + self.sigma**2/2
        res = self.delta0 * (-1 + np.exp(-r**2 * self.sigma**2 /2 + r*logdelta))
        
        return res/xiR

    def _compute_main_term(self, delta, sigma=None, delta0=None, delta02=1., xiR=None, sep=None, smoothing=1, **kwargs):
        self.set_params(sigma=sigma, delta0=delta0, delta02=delta02)
        if sep is not None:
            self.sep = sep
        if xiR is None:
            if not hasattr(self, 'smoothed_xi'):
                self.set_smoothed_xi_model(**kwargs)
            if smoothing == 1:
                xiR = self.smoothed_xi
            elif smoothing > 1:
                xiR = self.double_smoothed_xi
        if math.isfinite(delta):
            a = scipy.special.erf((np.log(1 + delta/self.delta0) + self.sigma**2/2. - np.log(1 + xiR/(self.delta0*self.delta02))) / (np.sqrt(2) * self.sigma))
            b = scipy.special.erf((np.log(1 + delta/self.delta0) + self.sigma**2/2.) / (np.sqrt(2) * self.sigma))
        else:
            if delta > 0:
                a = np.full_like(xiR, 1)
                b = np.full_like(xiR, 1)
            if delta < 0:
                a = np.full_like(xiR, -1)
                b = np.full_like(xiR, -1)
        return a, b
    
    def compute_dsplits(self, delta0=None, rsd=False, ells=[0, 2, 4], mu=None, density_bins=None, **kwargs):
        if delta0 is None:
            delta0 = self.delta0
        dsplits = list()
        if density_bins is not None:
            self.density_bins = density_bins
        self.logger.info('Computing lognormal density split model with bins {}.'.format(self.density_bins))
        for i in range(len(self.density_bins)-1):
            d1 = max(self.density_bins[i], -delta0)
            d2 = self.density_bins[i+1]
            a1, b1 = self._compute_main_term(d1, delta0=delta0, **kwargs)
            a2, b2 = self._compute_main_term(d2, delta0=delta0, **kwargs)
            main_term = (a2 - a1) / (b2 - b1)
            dsplits.append(self.delta02*(main_term - 1))
        if rsd:
            dsplits_rsd = list()
            for ds in range(len(dsplits)):
                res_ds = [(2*ell + 1)/2. * np.trapz(dsplits[ds] * scipy.special.legendre(ell)(mu), x=mu, axis=1) for ell in ells]
                dsplits_rsd.append(res_ds)
            return dsplits_rsd
        return dsplits
        
    def compute_dsplits_shotnoise(self, xi, norm, nsplits=None, density_bins=None, delta=None, k=None, **kwargs):
        if nsplits is not None:
            self.nsplits = nsplits
        if density_bins is not None:
            self.density_bins = density_bins
        if delta is None:
            if k is not None:
                xvals = yvals = k/norm
            else:
                xvals = yvals = np.arange(max(0, 1-self.delta0)+0.001, 11, 0.01)
        else:
            xvals, yvals = 1+delta, 1+delta
        rho1, rho2 = np.meshgrid(xvals, yvals, indexing='ij')
        density_pdf_2D_list = list()
        for i in range(len(xi)):
            if np.isnan(xi[i]):
                xi[i] = 0
            cov = np.array([[self.sigma**2, xi[i]],
                            [xi[i], self.sigma**2]])
            density_pdf_2D = self.density2D_shotnoise(norm, delta=delta, k=k, cov=cov, **kwargs)
            density_pdf_2D_list.append(density_pdf_2D)
        density_pdf_2D = np.array(density_pdf_2D_list)
        if k is not None:
            innerint = np.sum(rho2[None, :] * density_pdf_2D, axis=-1)/norm
        else:
            #innerint = np.trapz(rho2[None, :] * density_pdf_2D, x=yvals, axis=-1)
            innerint = np.sum(rho2[None, :] * density_pdf_2D, axis=-1)/norm
        dsplits = list()
        for i in range(len(self.density_bins)-1):
            d1 = max(self.density_bins[i], -1)
            d2 = self.density_bins[i+1]
            self.logger.info('Computing lognormal density split model in density bin {:.2f}, {:.2f}'.format(d1, d2))
            t0 = time.time()
            ds_mask = (rho1[:, 0] >= 1 + d1) & (rho1[:, 0] < 1 + d2)
            if k is not None:
                outerint = np.sum(innerint[..., ds_mask], axis=-1)/norm
                denom =  np.sum(np.sum(density_pdf_2D, axis=-1)[..., ds_mask], axis=-1)/norm**2
            else:
                outerint = np.sum(innerint[..., ds_mask], axis=-1)/norm
                denom =  np.sum(np.sum(density_pdf_2D, axis=-1)[..., ds_mask], axis=-1)/norm**2
                #outerint = np.trapz(innerint[..., ds_mask], x=rho1[:, 0][ds_mask], axis=-1)
                #denom =  np.trapz(np.trapz(density_pdf_2D, x=rho2[0, :], axis=-1)[..., ds_mask], x=rho1[:, 0][ds_mask], axis=-1)
            res = outerint/denom - 1
            self.logger.info('Computed lognormal model in split {:.2f}, {:.2f} for {} xi values in elapsed time: {}s'.format(d1, d2, len(np.array(xi)), time.time()-t0))
            dsplits.append(res)
        return dsplits        