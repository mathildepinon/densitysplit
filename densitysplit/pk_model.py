import numpy as np
import scipy
import scipy.stats
from matplotlib import pyplot as plt
import matplotlib as mpl
import math

from iminuit import Minuit
from pycorr import TwoPointCorrelationFunction, TwoPointEstimator, NaturalTwoPointEstimator, project_to_multipoles, project_to_wp, utils, setup_logging
from cosmoprimo import *

from .utils import *


def pk_func(pk, k):
    """
    Function that evaluates a power spectrum interpolator pk at vector of scales k
    
    Parameters
    ----------
    pk : PowerSpectrumInterpolator1D
        Power spectrum interpolator
    k : array
        Values of k at which the power spectrum must be computed
        
    Returns
    -------
    array
        Power spectrum at each value of k
    """
    return pk(k=k)


def BAO_damping_func(mu, k, sigma_par, sigma_perp):
    """
    Computes BAO damping factor in power spectrum model
    """
    sigma_nl_sq = sigma_par**2 * mu**2 + sigma_perp**2 * (1 - mu**2)
    return np.exp(-k**2 * sigma_nl_sq / 2.)


def bias_damping_func(mu, k, f, b, sigma_s):
    """
    Computes overall bias and damping factor in power spectrum model
    """
    beta = f/b
    damping_factor = (1 + beta * mu**2)**2 / (1 + k**2 * mu**2 * sigma_s**2 / 2.)
    return b**2 * damping_factor


def AP_dilation(mu, k, alpha_par, alpha_perp):
    """
    Computes effective k, mu from Alcock & Paczynski dilation parameters alpha_par and alpha_perp
    """
    F = alpha_par/alpha_perp
    dil = np.sqrt(1 + mu**2 * (1/F**2 - 1))
    k_transposed = k[np.newaxis].T
    k_AP = k_transposed/alpha_perp * dil
    mu_AP = np.divide(mu/F, dil)
    return np.tile(mu_AP, (len(k), 1)), k_AP


# Project to Legendre multipoles
def weights_trapz(x):
    """Return weights for trapezoidal integration."""
    return np.concatenate([[x[1]-x[0]], x[2:]-x[:-2], [x[-1]-x[-2]]])/2.


def to_poles(pkmu, ells):
    mu = np.linspace(0, 1, 101)
    muw_trapz = weights_trapz(mu)
    weights = np.array([muw_trapz * (2*ell + 1) * scipy.special.legendre(ell)(mu) for ell in ells])/(mu[-1] - mu[0])

    return np.sum(pkmu * weights[:,None,:], axis=-1) # P0, P2, P4


def xi_model_poles_interp(k, pk_model, ells):
    """
    Computes correlation function interpolator for each pole in list ells, from power spectrum array pk_model (as returned by pk_model_poles)
    """
    powerToCorrelation = PowerToCorrelation(k, ells, complex=False)
    sep, xi_poles = powerToCorrelation(pk_model)
    return np.array([CorrelationFunctionInterpolator1D(sep[ill], xi=xi_poles[ill]) for ill in range(len(ells))])


def broadband(s, coeffs):
    res = coeffs[0] * s**(-2) + coeffs[1] * s**(-1) + coeffs[2]
    return res


class PkModel:
    """
    Class for power spectrum model, based on model from Bautista et. al (2020).
    """

    def __init__(self, sep, ells, xiell, cov, redshift, cosmology, k=None, split=False, nsplits=1):

        self.sep = sep
        self.ells = ells
        self.nells = len(ells)
        self.xiell = xiell
        self.cov = cov
        self.redshift = redshift
        self.cosmology = cosmology
        self.k = k
        self.default_params_dict = {'f': 0., 'b': 2., 'alpha_par': 1., 'alpha_perp': 1., 'sigma_par': 8., 'sigma_perp': 3., 'sigma_s': 4.}
        self.params_labels = {'f': r'$f$', 'b': r'$b$', 'alpha_par': r'$\alpha_{\parallel}$', 'alpha_perp': r'$\alpha_{\perp}$', 'sigma_par': r'$\Sigma_{\parallel}$', 'sigma_perp': r'$\Sigma_{\perp}$', 'sigma_s': r'$\Sigma_{s}$'}
        self.fitted = False
        
        if split:
            self.std = np.array_split(np.array(np.array_split(np.diag(cov_split)**0.5, nells)), nsplits, axis=1)
        else:
            self.std = np.array_split(np.diag(cov)**0.5, self.nells)
            
    def set_k(self, k):
        self.k = k
        
    def set_default_params(self, **params):
        for key in params.keys():
            self.default_params_dict[key] = params[key]
            
    def set_s_lower_limit(self, s_lower_limit):
        self.s_lower_limit = s_lower_limit
        
    def extract_split(self, split_index):
        xiell = self.xiell[split]
        cov = extract_subcovmatrix(self.sep, self.cov, self.ells, slef.nsplits, split_extract=split_index)
        std = np.array_split(np.diag(cov)**0.5, nells)
        return xiell, cov, std
    
    def set_pk_model(self):            
        fo = Fourier(self.cosmology, engine='camb')
        pk = fo.pk_interpolator(nonlinear=False, extrap_kmin=1e-10, extrap_kmax=1e4).to_1d(z=self.redshift)
        self.pk = pk
        
        pk_smooth = PowerSpectrumBAOFilter(pk, engine='wallish2018').smooth_pk_interpolator()
        self.pk_smooth = pk_smooth
        
    def _pk_model_mu(self, mu, bao_peak=True, **model_params):
        """
        Computes power spectrum model
        """
        for key in self.default_params_dict.keys():
            if not key in model_params.keys():
                if key == 'alpha_perp' and 'alpha_par' in model_params.keys():
                    model_params[key] = model_params['alpha_par']
                if key == 'alpha_par' and 'alpha_perp' in model_params.keys():
                    model_params[key] = model_params['alpha_perp']
                if key == 'sigma_par' and 'sigma_perp' in model_params.keys():
                    model_params[key] = model_params['sigma_perp']
                if key == 'sigma_perp' and 'sigma_par' in model_params.keys():
                    model_params[key] = model_params['sigma_par']
                else:
                    model_params[key] = self.default_params_dict[key]

        mu_AP, k_AP = AP_dilation(mu, self.k, model_params['alpha_par'], model_params['alpha_perp'])
                
        if bao_peak:
            pk_peak = pk_func(self.pk, k_AP) - pk_func(self.pk_smooth, k_AP)
        else:
            pk_peak = 0
        pk_model = bias_damping_func(mu_AP, k_AP, model_params['f'], model_params['b'], model_params['sigma_s']) * (pk_func(self.pk_smooth, k_AP) + BAO_damping_func(mu_AP, k_AP, model_params['sigma_par'], model_params['sigma_perp']) * pk_peak)

        return pk_model/(model_params['alpha_par']*model_params['alpha_perp']**2)
    
    def _pk_model_poles(self, bao_peak=True, **model_params):
        """
        Computes multipoles of the model from smoothed power spectrum pk_smooth and original power spectrum pk, for a given array of scales k
        """
        mu = np.linspace(0, 1, 101)
        pkmu = self._pk_model_mu(mu, bao_peak=bao_peak, **model_params)
        poles = to_poles(pkmu, self.ells)
        return poles

    def model(self, s=None, pk_model_params=None, broadband_coeffs=None, s_lower_limit=None, bao_peak=True, negative=False):
        if s is None:
            s = self.sep
        if s_lower_limit is not None:
            s = s[s>lower_s_limit]

        if pk_model_params is None:
            pk_model_params = self.default_params_dict
        if broadband_coeffs is None or len(broadband_coeffs)==0:
            broadband_coeffs = np.tile([0., 0., 0.], self.nells)

        pk_model = self._pk_model_poles(bao_peak=bao_peak, **pk_model_params)

        xi_model = xi_model_poles_interp(self.k, pk_model, self.ells)

        broadbands = np.concatenate([broadband(s, broadband_coeffs[3*ill:3*(ill+1)]) for ill in range(self.nells)])
        xi_model_poles = np.concatenate([xi_model[ill](s) for ill in range(self.nells)])
        
        if negative:
            return broadbands - xi_model_poles
        else:
            return broadbands + xi_model_poles
    
    def fit(self, fit_params_init, s_lower_limit=None, print_output=True, bao_peak=True, negative=False, fit_method='scipy', minos=False):
        fit_params_init_copy = copy.deepcopy(fit_params_init)
        
        self.s_lower_limit = s_lower_limit
        if s_lower_limit is not None:
            s, xiell, cov = truncate_xiell(s_lower_limit, self.sep, self.xiell, self.ells, self.cov)
        else:
            s = self.sep
            xiell = self.xiell
            cov = self.cov
            
        fit_params_names = list(fit_params_init_copy.keys())
        
        if 'broadband_coeffs' in fit_params_names:
            broadband_coeffs_init = fit_params_init_copy.pop('broadband_coeffs')
            ncoeffs = len(broadband_coeffs_init)
            fit_params_init_values = np.concatenate((list(fit_params_init_copy.values()), broadband_coeffs_init))
            fit_params_names = list(fit_params_init_copy.keys())
            for i in range(len(broadband_coeffs_init)):
                fit_params_names.append('broadband_coeff'+str(i))
        else:
            ncoeffs = 0
            fit_params_init_values = list(fit_params_init_copy.values())
            
        nparams = len(fit_params_names)
        model_params_names = fit_params_names[0:(nparams-ncoeffs)]
        
        def fitting_func(s, *fit_params):
            model_params = fit_params[0:(nparams-ncoeffs)]
            model_params_dict = {key: value for key, value in zip(model_params_names, model_params)}
            broadband_coeffs = fit_params[nparams-ncoeffs:]
            
            res = self.model(s=s, pk_model_params=model_params_dict, broadband_coeffs=broadband_coeffs, bao_peak=bao_peak, negative=negative)
            return res
        
        # Function to minimize with iminuit
        def iminuit_chisq(*fit_params):
            model = fitting_func(s, *fit_params)
            return compute_chisq(s, xiell.flatten(), cov, model)
        
        if fit_method == 'iminuit':
            m = Minuit(iminuit_chisq, *np.array(fit_params_init_values))
            m.migrad()
            imin = m.hesse()
            
            popt = list()
            for param in imin.params:
                popt.append(param.value)
                
            pcov = np.array(imin.covariance)
            
            if minos:
                iminos = m.minos()
                minos_errors = list()
                for param in iminos.params:
                    minos_errors.append(param.merror)
            
            self.minos = minos

        if fit_method == 'scipy':
            popt, pcov = scipy.optimize.curve_fit(fitting_func, s, xiell.flatten(), sigma=cov, p0=np.array(fit_params_init_values), absolute_sigma=True)

        popt_dict = {key: value for key, value in zip(model_params_names, popt[0:(nparams-ncoeffs)])}
        if ncoeffs > 0:
            popt_dict.update({'broadband_coeffs': popt[nparams-ncoeffs:]})

        self.popt = popt
        self.pcov = pcov
        self.popt_dict = copy.deepcopy(popt_dict)
        if 'broadband_coeffs' in popt_dict.keys():
            self.broadband_coeffs = popt_dict.pop('broadband_coeffs')
        self.model_popt_dict = popt_dict
        
        if minos:
            minos_err_dict = {key: value for key, value in zip(model_params_names, minos_errors[0:(nparams-ncoeffs)])}
            if ncoeffs > 0:
                minos_err_dict.update({'broadband_coeffs': minos_errors[nparams-ncoeffs:]})
                
            self.minos_errors = minos_err_dict

        if print_output:
            print('Optimal parameters:')
            print(self.popt_dict)

            print('\nCovariance matrix:')
            print(pcov)

            print('\nSigmas:')
            print(np.diag(pcov)**0.5)
            
            if minos:
                print('\nMinos errors:')
                print(self.minos_errors)

        self.fitted = True
        self.bao_peak = bao_peak
        self.negative = negative
        self.chi_square()

        return self.popt_dict, pcov
    
    def chi_square(self, reduced=True, bao_peak=True, negative=False):
        if hasattr(self, 's_lower_limit') and self.s_lower_limit is not None:
            s, xiell, cov = truncate_xiell(self.s_lower_limit, self.sep, self.xiell, self.ells, self.cov)
        else:
            s = self.sep
            xiell = self.xiell
            cov = self.cov
            
        if self.fitted:
            if hasattr(self, 'broadband_coeffs'):
                model = self.model(s=s, pk_model_params=self.model_popt_dict, broadband_coeffs=self.broadband_coeffs, bao_peak=self.bao_peak, negative=self.negative)
            else:
                model = self.model(s=s, pk_model_params=self.model_popt_dict, bao_peak=self.bao_peak, negative=self.negative)
            ndof = len(s)*self.nells-len(self.popt)
        else:
            model = self.model(s=s, pk_model_params=self.default_params_dict, bao_peak=bao_peak, negative=negative)
            ndof = len(s)*self.nells
            
        chisq = compute_chisq(np.tile(s, self.nells), xiell.ravel(), cov, model)
        self.chisq = chisq
        
        rchisq = chisq/ndof
        self.ndof = ndof
        self.rchisq = rchisq
            
        if reduced:
            return rchisq
        else:
            return chisq
            
    def plot_model(self, fig=None, ax=None, plot_data=False, show_broadband=False, show_info=False, bao_peak=True, negative=False):
        if ax is None:
            ax=plt.gca()
            
        s = self.sep
        if self.fitted:
            if hasattr(self, 'broadband_coeffs'):
                model = self.model(s=s, pk_model_params=self.model_popt_dict, broadband_coeffs=self.broadband_coeffs, bao_peak=self.bao_peak, negative=self.negative)
                bbd = [broadband(s, self.broadband_coeffs[ill*3:(ill+1)*3]) for ill in range(self.nells)]
            else:
                model = self.model(s=s, pk_model_params=self.model_popt_dict, bao_peak=self.bao_peak, negative=self.negative)
            free_params = self.model_popt_dict
            fixed_params = {pname: value for pname, value in zip(self.default_params_dict.keys(), self.default_params_dict.values()) if pname not in free_params.keys()}
            if 'alpha_par' in free_params.keys() or 'alpha_perp' in free_params.keys():
                fixed_params.pop('alpha_par', None)
                fixed_params.pop('alpha_perp', None)
            if 'sigma_par' in free_params.keys() or 'sigma_perp' in free_params.keys():
                fixed_params.pop('sigma_par', None)
                fixed_params.pop('sigma_perp', None)
        else:
            model = self.model(bao_peak=bao_peak, negative=negative)
            bbd = 0
            fixed_params = self.default_params_dict.keys()
                    
        if self.nells == 1:
            ax.plot(s, s**2 * model, ls='--', color='C0')
            if plot_data:
                ax.errorbar(s, s**2 * self.xiell[0], s**2 * self.std[0], fmt='-', color='C0')
                # For legend
                ax.errorbar([], [], [], linestyle='-', color='C0', label='Data')
                ax.plot([], [], linestyle='--', color='C0', label='Model')
                ax.legend()
            ax.set_ylabel(r'$s^2\xi_{}(s)$'.format(self.ells[0])+r'[$(\mathrm{Mpc}/h)^{2}$]')
            if show_broadband and 'broadband_coeffs' in self.popt_dict.keys():
                ax.plot(s, s**2 * bbd[0], ls='dotted', color='C0', label='Broadband')
                ax.legend()
                      
        else:
            for ill, ell in enumerate(self.ells):
                ax.plot(s, s**2 * model[ill*len(s):(ill+1)*len(s)], ls='--', color='C'+str(ill), label='$\ell = {:d}$'.format(ell))
                if plot_data:
                    ax.errorbar(s, s**2 * self.xiell[ill], s**2 * self.std[ill], fmt='-', color='C'+str(ill))
                if show_broadband:
                    ax.plot(s, s**2 * bbd[ill], ls='dotted', color='C'+str(ill))
                ax.set_ylabel(r'$s^{2}\xi(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
            if plot_data:
                # For legend
                ax.errorbar([], [], [], linestyle='-', color='black', label='Data')
                ax.plot([],[], linestyle='--', color='black', label='Model')
            if show_broadband and 'broadband_coeffs' in self.popt_dict.keys():
                ax.plot([], [], linestyle='dotted', color='black', label='Broadband')
            ax.legend()

        ax.set_xlabel('$s$ [$\mathrm{Mpc}/h$]')
        ax.grid(True)
        
        if hasattr(self, 's_lower_limit') and self.s_lower_limit is not None:
            ax.axvline(self.s_lower_limit, linestyle='dashed', color='r')
            
        if show_info:
            if self.fitted:
                if hasattr(self, 'minos') and self.minos:
                    minos_errors_string = '\nMinos errors:\n'+'\n'.join([r'{}: err- {:.2e}, err+ {:.2e}'.format(self.params_labels[pname], merror[0], merror[1]) for pname, merror in zip(free_params.keys(), self.minos_errors.values())])
                    
                else:
                    minos_errors_string = ''
                    
                plt.suptitle('Fit: $s$ > {:.0f} Mpc/$h$, '.format(self.s_lower_limit) 
                             + r'$\chi^2_{r}$=' +'{:.2e}'.format(self.rchisq)
                             + '\n\nFree parameters:\n' + '\n'.join([r'{}: {:.3e} $\pm$ {:.1e}'.format(self.params_labels[pname], value, std) for pname, value, std in zip(free_params.keys(), free_params.values(), np.diag(self.pcov)**0.5)])
                             + minos_errors_string
                             + '\n\nFixed parameters:\n' + '\n'.join([r'{}: {:.1e}'.format(self.params_labels[pname], value) for pname, value in zip(fixed_params.keys(), fixed_params.values())]),
                             ha='left', x=0.1, y=0, size=14)
            else:
                plt.suptitle('Fit: $s$ > {:.0f} Mpc/$h$, '.format(self.s_lower_limit) 
                             + r'$\chi^2_{r}$=' +'{:.2e}'.format(self.rchisq)
                             + '\n\nFixed parameters:\n' + '\n'.join([r'{}: {:.1e}'.format(self.params_labels[pname], value) for pname, value in zip(fixed_params.keys(), fixed_params.values())]),
                             ha='left', x=0.1, y=0, size=14)
                                                                    
                                                    
def plot_likelihood(pk_model, param_name, param_values, free_params_init, s_lower_limit=None, without_peak=True, fig=None, ax=None):
    pk_model.set_s_lower_limit(s_lower_limit)
    default_params = pk_model.default_params_dict
    
    if ax is None:
        ax=plt.gca()

    def compute_chi2(param_value, bao_peak=True):
        param_dict = {param_name: param_value}
        if param_name == 'alpha_par' or param_name == 'alpha_perp':
            param_dict.update({'alpha_par': param_value, 'alpha_perp': param_value})
        if param_name == 'sigma_par' or param_name == 'sigma_perp':
            param_dict.update({'sigma_par': param_value, 'sigma_perp': param_value})
    
        pk_model.set_default_params(**param_dict)
        pk_model.fit(fit_params_init=free_params_init, s_lower_limit=s_lower_limit, print_output=False, bao_peak=bao_peak)
        return pk_model.chisq

    if without_peak:
        chi2 = np.array([compute_chi2(param_value, bao_peak=False) for param_value in param_values])
    chi2_bao_peak = np.array([compute_chi2(param_value) for param_value in param_values])
    
    # Set default params back to their original value
    pk_model.set_default_params(**default_params)
    
    min_chi2 = np.min(chi2_bao_peak)
    nparams = len(pk_model.popt)
    conf_int = [scipy.stats.chi2.cdf(s**2, 1) for s in [1, 2, 3]]
    chi2_sigmas = [scipy.stats.chi2.ppf(ci, nparams) for ci in conf_int]
    print(chi2_sigmas)

    param_limits_idx = np.argwhere(np.diff(np.sign(chi2_bao_peak - (min_chi2+chi2_sigmas[0])))).flatten()

    if without_peak:
        ax.plot(param_values, chi2, label='Without BAO peak', color='C1')
    ax.plot(param_values, chi2_bao_peak, label='With BAO peak', color='C0')
    ax.axhline(min_chi2+chi2_sigmas[0], linestyle='dotted', color='C0')
    ax.axhline(min_chi2+chi2_sigmas[1], linestyle='dotted', color='C0')
    ax.axhline(min_chi2+chi2_sigmas[2], linestyle='dotted', color='C0')

    param_lower_bound = np.min(param_values)
    ax.text(param_lower_bound, min_chi2+chi2_sigmas[0]+0.1, r'1$\sigma$', color='C0')
    ax.text(param_lower_bound, min_chi2+chi2_sigmas[1]+0.1, r'2$\sigma$', color='C0')
    ax.text(param_lower_bound, min_chi2+chi2_sigmas[2]+0.1, r'3$\sigma$', color='C0')
    ax.set_xlabel(pk_model.params_labels[param_name])
    ax.set_ylabel(r'$\chi^2$')
    ax.set_ylim(bottom=min_chi2)
    if without_peak:
        ax.legend()
        
    d = np.diff(param_values)[0]
    ndigits = abs(int(math.log10(abs(d)))-1)

    plt.suptitle(r'Values at $\chi^2 \pm 1\sigma$: ' + ', '.join([r'{0:.{1}f} $\pm$ {2:.1e}'.format(param_values[i], ndigits, d) for i in param_limits_idx]),
                 ha='left', x=0.1, y=0, size=14)
    
    if without_peak:
        return chi2, chi2_bao_peak
    else:
        return chi2_bao_peak





    
    
