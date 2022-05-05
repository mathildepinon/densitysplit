import numpy as np
import scipy
from matplotlib import pyplot as plt
import matplotlib as mpl

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
        
    def _pk_model_mu(self, mu, **model_params):
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
                
        pk_peak = pk_func(self.pk, k_AP) - pk_func(self.pk_smooth, k_AP)
        pk_model = bias_damping_func(mu_AP, k_AP, model_params['f'], model_params['b'], model_params['sigma_s']) * (pk_func(self.pk_smooth, k_AP) + BAO_damping_func(mu_AP, k_AP, model_params['sigma_par'], model_params['sigma_perp']) * pk_peak)

        return pk_model/(model_params['alpha_par']*model_params['alpha_perp']**2)
    
    def _pk_model_poles(self, **model_params):
        """
        Computes multipoles of the model from smoothed power spectrum pk_smooth and original power spectrum pk, for a given array of scales k
        """
        mu = np.linspace(0, 1, 101)
        pkmu = self._pk_model_mu(mu, **model_params)
        poles = to_poles(pkmu, self.ells)
        return poles

    def model(self, s=None, pk_model_params=None, broadband_coeffs=None, s_lower_limit=None):
        if s is None:
            s = self.sep
        if s_lower_limit is not None:
            s = s[s>lower_s_limit]

        if pk_model_params is None:
            pk_model_params = self.default_params_dict
        if broadband_coeffs is None or len(broadband_coeffs)==0:
            broadband_coeffs = np.tile([0., 0., 0.], self.nells)

        pk_model = self._pk_model_poles(**pk_model_params)

        xi_model = xi_model_poles_interp(self.k, pk_model, self.ells)

        broadbands = np.concatenate([broadband(s, broadband_coeffs[3*ill:3*(ill+1)]) for ill in range(self.nells)])
        xi_model_poles = np.concatenate([xi_model[ill](s) for ill in range(self.nells)])

        return broadbands + xi_model_poles
    
    def fit(self, fit_params_init, s_lower_limit=None):
        self.s_lower_limit = s_lower_limit
        if s_lower_limit is not None:
            s, xiell, cov = truncate_xiell(s_lower_limit, self.sep, self.xiell, self.ells, self.cov)
        else:
            s = self.sep
            xiell = self.xiell
            cov = self.cov
            
        fit_params_names = list(fit_params_init.keys())
        
        if 'broadband_coeffs' in fit_params_names:
            broadband_coeffs_init = fit_params_init.pop('broadband_coeffs')
            ncoeffs = len(broadband_coeffs_init)
            fit_params_init_values = np.concatenate((list(fit_params_init.values()), broadband_coeffs_init))
            fit_params_names = list(fit_params_init.keys())
            for i in range(len(broadband_coeffs_init)):
                fit_params_names.append('broadband_coeff'+str(i))
        else:
            ncoeffs = 0
            fit_params_init_values = list(fit_params_init.values())
            
        nparams = len(fit_params_names)
        model_params_names = fit_params_names[0:(nparams-ncoeffs)]
        
        def fitting_func(s, *fit_params):
            model_params = fit_params[0:(nparams-ncoeffs)]
            model_params_dict = {key: value for key, value in zip(model_params_names, model_params)}
            broadband_coeffs = fit_params[nparams-ncoeffs:]
            
            res = self.model(s=s, pk_model_params=model_params_dict, broadband_coeffs=broadband_coeffs)
            return res
        
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
        
        print('Optimal parameters:')
        print(self.popt_dict)

        print('\nCovariance matrix:')
        print(pcov)

        print('\nSigmas:')
        print(np.diag(pcov)**0.5)
        
        self.fitted = True
        self.chi_square()
        
        return self.popt_dict, pcov
    
    def chi_square(self, reduced=True):
        if hasattr(self, 's_lower_limit') and self.s_lower_limit is not None:
            s, xiell, cov = truncate_xiell(self.s_lower_limit, self.sep, self.xiell, self.ells, self.cov)
        else:
            s = self.sep
            xiell = self.xiell
            cov = self.cov
            
        if self.fitted:
            if hasattr(self, 'broadband_coeffs'):
                model = self.model(s=s, pk_model_params=self.model_popt_dict, broadband_coeffs=self.broadband_coeffs)
            else:
                model = self.model(s=s, pk_model_params=self.model_popt_dict)
            ndof = len(s)*self.nells-len(self.popt)
        else:
            model = self.model(s=s, pk_model_params=self.default_params_dict)
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
    
    def plot_model(self, fig=None, ax=None, plot_data=False, show_broadband=False, show_info=False):
        if ax is None:
            ax=plt.gca()
            
        s = self.sep
        if self.fitted:
            if hasattr(self, 'broadband_coeffs'):
                model = self.model(s=s, pk_model_params=self.model_popt_dict, broadband_coeffs=self.broadband_coeffs)
                bbd = [broadband(s, self.broadband_coeffs[ill*3:(ill+1)*3]) for ill in range(self.nells)]
            else:
                model = self.model(s=s, pk_model_params=self.model_popt_dict)
            params = self.model_popt_dict
        else:
            model = self.model()
            params = self.default_params_dict
            bbd = 0
                    
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
            plt.suptitle('Fit: $s$ > {:.0f} Mpc/$h$, '.format(self.s_lower_limit) 
                         + r'$\chi^2_{r}$=' +'{:.2e}'.format(self.rchisq)
                         + '\n' + '\n'.join([r'{}: {:.3e} $\pm$ {:.1e}'.format(pname, value, std) for pname, value, std in zip(params.keys(), params.values(), np.diag(self.pcov)**0.5)]),
                         ha='left', x=0.1, y=0)

        





    
    
