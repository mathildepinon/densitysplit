import numpy as np
import scipy
import scipy.stats
from matplotlib import pyplot as plt
import matplotlib as mpl
import math
import re

from iminuit import Minuit
from pycorr import TwoPointCorrelationFunction, TwoPointEstimator, NaturalTwoPointEstimator, project_to_multipoles, project_to_wp, utils, setup_logging
from cosmoprimo import *

from .utils import *


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


class BAOModel:
    """
    Class for power spectrum model, based on model from Bautista et. al (2020). 
    """

    def __init__(self, sep, k, ells, redshift, cosmology, iso=True, nsplits=1, 
                 model_params=None, model_params_labels=None, signature=None):

        self.sep = sep
        self.k = k
        self.ells = ells
        self.nells = len(ells)
        self.redshift = redshift
        self.cosmology = cosmology
        self.iso = iso
        self.bao_peak = True
        self.nsplits = nsplits
        
        if model_params is not None:
            self.model_params = model_params
            if model_params_labels is not None:
                self.model_params_labels = model_params_labels
        else:            
            model_params = {'f': 0., 'sigma_s': 4.}            
            model_params_labels = {'f': r'$f$', 'sigma_s': r'$\Sigma_{s}$'}
            
            if self.iso:
                model_params.update({'alpha_iso': 1., 'sigma_iso': 5})
                model_params_labels.update({'alpha_iso': r'$\alpha_{iso}$', 'sigma_iso': r'$\Sigma_{iso}$'})
                
            else:
                model_params.update({'alpha_par': 1., 'alpha_perp': 1., 'sigma_par': 8., 'sigma_perp': 3.})
                model_params_labels.update({'alpha_par': r'$\alpha_{\parallel}$', 'alpha_perp': r'$\alpha_{\perp}$', 'sigma_par': r'$\Sigma_{\parallel}$', 'sigma_perp': r'$\Sigma_{\perp}$'})
            
            # Bias parameters
            for i in range(self.nsplits):
                model_params.update({'b_DS'+str(i+1): 2.})
                model_params_labels.update({'b_DS'+str(i+1): r'$b\,(DS{split})$'.format(split=i+1)})
                
            # Broadband coefficients
            for i in range(self.nsplits):
                for j in range(self.nells):
                    model_params.update({'bb{}_{}_DS{}'.format(k, j, i+1): 0 for k in range(3)})
                    model_params_labels.update({'bb{}_{}_DS{}'.format(k, j, i+1): r'$a^{k}_{{\ell = {ell}}}\,(DS{split})$'.format(k=k, ell=self.ells[j], split=i+1) for k in range(3)})
                    
        self.model_params = model_params
        self.set_params(**model_params)
        self.model_params_labels = model_params_labels
            
        if signature is None:
            self.signature = [False for split in range(self.nsplits)]
        else:
            self.signature = signature
            
        self.set_pk_model()
    
    def set_sep(self, s):
        self.sep = s
        
    def set_k(self, k):
        self.k = k
        
    def set_params(self, **params):
        for key in params.keys():
            self.model_params[key] = params[key]
            
        broadband_coeffs = list()
        for ill in range(self.nells):
            for split in range(self.nsplits):
                for k in range(3):
                    if 'bb{}_{}_DS{}'.format(k, ill, split+1) in self.model_params.keys():
                        broadband_coeffs.append(self.model_params['bb{}_{}_DS{}'.format(k, ill, split+1)])
                    else:
                        broadband_coeffs.append(0)
        broadband_coeffs = np.array(broadband_coeffs)
        self.broadband_coeffs = broadband_coeffs

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
        for key in self.model_params.keys():
            if not key in model_params.keys():
                model_params[key] = self.model_params[key]
                
        if self.iso:
            model_params['alpha_par'] = model_params['alpha_iso']
            model_params['alpha_perp'] = model_params['alpha_iso']
            model_params['sigma_par'] = model_params['sigma_iso']
            model_params['sigma_perp'] = model_params['sigma_iso']
        
        mu_AP, k_AP = AP_dilation(mu, self.k, model_params['alpha_par'], model_params['alpha_perp'])
                
        if self.bao_peak:
            pk_peak = self.pk(k=k_AP) - self.pk_smooth(k=k_AP)
        else:
            pk_peak = 0
        pk_model = bias_damping_func(mu_AP, k_AP, model_params['f'], model_params['b'], model_params['sigma_s']) * (self.pk_smooth(k=k_AP) + BAO_damping_func(mu_AP, k_AP, model_params['sigma_par'], model_params['sigma_perp']) * pk_peak)

        return pk_model/(model_params['alpha_par']*model_params['alpha_perp']**2)
    
    def _pk_model_poles(self, **model_params):
        """
        Computes multipoles of the model from smoothed power spectrum pk_smooth and original power spectrum pk, for a given array of scales k
        """
        mu = np.linspace(0, 1, 101)
        pkmu = self._pk_model_mu(mu, **model_params)
        poles = to_poles(pkmu, self.ells)
        return poles

    def model(self, s=None, model_params=None, negative=False):
        if s is None:
            s = self.sep

        if model_params is None:
            model_params = self.model_params

        pk_model = self._pk_model_poles(**model_params)

        xi_model = xi_model_poles_interp(self.k, pk_model, self.ells)

        broadbands = np.concatenate([broadband(s, self.broadband_coeffs[3*ill:3*(ill+1)]) for ill in range(self.nells)])
        xi_model_poles = np.concatenate([xi_model[ill](s) for ill in range(self.nells)])
        
        if negative:
            return broadbands - xi_model_poles
        else:
            return broadbands + xi_model_poles
        
    def split_model(self, s=None, params=None):
        if s is None:
            s = self.sep
            
        if params is not None:
            self.set_params(**params)
        else:
            params = copy.deepcopy(self.model_params)

        split_models_list = list()
        
        common_params = {key: val for (key, val) in params.items() if not 'DS' in key}
        
        for split in range(self.nsplits):
            split_specific_params = {re.sub('_DS{}'.format(split+1), '', key): val for (key, val) in params.items() if '_DS{}'.format(split+1) in key}
            
            split_params = copy.deepcopy(common_params)
            split_params.update(split_specific_params)

            split_model = self.model(s=s, model_params=split_params, negative=self.signature[split])
            split_models_list.append(split_model)
        
        return np.concatenate(split_models_list) 

    def plot_split_model(self, fig=None, axes=None, show_broadband=False, show_info=False):
        if axes is None:
            axes=[plt.gca() for i in range(self.nsplits)]
            
        s = self.sep
        ns = len(s)
        
        split_model = self.split_model(s=s)
        models = np.array_split(split_model, self.nsplits)
        bb_coeffs = np.array_split(self.broadband_coeffs, self.nsplits)
                
        for split in range(self.nsplits):
            if self.nsplits == 1:
                ax = axes
            else:
                ax = axes[split]
            
            for ill, ell in enumerate(self.ells):
                ax.plot(s, s**2 * models[split][ill*ns:(ill+1)*ns], ls='--', color='C'+str(ill), label='$\ell = {:d}$'.format(ell))

                if show_broadband:
                    coeffs = bb_coeffs[split][3*ill:3*(ill+1)]
                    bb = broadband(s, coeffs)
                    ax.plot(s, s**2 * bb, ls='dotted', color='C'+str(ill))

                ax.set_ylabel(r'$s^{2}\xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')

            ax.set_xlabel('$s$ [$\mathrm{Mpc}/h$]')
            ax.grid(True)
                
            ax.set_title('DS{}'.format(split+1))
        
        ax.plot([],[], linestyle='--', color='black', label='Model')
        ax.plot([], [], linestyle='dotted', color='black', label='Broadband')
        ax.legend()



    
    
