import numpy as np
import scipy

from pycorr import TwoPointCorrelationFunction, TwoPointEstimator, NaturalTwoPointEstimator, project_to_multipoles, project_to_wp, utils, setup_logging
from cosmoprimo import *

# Parameters used in power spectrum model
sigma_s = 4
sigma_par = 8
sigma_perp = 3

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

def pk_model_mu(mu, k, pk_smooth, pk, f, b, alpha_par=1, alpha_perp=None, sigma_par=sigma_par, sigma_perp=sigma_perp, sigma_s=sigma_s):
    """
    Computes power spectrum model
    """
    if alpha_perp is None:
        alpha_perp = alpha_par
    
    mu_AP, k_AP = AP_dilation(mu, k, alpha_par, alpha_perp)
    
    pk_peak_func = lambda kk: pk_func(pk, kk) - pk_func(pk_smooth, kk)
    pk_model_func = lambda kk: bias_damping_func(mu_AP, kk, f, b, sigma_s) * (pk_func(pk_smooth, kk) + BAO_damping_func(mu_AP, kk, sigma_par, sigma_perp) * pk_peak_func(kk))
    
    return pk_model_func(k_AP)/(alpha_par*alpha_perp**2)

# Project to Legendre multipoles
def weights_trapz(x):
    """Return weights for trapezoidal integration."""
    return np.concatenate([[x[1]-x[0]], x[2:]-x[:-2], [x[-1]-x[-2]]])/2.

def to_poles(pkmu, ells):
    mu = np.linspace(0, 1, 101)
    muw_trapz = weights_trapz(mu)
    weights = np.array([muw_trapz * (2*ell + 1) * scipy.special.legendre(ell)(mu) for ell in ells])/(mu[-1] - mu[0])

    return np.sum(pkmu * weights[:,None,:], axis=-1) # P0, P2, P4

def pk_model_poles(k, pk_smooth, pk, ells, f, b, **kwargs):
    """
    Computes multipoles of the model from smoothed power spectrum pk_smooth and original power spectrum pk, for a given array of scales k
    """
    mu = np.linspace(0, 1, 101)
    pkmu = pk_model_mu(mu, k, pk_smooth, pk, f, b, **kwargs)
    poles = to_poles(pkmu, ells)
    return poles

def xi_model_poles_interp(k, pk_model, ells):
    """
    Computes correlation function interpolator for each pole in list ells, from power spectrum array pk_model (as returned by pk_model_poles)
    """
    powerToCorrelation = PowerToCorrelation(k, ells, complex=False)
    sep, xi_poles = powerToCorrelation(pk_model)
    return np.array([CorrelationFunctionInterpolator1D(sep[ill], xi=xi_poles[ill]) for ill in range(len(ells))])