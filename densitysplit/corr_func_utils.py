import os
import copy

import numpy as np
import scipy

from pycorr import TwoPointCorrelationFunction, setup_logging
from cosmoprimo import *



def get_poles(mocks_results, ells):
    """
    Get multipoles and covariance matrix of correlation function from mocks pycorr TwoPointCorrelationFunction output
    """
    nells = len(ells)
    n = len(mocks_results)
    results_poles = [np.ravel(res.get_corr(ells=ells, return_sep=False)) for res in mocks_results]
    poles = np.mean(results_poles, axis=0)
    xiell = poles.reshape((nells, len(poles)//nells))
    cov = np.cov(results_poles, rowvar=False)

    return xiell, cov


def get_split_poles(results, ells, nsplits):
    nells = len(ells)
    n = len(results)
    nsplits = len(results[0])

    xiell = list()
    cov = list()

    for i in range(nsplits):
        results_poles = [np.ravel(res[i].get_corr(ells=ells, return_sep=False)) for res in results]
        poles = np.mean(results_poles, axis=0)
        xiell.append(poles.reshape((nells, len(poles)//nells)))

    cov = np.cov([np.ravel([res[i].get_corr(ells=ells, return_sep=False) for i in range(nsplits)]) for res in results], rowvar=False)

    return xiell, cov


# Project to Legendre multipoles
def weights_trapz(x):
    """Return weights for trapezoidal integration."""
    return np.concatenate([[x[1]-x[0]], x[2:]-x[:-2], [x[-1]-x[-2]]])/2.


def to_poles(pkmu, ells, mu = np.linspace(0, 1, 101)):
    muw_trapz = weights_trapz(mu)
    weights = np.array([muw_trapz * (2*ell + 1) * scipy.special.legendre(ell)(mu) for ell in ells])/(mu[-1] - mu[0])

    return np.sum(pkmu * weights[:,None,:], axis=-1) # P0, P2, P4


def xi_model_poles_interp(k, pk_model, ells):
    """
    Computes correlation function interpolator for each pole in list ells, from power spectrum array pk_model (as returned by to_poles)
    """
    powerToCorrelation = PowerToCorrelation(k, ells, complex=False)
    sep, xi_poles = powerToCorrelation(pk_model)
    return np.array([CorrelationFunctionInterpolator1D(sep[ill], xi=xi_poles[ill]) for ill in range(len(ells))])
