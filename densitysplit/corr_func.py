import os
import copy

import numpy as np
from pycorr import TwoPointCorrelationFunction, setup_logging



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