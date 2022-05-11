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