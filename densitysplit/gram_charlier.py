import math
import scipy
import logging 
import time
import numpy as np
from numpy.polynomial.hermite_e import HermiteE
from scipy.special import factorial, hermite
from scipy.stats import rv_continuous, multivariate_normal
import scipy.special as special

from pycorr import setup_logging
from .utils import BaseClass


class GramCharlier1D(BaseClass):
    """
    Class computing Gram-Charlier 1D distribution to approximate near-Gaussian distribution from input sample.
    """
    def __init__(self, sample, n=3):
        self.logger = logging.getLogger('GramCharlier1D')
        self.logger.info('Initializing GramCharlier1D distribution')
        self.sample = sample
        self.mu = np.mean(sample)
        self.sigma = np.std(sample)
        self.standsample = (sample - self.mu) / self.sigma
        self.order = n

    def hermite_proba(self, i, x):
        return 2**(-i/2) * hermite(i)(x/np.sqrt(2))
    
    def termi(self, x, i):
        GCcum = np.mean(self.hermite_proba(i, self.standsample))
        hermite_i = self.hermite_proba(i, x)
        term = GCcum * hermite_i / factorial(i)
        return term

    def pdf(self, x):
        mu = self.mu
        sigma = self.sigma
        X = (x - mu) / sigma
        gauss = multivariate_normal(0, 1).pdf(X) / sigma
        if self.order <= 2:
            return gauss
        res = 1
        for i in range(3, self.order+1):
            self.logger.info('Computing expansion term of order {}'.format(i))
            term = self.termi(X, i)
            res += term
        return gauss * res


class GramCharlier2D(BaseClass):
    """
    Class computing Gram-Charlier 2D distribution to approximate near-Gaussian distribution from input sample.
    """
    def __init__(self, sample, n=3):
        self.logger = logging.getLogger('GramCharlier2D')
        self.logger.info('Initializing GramCharlier2D distribution')
        self.xsample = sample[0]
        self.ysample = sample[1]
        self.mu = np.mean(sample, axis=1)
        self.cov = np.cov(sample)
        self.sigma = np.diag(self.cov)**0.5
        self.standxsample = (sample[0] - self.mu[0]) / self.sigma[0]
        self.standysample = (sample[1] - self.mu[1]) / self.sigma[1]
        self.order = n
        self.hermite_x = np.array([self.hermite_proba(i, self.standxsample) for i in range(n+1)])
        self.hermite_y = np.array([self.hermite_proba(i, self.standysample) for i in range(n+1)])

    def hermite_proba(self, i, x):
        return 2**(-i/2) * hermite(i)(x/np.sqrt(2))
    
    def termij(self, x, y, i, j):
        GCcum = np.mean(self.hermite_x[i]*self.hermite_y[j])
        hermite_i = self.hermite_proba(i, x)
        hermite_j = self.hermite_proba(j, y)
        term = GCcum * hermite_i * hermite_j / (factorial(i) * factorial(j))
        return term

    def pdf(self, x):
        mu = self.mu
        sigma = self.sigma
        X = (x - mu) / sigma
        gauss = multivariate_normal([0, 0], self.cov / sigma**2).pdf(X) / (sigma[0] * sigma[1])
        if self.order <= 2:
            return gauss
        res = 1
        for k in range(3, self.order+1):
            self.logger.info('Computing expansion term of order {}'.format(k))
            for i in range(k+1):
                j = k - i
                self.logger.info('i = {}, j = {}'.format(i, j))
                term = self.termij(X[..., 0], X[..., 1], i, j)
                res += term
        print('Deviation from Gaussian: {}'.format(np.mean(res-1)))
        return gauss * res
        
        



    
