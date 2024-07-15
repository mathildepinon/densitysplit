import math
import logging 
import time
import numpy as np

from jax import random
from pycorr import setup_logging
from .utils import BaseClass
from .lognormal_model import LognormalDensityModel
from CombineHarvesterFlow import Harvest


class NormalizingFlows(BaseClass):
    """
    Class computing normalizing flows to approximate distribution from near-Gaussian input sample.
    """
    def __init__(self, data, n_flows=7, output=None):
        self.logger = logging.getLogger('NormalizingFlows')
        self.logger.info('Initializing NormalizingFlows')
        self.data = data
        self.flow = Harvest(output, chain=data, n_flows=n_flows) 
 
    def train(self):
        # train the flow
        self.flow.harvest()

    def sample(self, size=1000, seed=0):
        key = random.PRNGKey(seed)
        flow_sample_list = [np.asarray(self.flow.flow_list[i].sample(key, sample_shape=(size, ))) for i in range(self.flow.n_flows)]
        flow_sample = np.vstack(flow_sample_list)
        flow_sample_nonorm = flow_sample * self.flow.std + self.flow.mean
        return flow_sample_nonorm

    def pdf(self, x):
        flow_weight_list = [np.asarray(self.flow.flow_list[i].log_prob(x)) for i in range(self.flow.n_flows)]
        ln_weights = np.sum(np.vstack(flow_weight_list), axis=0) / self.flow.n_flows
        weights = np.exp(ln_weights - np.max(ln_weights))
        return weights / (self.flow.std[0] * self.flow.std[1])

    def save(self):
         self.flow.save_models()
