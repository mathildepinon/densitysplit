from os.path import exists
import sys
import copy

import numpy as np

from cosmoprimo import *
from pycorr import TwoPointCorrelationFunction, setup_logging
from mockfactory import EulerianLinearMock, LagrangianLinearMock, RandomBoxCatalog, setup_logging
from pypower.fft_power import project_to_basis

from densitysplit import catalog_data, density_split

# Set up logging
setup_logging()


def compute_2PCF(data, edges, seed=0, use_rsd=False, los=None, hz=None, use_weights=False, nthreads=128):
    if use_rsd:
        if data.positions_rsd is None:
            data.set_rsd(hz=hz, los=los)
        positions = data.positions_rsd
    else:
        positions = data.positions

    if use_weights and (data.weights is not None):
        weights = data.weights
    else:
        weights = None

    result = TwoPointCorrelationFunction('smu', edges,
                                        data_positions1=positions, data_weights1=weights,
                                        boxsize=data.boxsize,
                                        engine='corrfunc', nthreads=nthreads,
                                        los = los)

    return result

def main():
    # Mock parameters
    boxsize = 2000
    boxcenter = 1000
    nmesh = 1024
    cosmology=fiducial.AbacusSummitBase()

    # For RSD
    bg = cosmology.get_background()
    # f = bg.growth_rate(z)
    z = 0.8
    hz = 100*bg.efunc(z)

    name = 'AbacusSummit_2Gpc_z{:.3f}_downsampled_particles_nbar0.003'.format(z)
    abacus_mock = catalog_data.Data.load('/feynman/work/dphp/mp270220/data/'+name+'.npy')
                
    # Edges (s, mu) to compute correlation function at
    edges = (np.linspace(0., 150., 151), np.linspace(-1, 1, 201))
    los = 'x'
    
    mock_2PCF = compute_2PCF(abacus_mock, edges, seed=0, use_rsd=True, los=los, hz=hz, use_weights=False, nthreads=64)

    output_dir = '/feynman/work/dphp/mp270220/outputs/correlation_functions/'
    np.save(output_dir+name+'_2PCF_RSD', mock_2PCF)


if __name__ == "__main__":
    main()
