from os.path import exists
import copy

import numpy as np

from cosmoprimo import *
from pycorr import TwoPointCorrelationFunction, setup_logging
from mockfactory import EulerianLinearMock, LagrangianLinearMock, RandomBoxCatalog, setup_logging
from pypower.fft_power import project_to_basis

from densitysplit import catalog_data, density_split

# Set up logging
setup_logging()


# Function to damp the power spectrum for k higher than a given frequency kN
def damping_function(k, kN):
    k_lambda=0.8*kN
    sigma_lambda=0.05*kN

    if k < k_lambda:
        return 1
    else:
        return np.exp(-(k-k_lambda)**2/(2*sigma_lambda**2))


# Linear pk damped at frequency higher than Nyquist frequency of the mock
def pk_damped(cosmology, z, kN, k=np.logspace(-5, 2, 1000000)):
    pklin = cosmology.get_fourier().pk_interpolator().to_1d(z)
    pklin_array = pklin(k)
    pkdamped_func = lambda k: pklin(k) * np.array([damping_function(kk, kN) for kk in k])
    pkdamped = PowerSpectrumInterpolator1D.from_callable(k, pkdamped_func)
    return pkdamped


def generate_gaussian_mock(nmesh, boxsize, boxcenter, seed, nbar, cosmology=fiducial.AbacusSummitBase(), z=1.175, bias=1.8, rsd=False, los=None, f=None, save=False, output_dir='/', name='gaussian_mock', mpi=False):

    pk = pk_damped(cosmology, z, np.pi*nmesh/boxsize)

    mock = EulerianLinearMock(pk, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=False)
    mock.set_real_delta_field(bias=bias)

    data = RandomBoxCatalog(nbar=nbar, boxsize=boxsize, boxcenter=boxcenter, seed=seed)
    data['Weight'] = mock.readout(data['Position'], field='delta', resampler='tsc', compensate=True) + 1.
    offset = boxcenter - boxsize/2.
    positions = copy.deepcopy((data['Position'] - offset) % boxsize + offset)
    weights = copy.deepcopy(data['Weight'])

    if mpi:
        positions = positions.gather()
        weights = weights.gather()
        mock_catalog = None
        if mock.mpicomm.rank == 0:
            mock_catalog = catalog_data.Data(positions.T, z, boxsize, boxcenter, name=name, weights=weights)
            #if rsd:
                ## to complete (weights_rsd in catalog_data.Data ?)
            if save:
                mock_catalog.save(output_dir+name)

    else:
        mock_catalog = catalog_data.Data(positions.T, z, boxsize, boxcenter, name=name, weights=weights)
        #if rsd:
            ## to complete (weights_rsd in catalog_data.Data ?)
        if save:
            mock_catalog.save(output_dir+name)

    return mock_catalog


def generate_N_gaussian_mocks(nmocks, nmesh, boxsize, boxcenter, nbar, cosmology=fiducial.AbacusSummitBase(), z=1.175, bias=1.8, rsd=False, los=None, f=None, output_dir='/', name='gaussianMock', mpi=False, overwrite=True):
    for i in range(10, nmocks):
        filename = name+'_z{:.3f}_bias{:.1f}_boxsize{:d}_nmesh{:d}_nbar{:.3f}_mock{:d}'.format(z, bias, boxsize, nmesh, nbar, i)
        if not exists(output_dir+filename+'.npy') or overwrite:
            generate_gaussian_mock(nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=i,
                                     nbar=nbar, cosmology=cosmology,
                                     z=z, bias=bias, rsd=rsd, los=los, f=f,
                                     save=True, output_dir=output_dir, name=filename,
                                     mpi=mpi)
    return None


def generate_batch_2PCF(edges, nmocks, nmesh, boxsize, boxcenter, nbar,
                        cosmology=fiducial.AbacusSummitBase(), z=1.175, bias=1.8,
                        rsd=False, los=None, f=None,
                        use_weights=False,
                        nthreads=128, batch_size=None, batch_index=0,
                        output_dir='/', name='', mpi=False, overwrite=True):

    results = list()

    if batch_size is None:
        batch_size = nmocks

    mocks_indices = range(batch_index*batch_size, (batch_index+1)*batch_size)

    for i in mocks_indices:
        print('Mock '+str(i))
        filename = name+'_z{:.3f}_bias{:.1f}_boxsize{:d}_nmesh{:d}_nbar{:.3f}_mock{:d}'.format(z, bias, boxsize, nmesh, nbar, i)
        print(output_dir+filename+'.npy')
        if exists(output_dir+filename+'.npy'):
            print('Mock already exists. Loading mock...')
            mock_catalog = catalog_data.Data.load(output_dir+filename+'.npy')
        else:
            print('Mock does not exist')

        if mock_catalog is not None:
            print('Computing correlation function...')
            if rsd:
                positions = mock_catalog.positions_rsd
            else:
                positions = mock_catalog.positions

            if use_weights:
                weights = mock_catalog.weights
            else:
                weights = None

            result = TwoPointCorrelationFunction('smu', edges,
                                                data_positions1=positions, data_weights1=weights,
                                                boxsize=mock_catalog.boxsize,
                                                engine='corrfunc', nthreads=nthreads,
                                                los=los)

            results.append(result)

    return results


def main():
    # Output directory for generated mocks
    output_dir = '/feynman/work/dphp/mp270220/outputs/mocks/gaussian/'

    # Mock parameters
    boxsize = 2000
    boxcenter = 1000
    nmesh = 1024
    nbar = 0.01
    cosmology=fiducial.AbacusSummitBase()
    z = 1.175
    bias = 1.8

    # Mocks
    nmocks = 10

    # For RSD
    # bg = cosmology.get_background()
    # f = bg.growth_rate(z)
    # hz = 100*bg.efunc(z)

    # Edges (s, mu) to compute correlation function at
    edges = (np.linspace(0., 150., 151), np.linspace(-1, 1, 201))
    los = 'x'

    # generate_N_gaussian_mocks(nmocks, nmesh, boxsize, boxcenter, nbar,
    #                  cosmology=cosmology, z=z, bias=bias, rsd=False, los=None, f=None,
    #                  output_dir=output_dir, name='gaussianMock_pkdamped',
    #                  mpi=True, overwrite=True)

    results = generate_batch_2PCF(edges, nmocks, nmesh, boxsize, boxcenter,
                                 nbar, cosmology=cosmology, z=z, bias=bias,
                                 los=los, rsd=False, use_weights=True,
                                 nthreads=128, output_dir=output_dir, name='gaussianMock_pkdamped')

    name = '{:d}gaussianMocks_pkdamped_z{:.3f}_bias{:.1f}_boxsize{:d}_nmesh{:d}_nbar{:.3f}'.format(nmocks, z, bias, boxsize, nmesh, nbar)
    output_dir = '/feynman/work/dphp/mp270220/outputs/correlation_functions/'
    np.save(output_dir+name+'_2PCF', results)

if __name__ == "__main__":
    main()
