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


def compute_delta_R(mocks, cellsize, resampler, use_rsd=False, use_weights=False):
    nmocks = len(mocks)
    mocks_densities = list()
    for i in range(nmocks):
        mock_catalog = mocks[i]
        mock_density = density_split.DensitySplit(mock_catalog)
        mock_density.compute_density(cellsize=cellsize, resampler=resampler, use_rsd=use_rsd, use_weights=use_weights)
        ## Generate random particles and readout density at each particle
        rng = np.random.RandomState(seed=i)
        positions = [o + rng.uniform(0., 1., mock_density.data.size)*b for o, b in zip((mock_density.offset,)*3, (mock_density.boxsize,)*3)]
        shifted_positions = np.array(positions) - mock_density.offset
        densities = mock_density.density_mesh.readout(shifted_positions.T, resampler=resampler)
        mocks_densities.append(densities)
    return mocks_densities


def compute_xi_R(data_density, edges, seed=0, los=None, use_rsd=False, use_weights=False, nthreads=128):
    data = data_density.data

    if use_rsd and data.positions_rsd is not None:
        positions2 = data.positions_rsd
    else:
        positions2 = data.positions

    if use_weights and (data.weights is not None):
        weights2 = data.weights
    else:
        weights2 = None

    ## Generate random particles and readout density at each particle
    rng = np.random.RandomState(seed=seed)
    positions1 = [o + rng.uniform(0., 1., data_density.data.size)*b for o, b in zip((data_density.offset,)*3, (data_density.boxsize,)*3)]
    shifted_positions1 = np.array(positions1) - data_density.offset
    densities = data_density.density_mesh.readout(shifted_positions1.T, resampler=data_density.resampler)

    weights1 = 1 + densities

    result = TwoPointCorrelationFunction('smu', edges,
                                        data_positions1=positions1, data_positions2=positions2,
                                        data_weights1=weights1, data_weights2=weights2,
                                        boxsize=data_density.boxsize,
                                        engine='corrfunc', nthreads=nthreads,
                                        los = los)

    return result


def generate_batch_xi_R(edges, nmocks, nmesh, boxsize, boxcenter,
                        nbar, cosmology, z, bias,
                        cellsize, resampler, los=None, rsd=False, use_rsd=False, use_weights=False,
                        nthreads=128, batch_size=None, batch_index=0,
                        output_dir='/', name=''):

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
            print('Mock does not exist.')

        if mock_catalog is not None:
            print('Computing density...')
            mock_density = density_split.DensitySplit(mock_catalog)
            mock_density.compute_density(cellsize=cellsize, resampler=resampler, use_rsd=use_rsd, use_weights=use_weights)

            print('Computing correlation function...')
            mock_xi_R = compute_xi_R(mock_density, edges, seed=i, los=los, use_rsd=rsd, use_weights=use_weights, nthreads=nthreads)

            results.append(mock_xi_R)

    return results


def main():
    # Output directory for generated mocks
    # output_dir = '/feynman/work/dphp/mp270220/outputs/mocks/gaussian/'
    output_dir = '/feynman/work/dphp/mp270220/outputs/mocks/lognormal/'

    # Mock parameters
    boxsize = 2000
    boxcenter = 1000
    nmesh = 1024
    #nbar = 0.003
    cosmology=fiducial.AbacusSummitBase()
    #z = 1.175
    #bias = 1.8

    # Mocks
    nmocks = 10

    # For RSD
    # bg = cosmology.get_background()
    # f = bg.growth_rate(z)
    # hz = 100*bg.efunc(z)

    # Density smoothing parameters
    cellsize = 10
    resampler = 'tsc'

    #mocks_list = list()
    #name = 'gaussianMock_pkdamped_z{:.3f}_bias{:.1f}_boxsize{:d}_nmesh{:d}_nbar{:.3f}'.format(z, bias, boxsize, nmesh, nbar)
    #name = 'lognormal_mock_z{:.3f}_bias{:.1f}_boxsize{:d}_nmesh{:d}_nbar{:.3f}'.format(z, bias, boxsize, nmesh, nbar)

    #for i in range(nmocks):
    #    filename = name+'_mock{:d}'.format(i)
    #    mock = catalog_data.Data.load(output_dir+filename+'.npy')
    #    mocks_list.append(mock)

    #densities = compute_delta_R(mocks_list, cellsize, resampler, use_rsd=False, use_weights=True)
    #name = '{:d}gaussianMocks_pkdamped_z{:.3f}_bias{:.1f}_boxsize{:d}_nmesh{:d}_nbar{:.3f}'.format(nmocks, z, bias, boxsize, nmesh, nbar)
    #name = '{:d}_lognormal_mocks_z{:.3f}_bias{:.1f}_boxsize{:d}_nmesh{:d}_nbar{:.3f}'.format(nmocks, z, bias, boxsize, nmesh, nbar)
    #np.save(output_dir+name+'_cellsize{:d}_resampler'.format(cellsize)+resampler+'_delta_R', densities)
    
    abacus_mock = catalog_data.Data.load('/feynman/work/dphp/mp270220/data/AbacusSummit_2Gpc_z1.175_ph003.npy')
    abacus_density = compute_delta_R([abacus_mock], cellsize, resampler, use_rsd=False, use_weights=False)

    np.save('/feynman/work/dphp/mp270220/data/AbacusSummit_2Gpc_z1.175_ph003_cellsize{:d}_resampler{}_delta_R'.format(cellsize, resampler), abacus_density[0])
                
    # Edges (s, mu) to compute correlation function at
    edges = (np.linspace(0., 150., 151), np.linspace(-1, 1, 201))
    los = 'x'
    
    mock_density = density_split.DensitySplit(abacus_mock)
    mock_density.compute_density(cellsize=cellsize, resampler=resampler, use_rsd=False, use_weights=False)
    print('Computing correlation function...')
    mock_xi_R = compute_xi_R(mock_density, edges, seed=0, los=los, use_rsd=False, use_weights=False, nthreads=64)

    #batch_index = int(sys.argv[1])
    #batch_size = 1

    #results = generate_batch_xi_R(edges, nmocks, nmesh, boxsize, boxcenter,
    #                              nbar, cosmology=cosmology, z=z, bias=bias,
    #                              cellsize=cellsize, resampler=resampler,
    #                              los=los, rsd=False, use_rsd=False, use_weights=True,
    #                              nthreads=64, batch_size=batch_size, batch_index=batch_index,
    #                              output_dir=output_dir, name='lognormal_mock')
    
    # name = '{:d}gaussianMocks_batch{:d}_pkdamped_z{:.3f}_bias{:.1f}_boxsize{:d}_nmesh{:d}_nbar{:.3f}'.format(batch_size, batch_index, z, bias, boxsize, nmesh, nbar)
    # name = 'lognormalMock{:d}_z{:.3f}_bias{:.1f}_boxsize{:d}_nmesh{:d}_nbar{:.3f}'.format(batch_index, z, bias, boxsize, nmesh, nbar)
    output_dir = '/feynman/work/dphp/mp270220/outputs/correlation_functions/'
    #np.save(output_dir+name+'_cellsize{:d}_resampler{}'.format(cellsize, resampler)+'_xi_R', results)
    np.save(output_dir+'AbacusSummit_2Gpc_z1.175_ph003'+'_cellsize{:d}_resampler{}'.format(cellsize, resampler)+'_xi_R', mock_xi_R)


if __name__ == "__main__":
    main()
