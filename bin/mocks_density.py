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


def compute_delta_R(mocks, cellsize, resampler, use_rsd=False, los=None, hz=None, use_weights=False):
    nmocks = len(mocks)
    mocks_densities = list()
    for i in range(nmocks):
        mock_catalog = mocks[i]
        mock_density = density_split.DensitySplit(mock_catalog)
        mock_density.compute_density(cellsize=cellsize, resampler=resampler, use_rsd=use_rsd, los=los, hz=hz, use_weights=use_weights)
        ## Generate random particles and readout density at each particle
        rng = np.random.RandomState(seed=i)
        positions = [o + rng.uniform(0., 1., mock_density.data.size)*b for o, b in zip((mock_density.offset,)*3, (mock_density.boxsize,)*3)]
        shifted_positions = np.array(positions) - mock_density.offset
        densities = mock_density.density_mesh.readout(shifted_positions.T, resampler=resampler)
        mocks_densities.append(densities)
    return mocks_densities


def compute_jointPDF_delta_R1_R2(mock, cellsize1, cellsize2, resampler, s=None, use_rsd=False, los=None, hz=None, use_weights=False, seed=0):
    mock_catalog = mock
    mock_density = density_split.DensitySplit(mock_catalog)
    mock_density.compute_density(cellsize=cellsize1, resampler=resampler, use_rsd=use_rsd, los=los, hz=hz, use_weights=use_weights)
    mesh1 = mock_density.density_mesh
    mock_density.compute_density(cellsize=cellsize2, resampler=resampler, use_rsd=use_rsd, los=los, hz=hz, use_weights=use_weights)
    mesh2 = mock_density.density_mesh

    ## Generate random particles and readout density at each particle
    rng = np.random.RandomState(seed=seed)
    positions1 = np.array([rng.uniform(0., 1., mock_density.data.size)*b for b in (mock_density.boxsize,)*3])

    theta = rng.uniform(0., np.pi, mock_density.data.size)
    phi = rng.uniform(0., 2*np.pi, mock_density.data.size)
    relpos = np.array([s*np.sin(theta)*np.cos(phi), s*np.sin(theta)*np.sin(phi), s*np.cos(theta)])
    positions2 = (positions1 + relpos) % mock_density.boxsize
    
    delta_R1 = mesh1.readout(positions1.T, resampler=resampler)
    delta_R2 = mesh2.readout(positions2.T, resampler=resampler)

    return np.array([delta_R1, delta_R2])


def compute_xi_R(data_density, edges, seed=0, use_rsd=False, los=None, hz=None, use_weights=False, nthreads=128):
    data = data_density.data

    if use_rsd:
        if data.positions_rsd is None:
            data.set_rsd(hz=hz, los=los)
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


def compute_xi_RR(data_density, edges, seed=0, use_rsd=False, los=None, hz=None, use_weights=False, nthreads=128):
    data = data_density.data

    if use_rsd:
        if data.positions_rsd is None:
            data.set_rsd(hz=hz, los=los)
        positions2 = data.positions_rsd
    else:
        positions2 = data.positions

    if use_weights and (data.weights is not None):
        weights2 = data.weights
    else:
        weights2 = None

    ## Generate random particles and readout density at each particle
    rng = np.random.RandomState(seed=seed)
    positions1 = np.array([rng.uniform(0., 1., data_density.data.size)*b for b in (data_density.boxsize,)*3])
    densities1 = data_density.density_mesh.readout(positions1.T, resampler=data_density.resampler)

    weights1 = 1 + densities1

    result = TwoPointCorrelationFunction('smu', edges,
                                        data_positions1=positions1, #data_positions2=positions2,
                                        data_weights1=weights1, #data_weights2=weights2,
                                        boxsize=data_density.boxsize,
                                        engine='corrfunc', nthreads=nthreads,
                                        los = los)

    return result



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
    z = 0.8
    #bias = 1.8

    # Mocks
    nmocks = 10

    # For RSD
    rsd = False
    bg = cosmology.get_background()
    # f = bg.growth_rate(z)
    z = 0.8
    hz = 100*bg.efunc(z)

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

    # Edges (s, mu) to compute correlation function at
    edges = (np.linspace(0., 150., 151), np.linspace(-1, 1, 201))
    los = 'x'

    name = 'AbacusSummit_2Gpc_z{:.3f}_ph000_downsampled_particles_nbar0.0034'.format(z)
    #name = 'AbacusSummit_2Gpc_z{:.3f}_ph000'.format(z)
    abacus_mock = catalog_data.Data.load('/feynman/scratch/dphp/mp270220/abacus/'+name+'.npy')
    #abacus_mock = catalog_data.Data.load('/feynman/work/dphp/mp270220/data/'+name+'.npy')
    #abacus_density = compute_delta_R([abacus_mock], cellsize, resampler, use_rsd=rsd, los=los, hz=hz, use_weights=False)[0]

    #s = np.array([20, 30, 50, 100])
    #r1 = 10
    #r2 = 3
    #res = list()
    #for sep in s:
    #    joint_deltaR = compute_jointPDF_delta_R1_R2(abacus_mock, cellsize1=r1, cellsize2=r2, resampler=resampler, s=sep)
    #    res.append(joint_deltaR)
    #output_dir = '/feynman/work/dphp/mp270220/outputs/density/'
    #np.save(output_dir+name+'_cellsize1{:d}_cellsize2{:d}_resampler{}'.format(r1, r2, resampler)+'_joint_delta_R{}'.format('_RSD' if rsd else ''), res)

    #np.save('/feynman/work/dphp/mp270220/outputs/density/'+name+'_cellsize{:d}_resampler{}_delta_R{}'.format(cellsize, resampler, '_RSD' if rsd else ''), abacus_density)
                    
    mock_density = density_split.DensitySplit(abacus_mock)
    mock_density.compute_density(cellsize=cellsize, resampler=resampler, use_rsd=rsd, los=los, hz=hz, use_weights=False)
    print('Computing correlation function...')
    #mock_xi_R = compute_xi_R(mock_density, edges, seed=0, use_rsd=rsd, los=los, hz=hz, use_weights=False, nthreads=64)
    mock_xi_RR = compute_xi_RR(mock_density, edges, seed=0, use_rsd=rsd, los=los, hz=hz, use_weights=False, nthreads=64)

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
    np.save(output_dir+name+'_cellsize{:d}_resampler{}'.format(cellsize, resampler)+'_xi_RR{}'.format('_RSD' if rsd else ''), mock_xi_RR)


if __name__ == "__main__":
    main()
