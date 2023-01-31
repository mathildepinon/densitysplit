from os.path import exists, join
import sys
import copy

import numpy as np

from cosmoprimo import *
from pycorr import TwoPointCorrelationFunction, setup_logging
from mockfactory import EulerianLinearMock, LagrangianLinearMock, BoxCatalog, RandomBoxCatalog, setup_logging
from pypower.fft_power import project_to_basis, CatalogFFTPower, MeshFFTPower

# Set up logging
setup_logging()


def generate_lagrangian_mock(nmesh, boxsize, boxcenter, seed, nbar, cosmology=fiducial.AbacusSummitBase(), z=1., bias=1.4, rsd=False, los=None, f=None, save=False, output_dir='.', name='lagrangian_mock', mpi=False):
    pklin = cosmology.get_fourier().pk_interpolator().to_1d(z)

    mock = LagrangianLinearMock(pklin, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=False)

    # this is Lagrangian bias, Eulerian bias - 1
    mock.set_real_delta_field(bias=bias - 1)
    mock.set_analytic_selection_function(nbar=nbar)
    mock.poisson_sample(seed=seed+1)
    if rsd:
        mock.set_rsd(f=f, los=los)
    data = mock.to_catalog()

    fn = join(output_dir, name+'.fits')
    data.write(fn)

    return data


def generate_N_lagrangian_mocks(nmocks, nmesh, boxsize, boxcenter, nbar, cosmology=fiducial.AbacusSummitBase(), z=1., bias=1.4, rsd=False, los=None, f=None, output_dir='.', name='lagrangian_mock', mpi=False, overwrite=True):
    for i in range(nmocks):
        filename = name+'_z{:.3f}_bias{:.1f}_boxsize{:d}_nmesh{:d}_nbar{:.3f}_mock{:d}'.format(z, bias, boxsize, nmesh, nbar, i)
        if not exists(output_dir+filename+'.fits') or overwrite:
            generate_lagrangian_mock(nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=i,
                                     nbar=nbar, cosmology=cosmology,
                                     z=z, bias=bias, rsd=rsd, los=los, f=f,
                                     save=True, output_dir=output_dir, name=filename,
                                     mpi=mpi)
    return None


def generate_batch_pk(edges, nmocks, nmesh, boxsize, boxcenter, nbar,
                        cosmology=fiducial.AbacusSummitBase(), z=1., bias=1.4,
                        rsd=False, los=None, f=None,
                        use_weights=False,
                        nthreads=128, batch_size=None, batch_index=0,
                        output_dir='/', name='', mpi=False, overwrite=True):

    results_ic = list()
    results = list()

    if batch_size is None:
        batch_size = nmocks

    mocks_indices = range(batch_index*batch_size, (batch_index+1)*batch_size)

    for i in mocks_indices:
        print('Mock '+str(i))
        filename = name+'_z{:.3f}_bias{:.1f}_boxsize{:d}_nmesh{:d}_nbar{:.3f}_mock{:d}'.format(z, bias, boxsize, nmesh, nbar, i)
        print(output_dir+filename+'.fits')
        if exists(output_dir+filename+'.fits'):
            print('Mock already exists. Loading mock...')
            mock_catalog = BoxCatalog.read(output_dir+filename+'.fits', boxsize=boxsize, boxcenter=boxcenter, position='Position', velocity='Displacement')
        else:
            print('Mock does not exist')

        if mock_catalog is not None:
            print('Computing power spectrum...')
            positions = mock_catalog['Position']

            if use_weights:
                weights = mock_catalog['Weight']
            else:
                weights = None

            ## Pk of initial conditions
            pklin = cosmology.get_fourier().pk_interpolator().to_1d(z)
            mock = LagrangianLinearMock(pklin, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=i, unitary_amplitude=False)
            pk_ic = MeshFFTPower(mock.mesh_delta_k, edges=edges, los=los)
            results_ic.append(pk_ic)

            pk = CatalogFFTPower(data_positions1=positions, data_weights1=weights,
                                 edges=edges, boxsize=boxsize, nmesh=nmesh, resampler='tsc',
                                 los=los, position_type='pos', mpiroot=0)

            results.append(pk)

    return results_ic, results


def main():
    # Output directory for generated mocks
    output_dir = '/feynman/work/dphp/mp270220/outputs/mocks/lognormal/'

    # Mock parameters
    boxsize = 2000
    boxcenter = 1000
    nmesh = 1024
    nbar = 0.001
    cosmology=fiducial.AbacusSummitBase()
    z = 1.
    bias = 1.4

    # Mocks
    nmocks = 500

    # For RSD
    bg = cosmology.get_background()
    f = bg.growth_rate(z)
    hz = 100*bg.efunc(z)
    los = 'x'

    # Edges (k, mu) to compute correlation function at
    edges = {'step': 0.005}

    # generate_N_lagrangian_mocks(nmocks, nmesh, boxsize, boxcenter, nbar,
    #                              cosmology=cosmology, z=z, bias=bias, rsd=False, los=los, f=f,
    #                              output_dir=output_dir, name='lagrangian_mock',
    #                              mpi=True, overwrite=True)

    batch_size = 50
    batch_index = int(sys.argv[1])

    results_ic, results = generate_batch_pk(edges, nmocks, nmesh, boxsize, boxcenter,
                                             nbar, cosmology=cosmology, z=z, bias=bias,
                                             los=los, rsd=False, use_weights=False,
                                             nthreads=64, batch_size=batch_size, batch_index=batch_index,
                                             output_dir=output_dir, name='lagrangian_mock')

    name = '{:d}lagrangianMocks_batch{:d}_z{:.3f}_bias{:.1f}_boxsize{:d}_nmesh{:d}_nbar{:.3f}'.format(batch_size, batch_index, z, bias, boxsize, nmesh, nbar)
    output_dir = '/feynman/work/dphp/mp270220/outputs/power_spectrum/'
    np.save(output_dir+name+'_pk_ic', results_ic)
    np.save(output_dir+name+'_pk', results)

if __name__ == "__main__":
    main()
