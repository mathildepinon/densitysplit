from os.path import exists, join
import sys
import copy

import numpy as np

from cosmoprimo import *
from pycorr import TwoPointCorrelationFunction, setup_logging
from mockfactory import Catalog, EulerianLinearMock, LagrangianLinearMock, BoxCatalog, RandomBoxCatalog, setup_logging
from pypower.fft_power import project_to_basis, CatalogFFTPower, MeshFFTPower
from densitysplit import catalog_data

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


def generate_lognormal_mock(nmesh, boxsize, boxcenter, seed, nbar, cosmology=fiducial.AbacusSummitBase(), z=1., bias=1., rsd=False, los=None, f=None, damping=False,
                            save=False, output_dir='.', name='lognormal_mock', mpi=False):
    if damping:
        pklin = pk_damped(cosmology, z, np.pi*nmesh/boxsize)
    else:
        pklin = cosmology.get_fourier().pk_interpolator().to_1d(z)
    
    mock = EulerianLinearMock(pklin, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=False)

    mock.set_real_delta_field(bias=bias, lognormal_transform=True)
    mock.set_analytic_selection_function(nbar=nbar)
    mock.poisson_sample(seed=seed+1)
    data = mock.to_catalog()
    fn = join(output_dir, name+'.fits')
    data.write(fn)
    
    offset = boxcenter - boxsize/2.
    positions = copy.deepcopy((data['Position'] - offset) % boxsize + offset)
    
    if mpi:
        positions = positions.gather()
        mock_catalog = None
        if mock.mpicomm.rank == 0:
            mock_catalog = catalog_data.Data(positions.T, z, boxsize, boxcenter, name=name)
            mock_catalog.save(output_dir+name)
            
    else:
        mock_catalog = catalog_data.Data(positions.T, z, boxsize, boxcenter, name=name)
        mock_catalog.save(output_dir+name)        

    return data


def generate_N_lagrangian_mocks(nmocks, nmesh, boxsize, boxcenter, nbar, cosmology=fiducial.AbacusSummitBase(), z=1., bias=1.4, rsd=False, los=None, f=None,
                                damping=False, output_dir='.', name='lagrangian_mock', mpi=False, overwrite=True):
    for i in range(nmocks):
        filename = name+'_z{:.3f}_bias{:.1f}_boxsize{:d}_nmesh{:d}_nbar{:.3f}_mock{:d}'.format(z, bias, boxsize, nmesh, nbar, i)
        if not exists(output_dir+filename+'.fits') or overwrite:
            generate_lognormal_mock(nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=i,
                                     nbar=nbar, cosmology=cosmology,
                                     z=z, bias=bias, rsd=rsd, los=los, f=f, damping=damping,
                                     save=True, output_dir=output_dir, name=filename,
                                     mpi=mpi)
#        data = Catalog.read(output_dir+filename+'.fits', filetype='fits')
#        offset = boxcenter - boxsize/2.
#        positions = copy.deepcopy((data['Position'] - offset) % boxsize + offset)
 #       positions = copy.deepcopy(data['Position'])
 #       mock_catalog = catalog_data.Data(positions.T, z, boxsize, boxcenter, name=filename)
 #       mock_catalog.save(output_dir+filename)

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
    output_dir = '/feynman/work/dphp/mp270220/outputs/mocks/lognormal/'

    # Mock parameters
    boxsize = 2000
    boxcenter = 1000
    nmesh = 1024
    nbar = 0.003
    cosmology=fiducial.AbacusSummitBase()
    z = 1.175
    bias = 1.

    # Mocks
    nmocks = 1

    # For RSD
    bg = cosmology.get_background()
    f = bg.growth_rate(z)
    hz = 100*bg.efunc(z)
    los = 'x'
    
    generate_N_lagrangian_mocks(nmocks, nmesh, boxsize, boxcenter, nbar,
                                 cosmology=cosmology, z=z, bias=bias, rsd=False, los=los, f=f,
                                 output_dir=output_dir, name='lognormal_mock',
                                 mpi=False, overwrite=False)

    #batch_size = 1
    #batch_index = int(sys.argv[1])
    
    # Edges (s, mu) to compute correlation function at
    edges = (np.linspace(0., 150., 151), np.linspace(-1, 1, 201))
    
    results = generate_batch_2PCF(edges, nmocks, nmesh, boxsize, boxcenter,
                             nbar, cosmology=cosmology, z=z, bias=bias,
                             los=los, rsd=False, use_weights=False,
                             nthreads=100, output_dir=output_dir, name='lognormal_mock')

    # Edges (k, mu) to compute power spectrum at
    # edges = {'step': 0.005}

    #results_ic, results = generate_batch_pk(edges, nmocks, nmesh, boxsize, boxcenter,
    #                                         nbar, cosmology=cosmology, z=z, bias=bias,
    #                                         los=los, rsd=False, use_weights=False,
    #                                         nthreads=128,
    #                                         output_dir=output_dir, name='lagrangian_mock')

    output_dir = '/feynman/work/dphp/mp270220/outputs/correlation_functions/'
    name = '{:d}_lognormal_mocks_z{:.3f}_bias{:.1f}_boxsize{:d}_nmesh{:d}_nbar{:.3f}'.format(nmocks, z, bias, boxsize, nmesh, nbar)
    #output_dir = '/feynman/work/dphp/mp270220/outputs/power_spectrum/'
    #np.save(output_dir+name+'_pk_ic', results_ic)
    #np.save(output_dir+name+'_pk', results)
    np.save(output_dir+name+'_2PCF', results)


if __name__ == "__main__":
    main()
