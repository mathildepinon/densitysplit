from os.path import exists

import numpy as np

from cosmoprimo import *
from pycorr import TwoPointCorrelationFunction, setup_logging
from mockfactory import EulerianLinearMock, LagrangianLinearMock, RandomBoxCatalog, setup_logging

from densitysplit import catalog_data, density_split

# Set up logging
setup_logging()



def generate_mock(nmesh, boxsize, boxcenter, seed, cosmology, nbar, z, bias, rsd=False, los=None, f=None, save=False, output_dir='', name='mock', mpi=False):
    pklin = cosmology.get_fourier().pk_interpolator().to_1d(z)

    # unitary_amplitude forces amplitude to 1
    mock = LagrangianLinearMock(pklin, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=False)

    # this is Lagrangian bias, Eulerian bias - 1
    mock.set_real_delta_field(bias=bias - 1)
    mock.set_analytic_selection_function(nbar=nbar)
    mock.poisson_sample(seed=seed+1)
    data = mock.to_catalog()

    mock.set_rsd(f=f, los=los)
    data_rsd = mock.to_catalog()

    # Create Data instance
    offset = boxcenter - boxsize/2.
    positions = (data['Position'] - offset) % boxsize + offset
    positions_rsd = (data_rsd['Position'] - offset) % boxsize + offset

    if mpi:
        positions = positions.gather()
        positions_rsd = positions_rsd.gather()
        mock_catalog = None
        if mock.mpicomm.rank == 0:
            mock_catalog = catalog_data.Data(positions.T, z, boxsize, boxcenter, name=name)
            if rsd:
                mock_catalog.set_rsd(positions_rsd=positions_rsd.T)
            if save:
                mock_catalog.save(output_dir+name)

    else:
        mock_catalog = catalog_data.Data(positions.T, z, boxsize, boxcenter, name=name)
        if rsd:
            mock_catalog.set_rsd(positions_rsd=positions_rsd.T)
        if save:
            mock_catalog.save(output_dir+name)

    return mock_catalog


def generate_gaussian_mock(nmesh, boxsize, boxcenter, seed, cosmology, nbar, z, bias, rsd=False, los=None, f=None, save=False, output_dir='', name='gaussian_mock', mpi=False):
    pklin = cosmology.get_fourier().pk_interpolator().to_1d(z)

    # unitary_amplitude forces amplitude to 1
    mock = EulerianLinearMock(pklin, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=False)

    # this is Lagrangian bias, Eulerian bias - 1
    mock.set_real_delta_field(bias=bias)

    data = RandomBoxCatalog(nbar=nbar, boxsize=boxsize, boxcenter=boxcenter, seed=seed)
    data['Weight'] = mock.readout(data['Position'], field='delta', resampler='tsc', compensate=True) + 1.
    positions = data['Position'] % boxsize
    weights = data['Weight']

    # Set rsd
    mock.set_rsd(f=f, los=los)
    data_rsd = RandomBoxCatalog(nbar=nbar, boxsize=boxsize, boxcenter=boxcenter, seed=seed)
    data_rsd['Weight'] = mock.readout(data_rsd['Position'], field='delta', resampler='tsc', compensate=True) + 1.

    positions_rsd = data_rsd['Position'] % boxsize

    offset = boxcenter - boxsize/2.
    positions = (data['Position'] - offset) % boxsize + offset
    positions_rsd = (data_rsd['Position'] - offset) % boxsize + offset

    if mpi:
        positions = positions.gather()
        positions_rsd = positions_rsd.gather()
        weights = weights.gather()
        mock_catalog = None
        if mock.mpicomm.rank == 0:
            mock_catalog = catalog_data.Data(positions.T, z, boxsize, boxcenter, name=name, weights=weights)
            if rsd:
                mock_catalog.set_rsd(positions_rsd=positions_rsd.T)
            if save:
                mock_catalog.save(output_dir+name)

    else:
        mock_catalog = catalog_data.Data(positions.T, z, boxsize, boxcenter, name=name, weights=weights)
        if rsd:
            mock_catalog.set_rsd(positions_rsd=positions_rsd.T)
        if save:
            mock_catalog.save(output_dir+name)

    return mock_catalog


def generate_N_mocks(catalog, nmocks, nmesh, bias, rsd=False, los=None, f=None, nbar=None, output_dir='', mpi=False, overwrite=True, type='lagrangian'):
    if nbar is None:
        nbar=catalog.size/catalog.boxsize**3

    for i in range(nmocks):
        filename = catalog.name+'_gaussianMock'+str(i)+'_nbar'+str(nbar)
        if not exists(output_dir+filename+'.npy') or overwrite:
            if type == 'langrangian':
                generate_mock(nmesh=nmesh, boxsize=catalog.boxsize, boxcenter=catalog.boxcenter, seed=i,
                             cosmology=fiducial.AbacusSummitBase(), nbar=nbar,
                             z=catalog.redshift, bias=bias, rsd=rsd, los=los, f=f,
                             save=True, output_dir=output_dir, name=filename,
                             mpi=mpi)
            elif type == 'gaussian':
                generate_gaussian_mock(nmesh=nmesh, boxsize=catalog.boxsize, boxcenter=catalog.boxcenter, seed=i,
                                     cosmology=fiducial.AbacusSummitBase(), nbar=nbar,
                                     z=catalog.redshift, bias=bias, rsd=rsd, los=los, f=f,
                                     save=True, output_dir=output_dir, name=filename,
                                     mpi=mpi)


def split_density(catalog, cellsize, resampler, nsplits, use_rsd=False, use_weights=False, save=False, output_dir=''):
    catalog_density = density_split.DensitySplit(catalog)
    catalog_density.compute_density(cellsize=cellsize, resampler=resampler, use_rsd=use_rsd, use_weights=use_weights)
    catalog_density.split_density(nsplits)

    if save:
        catalog_density.save(output_dir+catalog.name+'_density')

    return catalog_density


def compute_densitySplit_CCF(data_density_splits, edges, los, use_rsd=False, use_weights=False, save=False, output_dir='', name='mock', randoms_size=1, nthreads=128):
    data = data_density_splits.data

    if use_rsd and data.positions_rsd is not None:
        rsd_info = '_RSD'
        positions = data.positions_rsd
        split_positions = data_density_splits.split_positions_rsd
    else:
        rsd_info = ''
        positions = data.positions
        split_positions = data_density_splits.split_positions

    if use_weights and (data.weights is not None):
        weights = data.weights
        split_weights = [weights[data_density_splits.split_indices[split]] for split in range(data_density_splits.nsplits)]
    else:
        weights = None
        split_weights = [None for split in range(data_density_splits.nsplits)]

    split_samples = data_density_splits.sample_splits(size=randoms_size*data_density_splits.data.size, seed=0, update=False)
    cellsize = data_density_splits.cellsize

    results_hh_auto = list()
    results_hh_cross = list()
    results_rh = list()

    for i in range(data_density_splits.nsplits):
        result_hh_auto = TwoPointCorrelationFunction('smu', edges,
                                                data_positions1=split_positions[i], data_weights1=split_weights[i],
                                                boxsize=data_density_splits.boxsize,
                                                engine='corrfunc', nthreads=128,
                                                los = los)

        cross_indices = [j for j in range(data_density_splits.nsplits) if j!=i]
        cross_positions = np.concatenate([split_positions[j] for j in cross_indices], axis=1)
        cross_weights = np.concatenate([split_weights[j] for j in cross_indices])
        result_hh_cross = TwoPointCorrelationFunction('smu', edges,
                                                data_positions1=split_positions[i], data_positions2=cross_positions,
                                                data_weights1=split_weights[i], data_weights2=cross_weights,
                                                boxsize=data_density_splits.boxsize,
                                                engine='corrfunc', nthreads=nthreads,
                                                los = los)

        result_rh = TwoPointCorrelationFunction('smu', edges,
                                                data_positions1=split_samples[i], data_positions2=positions,
                                                data_weights1=None, data_weights2=weights,
                                                boxsize=data_density_splits.boxsize,
                                                engine='corrfunc', nthreads=nthreads,
                                                los = los)

        results_hh_auto.append(result_hh_auto)
        results_hh_cross.append(result_hh_cross)
        results_rh.append(result_rh)

    if save:
        np.save(output_dir+name+'_densitySplit_hh_autoCFs_cellsize'+str(cellsize)+'_randomsize'+str(randoms_size)+rsd_info, results_hh_auto)
        np.save(output_dir+name+'_densitySplit_hh_crossCFs_cellsize'+str(cellsize)+'_randomsize'+str(randoms_size)+rsd_info, results_hh_cross)
        np.save(output_dir+name+'_densitySplit_rh_CCFs_cellsize'+str(cellsize)+'_randomsize'+str(randoms_size)+rsd_info, results_rh)

    return {'hh_auto': results_hh_auto, 'hh_cross': results_hh_cross, 'rh': results_rh}


def generate_batch_2PCF(catalog, nmocks, nmesh, bias, edges,
                        los=None, f=None, rsd=False, use_weights=False,
                        batch_size=None, batch_index=0,
                        nthreads=128,
                        save_each=False, output_dir='', mpi=False, overwrite=True):
    results = list()

    if batch_size is None:
        batch_size = nmocks

    mocks_indices = range(batch_index*batch_size, (batch_index+1)*batch_size)

    for i in mocks_indices:
        print('Mock '+str(i))
        filename = catalog.name+'_gaussianMock'+str(i)
        if exists(output_dir+filename+'.npy') and not overwrite:
            print('Mock already exists. Loading mock...')
            mock_catalog = catalog_data.Data.load(output_dir+filename+'.npy')
        else:
            print('Mock does not exist. Generating mock...')
            mock_catalog = generate_gaussian_mock(nmesh=nmesh, boxsize=catalog.boxsize, boxcenter=catalog.boxcenter, seed=i,
                                                 cosmology=fiducial.AbacusSummitBase(), nbar=catalog.size/catalog.boxsize**3,
                                                 z=catalog.redshift, bias=bias,
                                                 rsd=rsd, los=los, f=f,
                                                 save=save_each, output_dir=output_dir, name=filename,
                                                 mpi=mpi)

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
                                                los = los)
            results.append(result)

    return results


def generate_batch_densitySplit_CCF(catalog, nmocks, nmesh, bias,
                                    cellsize, resampler, nsplits, use_rsd,
                                    randoms_size, edges, los, f=None, rsd=False, use_weights=False, nbar=None,
                                    nthreads=128,
                                    batch_size=None, batch_index=0,
                                    save_each=False, output_dir='', mpi=False, overwrite=True):

    if nbar is None:
        nbar=catalog.size/catalog.boxsize**3

    results_hh_auto = list()
    results_hh_cross = list()
    results_rh = list()

    if batch_size is None:
        batch_size = nmocks

    mocks_indices = range(batch_index*batch_size, (batch_index+1)*batch_size)

    for i in mocks_indices:
        print('Mock '+str(i))
        filename = catalog.name+'_gaussianMock'+str(i)+'_nbar'+str(nbar)
        if exists(output_dir+filename+'.npy') and not overwrite:
            print('Mock already exists. Loading mock...')
            mock_catalog = catalog_data.Data.load(output_dir+filename+'.npy')
        else:
            print('Mock does not exist. Generating mock...')
            mock_catalog = generate_mock(nmesh=nmesh, boxsize=catalog.boxsize, boxcenter=catalog.boxcenter, seed=i,
                                         cosmology=fiducial.AbacusSummitBase(), nbar=catalog.size/catalog.boxsize**3,
                                         z=catalog.redshift, bias=bias,
                                         rsd=rsd, los=los, f=f,
                                         save=save_each, output_dir=output_dir, name=filename,
                                         mpi=mpi)

        if mock_catalog is not None:
            print('Computing density splits...')
            mock_density = split_density(mock_catalog, cellsize, resampler, nsplits, use_rsd=use_rsd, use_weights=use_weights, save=False)
            print('Computing correlation function...')
            mock_CCFs = compute_densitySplit_CCF(mock_density, edges, los, use_rsd=rsd, use_weights=use_weights, randoms_size=randoms_size, nthreads=nthreads)
            result_hh_auto = mock_CCFs['hh_auto']
            result_hh_cross = mock_CCFs['hh_cross']
            result_rh = mock_CCFs['rh']

            results_hh_auto.append(result_hh_auto)
            results_hh_cross.append(result_hh_cross)
            results_rh.append(result_rh)

    return results_hh_auto, results_hh_cross, results_rh
