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
    positions = data['Position'] % boxsize
    positions_rsd = data_rsd['Position'] % boxsize
            
    if mpi:
        positions = positions.gather()
        mock_catalog = None
        if mock.mpicomm.rank == 0:
            mock_catalog = catalog_data.Data(positions.T, z, boxsize, boxcenter, name=name)
            if rsd:
                mock_catalog.set_rsd(positions_rsd=positions_rsd)
            if save:
                mock_catalog.save(output_dir+name)
                
    else:
        mock_catalog = catalog_data.Data(positions.T, z, boxsize, boxcenter, name=name)
        if rsd:
            mock_catalog.set_rsd(positions_rsd=positions_rsd)
        if save:
            mock_catalog.save(output_dir+name)
    
    return mock_catalog


def generate_N_mocks(catalog, nmocks, nmesh, bias, rsd=False, los=None, f=None, output_dir='', mpi=False, overwrite=True):
    for i in range(nmocks):
        filename = catalog.name+'_mock'+str(i)
        if not exists(output_dir+filename+'.npy') or overwrite:
            generate_mock(nmesh=nmesh, boxsize=catalog.boxsize, boxcenter=catalog.boxcenter, seed=i,
                         cosmology=fiducial.AbacusSummitBase(), nbar=catalog.size/catalog.boxsize**3,
                         z=catalog.redshift, bias=bias, rsd=rsd, los=los, f=f,
                         save=True, output_dir=output_dir, name=filename,
                         mpi=mpi)


def split_density(catalog, cellsize, resampler, nsplits, save=False, output_dir=''):
    catalog_density = density_split.DensitySplit(catalog)
    catalog_density.compute_density(cellsize=cellsize, resampler=resampler, use_rsd=False, use_weights=False)
    catalog_density.split_density(nsplits)
    
    if save:
        catalog_density.save(output_dir+catalog.name+'_density')
        
    return catalog_density


def compute_densitySplit_CCF(data_density_splits, edges, los, save=False, name='mock'):
    
    split_samples = data_density_splits.sample_splits(size=data_density_splits.data.size, seed=0, update=False)
    
    results_gg = list()
    results_dg = list()
    
    for i in range(data_density_splits.nsplits):
        result_gg = TwoPointCorrelationFunction('smu', edges,
                                                data_positions1=data_density_splits.split_positions[i], data_positions2=data_density_splits.data.positions,
                                                boxsize=data_density_splits.boxsize,
                                                engine='corrfunc', nthreads=128,
                                                los = los)
        
        result_dg = TwoPointCorrelationFunction('smu', edges,
                                                data_positions1=split_samples[i], data_positions2=data_density_splits.data.positions,
                                                boxsize=data_density_splits.boxsize,
                                                engine='corrfunc', nthreads=128,
                                                los = los)
        results_gg.append(result_gg)
        results_dg.append(result_dg)
    
    if save:
        np.save(output_dir+name+'_densitySplit_gg_CCFs', results_gg)
        np.save(output_dir+name+'_densitySplit_dg_CCFs', results_dg)
    
    return {'gg': results_gg, 'dg': results_dg}


def generate_N_2PCF(catalog, nmocks, nmesh, bias, edges, los, save_each=False, output_dir='', mpi=False, overwrite=True):
    results = list()
    
    for i in range(nmocks):
        print('Mock '+str(i))
        filename = catalog.name+'_mock'+str(i)
        if exists(output_dir+filename+'.npy') and not overwrite:
            print('Mock already exists. Loading mock...')
            mock_catalog = catalog_data.Data.load(output_dir+filename+'.npy')
        else:
            print('Mock does not exist. Generating mock...')
            mock_catalog = generate_mock(nmesh=nmesh, boxsize=catalog.boxsize, boxcenter=catalog.boxcenter, seed=i,
                                         cosmology=fiducial.AbacusSummitBase(), nbar=catalog.size/catalog.boxsize**3,
                                         z=catalog.redshift, bias=bias,
                                         save=save_each, output_dir=output_dir, name=filename,
                                         mpi=mpi)
        
        if mock_catalog is not None:
            print('Computing correlation function...')
            result = TwoPointCorrelationFunction('smu', edges,
                                                data_positions1=mock_catalog.positions,
                                                boxsize=mock_catalog.boxsize,
                                                engine='corrfunc', nthreads=128,
                                                los = los)
            results.append(result)
        
    return results
        

def generate_N_densitySplit_CCF(catalog, nmocks, nmesh, bias, cellsize, resampler, nsplits, edges, los, save_each=False, output_dir='', mpi=False, overwrite=True):
    results_gg = list()
    results_dg = list()
    
    for i in range(nmocks):
        print('Mock '+str(i))
        filename = catalog.name+'_mock'+str(i)
        if exists(output_dir+filename+'.npy') and not overwrite:
            print('Mock already exists. Loading mock...')
            mock_catalog = catalog_data.Data.load(output_dir+filename+'.npy')
        else:
            print('Mock does not exist. Generating mock...')
            mock_catalog = generate_mock(nmesh=nmesh, boxsize=catalog.boxsize, boxcenter=catalog.boxcenter, seed=i,
                                         cosmology=fiducial.AbacusSummitBase(), nbar=catalog.size/catalog.boxsize**3,
                                         z=catalog.redshift, bias=bias,
                                         save=save_each, output_dir=output_dir, name=filename,
                                         mpi=mpi)
        
        if mock_catalog is not None:
            print('Computing density splits...')
            mock_density = split_density(mock_catalog, cellsize, resampler, nsplits, save=False)
            print('Computing correlation function...')
            mock_CCFs = compute_densitySplit_CCF(mock_density, edges, los)
            result_gg = mock_CCFs['gg']
            result_dg = mock_CCFs['dg']

            results_gg.append(result_gg)
            results_dg.append(result_dg)
        
    return results_gg, results_dg