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


def compute_densitySplit_CCF(data_density_splits, edges, los=None, use_rsd=False, use_weights=False, seed=0, output_dir='', randoms_size=1, nthreads=128):
    data = data_density_splits.data

    if use_rsd and data.positions_rsd is not None:
        positions = data.positions_rsd
        split_positions = data_density_splits.split_positions_rsd
    else:
        positions = data.positions
        split_positions = data_density_splits.split_positions

    if use_weights and (data.weights is not None):
        weights = data.weights
        split_weights = [weights[data_density_splits.split_indices[split]] for split in range(data_density_splits.nsplits)]
    else:
        weights = None
        split_weights = [None for split in range(data_density_splits.nsplits)]

    split_samples = data_density_splits.sample_splits(size=randoms_size*data_density_splits.data.size, seed=seed, update=False)
    cellsize = data_density_splits.cellsize

    # results_hh_auto = list()
    # results_hh_cross = list()
    # results_hh = list()
    results_rh = list()
    # results_rr = list()

    for i in range(data_density_splits.nsplits):
        # result_hh_auto = TwoPointCorrelationFunction('smu', edges,
        #                                         data_positions1=split_positions[i], data_weights1=split_weights[i],
        #                                         boxsize=data_density_splits.boxsize,
        #                                         engine='corrfunc', nthreads=128,
        #                                         los = los)
        #
        # cross_indices = [j for j in range(data_density_splits.nsplits) if j!=i]
        # cross_positions = np.concatenate([split_positions[j] for j in cross_indices], axis=1)
        # if use_weights and (data.weights is not None):
        #     cross_weights = np.concatenate([split_weights[j] for j in cross_indices])
        # else:
        #     cross_weights = None
        # result_hh_cross = TwoPointCorrelationFunction('smu', edges,
        #                                         data_positions1=split_positions[i], data_positions2=cross_positions,
        #                                         data_weights1=split_weights[i], data_weights2=cross_weights,
        #                                         boxsize=data_density_splits.boxsize,
        #                                         engine='corrfunc', nthreads=nthreads,
        #                                         los = los)

        # result_hh = TwoPointCorrelationFunction('smu', edges,
        #                                         data_positions1=split_positions[i], data_positions2=positions,
        #                                         data_weights1=split_weights[i], data_weights2=weights,
        #                                         boxsize=data_density_splits.boxsize,
        #                                         engine='corrfunc', nthreads=nthreads,
        #                                         los = los)
        #
        result_rh = TwoPointCorrelationFunction('smu', edges,
                                                data_positions1=split_samples[i], data_positions2=positions,
                                                data_weights1=None, data_weights2=weights,
                                                boxsize=data_density_splits.boxsize,
                                                engine='corrfunc', nthreads=nthreads,
                                                los = los)

        #result_rr = TwoPointCorrelationFunction('smu', edges,
        #                                        data_positions1=split_samples[i],
        #                                        boxsize=data_density_splits.boxsize,
        #                                        engine='corrfunc', nthreads=nthreads,
        #                                        los = los)

        # results_hh_auto.append(result_hh_auto)
        # results_hh_cross.append(result_hh_cross)
        # results_hh.append(result_hh)
        results_rh.append(result_rh)
        # results_rr.append(result_rr)

    # return {'hh_auto': results_hh_auto, 'hh_cross': results_hh_cross, 'rh': results_rh}
    return {'rh': results_rh}


def generate_batch_densitySplit_CCF(edges, nmocks, nmesh, boxsize, boxcenter,
                                    nbar, cosmology, z, bias,
                                    cellsize, resampler, nsplits, bins, randoms_size,
                                    los=None, rsd=False, use_rsd=False, use_weights=False,
                                    nthreads=128, batch_size=None, batch_index=0,
                                    output_dir='/', name=''):

    # results_hh_auto = list()
    # results_hh_cross = list()
    # results_hh = list()
    results_rh = list()
    # results_rr = list()

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
            print('Computing density splits...')
            mock_density = density_split.DensitySplit(mock_catalog)
            mock_density.compute_density(cellsize=cellsize, resampler=resampler, use_rsd=use_rsd, use_weights=use_weights)
            mock_density.split_density(nsplits, bins=bins)

            print('Computing correlation function...')
            mock_CCFs = compute_densitySplit_CCF(mock_density, edges, los, use_rsd=rsd, use_weights=use_weights, seed=i, randoms_size=randoms_size, nthreads=nthreads)
            # result_hh_auto = mock_CCFs['hh_auto']
            # result_hh_cross = mock_CCFs['hh_cross']
            # result_hh = mock_CCFs['hh']
            result_rh = mock_CCFs['rh']
            # result_rr = mock_CCFs['rr']

            # results_hh_auto.append(result_hh_auto)
            # results_hh_cross.append(result_hh_cross)
            # results_hh.append(result_hh)
            results_rh.append(result_rh)
            # results_rr.append(result_rr)

    # return results_hh_auto, results_hh_cross, results_rh
    return results_rh


def main():
    # Output directory for generated mocks
    output_dir = '/feynman/work/dphp/mp270220/outputs/mocks/lognormal/'

    # Mock parameters
    boxsize = 2000
    boxcenter = 1000
    nmesh = 1024
    #nbar = 0.01
    cosmology=fiducial.AbacusSummitBase()
    #z = 1.
    #bias = 1.

    # Mocks
    nmocks = 10

    # For RSD
    # bg = cosmology.get_background()
    # f = bg.growth_rate(z)
    # hz = 100*bg.efunc(z)

    # Density smoothing parameters
    cellsize = 10
    resampler = 'tsc'

    # Edges (s, mu) to compute correlation function at
    edges = (np.linspace(0., 150., 151), np.linspace(-1, 1, 201))
    los = 'x'

    # Density split parameters
    nsplits = 3
    randoms_size = 4
    # gaussian
    # bins = np.array([-np.inf, -0.21875857,  0.21875857, np.inf])
    # lognormal
    #bins = np.array([-1., -0.18346272,  0.09637895, np.inf])
    bins = np.array([-1, -0.29346216, 0.14210049, np.inf]) # for z = 1.175

    name = 'AbacusSummit_2Gpc_z1.175_ph003'
    abacus_mock = catalog_data.Data.load('/feynman/work/dphp/mp270220/data/'+name+'.npy')
    mock_density = density_split.DensitySplit(abacus_mock)
    mock_density.compute_density(cellsize=cellsize, resampler=resampler, use_rsd=False, use_weights=False)
    mock_density.split_density(nsplits, bins=bins)
    results_rh = compute_densitySplit_CCF(mock_density, edges, los, use_rsd=False, use_weights=False, seed=0, randoms_size=randoms_size, nthreads=32)
    
    #batch_size = 1
    #batch_index = int(sys.argv[1])

    #results_rh = generate_batch_densitySplit_CCF(edges, nmocks, nmesh, boxsize, boxcenter,
    #                                             nbar, cosmology, z, bias,
    #                                             cellsize, resampler, nsplits, bins, randoms_size,
    #                                             los=los, rsd=False, use_rsd=False, use_weights=True,
    #                                             nthreads=64, batch_size=batch_size, batch_index=batch_index,
    #                                             output_dir=output_dir, name='lognormal_mock')

    output_dir = '/feynman/work/dphp/mp270220/outputs/correlation_functions/'
    #name = 'lognormal_mock{:d}_z{:.3f}_bias{:.1f}_boxsize{:d}_nmesh{:d}_nbar{:.3f}'.format(batch_index, z, bias, boxsize, nmesh, nbar)
    
    # np.save(output_dir+name+'_cellsize{:d}_resampler{}_{:d}splits_randoms_size{:d}'.format(cellsize, resampler, nsplits, randoms_size)+'_HH_autoCF', results_hh_auto)
    # np.save(output_dir+name+'_cellsize{:d}_resampler{}_{:d}splits_randoms_size{:d}'.format(cellsize, resampler, nsplits, randoms_size)+'_HH_crossCF', results_hh_cross)
    np.save(output_dir+name+'_cellsize{:d}_resampler{}_{:d}splits_randoms_size{:d}'.format(cellsize, resampler, nsplits, randoms_size)+'_RH_CCF', results_rh)
    # np.save(output_dir+name+'_cellsize{:d}_resampler{}_{:d}splits_randoms_size{:d}'.format(cellsize, resampler, nsplits, randoms_size)+'_HH_CCF', results_hh)
    # np.save(output_dir+name+'_cellsize{:d}_resampler{}_{:d}splits_randoms_size{:d}'.format(cellsize, resampler, nsplits, randoms_size)+'_RR_CCF', results_rr)

if __name__ == "__main__":
    main()
