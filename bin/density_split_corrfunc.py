#!/usr/bin/python
import sys
import numpy as np

from cosmoprimo import *
from pycorr import TwoPointCorrelationFunction, setup_logging
from mockfactory import EulerianLinearMock, LagrangianLinearMock, RandomBoxCatalog, setup_logging

from densitysplit import catalog_data, density_split
from bin.density_split_mocks_functions import generate_N_mocks, generate_batch_2PCF, generate_batch_densitySplit_CCF, generate_batch_xi_R

# Set up logging
setup_logging()

# Mock batch
batch_index = int(sys.argv[1])
#batch_index = 0

# Data and output directories
data_dir = '/feynman/work/dphp/mp270220/data/'
output_dir = '/feynman/work/dphp/mp270220/outputs/'


# Get data
catalog_name = 'AbacusSummit_1Gpc_z1.175'
bias = 1.8

#catalog_name = 'AbacusSummit_2Gpc_z1.175'
#bias = 3.

#catalog_name = 'AbacusSummit_2Gpc_z0.800'
#catalog_name = 'mock'

catalog = catalog_data.Data.load(data_dir+catalog_name+'.npy')
catalog.shift_boxcenter(-catalog.offset)

# Parameters

# Density mesh
cellsize = 10
resampler = 'tsc'
nsplits = 3

# Correlation function
randoms_size = 4
edges = (np.linspace(0., 150., 51), np.linspace(-1, 1, 201))
los = 'x'

# Mocks
nmocks = 10
nmesh = 512
nbar = 0.01 #catalog.size/catalog.boxsize**3

# For RSD
cosmology=fiducial.AbacusSummitBase()
bg = cosmology.get_background()
f = bg.growth_rate(catalog.redshift)

# Set RSD
hz = 100*bg.efunc(catalog.redshift)
catalog.set_rsd(hz=hz)

# Generate mocks and save them
# generate_N_mocks(catalog, nmocks=nmocks, nmesh=nmesh,
#                  bias=bias,
#                  rsd=True, los=los, f=f, nbar=nbar,
#                  output_dir=output_dir+'mocks/', mpi=True, overwrite=True,
#                  type = 'lagrangian')

# results = generate_batch_2PCF(catalog, nmocks=nmocks, nmesh=nmesh,
#                              bias=bias,
#                              edges=edges, los=los,
#                              rsd=False, use_weights=True,
#                              nthreads=128, batch_size=1, batch_index=batch_index,
#                              save_each=False, output_dir=output_dir+'mocks/gaussian/', mpi=False, overwrite=False)

results_hh_auto, results_hh_cross, results_rh = generate_batch_densitySplit_CCF(catalog, nmocks=nmocks,
                                                                                 nmesh=nmesh,
                                                                                 bias=bias,
                                                                                 cellsize=cellsize, resampler=resampler, nsplits=nsplits, use_rsd=False,
                                                                                 randoms_size=randoms_size,
                                                                                 edges=edges, los=los, f=f, rsd=False, use_weights=True,
                                                                                 bins=np.array([-np.inf, -0.21870731,  0.21870731, np.inf]), nbar=nbar,
                                                                                 nthreads=128, batch_size=1, batch_index=batch_index,
                                                                                 save_each=False, output_dir=output_dir+'mocks/gaussian/', mpi=False, overwrite=False)

# results = generate_batch_xi_R(catalog, nmocks=10,
#                              nmesh=nmesh,
#                              bias=bias,
#                              cellsize=cellsize, resampler=resampler, nsplits=nsplits, use_rsd=False,
#                              edges=edges, los=los, f=f, rsd=False, use_weights=True, nbar=nbar,
#                              nthreads=128, batch_size=10, batch_index=batch_index,
#                              save_each=True, output_dir=output_dir+'mocks_rsd/', mpi=False, overwrite=False)

# np.save(output_dir+catalog.name+'_10_gaussianMocks_truncatedPk_nbarx5_cellsize'+str(cellsize)+'_xi_R', results)
# np.save(output_dir+catalog.name+'_10_gaussianMocks_truncatedPk_nbarx5_cellsize'+str(cellsize)+'_2PCF', results)
# np.save(output_dir+catalog.name+'_5000_mocks_2PCF_batch'+str(batch_index), results)
# np.save(output_dir+catalog.name+'_5000_mocks_densitySplit_hh_autoCF_cellsize'+str(cellsize)+'_randomsize'+str(randoms_size)+'_batch'+str(batch_index), results_hh_auto)
# np.save(output_dir+catalog.name+'_5000_mocks_densitySplit_hh_crossCF_cellsize'+str(cellsize)+'_randomsize'+str(randoms_size)+'_batch'+str(batch_index), results_hh_cross)
# np.save(output_dir+catalog.name+'_5000_mocks_densitySplit_rh_CCF_cellsize'+str(cellsize)+'_randomsize'+str(randoms_size)+'_batch'+str(batch_index), results_rh)
np.save(output_dir+catalog.name+'_10_gaussianMocksWeightedByDelta_truncatedPk_nbar0.01_densitySplit_fixedBins_hh_autoCF_cellsize'+str(cellsize)+'_randomsize'+str(randoms_size)+'_mock'+str(batch_index), results_hh_auto)
np.save(output_dir+catalog.name+'_10_gaussianMocksWeightedByDelta_truncatedPk_nbar0.01_densitySplit_fixedBins_hh_crossCF_cellsize'+str(cellsize)+'_randomsize'+str(randoms_size)+'_mock'+str(batch_index), results_hh_cross)
np.save(output_dir+catalog.name+'_10_gaussianMocksWeightedByDelta_truncatedPk_nbar0.01_densitySplit_fixedBins_rh_CCF_cellsize'+str(cellsize)+'_randomsize'+str(randoms_size)+'_mock'+str(batch_index), results_rh)
# np.save(output_dir+catalog.name+'_10_gaussianMocksWeightedByDelta_truncatedPk_nbar0.01_2PCF_mock'+str(batch_index), results)
