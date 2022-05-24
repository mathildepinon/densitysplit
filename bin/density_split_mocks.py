#!/usr/bin/python
import sys
import numpy as np

from cosmoprimo import *
from pycorr import TwoPointCorrelationFunction, setup_logging
from mockfactory import EulerianLinearMock, LagrangianLinearMock, RandomBoxCatalog, setup_logging

from densitysplit import catalog_data, density_split
from bin.density_split_mocks_functions import generate_N_mocks, generate_batch_2PCF, generate_batch_densitySplit_CCF

# Set up logging
setup_logging()

# Mock batch
batch_index = int(sys.argv[1])

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
nmocks = 1000
nmesh = 512

# For RSD
cosmology=fiducial.AbacusSummitBase()
bg = cosmology.get_background()
f = bg.Omega_m(catalog.redshift)**0.55

# Set RSD
#hz = bg.hubble_function(catalog.redshift)
hz = 100*bg.efunc(catalog.redshift)
catalog.set_rsd(hz=hz)

# Generate mocks and save them
#generate_N_mocks(catalog, nmocks=nmocks, nmesh=nmesh, 
#                 bias=bias, 
#                 rsd=True, los=los, f=f,
#                 output_dir=output_dir+'mocks_rsd/', mpi=True, overwrite=False)

#results = generate_batch_2PCF(catalog, nmocks=nmocks, nmesh=nmesh,
#                          bias=bias,
#                          edges=edges, los=los,
#                          rsd=True,
#                          nthreads=10, batch_size=10, batch_index=batch_index,
#                          save_each=True, output_dir=output_dir+'mocks_rsd/', mpi=False, overwrite=False)

results_hh_auto, results_hh_cross, results_rh = generate_batch_densitySplit_CCF(catalog, nmocks=nmocks,
                                                                                 nmesh=nmesh,
                                                                                 bias=bias,
                                                                                 cellsize=cellsize, resampler=resampler, nsplits=nsplits, use_rsd=False,
                                                                                 randoms_size=randoms_size,
                                                                                 edges=edges, los=los, f=f, rsd=True,
                                                                                 nthreads=10, batch_size=10, batch_index=batch_index,
                                                                                 save_each=True, output_dir=output_dir+'mocks_rsd/', mpi=False, overwrite=False)

#np.save(output_dir+catalog.name+'_1000_mocks_2PCF_RSD_batch'+str(batch_index), results)
np.save(output_dir+catalog.name+'_1000_mocks_densitySplit_hh_autoCF_cellsize'+str(cellsize)+'_randomsize'+str(randoms_size)+'_realDensity_RSD_batch'+str(batch_index), results_hh_auto)
np.save(output_dir+catalog.name+'_1000_mocks_densitySplit_hh_crossCF_cellsize'+str(cellsize)+'_randomsize'+str(randoms_size)+'_realDensity_RSD_batch'+str(batch_index), results_hh_cross)
np.save(output_dir+catalog.name+'_1000_mocks_densitySplit_rh_CCF_cellsize'+str(cellsize)+'_randomsize'+str(randoms_size)+'_realDensity_RSD_batch'+str(batch_index), results_rh)
