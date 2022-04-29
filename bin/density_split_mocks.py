import numpy as np

from cosmoprimo import *
from pycorr import TwoPointCorrelationFunction, setup_logging
from mockfactory import EulerianLinearMock, LagrangianLinearMock, RandomBoxCatalog, setup_logging

import catalog_data
import density_split
from density_split_mocks_functions import generate_N_mocks, generate_N_densitySplit_CCF

# Set up logging
setup_logging()



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
cellsize = 20
resampler = 'tsc'
nsplits = 2

# Correlation function
edges = (np.linspace(0., 150., 51), np.linspace(-1, 1, 201))
los = 'x'

# Mocks
nmocks = 1000
nmesh = 512

# Generate mocks and save them
#generate_N_mocks(catalog, nmocks=nmocks, nmesh=nmesh, 
#                 bias=bias, 
#                 output_dir=output_dir+'mocks/', mpi=True, overwrite=False)

results_gg, results_dg = generate_N_densitySplit_CCF(catalog, nmocks=nmocks, nmesh=nmesh, 
                                                     bias=bias, 
                                                     cellsize=cellsize, resampler=resampler, nsplits=nsplits,
                                                     edges=edges, los=los,
                                                     save_each=True, output_dir=output_dir+'mocks/', mpi=False, overwrite=False)

np.save(output_dir+catalog.name+'_1000_mocks_densitySplit_gg_CCF', results_gg)
np.save(output_dir+catalog.name+'_1000_mocks_densitySplit_dg_CCF', results_dg)
