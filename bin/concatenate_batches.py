#!/usr/bin/python
import sys
import numpy as np

from cosmoprimo import *
from pycorr import TwoPointCorrelationFunction, setup_logging

# Concatenate CCF results of mocks computed on separate jobs

# Data and output directories
data_dir = '/feynman/work/dphp/mp270220/data/'
output_dir = '/feynman/work/dphp/mp270220/outputs/'

results_list = list()

catalog_name = 'AbacusSummit_2Gpc_z1.175'
cellsize = 10
randoms_size = 4

#results = np.load(output_dir+catalog_name+'_1000_mocks_densitySplit_hh_crossCF_cellsize'+str(cellsize)+'_randomsize'+str(randoms_size)+'_RSD_all.npy', allow_pickle=True)
# results = np.load(output_dir+catalog_name+'_1000_mocks_2PCF_RSD_all.npy', allow_pickle=True)
# results_list = results.tolist()

for i in range(9):
    batch_index = i

    batch_results = np.load(output_dir+catalog_name+'_10_gaussianMocks_truncatedPk_nbarx5_densitySplit_fixedBins_hh_autoCF_cellsize'
                            +str(cellsize)+'_randomsize'+str(randoms_size)+'_mock'+str(batch_index)+'.npy',
                            allow_pickle=True)

    # batch_results = np.load(output_dir+catalog_name+'_5000_mocks_2PCF_batch'+str(batch_index)+'.npy', allow_pickle=True)

    for res in batch_results:
        results_list.append(res)

np.save(output_dir+catalog_name+'_10_gaussianMocks_truncatedPk_nbarx5_densitySplit_fixedBins_hh_autoCF_cellsize'+str(cellsize)+'_randomsize'+str(randoms_size)+'_all', results_list)
#np.save(output_dir+catalog_name+'_5000_mocks_2PCF_all', results_list)
