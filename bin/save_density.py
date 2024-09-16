import os
import sys
import argparse
import numpy as np

from densitysplit import DensitySplit

for i in range(1, 25):
    print('mock ', i)
    ds_dir = '/feynman/work/dphp/mp270220/outputs/densitysplit/'
    ds_fn = 'AbacusSummit_2Gpc_z0.800_ph0{:02d}_downsampled_particles_nbar0.0034_cellsize2_cellsize2_resamplerngp_smoothingR10_3splits_randoms_size4_RH_CCF.npy'.format(i)
    
    densitysplit = DensitySplit.load(os.path.join(ds_dir, ds_fn))
    
    density_fn = os.path.join('/feynman/scratch/dphp/mp270220/outputs', 'AbacusSummit_2Gpc_z0.800_ph0{:02d}_downsampled_particles_nbar0.0034_cellsize2_cellsize2_resamplerngp_smoothingR10_density_mesh'.format(i))
    densitysplit.save_mesh(density_fn)
    print('density mesh saved')

    new_ds_fn =  'AbacusSummit_2Gpc_z0.800_ph0{:02d}_downsampled_particles_nbar0.0034_cellsize2_cellsize2_resamplerngp_smoothingR10_3splits_randoms_size4_RH_CCF'.format(i)
    densitysplit.save(os.path.join(ds_dir, new_ds_fn), save_density_mesh=False)
    print('density split saved without density mesh')