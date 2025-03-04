import time
import os
import numpy as np
from pathlib import Path
from abacusnbody.data import read_abacus
import astropy.table

from cosmoprimo import *

from densitysplit import catalog_data
from densitysplit.cic_density import compute_cic_density, jax_cic_density


# Mock parameters
cosmology = fiducial.AbacusSummitBase()
bg = cosmology.get_background()
z = 0.800
#downsampling = 0.0027801453496639457
#downsampling = 0.1
downsampling = 1.1
boxsize = 2000
cellsize = 5

output_dir = '/pscratch/sd/m/mpinon/abacus/'
path_to_sim = '/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/AbacusSummit_base_c000_ph000/halos/z{:.3f}/'.format(z)
print(Path(path_to_sim))

fields = ['halo', 'field']
nfiles = 34

mesh = None
ntot = 0

for f in fields:
    for i in range(nfiles): 
        fn = Path(path_to_sim)/ '{}_rv_A/{}_rv_A_{:03d}.asdf'.format(f, f, i)
        print('reading file:', fn)
        t0 = time.time()
        batch = read_abacus.read_asdf(fn, load=['pos', 'vel'])
        print('file read in elapsed time {}s'.format(time.time() - t0))
        batch_size = len(batch)
        print('file contains {} particles'.format(batch_size))

        np.random.seed(0)
        sample = np.random.uniform(0., 1., batch_size) < downsampling
        
        part = batch[sample]
        positions = part['pos']
        velocities = part['vel']
        print('downsampled file contains {} particles'.format(len(part)))
        ntot += len(part)
    
        # Transpose arrays
        positions_reshaped = np.array(positions.T, dtype='f8')
        velocities_reshaped = np.array(velocities.T, dtype='f8')

        # last file
        if (f=='field') & (i==33):
            return_inter_mesh = False
        else:
            return_inter_mesh = True
    
        print('computing density...')
        t1 = time.time()
        mesh = jax_cic_density(positions_reshaped, boxsize=boxsize, boxcenter=0, cellsize=cellsize, smoothing_radius=10, return_counts=True, mesh=mesh, return_inter_mesh=return_inter_mesh)
        print('density computed in elapsed time: {}s'.format(time.time()-t1))

#print('total mesh', mesh.value)
#print('mesh sum', np.sum(mesh.value))

nbar = ntot / (boxsize**3)
print('Total number of particles: {}, nbar = {}'.format(ntot, nbar))
delta_R = mesh.value.flatten() / (nbar * 4/3 * np.pi * 10**3) - 1

simname0 = 'AbacusSummit_2Gpc_z{:.3f}_ph0{{:02d}}_particles_nbar{:.4f}'.format(z, nbar)
simname = simname0.format(0)
outputname = simname + '_cellsize{:d}_resampler{}{}_delta_R_jax'.format(cellsize, 'tophat', '_smoothingR{:02d}'.format(10))

print('Save density at: {}'.format(os.path.join('/pscratch/sd/m/mpinon/densitysplits/', outputname)))
np.save(os.path.join('/pscratch/sd/m/mpinon/densitysplits', outputname), delta_R)

