import numpy as np
from pathlib import Path
from abacusnbody.data import read_abacus
import astropy.table

from cosmoprimo import *

from densitysplit import catalog_data


# Mock parameters
cosmology = fiducial.AbacusSummitBase()
bg = cosmology.get_background()
z = 0.800
downsampling = 0.0027801453496639457

output_dir = '/feynman/scratch/dphp/mp270220/abacus/'
path_to_sim = '/feynman/scratch/dphp/ar264273/Abacus/AbacusSummit_base_c000_ph000/halos/z{:.3f}/'.format(z)
print(Path(path_to_sim))

allp = []
for fn in Path(path_to_sim).glob('*_rv_*/*.asdf'):
    batch = read_abacus.read_asdf(fn, load=['pos', 'vel'])
    batch_size = len(batch)
    print('batch size: ', batch_size)
    sample = np.random.uniform(0., 1., batch_size) < downsampling
    
    allp += [batch[sample]]
print(len(allp), ' particles')

allp = astropy.table.vstack(allp)

print(len(allp))

boxsize = 2000
positions = allp['pos']
velocities = allp['vel']
nbar = len(positions) / (boxsize**3)
print('nbar: ', nbar)

# Transpose arrays
positions_reshaped = np.array(positions.T, dtype='f8')
velocities_reshaped = np.array(velocities.T, dtype='f8')

# Create Data instance
name = 'AbacusSummit_2Gpc_z{:.3f}_ph000_downsampled_particles_nbar{:.4f}'.format(z, nbar)
catalog = catalog_data.Data(positions_reshaped, z, boxsize, boxcenter=0, name=name,
               weights=None, velocities=velocities_reshaped, mass_cut=None)

# Save Data instance
catalog.save(output_dir+name)