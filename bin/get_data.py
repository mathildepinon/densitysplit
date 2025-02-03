import numpy as np

from fitsio import read
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from densitysplit import catalog_data

output_dir = '/feynman/scratch/dphp/mp270220/abacus/'

# 1 Gpc/h
path_to_sim1 = '/feynman/scratch/dphp/ar264273/Abacus/AbacusSummit_highbase_c000_ph100/halos/z1.175'

# 2 Gpc/h
path_to_sim2_z1 = '/feynman/scratch/dphp/ar264273/Abacus/AbacusSummit_base_c000_ph0{:02d}/halos/z1.175'
path_to_sim2_z2 = '/feynman/scratch/dphp/ar264273/Abacus/AbacusSummit_base_c000_ph000/halos/z0.800'

# Galaxies
path_to_sim3 = '/feynman/scratch/dphp/mp270220/abacus/ELG_LRG_crossfit_z0.8-1.1_Abacussummit_highbase_c000_ph100.fits'
path_to_sim4 = '/feynman/scratch/dphp/mp270220/abacus/ELG_z0.950_ph0{:02d}.fits'
path_to_sim5 = '/feynman/scratch/dphp/mp270220/abacus/AbacusSummit_2Gpc_LRG_z0.800_ph0{:02d}.fits'
path_to_sim6 = '/feynman/scratch/dphp/mp270220/abacus/AbacusSummit_2Gpc_ELG_z0.800_ph0{:02d}.fits'

path_to_sim = [path_to_sim1,
               path_to_sim2_z1,
               path_to_sim2_z2,
               path_to_sim3,
               path_to_sim4,
               path_to_sim5,
               path_to_sim6]

catalog_names = ['AbacusSummit_1Gpc_z1.175',
                 'AbacusSummit_2Gpc_z1.175_ph0{:02d}',
                 'AbacusSummit_2Gpc_z0.800',
                 'AbacusSummit_1Gpc_z0.8-1.1',
                 'AbacusSummit_2Gpc_ELG_z0.950_ph0{:02d}',
                 'AbacusSummit_2Gpc_LRG_z0.800_ph0{:02d}',
                 'AbacusSummit_2Gpc_ELG_z0.800_ph0{:02d}']

# effective redshift
sim_z = [1.175, 
         1.175, 
         0.800,
         0.95,
         0.95,
         0.800,
         0.800]

sim_boxsizes = [1000, 
                2000, 
                2000,
                1000,
                2000,
                2000,
                2000]

def get_halos(path, name, boxsize, z, mass_cut=None):
    halo_catalog = CompaSOHaloCatalog(path, cleaned=True, fields=['id', 'x_L2com','v_L2com','N'])
    positions = halo_catalog.halos['x_L2com']
    velocities = halo_catalog.halos['v_L2com']
    weights = halo_catalog.halos['N']
    
    # Transpose arrays
    positions_reshaped = np.array(positions.T, dtype='f8')
    weights_reshaped = np.array(weights.T, dtype='f8')
    velocities_reshaped = np.array(velocities.T, dtype='f8')
    
    # Create Data instance
    data_catalog = catalog_data.Data(positions_reshaped, z, boxsize, boxcenter=0, name=name, 
                                     weights=weights_reshaped, velocities=velocities_reshaped,
                                     mass_cut=mass_cut)

    # Save Data instance
    data_catalog.save(output_dir+name)

def get_galaxies(path, name, boxsize, z, tracer=None):
    #data = read(path, columns=['x', 'y', 'z', 'vx', 'vy', 'vz', 'tracer'])
    data = read(path, columns=['x', 'y', 'z', 'vx', 'vy', 'vz'])
    # if tracer is not None:
    #     data = data[data['Tracer']==tracer]
    #     name += '_' + tracer
    # else:
    #     name += '_ELG_LRG'
    
    positions = np.array([data['x'], data['y'], data['z']])
    velocities = np.array([data['vx'], data['vy'], data['vz']])
    weights = None
    
    # Create Data instance
    data_catalog = catalog_data.Data(positions, z, boxsize, boxcenter=0, name=name, 
                                     weights=weights, velocities=velocities)

    # Save Data instance
    data_catalog.save(output_dir+name)

def apply_rsd(cat, z, boxsize, H_0=100, los='z', vsmear=None, cosmo=None):
    if cosmo is None :
        from cosmoprimo.fiducial import DESI
        cosmo = DESI(engine='class')
    rsd_factor = 1 / (1 / (1 + z) * H_0 * cosmo.efunc(z))
    pos_rsd = [cat[p] % boxsize if p !=los else (cat[p] + (cat['v'+p] + np.random.normal(0,vsmear, size=len(cat[p])))*rsd_factor) %boxsize if vsmear is not None else (cat[p] + cat['v'+p]*rsd_factor) %boxsize for p in 'xyz']
    return pos_rsd

sim = 6
mass_cut = 500 #for halos

for phase in np.arange(11, 25):
    get_galaxies(path_to_sim[sim].format(phase), catalog_names[sim].format(phase), sim_boxsizes[sim], sim_z[sim])
  