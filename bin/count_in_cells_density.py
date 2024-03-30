import os
import copy
import numpy as np

from densitysplit import catalog_data


# Count-in-cells in cubic box
def compute_cic_density(data, smoothing_radius, cellsize=None, use_rsd=False, use_weights=False):
    boxsize = data.boxsize
    offset = data.boxcenter - data.boxsize/2.
    
    if use_rsd and data.positions_rsd is not None:
        positions = data.positions_rsd
    else:
        positions = data.positions
        
    if use_weights and data.weights is not None:
        weights = data.weights
        norm = np.sum(weights) * (4/3 * np.pi * smoothing_radius**3) / boxsize**3
    else:
        weights = None
        norm = data.size * (4/3 * np.pi * smoothing_radius**3) / boxsize**3

    if cellsize is None:
        cellsize = smoothing_radius * 2
    else:
        if cellsize < 2 * smoothing_radius:
            print("Cellsize must be bigger than twice the smoothing radius.")
    
    def compute_density_mesh(pos):
        indices_in_grid = ((pos - offset) / cellsize + 0.5).astype('i4')
        grid_pos = indices_in_grid * cellsize + offset
        dist_to_nearest_node = np.sum((grid_pos - pos)**2, axis=0)**0.5
        mask_particles = dist_to_nearest_node < smoothing_radius

        nmesh = np.int32(boxsize / cellsize)
        mask_particles &= np.all((indices_in_grid > 0) & (indices_in_grid < nmesh), axis=0)
        mesh = np.zeros((nmesh - 1,)*3, dtype='f8')
        np.add.at(mesh, tuple(indices_in_grid[:, mask_particles] - 1), weights[mask_particles] if use_weights else 1.)
        return mesh

    data_mesh = compute_density_mesh(positions)
    #mesh = data_mesh / norm - 1

    return data_mesh
    
    
if __name__ == '__main__':
    from cosmoprimo import fiducial
    cosmo = fiducial.DESI()
    
    from pathlib import Path
    from abacusnbody.data import read_abacus
    import astropy.table

    path_to_sim = '/feynman/scratch/dphp/ar264273/Abacus/AbacusSummit_base_c000_ph000/halos/z0.800/'

    allp = []
    for fn in Path(path_to_sim).glob('*_rv_*/*.asdf'):
        batch = read_abacus.read_asdf(fn, load=['pos', 'vel'])
        batch_size = len(batch)
        sample = np.random.randint(0, batch_size, np.int(batch_size/50))
        allp += [batch[sample]]
    allp = astropy.table.vstack(allp)
    
    output_dir = '/feynman/work/dphp/mp270220/data/'
    name = 'AbacusSummit_2Gpc_z0.800_downsampled_particles'
    boxsize = 2000
    z = 0.8
    positions = allp['pos']
    velocities = allp['vel']

    # Transpose arrays
    positions_reshaped = np.array(positions.T, dtype='f8')
    velocities_reshaped = np.array(velocities.T, dtype='f8')

    # Create Data instance
    abacus_mock = catalog_data.Data(positions_reshaped, z, boxsize, boxcenter=0, name=name, 
                                    weights=None, velocities=velocities_reshaped,
                                    mass_cut=None)
    nbar = abacus_mock.size / abacus_mock.boxsize**3
    
    # Density smoothing parameters
    cellsize = 20
    R = 10
    
    #data_name = 'AbacusSummit_2Gpc_z0.800'
    data_name = 'AbacusSummit_2Gpc_z0.800_downsampled_particles_nbar{:.3f}'.format(nbar)
    #abacus_mock = catalog_data.Data.load('/feynman/work/dphp/mp270220/data/'+data_name+'.npy')
    
    density_cic = compute_cic_density(abacus_mock, R, cellsize=cellsize)
    
    output_dir = '/feynman/work/dphp/mp270220/outputs/'
    np.save(output_dir+data_name+'_density_cic_R{:02d}Mpc'.format(R), density_cic)
    #np.save(output_dir+data_name+'_density_contrast_cic_R{:02d}Mpc'.format(R), density_cic)