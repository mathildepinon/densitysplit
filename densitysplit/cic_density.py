import os
import copy
import numpy as np

from pmesh import ParticleMesh


# Count-in-cells in cubic box (new version, using pmesh)
def compute_cic_density(positions, boxsize, boxcenter, cellsize, smoothing_radius=None, weights=None, return_counts=False):
    ndim = positions.shape[0] # 2 or 3
    
    if cellsize is None:
        cellsize = smoothing_radius * 2
        subcellsize = cellsize
    else:
        if cellsize < 2 * smoothing_radius:
            subcellsize = cellsize
            nshifts = np.ceil(2 * smoothing_radius / subcellsize).astype('i4')
            cellsize = nshifts * subcellsize
            print("Cellsize ({}) is smaller than the smoothing diameter, shifting the mesh {} times by {} (main grid has cellsize {}).".format(subcellsize, nshifts, subcellsize, cellsize))
        else:
            subcellsize = cellsize
            nshifts = 1

    if ndim==2:
        shift_list = np.array([np.array([i, j]) for i in range(nshifts) for j in range(nshifts)])
        v = 4 * np.pi * smoothing_radius**2
    if ndim==3:
        shift_list = np.array([np.array([i, j, k]) for i in range(nshifts) for j in range(nshifts) for k in range(nshifts)])
        v = 4/3 * np.pi * smoothing_radius**3

    subnmesh = np.int32(boxsize / subcellsize)
    nmesh = np.int32(boxsize / cellsize)
    print('main cellsize:', cellsize)
    
    if weights is not None:
        norm = np.sum(weights) * v / boxsize**ndim
    else:
        norm = positions.shape[1] * v / boxsize**ndim
    print(norm)

    mesh = np.zeros((subnmesh,)*ndim, dtype='f8')
    offset = boxcenter - boxsize/2.
    posi = positions - offset

    # apply padding on the edges of the box (assuming periodicity of the box)
    for idim in range(ndim):
        os0 = np.zeros(ndim)[:, None]
        osL = np.zeros(ndim)[:, None]
        os0[idim] = boxsize
        osL[idim] = -boxsize
        pos0 = os0 + posi[:, np.where(positions[idim, :] <= smoothing_radius)[0]]
        posL = osL + posi[:, np.where((boxsize - smoothing_radius < posi[idim, :]))[0]]
        posi = np.concatenate([posi, pos0, posL], axis=1)

    for shift in shift_list:
        print('Shifting the mesh by {} cell units.'.format(shift))
        grid_idx = ((posi + 0.5*(cellsize-subcellsize) - shift[:, None]*subcellsize) / cellsize).astype('i4') # indices on the larger grid
        subgrid_idx = (grid_idx * cellsize / subcellsize + shift[:, None]).astype('i4') % subnmesh # indices on the smaller grid
        grid_pos = grid_idx * cellsize + (shift[:, None] + 0.5) * subcellsize
        dist_to_nearest_node = np.sum((grid_pos - posi)**2, axis=0)**0.5
        mask_particles = (dist_to_nearest_node < smoothing_radius) & np.all(grid_pos >= 0, axis=0) & np.all(grid_pos <= boxsize, axis=0)
        mask_grid_idx = np.argwhere(mask_particles).flatten()
        np.add.at(mesh, tuple(subgrid_idx[:, mask_grid_idx]), weights[mask_particles] if weights is not None else 1.)

    pm = ParticleMesh(BoxSize=(boxsize, )*ndim, Nmesh=(subnmesh, )*ndim, dtype=float)
    pmesh = pm.create(type='real')
    pmesh.value = mesh

    norm = np.mean(pmesh.value)
    print(np.mean(pmesh.value))
    density_mesh = pmesh / norm - 1

    if return_counts:
        toret = pmesh
    else:
        toret = density_mesh

    return toret


## old version
def compute_cic_density_old(positions, boxsize, boxcenter, cellsize, smoothing_radius=None, weights=None, return_counts=False, return_mesh_positions=False):
    offset = boxcenter - boxsize/2.
        
    shifts = [0]
    if cellsize is None:
        cellsize = smoothing_radius * 2
    else:
        if cellsize < 2 * smoothing_radius:
            effcellsize = 2 * smoothing_radius
            shifts = np.arange(0, effcellsize, cellsize)
            print("Cellsize ({}) is smaller than the smoothing diameter, shifting the mesh {} times by {}.".format(cellsize, len(shifts), effcellsize))
            cellsize = effcellsize

    nmesh = np.int32(boxsize / cellsize)
    
    if weights is not None:
        norm = np.sum(weights) * (4/3 * np.pi * smoothing_radius**3) / boxsize**3
    else:
        norm = positions.shape[1] * (4/3 * np.pi * smoothing_radius**3) / boxsize**3
    print(norm)

    positions = positions - offset
        
    def compute_density_mesh(pos):
        indices_in_grid = (pos / cellsize + 0.5).astype('i4')
        grid_pos = indices_in_grid * cellsize
        dist_to_nearest_node = np.sum((grid_pos - pos)**2, axis=0)**0.5
        mask_particles = dist_to_nearest_node < smoothing_radius

        mesh = np.zeros((nmesh,)*3, dtype='f8')
        mesh_pos = (np.array([np.indices(mesh.shape)[i].ravel() for i in range(3)]) * 1.5 * cellsize % boxsize) + offset
        
        # periodic box
        indices_in_grid = indices_in_grid % nmesh
        np.add.at(mesh, tuple(indices_in_grid[:, mask_particles]), weights[mask_particles] if weights is not None else 1.)

        return mesh.ravel(), mesh_pos

    data_meshes = list()
    mesh_positions = list()
    
    for shift in shifts:
        pos = (positions - shift) % boxsize
        data_mesh, mesh_pos = compute_density_mesh(pos)
        data_meshes.append(data_mesh)
        
        mesh_positions.append(mesh_pos)

    mesh_positions = np.concatenate(mesh_positions, axis=1)
    data_mesh = np.concatenate(data_meshes)
    
    norm = np.mean(data_mesh)
    print(np.mean(data_mesh))
    mesh = data_mesh / norm - 1

    if return_counts:
        toret = data_mesh
    else:
        toret = mesh

    if return_mesh_positions:
        return toret, mesh_positions
    else:
        return toret
