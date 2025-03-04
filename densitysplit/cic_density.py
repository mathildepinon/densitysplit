import time
import os
import copy
import numpy as np

from jax import jit
from jax import numpy as jnp
from jax import vmap
from functools import partial

from pmesh import ParticleMesh


# Count-in-cells in cubic box (new version, using pmesh)
def compute_cic_density(positions, boxsize, boxcenter, cellsize, smoothing_radius=None, weights=None, return_counts=False, mesh=None, return_inter_mesh=False):
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

    if mesh is None:
        mesh = np.zeros((subnmesh,)*ndim, dtype='f8')
    # user can provide mesh with pre-computed particle densities
    offset = boxcenter - boxsize/2.
    posi = positions - offset

    # apply padding on the edges of the box (assuming periodicity of the box)
    t0 = time.time()
    for idim in range(ndim):
        os0 = np.zeros(ndim)[:, None]
        osL = np.zeros(ndim)[:, None]
        os0[idim] = boxsize
        osL[idim] = -boxsize
        pos0 = os0 + posi[:, np.where(posi[idim, :] <= smoothing_radius)[0]]
        posL = osL + posi[:, np.where((boxsize - smoothing_radius < posi[idim, :]))[0]]
        posi = np.concatenate([posi, pos0, posL], axis=1)
    t1 = time.time()
    print('padding in elapsed time: {}'.format(t1-t0))

    t0 = time.time()
    for shift in shift_list:
        print('Shifting the mesh by {} cell units.'.format(shift))
        grid_idx = ((posi + 0.5*(cellsize-subcellsize) - shift[:, None]*subcellsize) / cellsize).astype('i4') # indices on the larger grid
        subgrid_idx = (grid_idx * cellsize / subcellsize + shift[:, None]).astype('i4') % subnmesh # indices on the smaller grid
        grid_pos = grid_idx * cellsize + (shift[:, None] + 0.5) * subcellsize
        dist_to_nearest_node = np.sum((grid_pos - posi)**2, axis=0)**0.5
        mask_particles = (dist_to_nearest_node < smoothing_radius) & np.all(grid_pos >= 0, axis=0) & np.all(grid_pos <= boxsize, axis=0)
        mask_grid_idx = np.argwhere(mask_particles).flatten()
        np.add.at(mesh, tuple(subgrid_idx[:, mask_grid_idx]), weights[mask_particles] if weights is not None else 1.)
    t1 = time.time()
    print('painted mesh in elapsed time: {}'.format(t1-t0))

    if return_inter_mesh:
        return mesh

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


@jit
def paint(field, positions, boxsize, cellsize, subcellsize, subnmesh, smoothing_radius, values, grid_shift):
    grid_idx = ((positions + 0.5*(cellsize-subcellsize) - grid_shift*subcellsize) / cellsize).astype('i4') # indices on the larger grid
    subgrid_idx = (grid_idx * cellsize / subcellsize + grid_shift).astype('i4') % subnmesh # indices on the smaller grid
    grid_pos = grid_idx * cellsize + (grid_shift + 0.5) * subcellsize
    dist_to_nearest_node = jnp.linalg.norm(grid_pos - positions, axis=0)
    mask_particles = (dist_to_nearest_node < smoothing_radius) & jnp.all(grid_pos >= 0, axis=0) & jnp.all(grid_pos <= boxsize, axis=0)
    values = values * mask_particles
    field = field.at[tuple(subgrid_idx)].add(values)
    return field

@partial(jit, static_argnums=2)
def padding_shift(idim, shift, ndim):
    pad = jnp.zeros(ndim)[:, None]
    pad = pad.at[idim].set(shift)
    return pad

@jit
def padding(positions, pad):
    padded_pos = positions + pad
    return padded_pos

def paint_padding(padding_args, field, positions, boxsize, cellsize, subcellsize, subnmesh, smoothing_radius, values, grid_shift):
    pad = padding_shift(padding_args['idim'], padding_args['shift'], padding_args['ndim'])
    padded_pos = padding(positions, pad)
    field = paint(field, padded_pos, boxsize, cellsize, subcellsize, subnmesh, smoothing_radius, values, grid_shift)
    return field

@jit
def sum_meshes(meshes):
    return jnp.sum(meshes, axis=0)


# Count-in-cells in cubic box (new version, using pmesh)
def jax_cic_density(positions, boxsize, boxcenter, cellsize, smoothing_radius=None, weights=None, return_counts=False, mesh=None, return_inter_mesh=False):
    t0 = time.time()
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
        weights = np.ones(positions.shape[1])
    print(norm)

    if mesh is None:
        mesh = np.zeros((subnmesh,)*ndim, dtype='f8')
    # user can provide mesh with pre-computed particle densities
    else:
        mesh = np.array(mesh)
    offset = boxcenter - boxsize/2.
    posi = positions - offset

    # apply padding on the edges of the box (assuming periodicity of the box)
    t0 = time.time()
    
    for grid_shift in shift_list:
        print('Shifting the mesh by {} cell units.'.format(grid_shift))
        global_args = [posi, boxsize, cellsize, subcellsize, subnmesh, smoothing_radius, weights, grid_shift[:, None]]
        mesh = paint(mesh, *global_args)
    
        idims, shifts = jnp.meshgrid(jnp.arange(ndim), jnp.array([-boxsize, boxsize]))
        padding_args = {'idim': idims.flatten(), 'shift': shifts.flatten(), 'ndim': ndim}
        mesh0 = np.zeros((subnmesh,)*ndim, dtype='f8')
        mesh_vec = vmap(paint_padding, in_axes=({'idim': 0, 'shift': 0, 'ndim': None}, *(None, )*9), out_axes=0)(padding_args, mesh0, *global_args)
        meshsum = sum_meshes(mesh_vec)
        
        mesh =  sum_meshes(jnp.array([meshsum, mesh]))

    #padding_args = {'idim': 0, 'shift': boxsize, 'ndim': ndim}
    #mesh = paint_padding(padding_args, mesh, posi, boxsize, cellsize, subcellsize, subnmesh, smoothing_radius, weights)
    
    t1 = time.time()
    print('painted mesh in elapsed time: {}s'.format(t1-t0))

    if return_inter_mesh:
        return mesh

    pm = ParticleMesh(BoxSize=(boxsize, )*ndim, Nmesh=(subnmesh, )*ndim, dtype=float)
    pmesh = pm.create(type='real')
    pmesh.value = np.array(mesh)

    norm = np.mean(pmesh.value)
    print(np.mean(pmesh.value))
    density_mesh = pmesh / norm - 1

    if return_counts:
        toret = pmesh
    else:
        toret = density_mesh

    return toret

