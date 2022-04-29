import os
import copy

import numpy as np
import pandas
import matplotlib as mpl
from matplotlib import pyplot as plt

from pypower import CatalogMesh

from . import catalog_data
from . import utils



def sample_splits(split_mesh, size, boxsize, offset, cellsize, seed=42):
    rng = np.random.RandomState(seed=seed)
    positions = [o + rng.uniform(0., 1., size)*b for o, b in zip((offset,)*3, (boxsize,)*3)]
    nmesh = round(boxsize/cellsize)
    positions_grid_indices = ((np.array(positions) - offset + cellsize/2.) // cellsize).astype(int) % nmesh

    split_samples = list()

    for i in np.unique(split_mesh):
        split = (split_mesh == i)
        sample_in_split = split[tuple(positions_grid_indices.tolist())]
        split_samples.append(np.array(positions).T[sample_in_split].T)

    return split_samples


class DensitySplit:
    """
    Class DensitySplit.
    """

    def __init__(self, data):

        self.data = data
        self.boxsize = data.boxsize
        self.boxcenter = data.boxcenter
        self.offset = data.boxcenter - data.boxsize/2.


    def shift_boxcenter(self, offset):
        self.data.shift_boxcenter(offset)
        self.boxsize = self.boxsize + offset
        self.boxcenter = self.boxcenter + offset
        self.offset =  self.offset + offset

        if hasattr(self, 'split_positions') and self.split_positions is not None:
            self.split_positions = [self.split_positions[i] + offset for i in range(len(self.split_positions))]

        if hasattr(self, 'split_samples') and self.split_samples is not None:
            self.split_samples = [self.split_samples[i] + offset for i in range(len(self.split_samples))]


    def compute_density(self, cellsize, resampler, use_rsd=False, use_weights=False):

        data = self.data

        if use_rsd and data.positions_rsd is not None:
            positions = data.positions_rsd
            self.use_rsd = True
        else:
            positions = data.positions
            self.use_rsd = False

        if use_weights and data.weights is not None:
            weights = data.weights
            self.use_weights = True
            norm = sum(weights)
        else:
            weights = None
            self.use_weights = False
            norm = data.size

        mesh = CatalogMesh(data_positions=positions, data_weights=weights,
                           interlacing=0,
                           boxsize=self.boxsize, boxcenter=self.boxcenter,
                           resampler=resampler,
                           cellsize=cellsize)

        painted_mesh = mesh.to_mesh(field='data')
        nmesh = mesh.nmesh[0]

        # Compute density contrast
        density_mesh = np.array(painted_mesh/(norm/(nmesh**3))) - 1

        # Get positions of catalog particles in the catalog mesh
        positions_grid_indices = ((np.array(positions) - self.offset + cellsize/2.) // cellsize).astype(int) % nmesh

        # Get densities at each point
        self.data_densities = density_mesh[tuple(positions_grid_indices.tolist())]

        self.density_mesh = density_mesh
        self.cellsize = cellsize
        self.resampler = resampler


    def split_density(self, nsplits=2, labels=None):
        if labels is None:
            #labels = ['DS{}'.format(i+1) for i in range(nsplits)]
            labels = [(i+1) for i in range(nsplits)]

        splits, bins = pandas.qcut(self.data_densities, nsplits, labels=labels, retbins=True)

        split_mesh = np.empty(np.shape(self.density_mesh))
        split_indices = list()
        split_densities = list()
        split_positions = list()

        for i in range(nsplits):
            split_min = bins[i]
            split_max = bins[i+1]
            if i == 0:
                split_mesh[self.density_mesh <= split_max] = 1
            elif i == nsplits-1:
                split_mesh[self.density_mesh > split_min] = nsplits
            else:
                split_mesh[np.logical_and(self.density_mesh > split_min, self.density_mesh <= split_max)] = i+1

            indices = (splits == i+1)

            split_indices.append(indices)
            split_densities.append(self.data_densities.T[indices].T)
            split_positions.append(self.data.positions.T[indices].T)

        self.nsplits = nsplits
        self.split_bins = bins
        self.split_labels = labels
        self.split_mesh = split_mesh
        self.split_indices = split_indices
        self.split_densities = split_densities
        self.split_positions = split_positions


    def sample_splits(self, size, seed=42, update=True):
        split_samples = sample_splits(self.split_mesh, size, self.boxsize, self.offset, self.cellsize, seed=seed)
        if update:
            self.split_samples = split_samples
        return split_samples


    def show_halos_map(self, fig, ax, cellsize, cut_direction, cut_idx, use_rsd=False, use_weights=False, split=False,
                       color='white', colors=None, density=True, cmap=mpl.cm.viridis):

        if use_rsd and self.data.positions_rsd is not None:
            positions = self.data.positions_rsd
        else:
            positions = self.data.positions

        cut_directions_dict = {'x': 0, 'y': 1, 'z': 2}
        plot_directions = [key for key, value in cut_directions_dict.items() if key != cut_direction]

        dir1_in_cut, dir2_cut, dir3_cut = utils.get_slice_from_3D_points(positions, cut_direction, cut_idx, cellsize, self.boxsize, self.offset, return_indices=True)

        if split:
            if colors is None:
                colors = plt.get_cmap('viridis', self.nsplits)

            for i in range(len(self.split_positions)):
                split_dir1_in_cut = np.logical_and(dir1_in_cut, self.split_indices[i])
                split_dir2_cut = (positions[cut_directions_dict[plot_directions[0]]][split_dir1_in_cut] - self.offset + cellsize/2.) % self.boxsize + self.offset
                split_dir3_cut = (positions[cut_directions_dict[plot_directions[1]]][split_dir1_in_cut] - self.offset + cellsize/2.) % self.boxsize + self.offset

                if use_weights and self.data.weights is not None:
                    split_weights_in_cut = self.data.weights[split_dir1_in_cut]
                    w = (split_weights_in_cut/np.max(split_weights_in_cut))*50000./self.boxsize
                else:
                    w = 2000./self.boxsize

                ax.scatter(split_dir2_cut, split_dir3_cut, s=w, color=colors[i], alpha=0.5, label='DS{}'.format(self.split_labels[i]))

        else:
            if use_weights and self.data.weights is not None:
                weights_in_cut = self.data.weights[dir1_in_cut]
                w = (weights_in_cut/np.max(weights_in_cut))*50000./self.boxsize
            else:
                w = 2000./self.boxsize

            if density:
                c = ax.scatter(dir2_cut, dir3_cut, s=w, alpha=0.5, c=self.data_densities[dir1_in_cut], cmap=cmap)
                fig.colorbar(c, ax=ax, label='$\delta$')
            else:
                ax.scatter(dir2_cut, dir3_cut, s=w, alpha=0.5, color=color)

        ax.set_xlabel(plot_directions[0]+' [Mpc/h]')
        ax.set_ylabel(plot_directions[1]+' [Mpc/h]')
        ax.set_aspect('equal', adjustable='box')


    def show_density_map(self, fig, ax, cut_direction, cut_idx, cmap=mpl.cm.viridis, show_halos=False, split=False):

        cut_directions_dict = {'x': 0, 'y': 1, 'z': 2}
        plot_directions = [key for key, value in cut_directions_dict.items() if key != cut_direction]

        cut = [slice(None)]*3
        cut[cut_directions_dict[cut_direction]] = cut_idx
        density_mesh_cut = self.density_mesh[tuple(cut)]

        extent = self.offset, self.offset + self.boxsize, self.offset, self.offset + self.boxsize
        c = ax.imshow(density_mesh_cut.T, cmap=cmap, extent=extent, origin='lower')
        ax.set_xlabel(plot_directions[0]+' [Mpc/h]')
        ax.set_ylabel(plot_directions[1]+' [Mpc/h]')
        fig.colorbar(c, ax=ax, label='$\delta$')

        if show_halos:
            self.show_halos_map(fig, ax, self.cellsize, cut_direction, cut_idx, use_rsd=self.use_rsd, use_weights=self.use_weights, split=split, density=False)


    def show_split_density_map(self, fig, ax, cut_direction, cut_idx, cmap=None, show_halos=False, split_halos=True, colors=('white', 'black')):
        if cmap is None:
            cmap = plt.get_cmap('viridis', self.nsplits)

        cut_directions_dict = {'x': 0, 'y': 1, 'z': 2}
        plot_directions = [key for key, value in cut_directions_dict.items() if key != cut_direction]

        cut = [slice(None)]*3
        cut[cut_directions_dict[cut_direction]] = cut_idx
        split_mesh_cut = self.split_mesh[tuple(cut)]

        extent = self.offset, self.offset + self.boxsize, self.offset, self.offset + self.boxsize
        c = ax.imshow(split_mesh_cut.T, cmap=cmap, extent=extent, origin='lower')
        ax.set_xlabel(plot_directions[0]+' [Mpc/h]')
        ax.set_ylabel(plot_directions[1]+' [Mpc/h]')

        # Discrete color bar for splits
        bounds = self.split_bins
        cbar = fig.colorbar(c, ax=ax)
        cbar_sep = np.linspace(1, self.nsplits, self.nsplits+1)
        ticks = [(cbar_sep[i] + cbar_sep[i+1])/2. for i in range(self.nsplits)]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(['DS{}'.format(l) for l in self.split_labels])

        if show_halos:
            self.show_halos_map(fig, ax, self.cellsize, cut_direction, cut_idx, color='white',
                                use_rsd=self.use_rsd, use_weights=self.use_weights, split=split_halos, colors=colors)


    def show_randoms_map(self, fig, ax, cellsize, cut_direction, cut_idx, colors):
        cut_directions_dict = {'x': 0, 'y': 1, 'z': 2}
        plot_directions = [key for key, value in cut_directions_dict.items() if key != cut_direction]

        for i, split_sample in enumerate(self.split_samples):
            dir2_cut, dir3_cut = utils.get_slice_from_3D_points(split_sample, cut_direction, cut_idx, cellsize, self.boxsize, self.offset, return_indices=False)
            ax.scatter(dir2_cut, dir3_cut, s=3, color=colors[i], alpha=0.5, label='DS{}'.format(self.split_labels[i]))

        ax.set_xlabel(plot_directions[0]+' [Mpc/h]')
        ax.set_ylabel(plot_directions[1]+' [Mpc/h]')
        ax.set_aspect('equal', adjustable='box')


    def __getstate__(self):
        state = {}
        for name in ['data', 'boxsize', 'boxcenter', 'offset',
                     'use_rsd', 'use_weights', 'cellsize', 'resampler', 'density_mesh', 'data_densities',
                     'nsplits', 'split_bins', 'split_labels', 'split_mesh', 'split_indices', 'split_densities', 'split_positions', 'split_samples']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


    def __setstate__(self, state):
        self.__dict__.update(state)


    def save(self, filename):
        np.save(filename, self.__getstate__(), allow_pickle=True)


    @classmethod
    def load(cls, filename):
        state = np.load(filename, allow_pickle=True)[()]
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new
