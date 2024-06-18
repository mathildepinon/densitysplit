import os
import copy
import logging 
import numpy as np
import pandas
import matplotlib as mpl
from matplotlib import pyplot as plt

from pypower import CatalogMesh
from pycorr import TwoPointCorrelationFunction, setup_logging

from . import catalog_data
from . import utils
from .utils import BaseClass



def sample_splits(density_mesh, resampler, split_bins, size, boxsize, offset, cellsize, seed=42):
    rng = np.random.RandomState(seed=seed)
    positions = [o + rng.uniform(0., 1., size)*b for o, b in zip((offset,)*3, (boxsize,)*3)]
    shifted_positions = np.array(positions) - offset
    densities = density_mesh.readout(shifted_positions.T, resampler=resampler)

    split_samples = list()

    nsplits = len(split_bins)-1

    for i in range(nsplits):
        split_min = split_bins[i]
        split_max = split_bins[i+1]
        if i == 0:
            split = (densities <= split_max)
        elif i == nsplits-1:
            split = (densities > split_min)
        else:
            split = np.logical_and((densities > split_min), (densities <= split_max))

        split_samples.append(np.array(positions).T[split].T)

    return split_samples


class DensitySplit(BaseClass):
    """
    Class DensitySplit.
    """
    
    def __init__(self, data):
        self.logger = logging.getLogger('DensitySplit')
        self.logger.info('Initializing DensitySplit')
        self.size = data.size
        self.boxsize = data.boxsize
        self.boxcenter = data.boxcenter
        self.offset = data.boxcenter - data.boxsize/2.


    def compute_density(self, data, cellsize, resampler, cellsize2=None, use_rsd=False, los=None, hz=None, use_weights=False):
        """Compute density contrast on a mesh."""
        self.logger.info('Compute density on a mesh with cellsize {}.'.format(cellsize))

        if use_rsd:
            if data.positions_rsd is None:
                data.set_rsd(hz=hz, los=los)
            positions = data.positions_rsd
        else:
            positions = data.positions

        if use_weights and data.weights is not None:
            weights = data.weights
            norm = sum(weights)
        else:
            weights = None
            norm = data.size

        mesh = CatalogMesh(data_positions=positions, data_weights=weights,
                           interlacing=0,
                           boxsize=self.boxsize, boxcenter=self.boxcenter,
                           resampler=resampler,
                           cellsize=cellsize)
        
        self.logger.info('Paint data to mesh.')
        painted_mesh = mesh.to_mesh(field='data')
        #painted_mesh = painted_mesh.r2c().apply(TopHat(r=20))
        #painted_mesh = painted_mesh.c2r()
        nmesh = mesh.nmesh[0]

        # Compute density contrast
        self.logger.info('Compute density contrast.')
        density_mesh = painted_mesh/(norm/(nmesh**3)) - 1
       
        if cellsize2 is not None and cellsize2 != cellsize:
            self.logger.info('Compute density with smoothing scale {}.'.format(cellsize2))
            mesh2 = CatalogMesh(data_positions=positions, data_weights=weights,
                               interlacing=0,
                               boxsize=self.boxsize, boxcenter=self.boxcenter,
                               resampler=resampler,
                               cellsize=cellsize2)      
            painted_mesh2 = mesh2.to_mesh(field='data')
            nmesh2 = mesh2.nmesh[0]
            density_mesh2 = painted_mesh2/(norm/(nmesh2**3)) - 1
            self.density_mesh2 = density_mesh2

        # Get densities at each point
        shifted_positions = positions - self.offset
        # resampler name conversions
        resampler_conversions = {'ngp': 'nnb', 'cic': 'cic', 'tsc': 'tsc', 'pcs': 'pcs'}
        self.logger.info('Read density contrast at data positions.')
        self.data_densities = density_mesh.readout(shifted_positions.T, resampler=resampler_conversions[resampler])

        self.density_mesh = density_mesh
        self.cellsize = cellsize
        self.resampler = resampler
        self.cellsize2 = cellsize2
        
    
    def readout_density(self, positions='randoms', rsd=False, resampler='tsc', seed=0, mesh=1):
        if mesh==1:
            density_mesh = self.density_mesh
        elif mesh==2:
            density_mesh = self.density_mesh2
        if isinstance(positions, str):
            if positions=='data':
                pos = self.data.positions - self.offset
            if positions=='data_rsd':
                pos = self.data.positions_rsd - self.offset
            if positions=='randoms':
                rng = np.random.RandomState(seed=seed)
                pos = np.array([rng.uniform(0., 1., self.data.size)*b for b in (self.data.boxsize,)*3])
        else:
            pos = positions
        densities = density_mesh.readout(pos.T, resampler=resampler)
        return densities


    def split_density(self, nsplits=2, bins=None, labels=None, return_incides=False):
        if labels is None:
            #labels = ['DS{}'.format(i+1) for i in range(nsplits)]
            labels = [(i+1) for i in range(nsplits)]

        if bins is None:
            # Use quantiles
            splits, bins = pandas.qcut(self.data_densities, nsplits, labels=labels, retbins=True)
        else:
            # Check consistency between nsplits and bins
            if nsplits + 1 == len(bins):
                # Use predefined bins
                splits, bins = pandas.cut(self.data_densities, bins, labels=labels, retbins=True)
            else:
                raise ValueError('bins must have length nsplits + 1.')

        split_mesh = np.empty(np.shape(self.density_mesh))
        split_indices = list()
        split_densities = list()

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

        self.nsplits = nsplits
        self.split_bins = bins
        self.split_labels = labels

        if return_incides:
            return split_mesh, split_indices
        else:
            return split_mesh


    def sample_splits(self, size, seed=42):
        split_samples = sample_splits(self.density_mesh, self.resampler, self.split_bins, size, self.boxsize, self.offset, self.cellsize, seed=seed)
        return split_samples


    def compute_smoothed_corr(self, edges, positions2=None, weights2=None, seed=0, nthreads=128):
        data = self.data

        ## Generate random particles and readout density at each particle
        rng = np.random.RandomState(seed=seed)
        positions1 = [o + rng.uniform(0., 1., self.size)*b for o, b in zip((self.offset,)*3, (self.boxsize,)*3)]
        shifted_positions1 = np.array(positions1) - self.offset
        densities = self.density_mesh.readout(shifted_positions1.T, resampler=self.resampler)
        weights1 = 1 + densities
    
        if self.cellsize2 is None:
            if positions2 is None:
                raise ValueError('Positions must be provided if cellsize2 is None.')
            else:
                self.logger.info('Use input positions.')
 
        else:
            self.logger.info('Use smoothed density contrast for second term (cellsize2 = {}).'.format(self.cellsize2))
            if self.cellsize2 == self.cellsize:
                positions2 = None
                weights2 = None
            else:
                positions2 = positions1
                densities2 = self.density_mesh2.readout(shifted_positions1.T, resampler=self.resampler)
                weights2 = 1 + densities2
    
        smoothed_corr = TwoPointCorrelationFunction('smu', edges,
                                            data_positions1=positions1, data_positions2=positions2,
                                            data_weights1=weights1, data_weights2=weights2,
                                            boxsize=self.boxsize,
                                            engine='corrfunc', nthreads=nthreads,
                                            los=los)

        self.smoothed_corr = smoothed_corr
        
        return smoothed_corr


    def compute_ds_data_corr(self, edges, positions2=None, weights2=None, seed=0, output_dir='', randoms_size=1, nthreads=128):
        """Compute cross-correlation of random points in density splits with data."""
        
        if self.cellsize2 is None:
            if positions2 is None:
                raise ValueError('Positions must be provided if cellsize2 is None.')
            else:
                self.logger.info('Use input positions.')
    
        else:
            self.logger.info('Use smoothed density contrast for data (cellsize2 = {}).'.format(self.cellsize2))
            rng = np.random.RandomState(seed=seed)
            positions = [o + rng.uniform(0., 1., self.data.size)*b for o, b in zip((self.offset,)*3, (self.boxsize,)*3)]
            shifted_positions = np.array(positions) - self.offset
            if self.cellsize2 == self.cellsize: 
                densities = self.density_mesh.readout(shifted_positions.T, resampler=self.resampler)
            else:
                densities = self.density_mesh2.readout(shifted_positions.T, resampler=self.resampler)
            weights = 1 + densities
            
        split_samples = self.sample_splits(size=randoms_size*data.size, seed=seed)
        cellsize = self.cellsize
    
        densitysplits = list()
    
        self.logger.info('Compute density splits.')
        for i in range(self.nsplits):
            self.logger.info('Density split {}'.format(i))
            dsplit = TwoPointCorrelationFunction('smu', edges,
                                                    data_positions1=split_samples[i], data_positions2=positions,
                                                    data_weights1=None, data_weights2=weights,
                                                    boxsize=self.boxsize,
                                                    engine='corrfunc', nthreads=nthreads,
                                                    los = los)
            densitysplits.append(dsplit)
    
        self.ds_data_corr = densitysplits
        
        return densitysplits

    
    def show_halos_map(self, fig, ax, cellsize, cut_direction, cut_idx, positions=None, weights=None, split=False,
                       color='white', colors=None, density=True, cmap=mpl.cm.viridis):

        cut_directions_dict = {'x': 0, 'y': 1, 'z': 2}
        plot_directions = [key for key, value in cut_directions_dict.items() if key != cut_direction]

        dir1_in_cut, dir2_cut, dir3_cut = utils.get_slice_from_3D_points(positions, cut_direction, cut_idx, cellsize, self.boxsize, self.offset, return_indices=True)

        if split:
            if colors is None:
                colors = plt.get_cmap('viridis', self.nsplits)

            split_mesh, split_indices = self.split_density(nsplits=self.nsplits, return_indices=True)

            for i in range(len(self.nsplits)):
                split_dir1_in_cut = np.logical_and(dir1_in_cut, split_indices[i])
                split_dir2_cut = (positions[cut_directions_dict[plot_directions[0]]][split_dir1_in_cut] - self.offset + cellsize/2.) % self.boxsize + self.offset
                split_dir3_cut = (positions[cut_directions_dict[plot_directions[1]]][split_dir1_in_cut] - self.offset + cellsize/2.) % self.boxsize + self.offset

                if weights is not None:
                    split_weights_in_cut = weights[split_dir1_in_cut]
                    w = (split_weights_in_cut/np.max(split_weights_in_cut))*50000./self.boxsize
                else:
                    w = 2000./self.boxsize

                ax.scatter(split_dir2_cut, split_dir3_cut, s=w, color=colors[i], alpha=0.5, label='DS{}'.format(self.split_labels[i]))

        else:
            if weights is not None:
                weights_in_cut = weights[dir1_in_cut]
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


    def show_density_map(self, fig, ax, cut_direction, cut_idx, cmap=mpl.cm.viridis, show_halos=False, positions=None, weights=None, split=False):

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
            self.show_halos_map(fig, ax, self.cellsize, cut_direction, cut_idx, positions=positions, weights=weights, split=split, density=False)


    def show_split_density_map(self, fig, ax, cut_direction, cut_idx, cmap=None, show_halos=False, positions=None, weights=None, split_halos=True, colors=('white', 'black')):
        if cmap is None:
            cmap = plt.get_cmap('viridis', self.nsplits)

        cut_directions_dict = {'x': 0, 'y': 1, 'z': 2}
        plot_directions = [key for key, value in cut_directions_dict.items() if key != cut_direction]

        cut = [slice(None)]*3
        cut[cut_directions_dict[cut_direction]] = cut_idx
        split_mesh_cut = self.split_density(nsplits=self.nsplits)[tuple(cut)]

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
                                positions=positions, weights=weights, split=split_halos, colors=colors)


    def show_randoms_map(self, fig, ax, cellsize, cut_direction, cut_idx, colors):
        cut_directions_dict = {'x': 0, 'y': 1, 'z': 2}
        plot_directions = [key for key, value in cut_directions_dict.items() if key != cut_direction]

        split_samples = self.sample_splits(self.size, seed=42):

        for i, split_sample in enumerate(split_samples):
            dir2_cut, dir3_cut = utils.get_slice_from_3D_points(split_sample, cut_direction, cut_idx, cellsize, self.boxsize, self.offset, return_indices=False)
            ax.scatter(dir2_cut, dir3_cut, s=3, color=colors[i], alpha=0.5, label='DS{}'.format(self.split_labels[i]))

        ax.set_xlabel(plot_directions[0]+' [Mpc/h]')
        ax.set_ylabel(plot_directions[1]+' [Mpc/h]')
        ax.set_aspect('equal', adjustable='box')


    def __getstate__(self):
        state = {}
        for name in ['boxsize', 'boxcenter', 'offset',
                     'cellsize', 'cellsize2', 'resampler', 'data_densities',
                     'nsplits', 'split_bins', 'split_labels',
                     'smoothed_corr', 'ds_data_corr']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        #if hasattr(self, 'data'):
        #    state['data'] = self.data.__getstate__()
        if hasattr(self, 'density_mesh'):
            state['density_mesh'] = {'array': self.density_mesh.value, 'boxsize': self.density_mesh.pm.BoxSize}
        if hasattr(self, 'density_mesh2'):
            state['density_mesh2'] = {'array': self.density_mesh2.value, 'boxsize': self.density_mesh2.pm.BoxSize}
        return state


    def __setstate__(self, state):
        self.__dict__.update(state)
        self.data = catalog_data.Data.from_state(self.data)
        from pmesh.pm import ParticleMesh
        pm = ParticleMesh(BoxSize=self.density_mesh['boxsize'], Nmesh=self.density_mesh['array'].shape, dtype=self.density_mesh['array'].dtype)
        mesh = pm.create(type='real')
        mesh.unravel(self.density_mesh['array'].ravel())
        self.density_mesh = mesh
        if hasattr(self, 'density_mesh2'):
            pm = ParticleMesh(BoxSize=self.density_mesh2['boxsize'], Nmesh=self.density_mesh2['array'].shape, dtype=self.density_mesh2['array'].dtype)
            mesh2 = pm.create(type='real')
            mesh2.unravel(self.density_mesh2['array'].ravel())
            self.density_mesh2 = mesh2


    @classmethod
    def load(cls, filename):
        state = np.load(filename, allow_pickle=True)[()]
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new


class TopHat(object):
    '''Top-hat filter in Fourier space
    adapted from https://github.com/bccp/nbodykit/

    Parameters
    ----------
    r : float
        the radius of the top-hat filter
    '''
    def __init__(self, r):
        self.r = r

    def __call__(self, k, v):
        r = self.r
        k = sum(ki ** 2 for ki in k) ** 0.5
        kr = k * r
        with np.errstate(divide='ignore', invalid='ignore'):
            w = 3 * (np.sin(kr) / kr ** 3 - np.cos(kr) / kr ** 2)
        w[k == 0] = 1.0
        return w * v


