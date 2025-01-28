import os
import copy
import pickle
import logging 
import numpy as np
import pandas
import matplotlib as mpl
from matplotlib import pyplot as plt

from pypower import CatalogMesh
from pycorr import TwoPointCorrelationFunction, setup_logging

from . import catalog_data
from . import utils
from .utils import BaseClass, mkdir
from densitysplit.corr_func_utils import get_poles, get_split_poles
from densitysplit.cic_density import compute_cic_density


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
        self.resampler_conversions = {'ngp': 'nnb', 'cic': 'cic', 'tsc': 'tsc', 'pcs': 'pcs'}

    def compute_density(self, data, cellsize, resampler, smoothing_radius=None, cellsize2=None, smoothing_radius2=None, use_rsd=False, los=None, hz=None, use_weights=False, return_counts=False):
        """Compute density contrast on a mesh."""
        self.logger.info('Compute density on a mesh with cellsize {}.'.format(cellsize))

        if use_rsd:
            self.logger.info('Using positions in redshift space.')
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
            norm = self.size

        if resampler=='tophat':
            # density is computed directly in spheres around mesh positions in configuration space
            density_mesh = compute_cic_density(positions, boxsize=self.boxsize, boxcenter=self.boxcenter, cellsize=cellsize, smoothing_radius=smoothing_radius, weights=weights, return_counts=return_counts)
            self.density_mesh = density_mesh
            self.data_densities = None

            if smoothing_radius2 is not None and smoothing_radius2 != smoothing_radius:
                density_mesh = compute_cic_density(positions, boxsize=self.boxsize, boxcenter=self.boxcenter, cellsize=cellsize2, smoothing_radius=smoothing_radius2, weights=weights, return_counts=return_counts)
                self.density_mesh2 = density_mesh2

        else:
            mesh = CatalogMesh(data_positions=positions, data_weights=weights,
                               interlacing=0,
                               boxsize=self.boxsize, boxcenter=self.boxcenter,
                               resampler=resampler,
                               cellsize=cellsize)
            
            self.logger.info('Paint data to mesh.')
            painted_mesh = mesh.to_mesh(field='data')
            if smoothing_radius is not None:
                painted_mesh = painted_mesh.r2c().apply(TopHat(r=smoothing_radius))
                painted_mesh = painted_mesh.c2r()
            nmesh = mesh.nmesh[0]
    
            # Compute density contrast
            self.logger.info('Compute density contrast.')
            print(norm/(nmesh**3))
            print(np.mean(painted_mesh))
            density_mesh = painted_mesh/(norm/(nmesh**3)) - 1
           
            if cellsize2 is not None and cellsize2 != cellsize:
                self.logger.info('Compute density with smoothing scale {}.'.format(cellsize2))
                mesh2 = CatalogMesh(data_positions=positions, data_weights=weights,
                                   interlacing=0,
                                   boxsize=self.boxsize, boxcenter=self.boxcenter,
                                   resampler=resampler,
                                   cellsize=cellsize2)      
                painted_mesh2 = mesh2.to_mesh(field='data')
                if smoothing_radius2 is not None:
                    painted_mesh2 = painted_mesh2.r2c().apply(TopHat(r=smoothing_radius2))
                    painted_mesh2 = painted_mesh2.c2r()
                nmesh2 = mesh2.nmesh[0]
                density_mesh2 = painted_mesh2/(norm/(nmesh2**3)) - 1
                self.density_mesh2 = density_mesh2
    
            # Get densities at each point
            shifted_positions = positions - self.offset
            # resampler name conversions
            self.logger.info('Read density contrast at data positions.')
            # NB: readout method assumes positions are in [0, boxsize]
            self.resampler_conversions = {'ngp': 'nnb', 'cic': 'cic', 'tsc': 'tsc', 'pcs': 'pcs'}
            self.data_densities = density_mesh.readout(shifted_positions.T, resampler=self.resampler_conversions[resampler])
            self.density_mesh = density_mesh
            
        self.cellsize = cellsize
        self.resampler = resampler
        self.smoothing_radius = smoothing_radius
        self.cellsize2 = cellsize2
        self.smoothing_radius2 = smoothing_radius2
        
    
    def readout_density(self, positions='randoms', size=None, resampler=None, seed=0, mesh=1, return_positions=False):
        if resampler is None:
            resampler = self.resampler
        self.resampler_conversions = {'ngp': 'nnb', 'cic': 'cic', 'tsc': 'tsc', 'pcs': 'pcs'}
        if size is None:
            size = self.size
        if mesh==1:
            density_mesh = self.density_mesh
            cellsize = self.cellsize
        elif mesh==2:
            density_mesh = self.density_mesh2
            cellsize = self.cellsize2
        if isinstance(positions, str):
            if positions=='randoms':
                rng = np.random.RandomState(seed=seed)
                pos = np.array([rng.uniform(0., 1., size)*b for b in (self.boxsize,)*3])
            elif positions=='mesh': # positions must be 'mesh' if resampler is 'tophat'
                pos = (0.5 + np.indices(density_mesh.value.shape)) * cellsize
                if return_positions:
                    return density_mesh.value.ravel(), self.offset + np.array([pos[i].ravel() for i in range(3)])
                else:
                    return density_mesh.value.ravel()
        else:
            pos = positions
        densities = density_mesh.readout(pos.T, resampler=self.resampler_conversions[resampler])
        if return_positions:
            return densities, self.offset + pos
        else:
            return densities


    def split_density(self, nsplits=2, bins=None, labels=None, return_indices=False):
        if labels is None:
            #labels = ['DS{}'.format(i+1) for i in range(nsplits)]
            labels = [(i+1) for i in range(nsplits)]

        self.nsplits = nsplits
        self.split_labels = labels

        if hasattr(self, 'data_densities') and self.data_densities is not None: # if resampler is not 'tophat'
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

        self.split_bins = bins
        split_mesh = np.empty(np.shape(self.density_mesh))
        split_indices = list()
        split_densities = list()

        for i in range(nsplits):
            split_min = bins[i]
            split_max = bins[i+1]
            if i == 0:
                split_mesh[self.density_mesh < split_max] = 1
            elif i == nsplits-1:
                split_mesh[self.density_mesh >= split_min] = nsplits
            else:
                split_mesh[np.logical_and(self.density_mesh >= split_min, self.density_mesh < split_max)] = i+1

            if hasattr(self, 'data_densities') and self.data_densities is not None: # if resampler is not 'tophat'
                indices = (splits == i+1)
                split_indices.append(indices)
                split_densities.append(self.data_densities.T[indices].T)

        if return_indices:
            return split_mesh, split_indices
        else:
            return split_mesh


    #def sample_splits(self, size, seed=42):
    #    split_samples = sample_splits(self.density_mesh, self.resampler_conversions[self.resampler], self.split_bins, size, self.boxsize, self.offset, self.cellsize, seed=seed)
    #    return split_samples

    def sample_splits(self, positions, size=None, seed=42, norm=None):
        densities, positions = self.readout_density(positions=positions, size=size, resampler=self.resampler, return_positions=True)

        if self.resampler=='tophat':
            densities = densities / norm - 1
    
        split_samples = list()
    
        nsplits = len(self.split_bins)-1
    
        for i in range(nsplits):
            split_min = self.split_bins[i]
            split_max = self.split_bins[i+1]
            if i == 0:
                split = (densities < split_max)
            elif i == nsplits-1:
                split = (densities >= split_min)
            else:
                split = np.logical_and((densities >= split_min), (densities < split_max))
    
            split_samples.append(np.array(positions)[..., split])
    
        return split_samples

    
    def compute_smoothed_corr(self, edges, positions2=None, weights2=None, seed=0, los='x', nthreads=128, norm=None, mode='smu'):

        if self.resampler=='tophat':
            densities, positions1 = self.readout_density(positions='mesh', resampler=self.resampler, return_positions=True)
            #positions1 = positions1 - self.offset
            densities = densities / norm - 1
        else:
            ## Generate random particles and readout density at each particle
            rng = np.random.RandomState(seed=seed)
            positions1 = [o + rng.uniform(0., 1., self.size)*b for o, b in zip((self.offset,)*3, (self.boxsize,)*3)]
            shifted_positions1 = np.array(positions1) - self.offset
            densities = self.readout_density(shifted_positions1, resampler=self.resampler)
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
                if self.resampler=='tophat':
                    densities2, positions2 = self.readout_density('mesh', resampler=self.resampler, mesh=2, return_positions=True)
                    densities2 = densities2 / norm - 1
                else:
                    densities2 = self.readout_density(shifted_positions1.T, resampler=self.resampler, mesh=2)
                weights2 = 1 + densities2

        #rng = np.random.RandomState(seed=42)
        #mask = rng.uniform(0., 1., positions1.shape[-1]) < 0.1

        smoothed_corr = TwoPointCorrelationFunction(mode, edges,
                                            data_positions1=positions1, data_positions2=positions2,
                                            data_weights1=weights1, data_weights2=weights2,
                                            #randoms_positions1=positions1, 
                                            estimator='natural', boxsize=self.boxsize,
                                            engine='corrfunc', nthreads=nthreads,
                                            los=los)

        # when positions are mesh positions, need to specify mesh positions as randoms
        if self.resampler=='tophat':
            smoothed_corr.R1R2.wcounts[...] = smoothed_corr.D1D2.ncounts[...]
            smoothed_corr.R1R2.wnorm = positions1.shape[1]*(positions1.shape[1]-1)

        self.smoothed_corr = smoothed_corr
        
        return smoothed_corr


    def compute_ds_data_corr(self, edges, positions2=None, weights2=None, seed=0, output_dir='', randoms_size=1, los='x', nthreads=128, norm=None, mode='smu'):
        """Compute cross-correlation of random points in density splits with data."""
        
        if self.cellsize2 is None:
            if positions2 is None:
                raise ValueError('Positions must be provided if cellsize2 is None.')
            else:
                self.logger.info('Use input positions.')
    
        else:
            self.logger.info('Use smoothed density contrast for data (cellsize2 = {}).'.format(self.cellsize2))
            if self.resampler=='tophat':
                densities, positions2 = self.readout_density(positions='mesh', resampler=self.resampler, return_positions=True)
                densities = densities / norm - 1 
                split_samples = self.sample_splits(positions='mesh', norm=norm)
            else:
                rng = np.random.RandomState(seed=seed)
                positions2 = [o + rng.uniform(0., 1., self.size)*b for o, b in zip((self.offset,)*3, (self.boxsize,)*3)]
                shifted_positions2 = np.array(positions2) - self.offset
                if self.cellsize2 == self.cellsize: 
                    densities = self.readout_density(positions=shifted_positions2, resampler=self.resampler, seed=seed, mesh=1)
                else:
                    densities = self.readout_density(positions=shifted_positions2, resampler=self.resampler, seed=seed, mesh=2)
                split_samples = self.sample_splits(positions='randoms', size=randoms_size*self.size, seed=seed)
        
        weights = 1 + densities    
        densitysplits = list()

        #rng = np.random.RandomState(seed=42)

        R1R2 = None
    
        self.logger.info('Compute density splits.')
        for i in range(self.nsplits):
            self.logger.info('Density split {}'.format(i))
            #maski = rng.uniform(0., 1., split_samples[i].shape[-1]) < 0.1
            #mask = rng.uniform(0., 1., positions2.shape[-1]) < 0.1
            
            dsplit = TwoPointCorrelationFunction(mode, edges,
                                                    data_positions1=split_samples[i], data_positions2=positions2,
                                                    data_weights1=None, data_weights2=weights,
                                                    randoms_positions1=positions2, randoms_positions2=positions2,
                                                    R1R2=R1R2,
                                                    boxsize=self.boxsize,
                                                    engine='corrfunc', nthreads=nthreads,
                                                    los=los)

            R1R2 = dsplit.R1R2

            densitysplits.append(dsplit)
    
        self.ds_data_corr = densitysplits
        
        return densitysplits


    def compute_biasfunction(self, deltabins, smul=1, norm=1, return_corr=False, sep=None, mu=None, swidth=None, method='mesh'):
        # if method=="sphere":
        #     deltaR1, deltaR2 = self.compute_jointpdf_delta_R1_R2(s=sep, sbin=(sep-swidth, sep+swidth), mu=mu, query_positions='mesh')
        #     print('deltaR1 shape: ', deltaR1.shape)
        #     deltaR1, deltaR2 =  deltaR1/norm - 1,  deltaR2/norm - 1
        #     corr = np.mean(deltaR1*deltaR2, axis=-1)
        #     bias_func = list()
        #     for i in range(len(deltabins)-1):
        #         mask = (deltaR1 >= deltabins[i]) & (deltaR1 < deltabins[i+1])
        #         bias = np.nanmean(np.where(mask, deltaR2, np.nan), axis=-1)
        #         bias_func.append(bias)
        #     if return_corr:
        #         return np.array(bias_func) / corr, corr
        #     else:
        #         return np.array(bias_func) / corr
 
        mesh_indices = np.array([np.indices(self.density_mesh.value.shape)[i].ravel() for i in range(3)])

        new_indices = mesh_indices.copy()
        nmesh = np.int32(self.boxsize/self.cellsize)

        delta_R1 = self.density_mesh.value / norm - 1

        delta_R2 = list()

        for i in range(3):
            new_indices[i] = (new_indices[i] + smul) % nmesh
            tmp = delta_R1[new_indices[0], new_indices[1], new_indices[2]].ravel()
            delta_R2.append(tmp)

        delta_R2 = np.array(delta_R2)
        delta_R1 = delta_R1.ravel()

        bias_func = list()

        for i in range(len(deltabins)-1):
            mask = (delta_R1 >= deltabins[i]) & (delta_R1 < deltabins[i+1])
            delta_R2_masked = delta_R2[:, mask]
            n1 = np.sum(mask)
            bias = np.sum(delta_R2_masked) / (3*n1)
            bias_func.append(bias)

        mean_delta_R2 = np.mean(delta_R2, axis=0)
        corr = np.mean(delta_R1*mean_delta_R2)

        if return_corr:
            return np.array(bias_func) / corr, corr
        else:
            return np.array(bias_func) / corr

    
    def compute_jointpdf_delta_R1_R2(self, s=None, sbin=None, query_positions='randoms', sample_size=None, mu=None, los='x', seed=0, split=None, norm=None):      
        if sample_size is None:
            sample_size = self.size

        resampler = self.resampler

        rng = np.random.RandomState(seed=seed)
        np.random.seed(seed)

        vlos = np.array([1. * (los == axis) for axis in 'xyz'])
        
        if resampler =='tophat':
            ndim = len(self.density_mesh.value.shape)
            mesh_indices = np.array([np.indices(self.density_mesh.value.shape)[i].ravel() for i in range(ndim)])
            mesh_positions = self.offset + (0.5 + np.indices(self.density_mesh.value.shape)) * self.cellsize
            mesh_positions_ravel = np.array([mesh_positions[i].ravel() for i in range(ndim)])
            start_position = mesh_positions_ravel[:, 0]
            if ndim==2:
                offsets = np.array([np.array([i, j]) for i in (0, self.boxsize) for j in (0, self.boxsize)])
            elif ndim==3:
                offsets = np.array([np.array([i, j, k]) for i in (0, self.boxsize) for j in (0, self.boxsize) for k in (0, self.boxsize)])
            mask = False
            for offset in offsets:
                pos_norm = np.sum((mesh_positions_ravel-start_position[:, None]-offset[:, None])**2, axis=0)**0.5
                #pos_mu = mesh_positions_ravel * vlos[:, None] / pos_norm
                mask = mask | ((pos_norm >= sbin[0]) & (pos_norm < sbin[1]))
            masked_loc = np.argwhere(mask).flatten()
            random_choices = np.random.choice(masked_loc, size=mesh_positions_ravel.shape[1])
            masked_indices = mesh_indices[:, random_choices]
            nmesh = np.int32(self.boxsize/self.cellsize)
            new_indices = (mesh_indices + masked_indices) % nmesh
            #return mesh_positions, mesh_positions_ravel, masked_indices, new_indices
            
            self.logger.info('Readout density at particles positions 1.')
            delta_R1 = self.density_mesh.value #/ norm - 1
            if self.cellsize2 != self.cellsize:
                pass #to do
            else:
                self.logger.info('Readout density at particles positions 2.')
                if ndim==2:
                    delta_R2 = delta_R1[new_indices[0], new_indices[1]]
                elif ndim==3:
                    delta_R2 = delta_R1[new_indices[0], new_indices[1], new_indices[2]]
                
            return np.array([delta_R1.ravel(), delta_R2.ravel()])

        else:
            ## Generate particles where to readout density, either on a mesh or random positions
            if query_positions=='mesh':
                nmesh = np.rint(sample_size**(1/3))
                self.logger.info('Generate {} regularly spaced particles (nmesh = {}).'.format(nmesh**3, nmesh))
                idxi = indxj = indxk = np.arange(-self.offset, self.boxsize - self.offset, self.boxsize/nmesh)
                grid_pos = np.meshgrid(idxi, indxj, indxk, indexing='ij')
                positions1 = np.array([grid_pos[i].flatten() for i in range(3)])
            elif query_positions=='randoms':
                self.logger.info('Generate {} random positions.'.format(sample_size))
                if split is not None:
                    split_samples = self.sample_splits(positions='randoms', size=sample_size, seed=seed)
                    positions1 = split_samples[split]
                else:
                    positions1 = np.array([rng.uniform(0., 1., sample_size)*b for b in (self.boxsize,)*3])
    
            sample_size = positions1.shape[1]
            phi = rng.uniform(0., 2*np.pi, sample_size)
            
            if mu is None:
                theta = rng.uniform(0., np.pi, sample_size)
            else:
                if len(list(mu)):
                    theta = np.tile(np.arccos(mu), sample_size)
                    positions1 = np.repeat(positions1, len(mu), axis=1)
                    phi = np.repeat(phi, len(mu))
                else:
                    theta = np.full(sample_size, np.arccos(mu))
            
            if los == 'x':
                relpos = np.array([s*np.cos(theta), s*np.sin(theta)*np.cos(phi), s*np.sin(theta)*np.sin(phi)])
            elif los == 'y':
                relpos = np.array([s*np.sin(theta)*np.sin(phi), s*np.cos(theta), s*np.sin(theta)*np.cos(phi)])
            elif los == 'z':
                relpos = np.array([s*np.sin(theta)*np.cos(phi), s*np.sin(theta)*np.sin(phi), s*np.cos(theta)])
            # periodic box
            positions2 = (positions1 + relpos) % self.boxsize
        
            self.logger.info('Readout density at particles positions 1.')
            delta_R1 = self.readout_density(positions=positions1, resampler=resampler, mesh=1)
    
            self.logger.info('Readout density at particles positions 2.')
            if self.cellsize2 != self.cellsize:
                delta_R2 = self.readout_density(positions=positions2, resampler=resampler, mesh=2)
            else:
                delta_R2 = self.readout_density(positions=positions2, resampler=resampler, mesh=1)
    
            if mu is not None and len(list(mu)):
                delta_R1 = delta_R1.reshape((sample_size, len(mu))).T
                delta_R2 = delta_R2.reshape((sample_size, len(mu))).T
    
        return np.array([delta_R1, delta_R2])

    
    def show_halos_map(self, fig, ax, cellsize, cut_direction, cut_idx, positions=None, weights=None, split=False,
                       color='white', colors=None, density=True, cmap=mpl.cm.viridis):

        cut_directions_dict = {'x': 0, 'y': 1, 'z': 2}
        plot_directions = [key for key, value in cut_directions_dict.items() if key != cut_direction]

        dir1_in_cut, dir2_cut, dir3_cut = utils.get_slice_from_3D_points(positions, cut_direction, cut_idx, cellsize, self.boxsize, self.offset, return_indices=True)

        if split:
            if colors is None:
                colors = plt.get_cmap('viridis', self.nsplits)

            split_mesh, split_indices = self.split_density(nsplits=self.nsplits, return_indices=True)

            for i in range(self.nsplits):
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
                cbar = fig.colorbar(c, ax=ax)
                cbar.set_label('$\delta_R$', rotation=90)
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
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('$\delta_R$', rotation=90)

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

        split_samples = self.sample_splits('randoms', self.size, seed=42)

        for i, split_sample in enumerate(split_samples):
            dir2_cut, dir3_cut = utils.get_slice_from_3D_points(split_sample, cut_direction, cut_idx, cellsize, self.boxsize, self.offset, return_indices=False)
            ax.scatter(dir2_cut, dir3_cut, s=3, color=colors[i], alpha=0.5, label='DS{}'.format(self.split_labels[i]))

        ax.set_xlabel(plot_directions[0]+' [Mpc/h]')
        ax.set_ylabel(plot_directions[1]+' [Mpc/h]')
        ax.set_aspect('equal', adjustable='box')


    def __getstate__(self, save_density_mesh=True):
        state = {}
        for name in ['boxsize', 'boxcenter', 'offset', 'size',
                     'cellsize', 'cellsize2', 'resampler', 'smoothing_radius', 'smoothing_radius2', 'data_densities',
                     'nsplits', 'split_bins', 'split_labels',
                     'smoothed_corr', 'ds_data_corr']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        #if hasattr(self, 'data'):
        #    state['data'] = self.data.__getstate__()
        if hasattr(self, 'density_mesh') & save_density_mesh:
            state['density_mesh'] = {'array': self.density_mesh.value, 'boxsize': self.density_mesh.pm.BoxSize}
        if hasattr(self, 'density_mesh2') & save_density_mesh:
            state['density_mesh2'] = {'array': self.density_mesh2.value, 'boxsize': self.density_mesh2.pm.BoxSize}
        return state
        

    def __setstate__(self, state):
        self.__dict__.update(state)
        #self.data = catalog_data.Data.from_state(self.data)
        self.logger = logging.getLogger('DensitySplit')
        if hasattr(self, 'density_mesh'):
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

            
    def save(self, filename, save_density_mesh=True):
        print('Saving {}.'.format(filename))
        mkdir(os.path.dirname(filename))
        try:
            np.save(filename, self.__getstate__(save_density_mesh=save_density_mesh), allow_pickle=True)
        except:
            # for large files, need pickle protocol 4:
            with open(filename+'.npy', 'wb') as f:
                pickle.dump(self.__getstate__(save_density_mesh=save_density_mesh), f, protocol=4)

    
    def save_mesh(self, filename, mesh=None):
        print('Saving {}.'.format(filename))
        mkdir(os.path.dirname(filename))
        if mesh is None:
            mesh = self.density_mesh
        state = {'array': mesh.value, 'boxsize': mesh.pm.BoxSize}
        with open(filename+'.npy', 'wb') as f:
            pickle.dump(state, f, protocol=4)

    
    @classmethod
    def load(cls, filename, mesh_filename=None):
        try:
            state = np.load(filename, allow_pickle=True)[()]
        except:
            # for large catalogs:
            with open(filename, 'rb') as f:
                state = pickle.load(f)
        # load density mesh separately
        if mesh_filename is not None:
            with open(mesh_filename, 'rb') as f:
                density_mesh = pickle.load(f)
            state['density_mesh'] = density_mesh
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


class CountInCellsDensitySplitMeasurement(BaseClass):
    """
    Class DensitySplitMeasurement, to store a colleciton of density-split measurements.
    """
    def __init__(self, densitysplits, mesh_fn=None):
        self.logger = logging.getLogger('CountInCellsDensitySplitMeasurement')
        self.logger.info('Initializing CountInCellsDensitySplitMeasurement')

        mocks = list()
        densitysplits = list(densitysplits)
        for i, mock_fn in enumerate(densitysplits):
            if isinstance(mock_fn, str):
                if os.path.isfile(mock_fn):
                    if mesh_fn is not None:
                        mock = DensitySplit.load(mock_fn, mesh_filename=mesh_fn[i])
                    else:
                        mock = DensitySplit.load(mock_fn)
                        #mock.smoothing_radius = 10
                        #mock.save(mock_fn)
                    mocks.append(mock)
                else:
                    self.logger.warning('File {} not found, ignoring this file.'.format(mock_fn))
            else:
                mocks.append(mock_fn)

        if len(mocks) == 0:
            self.logger.error('No file found.')

        self.mocks = mocks
        self.nmocks = len(mocks)
        self.logger.info('Loaded {} mocks.'.format(self.nmocks))
        
        self.nbar_all = np.array([mocks[i].size/mocks[i].boxsize**3 for i in range(self.nmocks)])
        self.nbar = np.mean(self.nbar_all)
        self.smoothing_radius = np.array([mocks[i].smoothing_radius for i in range(self.nmocks)])
        if len(set(self.smoothing_radius)) == 1:
            self.smoothing_radius = self.smoothing_radius[0]
        else:
            self.logger.warning('Smoothing radius is not the same for all mocks!')
        self.norm_all = self.nbar_all * 4/3 * np.pi * self.smoothing_radius**3
        self.norm = np.mean(self.norm_all)

        self.cellsize = np.array([self.mocks[i].cellsize for i in range(self.nmocks)])
        if len(set(self.cellsize)) == 1:
            self.cellsize  = self.cellsize[0]
        else:
            self.logger.warning('Cell size is not the same for all mocks!')

        self.kvals = np.arange(0, np.round(10*self.norm))
        self.dedges = 1/self.norm
        self.edges = self.kvals/self.norm - 1 - self.dedges/2
        self.deltavals = (self.edges[1:] + self.edges[:-1])/2

        self.bias_function_x = dict()
        self.bias_function = dict()
        self.bias_corr = dict()

        self.hist2D_x = dict()
        self.hist2D = dict()

        bins = list()
        for i, mock in enumerate(self.mocks):
            if hasattr(mock, 'split_bins'):
                bins.append(mock.split_bins)
            else:
                bins.append(None)
        self.bins = np.array(bins)
        
        if (not np.all(self.bins == None)) and (not np.all(np.all(self.bins == self.bins[0], axis=1))):
            self.logger.warning('Density split bins are not equal for all mocks!')
        if not np.all(self.bins == None):
            self.bins = np.mean(self.bins, axis=0)
            self.logger.info('Taking average bins: {}'.format(self.bins))
            self.nsplits = len(self.bins) - 1
        else:
            self.bins = None
            self.nsplits = None

    def set_pdf1D(self, xlim=None, rebin=None):
        N_R = np.array([self.mocks[i].density_mesh.value.flatten() for i in range(self.nmocks)])
        delta_R = N_R / self.norm - 1
        self.avg_all = np.mean(delta_R, axis=1)
        self.avg = np.mean(delta_R.flatten())
        self.sigma_all = np.std(delta_R, axis=1)
        self.sigma = np.std(delta_R.flatten())
        self.delta3_all = np.mean(delta_R**3, axis=1)
        self.delta3 = np.mean(delta_R.flatten()**3)
        edges = self.edges
        if rebin is not None:
            edges = edges[::rebin]
        if xlim is not None:
            mask = (edges >= xlim[0]) & (edges <= xlim[1])
            edges = edges[mask]
        self.pdf1D_x = (edges[1:] + edges[:-1])/2
        self.pdf1D = np.array([np.histogram(delta_R[i], bins=edges, density=True)[0] for i in range(self.nmocks)])

    def set_bias_function(self, sep, mu=None, xlim=None, rebin=None, swidth=None, method='mesh'):
        edges = self.edges
        if rebin is not None:
            edges = edges[::rebin]
        if xlim is not None:
            mask = (edges >= xlim[0]) & (edges <= xlim[1])
            edges = edges[mask]
        self.bias_function_x[str(sep)] = (edges[1:] + edges[:-1])/2
        corr = list()
        bias_function = list()
        for i in range(self.nmocks):   
            if method=='mesh':            
                tmp = self.mocks[i].compute_biasfunction(deltabins=edges, smul=sep//self.cellsize, norm=self.norm, return_corr=True, method=method)
            elif method=='sphere':
                tmp = self.mocks[i].compute_biasfunction(deltabins=edges, sep=sep, swidth=swidth, mu=mu, norm=self.norm, return_corr=True, method=method)
            bias_function.append(tmp[0])
            corr.append(tmp[1])
        self.bias_corr[str(sep)] = np.array(corr)
        print('bias func shape: ', np.array(bias_function).shape)
        self.bias_function[str(sep)] = np.array(bias_function)

    def set_pdf2D(self, sep, swidth, xlim=None, rebin=None):
        N_R = [self.mocks[i].compute_jointpdf_delta_R1_R2(s=sep, sbin=(sep-swidth, sep+swidth), query_positions='mesh') for i in range(self.nmocks)]
        N1_R = np.array([N_R[i][0] for i in range(self.nmocks)])
        N2_R = np.array([N_R[i][1] for i in range(self.nmocks)])
        delta_R1 = N1_R / self.norm - 1
        delta_R2 = N2_R / self.norm - 1
        edges = self.edges
        if rebin is not None:
            edges = edges[::rebin]
        if xlim is not None:
            mask = (edges >= xlim[0]) & (edges <= xlim[1])
            edges = edges[mask]
        hist2D = list()
        for i in range(self.nmocks):
            hist2D.append(np.histogram2d(delta_R1[i], delta_R2[i], bins=(edges, edges), density=True)[0])
        self.hist2D[str(sep)] = np.array(hist2D)
        self.hist2D_x[str(sep)] = edges

    def set_densitysplits(self, sep=None):
        ds_corr = np.array([self.mocks[i].ds_data_corr for i in range(self.nmocks)])
        #ds_poles, cov = get_split_poles(ds_corr, ells=None)
        #std = np.nan if cov.size == 1 else np.array_split(np.array(np.array_split(np.diag(cov)**0.5, 1)), self.nsplits, axis=1)
        if sep is None: 
            self.sep = ds_corr[0][0].get_corr(return_sep=True)[0].ravel()
        else:
            self.sep = sep
        self.smoothed_corr = np.array([self.mocks[i].smoothed_corr for i in range(self.nmocks)])
        self.ds_corr = ds_corr

    def __getstate__(self):
        state = {}
        for name in self.__dict__.keys():
            if (name != 'mocks') & (name != 'logger'): # don't save all data, don't save logger
                state[name] = getattr(self, name)
        return state



