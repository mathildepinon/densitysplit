import os
import copy
import pickle
import numpy as np
import pandas
import matplotlib as mpl
from matplotlib import pyplot as plt

from pypower import CatalogMesh
from . import utils

class Data:
    """
    Class Data.
    """
    def __init__(self, positions, redshift, boxsize, boxcenter, name='catalog', weights=None, velocities=None, mass_cut=None):

        self.name = name
        self.positions = positions
        self.size = np.shape(positions)[1]
        self.redshift = redshift
        self.boxsize = boxsize
        self.boxcenter = boxcenter
        self.offset = boxcenter - boxsize/2.
        self.velocities = velocities
        self.mass_cut = mass_cut
        self.weights = weights
        self.rsd = False
        self.positions_rsd = None

        self.cut_lower_mass(mass_cut)

    def cut_lower_mass(self, mass_cut):
        if mass_cut is not None:
            self.mass_cut = mass_cut

            self.positions = self.positions.T[self.weights > mass_cut].T
            self.size = np.shape(self.positions)[1]

            if self.velocities is not None:
                self.velocities = self.velocities.T[self.weights > mass_cut].T

            if self.positions_rsd is not None:
                self.positions_rsd = self.positions_rsd.T[self.weights > mass_cut].T

            if self.weights is not None:
                self.weights = self.weights[self.weights > mass_cut]

    def downsample(self, factor=1, seed=0):
        if factor != 1:
            np.random.seed(seed)
            sample = np.random.uniform(0., 1., self.positions.shape[1]) <= factor
            self.positions = self.positions[:, sample]
            self.size = np.shape(self.positions)[1]

            if self.velocities is not None:
                self.velocities = self.velocities[:, sample]

            if self.positions_rsd is not None:
                self.positions_rsd = self.positions_rsd[:, sample]

            if self.weights is not None:
                self.weights = self.weights[:, sample]

    def shift_boxcenter(self, offset):
        self.positions = self.positions + offset
        self.boxcenter = self.boxcenter + offset
        self.offset = self.offset + offset
        if self.positions_rsd is not None:
            self.positions_rsd = self.positions_rsd + offset


    def set_rsd(self, positions_rsd=None, hz=None, los='x'):
        a = 1/(1+self.redshift)

        if positions_rsd is not None:
            self.rsd = True
            self.positions_rsd = positions_rsd

        elif hz is not None and self.velocities is not None:
            self.rsd = True
            vlos = np.array([1. * (los == axis) for axis in 'xyz'])
            self.positions_rsd = (self.positions + (self.velocities * vlos[:, None])/(a*hz) - self.offset) % self.boxsize + self.offset


    def show_halos_map(self, fig, ax, cellsize, cut_direction, cut_idx, use_rsd=False, use_weights=False, **kwargs):
        if use_rsd and self.positions_rsd is not None:
            positions = self.positions_rsd
        else:
            positions = self.positions

        cut_directions_dict = {'x': 0, 'y': 1, 'z': 2}
        plot_directions = [key for key, value in cut_directions_dict.items() if key != cut_direction]

        dir1_in_cut, dir2_cut, dir3_cut = utils.get_slice_from_3D_points(positions, cut_direction, cut_idx, cellsize, self.boxsize, self.offset, return_indices=True)

        if use_weights and self.weights is not None:
            weights_in_cut = self.weights[dir1_in_cut]
            w = (weights_in_cut/np.max(weights_in_cut))*50000./self.boxsize
        else:
            w = 0.3#2000./self.boxsize

        ax.scatter(dir2_cut, dir3_cut, s=w, alpha=0.2, **kwargs)
        ax.set_xlabel(r'${} \; [h^{{-1}}\rm Mpc]$'.format(plot_directions[0]))
        ax.set_ylabel(r'${} \; [h^{{-1}}\rm Mpc]$'.format(plot_directions[1]))
        ax.set_aspect('equal', adjustable='box')

        
    def __getstate__(self):
        state = {}
        for name in ['name', 'positions', 'size', 'redshift',
                     'boxsize', 'boxcenter', 'offset',
                     'velocities', 'mass_cut', 'weights', 'rsd', 'positions_rsd']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    
    def __setstate__(self, state):
        self.__dict__.update(state)

        
    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new       


    def save(self, filename):
        try:
            np.save(filename, self.__getstate__(), allow_pickle=True)
        except:
            # for large catalogs, need pickle protocol 4:
            with open(filename+'.npy', 'wb') as f:
                pickle.dump(self.__getstate__(), f, protocol=4)


    @classmethod
    def load(cls, filename):
        try:
            state = np.load(filename, allow_pickle=True)[()]
        except:
            # for large catalogs:
            with open(filename, 'rb') as f:
                state = pickle.load(f)
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new
