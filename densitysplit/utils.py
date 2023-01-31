import os
import copy

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from pmesh import ParticleMesh


def get_slice_from_3D_points(points, cut_direction, cut_idx, cellsize, boxsize, offset, return_indices=False):
    """
    Get points in a given slice of a set of 3D positions.
    """
    cut_directions_dict = {'x': 0, 'y': 1, 'z': 2}
    plot_directions = [value for key, value in cut_directions_dict.items() if key != cut_direction]

    cut_direction_positions = points[cut_directions_dict[cut_direction]]

    cut_direction_grid_idx = (cut_direction_positions - offset + cellsize/2.) // cellsize

    dir1_in_cut = (cut_direction_grid_idx == cut_idx)

    dir2_cut = (points[plot_directions[0]][dir1_in_cut] - offset + cellsize/2.) % boxsize + offset
    dir3_cut = (points[plot_directions[1]][dir1_in_cut] - offset + cellsize/2.) % boxsize + offset

    if return_indices:
        return dir1_in_cut, dir2_cut, dir3_cut
    else:
        return dir2_cut, dir3_cut


def extract_subcovmatrix(s, cov, ells, nsplits, split_extract):
    """
    Extract sub-covariance matrix that corresponds to a given split index given by split_extract

    Arguments
    ---------
    s : 1D array, separations.

    cov : 2D array, original covariance matrix, including poles ells and nsplits splits

    ells : int list, multipoles contained in cov

    nsplits : integer, number of splits contained in cov

    split_extract : integer, index of the split to extract from cov
    """
    ns = len(s)
    nells = len(ells)

    cov_extract = cov[split_extract*nells*ns:(split_extract+1)*nells*ns,
                      split_extract*nells*ns:(split_extract+1)*nells*ns]

    return cov_extract


def truncate_xiell(lower_s_limit, s, xiell, ells, cov, split=False, nsplits=1):
    """
    Truncate separations, correlation function multipoles and covariance matrix above a given separation s_lower_limit.

    Arguments
    ---------
    lower_s_limit : float, spearation threshold in Mpc/h

    s : array, separations

    xiell : array, correlation function multipoles

    ells : int list, multipoles in xiell

    cov : 2D array, covariance matrix of the correlation function multipoles
    """
    s_truncated = s[s>lower_s_limit]
    ns = len(s)

    if split:
        xiell_toret = list()
        for split in range(nsplits):
            xiell_truncated_list = list()
            for ill, ell in enumerate(ells):
                xiell_truncated_list.append(xiell[split][ill][s>lower_s_limit])
                first_index = np.sum(np.logical_not(s>lower_s_limit))

            xiell_truncated = np.array(xiell_truncated_list)
            xiell_toret.append(xiell_truncated)

        xiell_toret = np.array(xiell_toret)

    else:
        nsplits = 1

        xiell_truncated_list = list()
        for ill, ell in enumerate(ells):
            xiell_truncated_list.append(xiell[ill][s>lower_s_limit])

        xiell_truncated = np.array(xiell_truncated_list)
        xiell_toret = xiell_truncated


    # Truncate the whole covariance matrix
    ns_trunc = len(s_truncated)
    nells = len(ells)
    cov_truncated_full = np.zeros((ns_trunc*nells*nsplits, ns_trunc*nells*nsplits))

    for i in range(nells*nsplits):
        for j in range(nells*nsplits):
            cov_truncated_full[i*ns_trunc:(i+1)*ns_trunc,j*ns_trunc:(j+1)*ns_trunc] = cov[(i+1)*ns-ns_trunc:(i+1)*ns,(j+1)*ns-ns_trunc:(j+1)*ns]

    return s_truncated, xiell_toret, cov_truncated_full


def compute_chisq(ydata, sigma, fitted_model):
    r = ydata - fitted_model

    chisq = r.T @ np.linalg.inv(sigma) @ r

    return chisq


def plot_corrcoef(cov, ells, s, nsplits):
    stddev = np.sqrt(np.diag(cov).real)
    corrcoef = cov / stddev[:, None] / stddev[None, :]

    ns = len(s)
    nells = len(ells)

    fig, lax = plt.subplots(nrows=nells*nsplits, ncols=nells*nsplits, sharex=False, sharey=False, figsize=(10, 8), squeeze=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    norm = Normalize(vmin=-1, vmax=1)
    for i in range(nells*nsplits):
        for j in range(nells*nsplits):
            ax = lax[nells*nsplits-1-i][j]
            mesh = ax.pcolor(s, s, corrcoef[i*ns:(i+1)*ns,j*ns:(j+1)*ns].T, norm=norm, cmap=plt.get_cmap('coolwarm'))
            if i>0: ax.xaxis.set_visible(False)
            else: ax.set_xlabel(r'$s$  [Mpc/h]'
                                +'\n'+r'$\ell={}$'.format(ells[j % nells])
                                +'\n''DS{}'.format(j//nells +1))
            if j>0: ax.yaxis.set_visible(False)
            else: ax.set_ylabel('DS{}'.format(i//nells +1)
                                +'\n'+r'$\ell={}$'.format(ells[i//nsplits])
                                +'\n'+r'$s$  [Mpc/h]')
    fig.colorbar(mesh, ax=lax, label=r'$r$')


def weights_trapz(x):
    """Return weights for trapezoidal integration."""
    return np.concatenate([[x[1]-x[0]], x[2:]-x[:-2], [x[-1]-x[-2]]])/2.


def integrate_pmesh_field(field):
    """Integrate field of class ParticleMesh.field over 3D space."""
    x_vals = list()
    for x in field.slabs.x:
        x_vals.append(x[0][0][0])

    # yvals = np.real(x[2][0])
    # zvals = np.real(x[1].transpose()[0])
    xvals = np.real(np.array(x_vals))
    dV = (xvals[1]-xvals[0])**3
    #xvals = np.real(field.slabs.optx[0][0][0])
    # yvals = xvals
    # zvals = xvals
    # x_sorted_indices = xvals.argsort()
    # y_sorted_indices = yvals.argsort()
    # z_sorted_indices = zvals.argsort()
    #
    # x_sorted_field = np.real(field[x_sorted_indices, :, :])
    # xy_sorted_field = x_sorted_field[:, y_sorted_indices, :]
    # xyz_sorted_field = xy_sorted_field[:, :, z_sorted_indices]
    #
    # sorted_xvals = xvals[x_sorted_indices]
    # sorted_yvals = yvals[y_sorted_indices]
    # sorted_zvals = zvals[z_sorted_indices]
    #
    # weights_x = weights_trapz(sorted_xvals)
    # weights_y = weights_trapz(sorted_yvals)
    # weights_z = weights_trapz(sorted_zvals)

    # intz = np.sum(xyz_sorted_field[:, :, :] * weights_z[None, None, :], axis=2)
    # intyz = np.sum(intz[:, :] * weights_y[None, :], axis=1)
    # intxyz = np.sum(intyz * weights_x)
    intxyz = np.sum(np.real(field)) * dV

    return intxyz
