import os
import copy

import numpy as np


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


def truncate_xiell(lower_s_limit, s, xiell, ells, cov):
    """
    Truncate separations, correlation function multipoles and covariance matrix above a given separation s_lower_limit.
    
    Arguments
    ---------
    lower_s_limit : float, spearation threshold in Mpc/h
    
    s : array, separations
    
    xiell : array, correlation function multipoles
    
    ells : int list, multipoles in xiell
    
    cov : 2D array, covariance matric of the correlation function multipoles
    """
    s_truncated = s[s>lower_s_limit]

    xiell_truncated_list = list()
    ns = len(s)

    for ill, ell in enumerate(ells):
        xiell_truncated_list.append(xiell[ill][s>lower_s_limit])
        first_index = np.sum(np.logical_not(s>lower_s_limit))

    xiell_truncated = np.array(xiell_truncated_list)
    
    # Truncate the whole covariance matrix
    ns_trunc = len(s_truncated)
    nells = len(ells)
    cov_truncated_full = np.zeros((ns_trunc*nells, ns_trunc*nells))

    for i in range(nells):
        for j in range(nells):
            cov_truncated_full[i*ns_trunc:(i+1)*ns_trunc,j*ns_trunc:(j+1)*ns_trunc] = cov[(i+1)*ns-ns_trunc:(i+1)*ns,(j+1)*ns-ns_trunc:(j+1)*ns]
    
    return s_truncated, xiell_truncated, cov_truncated_full


def compute_chisq(xdata, ydata, sigma, fitted_model):
    r = ydata - fitted_model
    
    chisq = r.T @ np.linalg.inv(sigma) @ r
    
    return chisq