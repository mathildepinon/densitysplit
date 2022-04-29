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