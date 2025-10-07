import math
import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import os

from cosmoprimo import fiducial

plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams.update({'font.size': 14})

from densitysplit import catalog_data, density_split

plots_dir = '/feynman/home/dphp/mp270220/plots/densitysplit'

plt.style.use(os.path.join(os.path.abspath('/feynman/home/dphp/mp270220/densitysplit/nb'), 'densitysplit.mplstyle'))

data_dir = '/feynman/scratch/dphp//mp270220/abacus/'
output_dir = '/feynman/work/dphp/mp270220/outputs/'

catalog_name = 'AbacusSummit_2Gpc_z0.800_ph000_downsampled_particles_nbar0.0034'
#catalog_name = 'mock'

catalog = catalog_data.Data.load(data_dir+catalog_name+'.npy')
catalog.shift_boxcenter(-catalog.offset)

cellsize = 5
cellsize2 = 5
smoothing_radius = 10
resampler = 'ngp'
catalog_density = density_split.DensitySplit(catalog)
catalog_density.compute_density(data=catalog, cellsize=cellsize, smoothing_radius=smoothing_radius, resampler=resampler, cellsize2=cellsize2, use_rsd=False, use_weights=False)
#density_without_masses = catalog_density.data_densities

idx=0
cut_direction = 'z'
cmap = mpl.colormaps['inferno']

# catalog_density.show_density_map(plt.figure(figsize=(3.4, 3.4)), plt.gca(), cut_direction=cut_direction, cut_idx=idx, cmap=cmap, log=False)
# plt.xlim(0, 1000)
# plt.ylim(0, 1000)
# plt.grid(visible=False)
# plt.gca().axis('off')
# plt.savefig(os.path.join(plots_dir, 'density_map.png'), dpi=2000)
# plt.close()

from circles import circles
mesh_pos = (0.5 + np.indices((25, 25))) * 40

catalog_density.show_halos_map(plt.figure(figsize=(3.4, 3.4)), plt.gca(), positions=catalog.positions, cellsize=20, cut_direction=cut_direction, cut_idx=idx, color='grey', density=False)
#plt.gca().scatter(mesh_pos[0], mesh_pos[1], s=30, edgecolors='C0', facecolors='none', linewidths=0.7)
plt.xlim(0, 1000)
plt.ylim(0, 1000)
#plt.gca().set_facecolor('black')
plt.grid(visible=False)
plt.gca().axis('off')
plt.savefig(os.path.join(plots_dir, 'density_particles.png'), dpi=2000)
plt.close()
print("saved density particles map")

nsplit = 3
if nsplit==3:
    splits = [-1, -0.29, 0.11, np.inf]
elif nsplits==5:
    splits = [-1, -0.44, -0.22, 0.02, 0.37, np.inf]
mesh, indices = catalog_density.split_density(nsplit, bins=splits, return_indices=True)

#cmap = plt.get_cmap('coolwarm', catalog_density.nsplits)
base_colors = ['cornflowerblue', '#ddb2c4', 'red']
cmap = LinearSegmentedColormap.from_list("mycmap", base_colors, N=catalog_density.nsplits)
colors = [cmap(i) for i in range(catalog_density.nsplits)]

ax = plt.gca()
nbar = catalog.size/catalog.boxsize**3
norm = nbar * 4/3 * np.pi * catalog_density.smoothing_radius**3
catalog_density.show_randoms_map(plt.figure(figsize=(3.4, 3.4)), plt.gca(), 2*catalog_density.cellsize, cut_direction=cut_direction, cut_idx=idx, colors=colors, positions='mesh', norm=norm)
plt.grid(visible=False)
plt.xlim(0, 1000)
plt.ylim(0, 1000)
plt.gca().axis('off')
plt.savefig(os.path.join(plots_dir, 'densitysplits_randoms.png'), dpi=500)
plt.close()
