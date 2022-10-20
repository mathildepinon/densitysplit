import numpy as np

from pmesh import ParticleMesh
from cosmoprimo.fiducial import DESI
from cosmoprimo import *
from pycorr import TwoPointCorrelationFunction, setup_logging
from mockfactory import EulerianLinearMock, LagrangianLinearMock, RandomBoxCatalog, setup_logging
from pypower.fft_power import project_to_basis

from densitysplit import catalog_data, density_split

# Data and output directories
data_dir = '/feynman/work/dphp/mp270220/data/'
output_dir = '/feynman/work/dphp/mp270220/outputs/'

catalog_name = 'AbacusSummit_1Gpc_z1.175'
bias = 1.8
catalog = catalog_data.Data.load(data_dir+catalog_name+'.npy')
catalog.shift_boxcenter(-catalog.offset)
z = catalog.redshift

# Theoretical linear 2PCF
cosmo = DESI()
pk_interp = cosmo.get_fourier().pk_interpolator().to_1d(z=z)

# Generate P(k) on the mesh
pm = ParticleMesh(BoxSize=[1000.] * 3, Nmesh=[512] * 3, dtype='f8')
cfield = pm.create('complex')
norm = bias**2 / pm.BoxSize.prod()
for kslab, delta_slab in zip(cfield.slabs.x, cfield.slabs):
    # The square of the norm of k on the mesh
    k2 = sum(kk**2 for kk in kslab)
    k = (k2**0.5).ravel()
    mask_nonzero = k != 0.
    pk = np.zeros_like(k)
    pk[mask_nonzero] = pk_interp(k[mask_nonzero])
    delta_slab[...].flat = pk * norm

# You can replace cfield by an array of initial conditions, e.g.
# cfield = ArrayMesh(array, boxsize=1000., type='complex', nmesh=256, mpiroot=0)

xifield = cfield.c2r()  # Xi(s)
from pypower.fft_power import project_to_basis
s, mu, xi = project_to_basis(xifield, edges=(np.linspace(0., 200., 51), np.array([-1., 1.])), exclude_zero=True)[0][:3]
s, xi = s.ravel(), xi.ravel()

# 2PCF from gaussian mock

# Parameters

# Density mesh
cellsize = 10
resampler = 'tsc'
nsplits = 3

# Correlation function
randoms_size = 4
edges = (np.linspace(0., 150., 51), np.linspace(-1, 1, 201))
los = 'x'

# Mocks
nmesh = 512
nbar = catalog.size/catalog.boxsize**3
boxsize = catalog.boxsize
boxcenter = catalog.boxcenter

# For RSD
cosmology=fiducial.AbacusSummitBase()
bg = cosmology.get_background()
f = bg.growth_rate(catalog.redshift)

# Generate mock
pklin = cosmology.get_fourier().pk_interpolator().to_1d(z)

kN = np.pi*nmesh/boxsize

#k=np.logspace(-5, 2, 1000000)
#pklin_array = pklin(k)
#pkdamped_func = lambda k: pklin(k) * np.array([damping_function(kk, kN) for kk in k])
#pkdamped = PowerSpectrumInterpolator1D.from_callable(k, pkdamped_func)

# unitary_amplitude forces amplitude to 1
mock = EulerianLinearMock(pklin, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=0, unitary_amplitude=False)

# this is Lagrangian bias, Eulerian bias - 1
mock.set_real_delta_field(bias=bias)
mock_rfield = mock.mesh_delta_r
mock_cfield = mock_rfield.r2c()
mock_cfield[...] = mock_cfield[...] * mock_cfield[...].conj()  # P(k)
mock_xifield = mock_cfield.c2r()  # Xi(s)
mock_s, mock_mu, mock_xi = project_to_basis(mock_xifield, edges=(np.linspace(0., 200., 51), np.array([-1., 1.])), exclude_zero=True)[0][:3]
mock_s, mock_xi = mock_s.ravel(), mock_xi.ravel()

if xifield.pm.comm.rank == 0:
    from matplotlib import pyplot as plt
    plt.plot(s, s**2 * xi, label='3D FFT')
    xi_interp = pk_interp.to_xi()
    plt.plot(s, s**2 * bias**2 * xi_interp(s), label='Hankel')
    plt.plot(mock_s, mock_s**2 * mock_xi, label='3D FFT from mock')
    plt.xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
    plt.ylabel(r'$s^2 \xi(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
    plt.legend()
    plt.show()
