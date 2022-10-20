import numpy as np

from pmesh import ParticleMesh
from cosmoprimo.fiducial import DESI


cosmo = DESI()
pk_interp = cosmo.get_fourier().pk_interpolator().to_1d(z=0)

# Generate P(k) on the mesh
pm = ParticleMesh(BoxSize=[1000.] * 3, Nmesh=[256] * 3, dtype='f8')
cfield = pm.create('complex')
norm = 1.0 / pm.BoxSize.prod()
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

if xifield.pm.comm.rank == 0:
    from matplotlib import pyplot as plt
    plt.plot(s, s**2 * xi, label='3D FFT')
    xi_interp = pk_interp.to_xi()
    plt.plot(s, s**2 * xi_interp(s), label='Hankel')
    plt.legend()
    plt.show()
