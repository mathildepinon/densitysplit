import numpy as np
import logging 
import time
from operator import mul
from functools import reduce # Valid in Python 2.6+, required in Python 3

from pycorr import TwoPointCorrelationFunction, project_to_multipoles, setup_logging
from cosmoprimo import *
from .corr_func_utils import *
from .utils import BaseClass, integrate_pmesh_field
from pmesh import ParticleMesh
from pypower.fft_power import project_to_basis


class BaseTwoPointCorrelationFunctionModel(BaseClass):
    """
    Class implementing two-point correlation function model.
    """
    _defaults = dict(redshift=None, cosmology=None, k=np.logspace(-5, 3, 100000), mu=np.linspace(-1, 1, 400), pk=None, 
                     damping=False, non_linear=False, b1=1., rsd=False, shotnoise=0,
                     boxsize=1000, nmesh=512)

    def __init__(self, **kwargs):
        self.logger = logging.getLogger('BaseTwoPointCorrelationFunctionModel')
        self.logger.info('Initializing BaseTwoPointCorrelationFunctionModel')
        super().__init__(**kwargs)
        if self.pk is None:
            self.logger.info('Initializing theoretical 1D power spectrum')
            fo = Fourier(self.cosmology, engine='camb')
            pk_callable = fo.pk_interpolator(non_linear=self.non_linear, extrap_kmin=1e-10, extrap_kmax=1e6).to_1d(z=self.redshift)
            pk_array = pk_callable(self.k)
            if self.damping:
                self.logger.info('Applying damping from 80% of Nyquist frequency')
                def damping_function(k, k_lambda, sigma_lambda):
                    if k < k_lambda:
                        return 1
                    else:
                        return np.exp(-(k-k_lambda)**2/(2*sigma_lambda**2))
                kN = np.pi*self.nmesh/self.boxsize
                pkdamped_func = lambda k: pk_callable(k) * np.array([damping_function(kk, 0.8*kN, 0.05*kN) for kk in k])
                pk = PowerSpectrumInterpolator1D.from_callable(self.k, pkdamped_func)
            else:
                pk = pk_callable
            self.pk = pk
        if self.rsd:
            f = cosmology.get_background().growth_rate(self.redshift)
            pk_rsd = (1 + f * self.mu**2)**2 * pk_array[:, None]
            self.pk_rsd = pk_rsd
        self.logger.info('Initializing ParticleMesh with boxisze {}, nmesh {}'.format(self.boxsize, self.nmesh))
        self.pm = ParticleMesh(BoxSize=[self.boxsize] * 3, Nmesh=[self.nmesh] * 3, dtype='c16')
        self.set_pk_3D()
        self.set_xi()
        self.compute_sigma()

    def set_pk_3D(self):
        """Generate P(k) on the mesh."""
        cfield = self.pm.create('complex')
        norm = 1 / self.pm.BoxSize.prod()
        self.logger.info('Painting 1D power spectrum on 3D mesh')
        t0 = time.time()
        for kslab, delta_slab in zip(cfield.slabs.x, cfield.slabs):
            # The square of the norm of k on the mesh
            k2 = sum(kk**2 for kk in kslab)
            k = (k2**0.5).ravel()
            mask_nonzero = k != 0.
            pk = np.zeros_like(k)
            pk[mask_nonzero] = self.pk(k[mask_nonzero])
            delta_slab[...].flat = (self.b1**2 * pk + self.shotnoise) * norm
        self.pk_3D = cfield
        self.logger.info("3D power spectrum calculated in {:.2f} seconds.".format(time.time() - t0))

    def set_xi(self, s=np.linspace(0., 150., 151)):
        self.s = s
        xifield = self.pk_3D.c2r()
        sep, mu, xi = project_to_basis(xifield, edges=(self.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
        self.sep = sep
        self.xi = xi

    def compute_sigma(self):
        val = np.real(np.sum(self.pk_3D))   
        self.sigma = np.sqrt(val)
        return self.sigma
 

class SmoothedTwoPointCorrelationFunctionModel(BaseTwoPointCorrelationFunctionModel):
    """
    Class implementing two-point correlation function model with a smoothing kernel.
    """

    def __init__(self, *args, smoothing_kernel=6, smoothing_scale=10, smoothing_scale2=10, nbar=None, **kwargs):
        if len(args) and type(args[0]) is BaseTwoPointCorrelationFunctionModel:
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(**kwargs)
        self.logger.info('Initializing SmoothedTwoPointCorrelationFunctionModel')
        self.smoothing_kernel = smoothing_kernel
        self.nbar = nbar
        self.set_smoothing_scale(smoothing_scale, smoothing_scale2)
        self.set_shotnoise(nbar)

    def set_shotnoise(self, nbar=None):
        self.logger.info('Setting nbar to {}'.format(nbar))
        if nbar is None:
            nbar = 0
        self.set_smoothed_xi(nbar)

    def set_smoothing_scale(self, smoothing_scale, smoothing_scale2=None):
        self.logger.info('Setting smoothing scale to {}'.format(smoothing_scale))
        if smoothing_scale2 is not None:
            self.logger.info('Setting second smoothing scale to {}'.format(smoothing_scale2))
        self.smoothing_scale = smoothing_scale
        self.smoothing_scale2 = smoothing_scale2
        self.set_smoothing_kernel_3D(p=self.smoothing_kernel)
        self.set_smoothed_pk_3D()
        self.set_smoothed_xi(nbar=self.nbar)

    def set_smoothing_kernel_3D(self, p=6):
        self.logger.info('Setting 3D smoothing kernel of order {}'.format(p))
        self.smoothing_kernel = p # order of the smoothing kernel
        cfield = self.pm.create('complex')
        for kslab, w_slab in zip(cfield.slabs.x, cfield.slabs):
            w = reduce(mul, (np.sinc(self.smoothing_scale * kk / 2. / np.pi)**p for kk in kslab), 1)
            w_slab[...].flat = w
        self.smoothing_kernel_3D = cfield
        if (self.smoothing_scale2 is not None) & (self.smoothing_scale2 != self.smoothing_scale):
            cfield2 = self.pm.create('complex')
            for kslab, w_slab in zip(cfield2.slabs.x, cfield2.slabs):
                w = reduce(mul, (np.sinc(self.smoothing_scale2 * kk / 2. / np.pi)**p for kk in kslab), 1)
                w_slab[...].flat = w
            self.smoothing_kernel2_3D = cfield2

    def set_smoothed_pk_3D(self):
        self.smoothed_pk_3D = self.pk_3D * self.smoothing_kernel_3D
        if self.smoothing_scale2 != self.smoothing_scale:
            self.double_smoothed_pk_3D = self.pk_3D * self.smoothing_kernel_3D * self.smoothing_kernel2_3D
        else:
            self.double_smoothed_pk_3D = self.pk_3D * self.smoothing_kernel_3D**2

    def set_smoothed_xi(self, nbar=None, smoothing_scale2=None):      
        xiRfield = self.smoothed_pk_3D.c2r()
        xiRfield.value = np.real(xiRfield.value)
        sep, mu, xiR = project_to_basis(xiRfield, edges=(self.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
        self.smoothed_xi = np.real(xiR)
        self.smoothed_sigma = np.sqrt(np.real(xiR))

        xiRRfield = self.double_smoothed_pk_3D.c2r()
        xiRRfield.value = np.real(xiRRfield.value)
        sep, mu, xiRR = project_to_basis(xiRRfield, edges=(self.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
        self.double_smoothed_xi = np.real(xiRR)

        ## shotnoise correction
        self.nbar = nbar if nbar is not None else 0
        if self.nbar:
            wfield = self.smoothing_kernel_3D.c2r() / self.boxsize**3
            sep, mu, w = project_to_basis(wfield, edges=(self.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
            shotnoise_corr = (1 + self.sigma**2) * w / self.nbar
            self.smoothed_xi  = self.smoothed_xi + np.real(shotnoise_corr)

            if self.smoothing_scale2 != self.smoothing_scale:
                wfield2 = self.smoothing_kernel2_3D.c2r() / self.boxsize**3
                sep, mu, w2 = project_to_basis(wfield*wfield2, edges=(self.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
            else:
                sep, mu, w2 = project_to_basis(wfield**2, edges=(self.s, np.array([-1., 1.])), exclude_zero=False)[0][:3]
            shotnoise_corr2 = (1 + self.sigma**2) * w2 / self.nbar
            self.double_smoothed_xi  = self.double_smoothed_xi + np.real(shotnoise_corr2)

        return self.smoothed_xi