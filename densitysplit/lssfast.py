import time
import logging
import numpy as np
#import sympy as sp
from operator import mul
from functools import reduce
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

from pycorr import setup_logging
from .utils import BaseClass
from pmesh import ParticleMesh
from pypower.fft_power import project_to_basis


class LDT(BaseClass):
    """
    Class implementing large deviation theory (LDT) model for the 2D PDF of the matter density field and the bias function.
    """
    _defaults = dict(redshift=None, cosmology=None, k=np.logspace(-5, 3, 100000), pk=None, 
                     damping=False, non_linear=False, b1=1., shotnoise=0, boxsize=1000, nmesh=512,
                     smoothing_kernel=6, smoothing_scale=10)

    def __init__(self, **kwargs):
        self.logger = logging.getLogger('LDT')
        self.logger.info('Initializing LDT')
        super().__init__(**kwargs)
        if self.cosmology is None:
            from cosmoprimo import fiducial
            self.cosmology = fiducial.AbacusSummitBase(non_linear='halofit' if self.non_linear else False)
        if self.pk is None:
            self.logger.info('Initializing theoretical 1D power spectrum')
            from cosmoprimo import Fourier
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
        self.logger.info('Initializing ParticleMesh with boxisze {}, nmesh {}'.format(self.boxsize, self.nmesh))
        self.pm = ParticleMesh(BoxSize=[self.boxsize] * 3, Nmesh=[self.nmesh] * 3, dtype='c16')
        self.set_pk_3D()
        self.set_smoothing_scale(self.smoothing_scale)

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

    def set_smoothing_scale(self, smoothing_scale):
        self.smoothing_scale = smoothing_scale
        self.set_smoothing_kernel_3D(p=self.smoothing_kernel)
        self.set_smoothed_pk_3D()
        self.compute_sigma()

    def set_smoothing_kernel_3D(self, p=6):
        self.smoothing_kernel = p # order of the smoothing kernel
        cfield = self.pm.create('complex')
        for kslab, w_slab in zip(cfield.slabs.x, cfield.slabs):
            w = reduce(mul, (np.sinc(self.smoothing_scale * kk / 2. / np.pi)**p for kk in kslab), 1)
            w_slab[...].flat = w
        self.smoothing_kernel_3D = cfield
 
    def set_smoothed_pk_3D(self):
        self.smoothed_pk_3D = self.pk_3D * self.smoothing_kernel_3D
        self.double_smoothed_pk_3D = self.pk_3D * self.smoothing_kernel_3D**2

    def compute_sigma(self):
        val = np.real(np.sum(self.double_smoothed_pk_3D))   
        self.sigma = np.sqrt(val)
        return self.sigma

    def get_sigma(self, smoothing_scale):
        tmp = self.smoothing_scale
        self.set_smoothing_scale(smoothing_scale)
        res = self.sigma
        # set smoothing scale back to the original value
        self.set_smoothing_scale(tmp)
        return res

    def interpolate_sigma(self, tab_fn=None):
        #r = sp.symbols('r')
        if tab_fn is not None:
            Rvals, sigmavals = np.load(tab_fn)
        else:
            logRvals = np.arange(np.log(0.01**(1/3)), np.log(10**(1 + 1/3 + 1)), 0.2)
            self.logger.info('Interpolating sigma for {} R log-spaced values between {} and {}'.format(len(logRvals), np.min(logRvals), np.max(logRvals)))
            Rvals = np.exp(logRvals)
            sigmavals = [self.get_sigma(Rval) for Rval in Rvals]
        self.Rvals = Rvals
        self.sigmavals = sigmavals
        #self.sigma_interp = sp.functions.special.bsplines.interpolating_spline(7, r, Rvals, sigmavals)
        self.sigma_interp = interp1d(Rvals, sigmavals, kind=7)

    def compute_ldt(self, sigma_val, nu_val=21/13):
        #x, y, nu, r, s = sp.symbols('x y nu r s')
        #Zeta = 1 / (1 - x / nu)**nu
        #self.Tau = sp.solve(y - Zeta, x)[0]
        #tau = sp.utilities.lambdify([y, nu], self.Tau, 'numpy')
        #self.Psi = (1 / 2) * self.Tau**2 / self.sigma_interp.subs(r, y**(1 / 3)*self.smoothing_scale)**2 * self.sigma_interp.subs(r, self.smoothing_scale)**2
        #self.dPsi = sp.diff(self.Psi, y)
        #self.ddPsi = sp.diff(self.Psi, y, y)
        #self.Lowlogrho = (1 / sp.sqrt(2 * sp.pi)) * (1 / s) * sp.sqrt(self.ddPsi + self.dPsi / y) * sp.exp(-1 / s**2 * self.Psi)
        self.yvals = np.linspace(0.1, 10, 1000)
        self.nu = nu_val
        self.Tau = lambda y: self.nu*(1 - (1/y)**(1/self.nu))
        self.tau = self.nu*(1 - (1/self.yvals)**(1/self.nu))
        self.psi = (1 / 2) * self.tau**2 / self.sigma_interp(self.yvals**(1 / 3)*self.smoothing_scale)**2 * self.sigma_interp(self.smoothing_scale)**2
        self.dpsi = np.gradient(self.psi, self.yvals, edge_order=2)
        self.ddpsi = np.gradient(self.dpsi, self.yvals, edge_order=2)
        self.lowlogrho = lambda s : (1 / np.sqrt(2 * np.pi)) * (1 / s) * np.sqrt(self.ddpsi + self.dpsi / self.yvals) * np.exp(-1 / s**2 * self.psi)
        self.sigma_val = sigma_val
        self.eff_sigma_log = self.effsiglog()
        self.expbiasnorm()

    def logs(self, order, s_val):
        #y, nu, s = sp.symbols('y nu s')
        #lowlogrho = self.Lowlogrho.subs({s: s_val, nu: nu_val})
        #func = sp.utilities.lambdify(y, y**order * lowlogrho, 'numpy') # returns a numpy-ready function
        #res = quad(func, 0.1, 10)[0]
        res = np.trapz(self.yvals**order * self.lowlogrho(s_val), x=self.yvals)
        return res

    def effsiglog(self):
        def func(ss):
            ss = ss[0] if len(ss) else float(ss)
            return self.logs(0, float(ss)) * self.logs(2, float(ss)) / self.logs(1, float(ss))**2 - 1 - self.sigma_val**2
        solution = fsolve(func, self.sigma_val)
        return solution[0]

    def density_pdf(self, rho):
        logs1 = self.logs(1, self.eff_sigma_log)
        self.logs1 = logs1
        logs0 = self.logs(0, self.eff_sigma_log)
        self.logs0 = logs0
        lowlogrho = interp1d(self.yvals, self.lowlogrho(self.eff_sigma_log), kind=7)(rho * logs1/logs0)
        return logs1 / logs0**2 * lowlogrho

    def lowrhobias(self, rho):
        return self.sigma_interp(self.smoothing_scale)**2 / (self.sigma_interp(rho**(1 / 3) * self.smoothing_scale)**2 * self.eff_sigma_log**2) * self.Tau(rho)

    def expbiasnorm(self):
        ymax = 9
        mask = self.yvals < ymax
        x = self.yvals[mask]
        self.xvals = x
        lowrhobias = self.lowrhobias(x)
        self.low_rho_bias = lowrhobias
        self.density_pdfvals = np.array([self.density_pdf(rho) for rho in x])
        self.exp_bias_norm = np.trapz(lowrhobias * self.density_pdfvals, x=x)
        return self.exp_bias_norm
    
    def exprhobiasnorm(self):
        integrand = self.xvals * (self.low_rho_bias - self.exp_bias_norm) * self.density_pdfvals
        return np.trapz(integrand, x=self.xvals)
    
    # bias function
    def bias(self, rho):
        rho_bias_norm = self.exprhobiasnorm()
        res = (self.lowrhobias(rho) - self.exp_bias_norm)  / rho_bias_norm
        return res





