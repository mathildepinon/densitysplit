import os 
import time
import logging
import numpy as np
#import sympy as sp
from operator import mul
from functools import reduce
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.special import factorial, loggamma

from pycorr import setup_logging
from .utils import BaseClass
from pmesh import ParticleMesh
from pypower.fft_power import project_to_basis


class LDT(BaseClass):
    """
    Class implementing large deviation theory (LDT) model for the 2D PDF of the matter density field and the bias function.
    Translated in python from Mathematica package LSSFast, see Uhlemann et al. 2017 (arxiv:1607.01026), and Codis et al. 2016 (arXiv:1602.03562).
    """
    _defaults = dict(redshift=None, cosmology=None, k=np.logspace(-5, 3, 100000), pk=None, 
                     damping=False, non_linear=False, b1=1., shotnoise=0, nbar=None, iso=True, boxsize=1000, nmesh=512,
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
        if not self.iso:
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
        if self.iso:
            self.set_smoothing_kernel()
        else:
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

    def set_smoothing_kernel(self):
        ngp_smoothing_kernel = np.sinc(self.smoothing_scale * self.k / 2. / np.pi)**2
        self.smoothing_kernel = 3 * (self.smoothing_scale * self.k * np.cos(self.smoothing_scale * self.k) - np.sin(self.smoothing_scale * self.k)) / (self.k**3 * self.smoothing_scale**3)
    
    def set_smoothed_pk_3D(self):
        self.smoothed_pk_3D = self.pk_3D * self.smoothing_kernel_3D
        self.double_smoothed_pk_3D = self.pk_3D * self.smoothing_kernel_3D**2

    def compute_sigma(self):
        if self.iso:
            val = np.trapz((self.pk(self.k)) * self.k**2 * self.smoothing_kernel**2, self.k)
        else: 
            val = np.real(np.sum(self.double_smoothed_pk_3D))   
        #if self.nbar is not None:
        #    val += 1 / (self.nbar * 4/3 * np.pi * self.smoothing_scale**3)
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
        if tab_fn is not None and os.path.isfile(tab_fn):
            Rvals, sigmavals = np.load(tab_fn)
        else:
            logRvals = np.arange(np.log(0.01**(1/3)), np.log(10**(1 + 1/3 + 1)), 0.1)
            #logRvals = np.arange(-10, np.log(10**(1 + 1/3 + 1)), 0.1)
            self.logger.info('Interpolating sigma for {} R log-spaced values between {} and {}'.format(len(logRvals), np.min(logRvals), np.max(logRvals)))
            Rvals = np.exp(logRvals)
            sigmavals = [self.get_sigma(Rval) for Rval in Rvals]
            if tab_fn is not None:
                self.logger.info("Saving tabluated sigma values: {}.".format(tab_fn))
                np.save(tab_fn, np.array([Rvals, sigmavals]))
        self.Rvals = Rvals
        self.sigmavals = sigmavals
        #self.sigma_interp = sp.functions.special.bsplines.interpolating_spline(7, r, Rvals, sigmavals)
        self.sigma_interp = interp1d(Rvals, sigmavals, kind=7, bounds_error=False)

    def compute_ldt(self, sigma_val, nu=21/13, k=None):
        self.yvals = np.arange(0.03, 15, 0.01)
        self.ymax = 12
        self.nu = nu
        self.Tau = lambda y: self.nu*(1 - (1/y)**(1/self.nu))
        self.tau = self.nu*(1 - (1/self.yvals)**(1/self.nu))
        self.psi = (1 / 2) * self.tau**2 / self.sigma_interp(self.yvals**(1 / 3)*self.smoothing_scale)**2 * self.sigma_interp(self.smoothing_scale)**2
        self.ldavals = sigma_val**2 * np.arange(-5, 5, 0.001)
        self.cgf = np.max(self.yvals[:, None]*self.ldavals[None, :] - self.psi[:, None], axis=0) / sigma_val**2
        self.dpsi = np.gradient(self.psi, self.yvals, edge_order=2)
        self.ddpsi = np.gradient(self.dpsi, self.yvals, edge_order=2)
        self.lowlogrho = lambda s : (1 / np.sqrt(2 * np.pi)) * (1 / s) * np.sqrt(self.ddpsi + self.dpsi / self.yvals) * np.exp(-1 / s**2 * self.psi)
        self.sigma_val = sigma_val
        self.norm = self.nbar * 4/3 * np.pi * self.smoothing_scale**3
        self.eff_sigma_log = self.effsiglog()
        self.kvals = k
        self.expbiasnorm()

    def pt_s3(self):
        todiff = np.log(np.array(self.sigmavals)**2)
        res = np.gradient(todiff, np.log(self.Rvals))
        return 34/7 + res

    def logs(self, order, s_val):
        res = np.trapz(self.yvals**order * self.lowlogrho(s_val), x=self.yvals)
        return res

    def effsiglog(self):
        def func(ss):
            ss = ss[0] if len(ss) else float(ss)
            return self.logs(0, float(ss)) * self.logs(2, float(ss)) / self.logs(1, float(ss))**2 - 1 - self.sigma_val**2
        solution = fsolve(func, self.sigma_val)
        return solution[0]

    def density_pdf_noshotnoise(self, rho, b1=1):
        rho = 1+(rho-1)/b1
        logs1 = self.logs(1, self.eff_sigma_log)
        self.logs1 = logs1
        logs0 = self.logs(0, self.eff_sigma_log)
        self.logs0 = logs0
        lowlogrho_func = interp1d(self.yvals, self.lowlogrho(self.eff_sigma_log), kind=7)
        if (isinstance(rho, list) or isinstance(rho, np.ndarray)) and len(rho) > 0:
            lowlogrho = np.array([lowlogrho_func(rhoo * logs1/logs0) for rhoo in rho])
        else:
            lowlogrho = lowlogrho_func(rho * logs1/logs0)
        return logs1 / logs0**2 * lowlogrho / b1

    def density_pdf(self, rho=None, k=None, **kwargs): # convolve density PDF with Poisson shot noise
        if self.nbar is None:
            return self.density_pdf_noshotnoise(rho, **kwargs)
        else:
            ymax = self.ymax
            mask = self.yvals < ymax
            x = self.yvals[mask]
            density_pdfvals = self.density_pdf_noshotnoise(x, **kwargs)
            norm = self.norm
            def func(N):
                log_poisson_pdf = N * np.log(norm * x[:, None]) - (norm * x[:, None]) - loggamma(N+1) # log to avoid overflow
                poisson_pdf = np.exp(log_poisson_pdf)
                res = np.trapz(poisson_pdf * density_pdfvals[:, None], x=x, axis=0)
                return res
            if (k is None) and (rho is not None):
                k = np.round(norm * rho)
                k = np.append(k, np.max(k)+1)
                return interp1d(k, func(k), bounds_error=False, fill_value=0)(norm * rho) * norm
                #return func(k.ravel()).reshape(k.shape) * norm
                #return interp1d(k.ravel(), func(k.ravel()), bounds_error=False, fill_value=0)(norm * rho) * norm
            else:
                if k is None:
                    k=self.kvals
                return func(k) * norm # k may not be flat

    def alpha(self, alpha0=1, alpha1=0, alpha2=0):
        x = self.yvals[self.yvals < self.ymax]
        delta_m = x - 1
        return alpha0 + alpha1 * delta_m + alpha2 * delta_m**2

    def delta_t_expect(self, bG1=1, bG2=-1):
        sigma_m = self.sigma_val
        def fL(delta_L):
            toret = np.exp(- bG1**2/(2*bG2)) / np.sqrt(1 + bG2 * sigma_m**2) * np.exp((bG2 * (bG1/bG2 + delta_L)**2)/(2 * (1 + bG2 * sigma_m**2)))
            return toret
        x = self.yvals[self.yvals < self.ymax]
        delta_m = x - 1
        delta_L = self.tau[self.yvals < self.ymax] - 1
        fLm = fL(delta_L)
        pdf_noshotnoise = self.density_pdf_noshotnoise(x)
        mean_term = np.trapz(x * fLm * pdf_noshotnoise, x=x)
        toret = (1 + delta_m) * fLm - mean_term - 1
        return toret

    def Nt_pdf(self, Nt, bG1=1, bG2=-1, alpha0=1, alpha1=0, alpha2=0):
        alpha = self.alpha(alpha0, alpha1, alpha2)[:, None]
        Nt_expect = self.norm * (1 + self.delta_t_expect(bG1, bG2)[:, None])
        logpdf = - np.log(alpha) - Nt_expect/alpha - loggamma(1+Nt/alpha) + Nt[None, :]/alpha * np.log(Nt_expect/alpha)
        return np.exp(logpdf)

    def tracer_density_pdf(self, rho=None, k=None, **kwargs): # convolve density PDF with Poisson shot noise
        x = self.yvals[self.yvals < self.ymax]
        density_pdfvals = self.density_pdf_noshotnoise(x)[:, None]
        norm = self.norm
        def func(N):
            #print('param_dict', kwargs)
            Nt_pdf = self.Nt_pdf(N, **kwargs)
            res = np.trapz(Nt_pdf * density_pdfvals, x=x, axis=0)
            return res
        if (k is None) and (rho is not None):
            k = np.round(norm * rho)
            k = np.append(k, np.max(k)+1)
            return interp1d(k, func(k), bounds_error=False, fill_value=0)(norm * rho) * norm
            #return func(k.ravel()).reshape(k.shape) * norm
            #return interp1d(k.ravel(), func(k.ravel()), bounds_error=False, fill_value=0)(norm * rho) * norm
        else:
            if k is None:
                k=self.kvals
            return func(k) * norm # k may not be flat

    def lowrhobias(self, rho):
        #return self.sigma_interp(self.smoothing_scale)**2 / (self.sigma_interp(rho**(1 / 3) * self.smoothing_scale)**2 * self.eff_sigma_log**2) * self.Tau(rho)
        return self.Tau(rho) / (self.sigma_interp(rho**(1 / 3) * self.smoothing_scale)**2)

    def expbiasnorm(self):
        ymax = self.ymax
        mask = self.yvals < ymax
        x = self.yvals[mask]
        self.xvals = x
        lowrhobias = self.lowrhobias(x)
        self.density_pdfvals_noshotnoise = self.density_pdf_noshotnoise(x)
        self.low_rho_bias = lowrhobias
        self.exp_bias_norm = np.trapz(lowrhobias * self.density_pdfvals_noshotnoise, x=x)
        return self.exp_bias_norm
    
    def exprhobiasnorm(self):
        integrand = self.xvals * (self.low_rho_bias - self.exp_bias_norm) * self.density_pdfvals_noshotnoise
        return np.trapz(integrand, x=self.xvals)
    
    # bias function
    def bias_noshotnoise(self, rho, b1=1):
        rho = 1 + (rho - 1) / b1
        rho_bias_norm = self.exprhobiasnorm()
        res = (self.lowrhobias(rho) - self.exp_bias_norm)  / rho_bias_norm
        return res / b1

    # bias function convolved with Poisson shot noise
    def bias(self, rho=None, **kwargs):
        ymax = self.ymax
        mask = self.yvals < ymax
        x = self.yvals[mask]
        norm = self.norm
        bias_func = self.bias_noshotnoise(x, **kwargs)
        density_pdfvals_noshotnoise = self.density_pdf_noshotnoise(x, **kwargs)
        prod = bias_func*density_pdfvals_noshotnoise
        density_pdfvals = self.density_pdf(rho=rho, **kwargs)
        def func(N):
            log_poisson_pdf = N * np.log(norm * x[:, None]) - (norm * x[:, None]) - loggamma(N+1) # log to avoid overflow
            poisson_pdf = np.exp(log_poisson_pdf)
            res = np.trapz(poisson_pdf * prod[:, None], x=x, axis=0)
            return res
        if (self.kvals is None) and (rho is not None):
            k = np.round(norm * rho)
            k = np.append(k, np.max(k)+1)
            return interp1d(k, func(k), bounds_error=False, fill_value=0)(norm * rho) * norm / density_pdfvals
        else:
            k=self.kvals
            return func(k) * norm / density_pdfvals
            

class LDTDensitySplitModel(LDT):
    """
    Class implementing LDT model for density-split statistics.
    """

    def __init__(self, *args, nsplits=3, density_bins=None, **kwargs):
        if len(args) and isinstance(args[0], LDT):
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(**kwargs)
        self.logger.info('Initializing LDTDensitySplitModel with {} density splits'.format(nsplits))
        self.nsplits = nsplits
        self._fixed_density_bins = True
        self.density_bins = density_bins
        if density_bins is None:
            self._fixed_density_bins = False

    def joint_density_pdf_nonorm(self, xi, rho1=None, rho2=None, b1=1):
        if (rho1 is None) & (self.kvals is not None):
            pdf1 = self.density_pdf(k=self.kvals, b1=b1)
            pdf2 = self.density_pdf(k=self.kvals, b1=b1)
            rho1, rho2 = self.kvals/self.norm, self.kvals/self.norm
        else:
            pdf1 = self.density_pdf(rho1, b1=b1)
            pdf2 = self.density_pdf(rho2, b1=b1)
        bias1 = self.bias(rho1, b1=b1)
        bias2 = self.bias(rho2, b1=b1)
        res = pdf1[..., :, None] * pdf2[..., None, :] * (1 + np.array(xi)[..., None, None] * bias1[..., :, None] * bias2[..., None, :])
        res[np.isnan(res)] = 0
        return res

    def joint_density_pdf(self, xi, rho1=None, rho2=None, **kwargs):
        if (rho1 is None) & (self.kvals is not None):
            k1, k2 = np.meshgrid(self.kvals, self.kvals, indexing='ij')
            rho1, rho2 = self.kvals/self.norm, self.kvals/self.norm
            res = self.joint_density_pdf_nonorm(xi, **kwargs)
            norm = np.sum(np.sum(res, axis=-1), axis=-1)/self.norm**2
            avg = np.sum(np.sum(rho2*res, axis=-1), axis=-1)/self.norm**2
        else:
            res = self.joint_density_pdf_nonorm(xi, rho1, rho2, **kwargs)
            norm = np.trapz(np.trapz(res, x=rho2, axis=-1), x=rho1, axis=-1)
            avg = np.trapz(np.trapz(rho2*res, x=rho2, axis=-1), x=rho1, axis=-1)
        beta = avg/norm
        alpha = avg**2 / norm**3
        alpha = alpha[..., None, None]
        beta = beta[..., None]
        res = alpha * self.joint_density_pdf_nonorm(xi, beta*rho1, beta*rho2, **kwargs)
        return res

    def compute_dsplits(self, xi, nsplits=None, density_bins=None, x=None, density_pdf=None, bias=None, **kwargs):
        if nsplits is not None:
            self.nsplits = nsplits
        if density_bins is not None:
            self.density_bins = density_bins
        if x is None:
            if self.kvals is not None:
                xvals = yvals = self.kvals/self.norm
            else:
                xvals = self.xvals
                xvals = yvals = xvals[xvals < 9]
        else:
            test = self.kvals/self.norm
            xvals, yvals = 1+x[0], 1+x[1]
        rho1, rho2 = np.meshgrid(xvals, yvals, indexing='ij')
        t0 = time.time()
        dsplits = list()
        if density_pdf is None:
            density_pdf = self.density_pdf(xvals, **kwargs)
        if bias is None:
            bias = self.bias(rho=xvals, **kwargs)
        print('bias shape: ', bias.shape)
        if bias.ndim > 1:
            toint = bias * density_pdf[:, None]
        else:
            toint = bias * density_pdf
        for i in range(len(self.density_bins)-1):
            d1 = max(self.density_bins[i], -1)
            d2 = self.density_bins[i+1]
            self.logger.info('Computing LDT density split model in density bin {:.2f}, {:.2f}'.format(d1, d2))
            ds_mask = (xvals >= 1 + d1) & (xvals < 1 + d2)
            res = np.nansum(toint[ds_mask], axis=0)/self.norm
            norm = np.nansum(density_pdf[ds_mask])/self.norm
            dsplits.append(res/norm * xi)
        self.logger.info('Computed LDT model for {} xi values in elapsed time: {}s'.format(len(np.array(xi)), time.time()-t0))
        return dsplits
 
    def compute_dsplits_test(self, nsplits=None, density_bins=None, joint_density_pdf=None, x1=None, x2=None):
        if nsplits is not None:
            self.nsplits = nsplits
        if density_bins is not None:
            self.density_bins = density_bins
        rho1, rho2 = 1+x1, 1+x2
        innerint = np.trapz(rho2 * joint_density_pdf, x=rho2[0, :], axis=-1)
        #norm_test = np.trapz(np.trapz(joint_density_pdf, x=rho1[:, 0], axis=-2), x=rho2[0, :], axis=-1)
        #print('2D pdf normalization: 1 = ', norm_test)
        #norm_test = np.trapz(np.trapz(rho2*joint_density_pdf, x=rho1[:, 0], axis=-2), x=rho2[0, :], axis=-1)
        #print('2D pdf avg: 1 = ', norm_test)
        #norm_test = np.trapz(density_pdf, x=rho1[:, 0])
        #print('1D pdf normalization: 1 = ', norm_test)
        #density_pdf /= norm_test
        dsplits = list()
        for i in range(len(self.density_bins)-1):
            d1 = self.density_bins[i]
            d2 = self.density_bins[i+1]
            self.logger.info('Computing density split model in density bin {:.2f}, {:.2f}'.format(d1, d2))
            t0 = time.time()
            ds_mask = (rho1[:, 0] >= 1 + d1) & (rho1[:, 0] <= 1 + d2)
            print(rho1[:, 0][ds_mask] - 1)
            outerint = np.trapz(innerint[..., ds_mask], x=rho1[:, 0][ds_mask], axis=-1)
            norm =  np.trapz(np.trapz(joint_density_pdf, x=rho2[0, :], axis=-1)[..., ds_mask], x=rho1[:, 0][ds_mask], axis=-1)
            #print(outerint)
            #norm = np.trapz(density_pdf[ds_mask], x=rho1[:, 0][ds_mask])
            #print(norm)
            #print(np.trapz(np.trapz(self.joint_density_pdf(rho1, rho2, xi), x=rho2[:, 0], axis=-2), x=rho1[0], axis=-1))
            res = outerint/norm - 1
            dsplits.append(res)
        return dsplits
    
        
        





