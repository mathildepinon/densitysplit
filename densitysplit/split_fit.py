import numpy as np
import scipy
import scipy.stats
from matplotlib import pyplot as plt
import matplotlib as mpl
import math

from iminuit import Minuit
from pycorr import TwoPointCorrelationFunction, TwoPointEstimator, NaturalTwoPointEstimator, project_to_multipoles, project_to_wp, utils, setup_logging
from cosmoprimo import *

from .utils import *


class DensitySplitFit:
    """
    Class to fit density splits correlation function with a given model.
    """

    def __init__(self, sep, k, ells, xiell, cov, xi_model=None, model_params=None, model_params_labels=None, nsplits=1):

        self.sep = sep
        self.k = k
        self.ells = ells
        self.nells = len(ells)
        self.xiell = np.array(xiell)
        self.cov = cov
        self.set_xi_model(xi_model, model_params, model_params_labels)
        self.fixed_params = None
        self.free_params = list(model_params.keys())
        self.fitted = False
        self.nsplits = nsplits
        self.std = np.array_split(np.array(np.array_split(np.diag(cov)**0.5, self.nells)), self.nsplits, axis=1)
                
    def set_xi_model(self, xi_model, model_params, model_params_labels=None):
        self.xi_model = xi_model
        self.xi_model.set_params(**model_params)
        self.model_params = model_params
        if model_params_labels is not None:
            self.model_params_labels = model_params_labels
        
    def extract_split(self, split_index):
        xiell = self.xiell[split]
        cov = extract_subcovmatrix(self.sep, self.cov, self.ells, slef.nsplits, split_extract=split_index)
        std = np.array_split(np.diag(cov)**0.5, nells)
        return xiell, cov, std
    
    def set_s_lower_limit(self, s_lower_limit):
        self.s_lower_limit = s_lower_limit
        s, xiell, cov = truncate_xiell(s_lower_limit, self.sep, self.xiell, self.ells, self.cov, split=self.nsplits>1, nsplits=self.nsplits)
        self.sep = s
        self.xiell = xiell
        self.cov = cov
        self.std = np.array_split(np.array(np.array_split(np.diag(cov)**0.5, self.nells)), self.nsplits, axis=1)
    
    def fit(self, fit_params_init, s_lower_limit=None, print_output=True, fit_method='scipy', 
            minos=False, mnprofile=False, profile_param=None, mncontour=False, contour_params=None):
        fit_params_init_copy = copy.deepcopy(fit_params_init)
        fit_params_init_values = list(fit_params_init_copy.values())
        fit_params_names = list(fit_params_init_copy.keys())
        self.free_params = fit_params_names
        self.fixed_params = {key: val for (key, val) in self.model_params.items() if key not in fit_params_names}
        
        if s_lower_limit is not None:
            self.set_s_lower_limit(s_lower_limit)
            
        s = self.sep
        xiell = self.xiell
        cov = self.cov

        def fitting_func(s, *fit_params):
            fit_params_dict = {key: value for key, value in zip(fit_params_names, fit_params)}
            fit_params_dict.update(self.fixed_params)
            
            res = self.xi_model.split_model(s=s, params=fit_params_dict)
            return res
        
        # Function to minimize with iminuit
        def iminuit_chisq(*fit_params):
            model = fitting_func(s, *fit_params)
            return compute_chisq(s, xiell.flatten(), cov, model)
        
        if fit_method == 'scipy':
            popt, pcov = scipy.optimize.curve_fit(fitting_func, s, xiell.flatten(), sigma=cov, p0=np.array(fit_params_init_values), absolute_sigma=True)
        
        if fit_method == 'iminuit':
            m = Minuit(iminuit_chisq, **fit_params_init_copy, name=fit_params_names)
            m.migrad()
            imin = m.hesse()
            
            popt = list()
            for param in imin.params:
                popt.append(param.value)
                
            pcov = np.array(imin.covariance)
            
            if minos:
                iminos = m.minos()
                minos_errors = list()
                for param in iminos.params:
                    minos_errors.append(param.merror)
                    
            if mnprofile:
                m.draw_mnprofile(vname=profile_param, size=50, bound=3)
                    
            if mncontour:
                m.draw_mncontour(x=contour_params[0], y=contour_params[1])
            
            self.minos = minos
            self.minuit = m
        
        self.popt = popt
        self.pcov = pcov
        self.popt_dict = {key: value for key, value in zip(fit_params_names, popt)}
        self.set_xi_model(xi_model=self.xi_model, model_params=self.popt_dict)
        
        if minos:
            self.minos_errors = {key: value for key, value in zip(model_params_names, minos_errors)}

        if print_output:
            print('Optimal parameters:')
            print(self.popt_dict)

            print('\nCovariance matrix:')
            print(pcov)

            print('\nSigmas:')
            print(np.diag(pcov)**0.5)
            
            if minos:
                print('\nMinos errors:')
                print(self.minos_errors)

        self.fitted = True
        self._chi_square()

        return self.popt_dict, pcov
    
    def _chi_square(self, reduced=True):
        s = self.sep
        xiell = self.xiell
        cov = self.cov

        if self.fitted:
            model = self.xi_model.split_model(s=s, params=self.popt_dict)
            ndof = len(s)*self.nells*self.nsplits-len(self.popt)
        else:
            model = self.xi_model.split_model(s=s, pk_model_params=self.model_params)
            ndof = len(s)*self.nells*self.nsplits

        chisq = compute_chisq(np.tile(s, self.nells*self.nsplits), xiell.flatten(), cov, model)
        self.chisq = chisq

        rchisq = chisq/ndof
        self.ndof = ndof
        self.rchisq = rchisq

        if reduced:
            return rchisq
        else:
            return chisq
        
    def plot_split_model(self, fig=None, axes=None, show_data=True, show_broadband=False, show_info=False):
        if axes is None:
            axes=[plt.gca() for i in range(self.nsplits)]

        s = self.sep
        ns = len(s)
        
        if show_data: 
            for split in range(self.nsplits):
                if self.nsplits == 1:
                    ax = axes
                    split_xiell = self.xiell
                else:
                    ax = axes[split]
                    split_xiell = self.xiell[split]

                for ill, ell in enumerate(self.ells):
                    ax.errorbar(s, s**2 * split_xiell[ill], s**2 * self.std[split][ill], fmt='-', color='C'+str(ill))
                
        if hasattr(self, 's_lower_limit') and self.s_lower_limit is not None:
            ax.axvline(self.s_lower_limit, linestyle='dashed', color='r', linewidth=0.7)
            s_lower_limit_info = self.s_lower_limit
        else:
            s_lower_limit_info = 0
            
        self.xi_model.plot_split_model(fig=fig, axes=axes, show_broadband=show_broadband, show_info=show_info)
        ax.errorbar([], [], [], linestyle='-', color='black', label='Data')
        ax.legend()

        if show_info and self.fitted:
            if hasattr(self, 'minos') and self.minos:
                minos_errors_string = '\nMinos errors:\n'+'\n'.join([r'{}: err- {:.2e}, err+ {:.2e}'.format(self.model_params_labels[pname], merror[0], merror[1]) for pname, merror in zip(self.free_params, self.minos_errors.values())])

            else:
                minos_errors_string = ''

            plt.suptitle('----------\n'
                         +'Fit info\n'
                         + '----------\n'
                         + 'Fit from $s$ > {:.0f} Mpc/$h$ \n'.format(s_lower_limit_info) 
                         + r'$\chi^2_{r}$=' +'{:.2e}'.format(self.rchisq)
                         + '\n\nFree parameters\n' 
                         + '------------------------\n' 
                         + '\n'.join([r'{}: {:.3e} $\pm$ {:.2e}'.format(self.model_params_labels[pname], value, std) for pname, value, std in zip(self.popt_dict.keys(), self.popt_dict.values(), np.diag(self.pcov)**0.5)])
                         + minos_errors_string
                         + '\n\nFixed parameters\n' 
                         + '-------------------------\n' 
                         + '\n'.join([r'{}: {:.1e}'.format(self.model_params_labels[pname], value) for pname, value in zip(self.fixed_params.keys(), self.fixed_params.values())]),
                         ha='left', x=0.1, y=0, size=14)      
            
    def plot_likelihood(self, param_name, param_values, free_params_init):
        chisq_vals = list()
        
        for param in param_values:
            self.set_xi_model(xi_model=self.xi_model, model_params={param_name: param})
            self.fit(fit_params_init=free_params_init, s_lower_limit=self.s_lower_limit, print_output=False)
            chisq_vals.append(self.chisq)
        chisq_vals = np.array(chisq_vals)
        
        min_chisq = np.min(chisq_vals)
        conf_int = [scipy.stats.chi2.cdf(s**2, 1) for s in [1, 2, 3]]
        chisq_sigmas = [scipy.stats.chi2.ppf(ci, 1) for ci in conf_int]

        param_limits_idx = np.argwhere(np.diff(np.sign(chisq_vals - (min_chisq+chisq_sigmas[0])))).flatten()

        plt.plot(param_values, chisq_vals, color='C0')
        plt.axhline(min_chisq+chisq_sigmas[0], linestyle='dotted', color='C0')
        plt.axhline(min_chisq+chisq_sigmas[1], linestyle='dotted', color='C0')
        plt.axhline(min_chisq+chisq_sigmas[2], linestyle='dotted', color='C0')
        plt.ylim(bottom=min_chisq)
        
    def draw_profile(self, profile_param, bounds, fit_params_init, s_lower_limit=None):
        fit_params_init_copy = copy.deepcopy(fit_params_init)
        fit_params_init_values = list(fit_params_init_copy.values())
        fit_params_names = list(fit_params_init_copy.keys())
        self.free_params = fit_params_names
        self.fixed_params = {key: val for (key, val) in self.model_params.items() if key not in fit_params_names}
        
        if s_lower_limit is not None:
            self.set_s_lower_limit(s_lower_limit)
            
        s = self.sep
        xiell = self.xiell
        cov = self.cov
        
        # Function to minimize with iminuit
        def iminuit_chisq(*fit_params):
            model = fitting_func(s, *fit_params)
            return compute_chisq(s, xiell.flatten(), cov, model)
        
        m = Minuit(iminuit_chisq, **fit_params_init_copy, name=fit_params_names)




        