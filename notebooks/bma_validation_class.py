import numpy as np
from astropy.table import Table, vstack
from astropy.io.fits import getdata
import scipy.stats as st

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)

class bmaValidation:
    """This class provides a set of metrics and plots to validate the bma code
    """
    
    def __init__(self):
        print('Welcome to BMA Validation')
    
    def add_model(self, name, mass, mass_true, abs_mag, abs_mag_true, redshift):
        self.model_name = name
        self.mass = np.array(mass)
        self.mass_true = np.array(mass_true)
        self.Mr = np.array(abs_mag)
        self.Mr_true = np.array(abs_mag_true)
        self.z = np.array(redshift)
        
        # new variables
        self.res_mass = self.mass-self.mass_true
        self.res_Mr = self.Mr-self.Mr_true
                
    def plot_residual_mass(self,name,mask=None, s=20, alpha=0.3,fontsize=20, ax=None, scatter=True, label = None):
        if ax is None: ax=plt.axes()
        if mask is None: mask = np.argsort(self.mass)
        ylabel = res_mass_label
        x, y = self.mass_true[mask], self.res_mass[mask]
        x_bin, x_bin_err, y_bin, y_bin_err = get_binned_variables(x,y)
        snmad, mad = get_nmad(y)
        name = r'%s :$\sigma_{nmad}=$ %.3f'%(name,snmad)
        
        if scatter:
            ax.scatter(x,y, s=s, alpha=alpha, label='_nolabel_', color = 'grey')
        ax.errorbar(x_bin,y_bin,xerr=x_bin_err,yerr=y_bin_err, fmt='o', linestyle='--', markersize=8, capsize=4, capthick=2, label = name)
        ax.set_xlabel(mass_true_label, fontsize=fontsize)
        ax.set_ylabel(res_mass_label, fontsize=fontsize)
        ax.set_title('Mass Residual Plot', fontsize=fontsize)
        
        #ax.show()
        pass
    
    def plot_residual_absolute_mag(self,name,mask=None, s=20, alpha=0.3, fontsize = 20, ax=None, scatter=True, label = None):
        if ax is None: ax=plt.axes()
        if mask is None: mask = np.argsort(self.Mr)
        ylabel = res_abs_mag_label
        x, y = self.Mr_true[mask], self.res_Mr[mask]
        x_bin, x_bin_err, y_bin, y_bin_err = get_binned_variables(x,y)
        snmad, mad = get_nmad(y)
        name = r'%s :$\sigma_{nmad}=$ %.3f'%(name,snmad)
        
        if scatter:
            ax.scatter(x,y, s=s, alpha=alpha, label='_nolabel_', color = 'grey')
        ax.errorbar(x_bin,y_bin,xerr=x_bin_err,yerr=y_bin_err, fmt='o', linestyle='--', markersize=8, capsize=4, capthick=2, label = name)
        ax.set_xlabel(abs_mag_label, fontsize=fontsize)
        ax.set_xlim(-26, -10)
        ax.set_ylabel(res_abs_mag_label, fontsize=fontsize)
        ax.set_ylim(-5, 5)
        ax.set_title('Absolute Magnitude Residual Plot', fontsize=fontsize)
        
        pass
    
    def plot_residual_redshift(self,name,mask=None, s=20, alpha=0.3,fontsize=20, ax=None, scatter=True, label = None):
        if ax is None: ax=plt.axes()
        if mask is None: mask = np.argsort(self.mass)
        ylabel = res_mass_label
        x, y = self.z[mask], self.res_mass[mask]
        x_bin, x_bin_err, y_bin, y_bin_err = get_binned_variables(x,y)
        ax.errorbar(x_bin,y_bin,xerr=x_bin_err,yerr=y_bin_err, fmt='o', linestyle='--', markersize=8, capsize=4, capthick=2, label = name)
        snmad, mad = get_nmad(y)
        name = r'%s :$\sigma_{nmad}=$ %.3f'%(name,snmad)
        ax.set_xlabel(mass_true_label)
        ax.set_ylabel('redshift')
        ax.set_title('Redshift Residual Plot', fontsize=fontsize)
        pass
        
    def plot_identity_mass_redshift(self,mask=None, ax=None):
        if ax is None: ax=plt.axes()
        if mask is None: mask = np.argsort(self.mass)
        tree_variable_plot(self.mass_true[mask],self.mass[mask],self.z[mask], ax=ax)
        ax.set_ylabel(mass_label)
        ax.set_xlabel(mass_true_label)
        pass

    def plot_identity_mass_chisqr(self,mask=None, ax=None):
        if ax is None: ax=plt.axes()
        if mask is None: mask = np.argsort(self.mass)
        tree_variable_plot(self.mass_true[mask],self.mass[mask],self.chisqr[mask],zlabel='Log(chisqr)', ax=ax)
        ax.set_ylabel(mass_label)
        ax.set_xlabel(mass_true_label)
        pass
        
        
def remove_nan(x):
            return np.logical_not(np.isnan(x))

mass_label = r'Log($M_{\star}^{BMA}$)'
mass_true_label = r'Log($M_{\star}^{COSMOS}$)'
res_mass_label = r'Log $\left(M_{\star}^{COSMOS} / M_{\star}^{BMA} \right)$'
res_abs_mag_label   = r'$M_r^{BMA}-M_r^{COSMOS}$'
abs_mag_label = r'$M_r^{COSMOS}$'

def tree_variable_plot(x1,x2,x3,zlabel='$z_{cls}$', ax=None):
    cut = remove_nan(x2)
    idx = np.argsort(-1*x3)
    
    xmin, xmax = np.nanmin(np.hstack([x1,x2])), np.nanmax(np.hstack([x1,x2]))
    ax.plot([xmin,xmax],[xmin,xmax],'k--',lw=3)
    ax.scatter(x1[idx],x2[idx],c=x3[idx],cmap='RdBu',s=5,alpha=0.8)
    if scatter:
        ax.scatter(x,y, s=s, alpha=alpha, label='_nolabel_', color = 'grey')
        ax.errorbar(x_bin,y_bin,xerr=x_bin_err,yerr=y_bin_err, fmt='o', linestyle='--', markersize=8, capsize=4, capthick=2, label = name)
        ax.set_xlabel('z', fontsize=fontsize)
        ax.set_ylabel(res_mass_label, fontsize=fontsize)
        ax.set_title('Residual as a function of Redshift', fontsize=fontsize)
        
        
        pass

    
# auxialiary variables and functions

def makeBin(variable,nbins=10.0,xvec=None):
    width = len(variable) / nbins
    if xvec is None:
        xmin, xmax = (variable.min()), (variable.max() + width)
        xvec = np.arange(xmin,xmax,width)

    #idx,xbins = [], []
    idx = [np.where((variable>=xlo)&(variable<=xhi))[0]
           for xlo,xhi in zip(xvec[:-1],xvec[1:])]
    xbins = 0.5*(xvec[1:]+xvec[:-1])
    return idx, xbins
    
def get_binned_variables(x,y,xedges=None):

    if xedges is None:
        xedges = np.nanpercentile(x, np.linspace(0, 100, 12))

    indices, x_bin = makeBin(x, xvec=xedges)
    x_bin_err = np.diff(xedges)/ 2
    y_bin = [np.mean(y[idx]) for idx in indices]
    y_bin_err = [np.std(y[idx]) for idx in indices]        
    return x_bin, x_bin_err, y_bin, y_bin_err

def get_nmad(x):
    mad = np.nanmedian(x)
    nmad = 1.48*np.median(np.abs(x-mad))
    return nmad, mad

def remove_nan(x):
    return np.logical_not(np.isnan(x))

def tree_variable_plot(x1,x2,x3,zlabel='$z_{cls}$', ax=None):
    cut = remove_nan(x2)
    idx = np.argsort(-1*x3)
    
    xmin, xmax = np.nanmin(np.hstack([x1,x2])), np.nanmax(np.hstack([x1,x2]))
    ax.plot([xmin,xmax],[xmin,xmax],'k--',lw=3)
    ax.scatter(x1[idx],x2[idx],c=x3[idx],cmap='RdBu',s=5,alpha=0.8)