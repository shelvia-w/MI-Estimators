# Adapted from:
# https://github.com/manuel-alvarez-chaves/unite_toolbox/blob/main/unite_toolbox/kde_estimators.py
# https://github.com/cbg-ethz/bmi/blob/main/src/bmi/estimators/_kde.py

import numpy as np
from scipy.integrate import nquad
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

def calc_kde_gaussian_mi(x, y, bandwidth='silverman', mode='resubstitution'):
    """
    Estimates the mutual information between x and y using KDE (gaussian kernel).
    
    Parameters
    ----------
    x : numpy.ndarray
        Array of shape (n_samples, dx_features).
    y : numpy.ndarray
        Array of shape (n_samples, dy_features).
    bandwidth : str or float, optional
        Bandwith of the kernel. Available options are 'scott','silverman'
        and a scalar [Default is 'silverman'].
    mode : str, optional
        Method for entropy calculation. Available options are 'resubstitution'
        and 'integral' [Default is 'resubstitution'].
    
    Returns
    -------
    mi : float
        Mutual information between x and y [in nats].
    
    """
    
    xy = np.hstack((x, y))
    
    kde_x = gaussian_kde(x.T, bw_method = bandwidth)
    kde_y = gaussian_kde(y.T, bw_method = bandwidth)
    kde_xy = gaussian_kde(xy.T, bw_method = bandwidth)
    
    if mode == 'resubstitution':
        p_x = kde_x.evaluate(x.T)
        p_y = kde_y.evaluate(y.T)
        p_xy = kde_xy.evaluate(xy.T)
        
        mi = np.mean(np.log(p_xy / (p_x * p_y)))
        return max(0.0, mi)
    
    elif mode == 'integral':
        bw = kde_xy.factor
        lims = np.vstack((xy.min(axis=0) - bw, xy.max(axis=0) + bw)).T

        def eval_mi(*args: float) -> float:
            p_x = kde_x.evaluate(np.vstack(args[:d]))
            p_y = kde_y.evaluate(np.vstack((args[d:])))
            p_xy = kde_xy.evaluate(np.vstack(args))
            return pxy * np.log(p_xy / (p_x * p_y))

        mi = nquad(eval_mi, ranges=lims)[0]
        return max(0.0, mi)
    
def calc_kde_mi(x, y, bandwidth='silverman', kernel='gaussian'):
    """
    Estimates the mutual information between x and y using KDE.
    
     Parameters
    ----------
    x : numpy.ndarray
        Array of shape (n_samples, dx_features).
    y : numpy.ndarray
        Array of shape (n_samples, dy_features).
    bandwidth : str or float, optional
        Bandwith of the kernel. Available options are 'scott', 'silverman'
        and a scalar [Default is 'silverman'].
    kernel: str, optional
        Kernel type. Available options are 'gaussian', 'tophat', 'epanechnikov',
        'exponential', 'linear', and 'cosine' [Default is 'gaussian'].
    
    Returns
    -------
    mi : float
        Mutual information between x and y [in nats].
    
    """
    
    xy = np.hstack((x, y))
    
    kde_x = KernelDensity(bandwidth = bandwidth, kernel = kernel).fit(x)
    kde_y = KernelDensity(bandwidth = bandwidth, kernel = kernel).fit(y)
    kde_xy = KernelDensity(bandwidth = bandwidth, kernel = kernel).fit(xy)
    
    h_x = -np.mean(kde_x.score_samples(x))
    h_y = -np.mean(kde_y.score_samples(y))
    h_xy = -np.mean(kde_xy.score_samples(xy))
    
    mi = h_x + h_y - h_xy
    return mi