# Adapted from: https://github.com/manuel-alvarez-chaves/unite_toolbox/blob/main/unite_toolbox/bin_estimators.py

import numpy as np

def calc_bin_mi(x, y, method='fd'):
    """
    Estimates the mutual information between x and y using binning.
    
    Parameters
    ----------
    x : numpy.ndarray
        Array of shape (n_samples, dx_features).
    y : numpy.ndarray
        Array of shape (m_samples, dy_features).
    method : str, optional
        Method for estimating the ideal number of bins. Available options
        are 'fd', 'doane', 'scott', 'rice', 'sturges', and 'sqrt' [Default is 'fd'].
    
    Returns
    -------
    mi : float
        Mutual information between x and y [in nats].
    
    """
    
    bins_x = estimate_ideal_bins(x, counts=False)[method]
    bins_y = estimate_ideal_bins(y, counts=False)[method]
    
    _, dx = x.shape
    _, dy = y.shape
    xy = np.hstack((x, y))
    
    p_xy, joint_edges = np.histogramdd(xy, bins=bins_x+bins_y, density=True)
    p_x, _ = np.histogramdd(x, bins=bins_x, density=True)
    p_y, _ = np.histogramdd(y, bins=bins_y, density=True)
    
    volume = calc_vol_array(joint_edges)
    
    mi = 0.0
    for idx in np.ndindex(p_xy.shape):
        if p_xy[idx] != 0.0:
            mi += (p_xy[idx] * volume[idx] * np.log(p_xy[idx] / (p_x[idx[:dx]] * p_y[idx[-dy:]])))
    return max(0.0, mi)

def estimate_ideal_bins(data, counts=True):
    """
    Estimates the number of ideal bins.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape (n_samples, d_features).
    counts : bool, optional
        Whether to return the number of bins (True) or the bin edges (False).

    Returns
    -------
    dict
        A dictionary with a key for each method, and the values are lists of
        number of bins or bin edges for each feature of the data (if counts=False).
        
    """
    
    _, d_features = data.shape
    
    methods = ['fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt']
    ideal_bins = []
    
    for m in methods:
        d_bins = []
        for d in range(d_features):
            n_bins = np.histogram_bin_edges(data[:, d], bins=m)
            n_bins = len(n_bins) if counts is True else n_bins
            d_bins.append(n_bins)
        ideal_bins.append(d_bins)
        
    return dict(zip(methods, ideal_bins, strict=True))

def calc_vol_array(edges):
    """
    Calculates the volume of a multidimensional array.

    Parameters
    ----------
    edges : list[np.ndarray]
        List of 1D NumPy arrays.

    Returns
    -------
    vol : numpy.ndarray
        Array of shape (len(arr0) - 1, len(arr1) - 1, ..., len(arrn) - 1).

    """
    
    vol = np.diff(edges[0])
    for e in edges[1:]:
        vol = np.stack([vol] * (len(e) - 1), axis=-1)
        for idx, val in enumerate(np.diff(e)):
            vol[..., idx] = vol[..., idx] * val
    return vol