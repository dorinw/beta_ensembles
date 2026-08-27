"""
This module contains the helper functions for plotting.
"""

from . import statistics as _st
import numpy as _np
import matplotlib.pyplot as _plt

###########################################
# Plotting functions: histograms
###########################################
def hist(values, **kwargs):
    """
    Plot a histogram of the provided values in a default style.
    
    The spacings can be a single array or a collection of arrays. They are flattened before plotting. By default, plots the density with 100 bins.
    
    Keyword arguments are passed directly to pyplot.hist. They can be used for styling or to override default behavior.
    
    Returns
    -------
    tuple
        Tuple containing the histogram values, bin edges, and the container of histogram patches returned by pyplot.hist.
    """
    values = _np.asarray(values)

    kwargs.setdefault('bins',100)
    kwargs.setdefault('density',True)

    h = _plt.hist(values.ravel(),**kwargs)

    return h


def hist_s(spacings, **kwargs):
    """
    Plot a histogram of the spacings in a default style.
    
    The spacings can be a single array or a collection of arrays. They are flattened before plotting. By default, plots the density with 100 bins and labels the axes as "s" and "p_s(s)".
    
    Keyword arguments are passed directly to pyplot.hist. They can be used for styling or to override default behavior.
    
    Returns
    -------
    tuple
        Tuple containing the histogram values, bin edges, and the container of histogram patches returned by pyplot.hist.
    """

    h = hist(spacings,**kwargs)
    _plt.xlabel(r'$s$')
    _plt.ylabel(r'$p_s(s)$')
    
    return h


def hist_r(ratios, **kwargs):
    """
    Plot a histogram of the spacing ratios in a default style.
    
    The ratios can be a single array or a collection of arrays. They are flattened before plotting. By default, plots the density with 100 bins and labels the axes as "r" and "p_r(r)".
    
    Keyword arguments are passed directly to pyplot.hist. They can be used for styling or to override default behavior.
    
    Returns
    -------
    tuple
        Tuple containing the histogram values, bin edges, and the container of histogram patches returned by pyplot.hist.
    """
    
    h = hist(ratios,**kwargs)
    _plt.xlabel(r'$r$')
    _plt.ylabel(r'$p_r(r)$')
    
    return h

###########################################
# Plotting functions: spacings distributions
###########################################
    
def pdf_s(beta, s_min=0, s_max=4, n_points = 400, **kwargs):
    """
    Plot the Wigner-Dyson spacing distribution for a given value of beta.
    
    The distribution is plotted over the range s_min to s_max using n_points points.
    
    Additional keyword arguments are passed directly to pyplot.plot. They can be used for styling or to override default behavior.
    
    Parameters
    ----------
    beta : float
        Dyson index of the Wigner-Dyson distribution.
    s_min : float, optional
        Minimum spacing value to plot. Default is 0.
    s_max : float, optional
        Maximum spacing value to plot. Default is 4.
    n_points : int, optional
        Number of points used to evaluate the distribution. Default is 400.
    **kwargs
        Additional keyword arguments passed directly to pyplot.plot.
    
    Returns
    -------
    list
        List of Line2D objects returned by pyplot.plot.
    """

    s = _np.linspace(s_min,s_max,n_points)

    kwargs.setdefault("label", fr"$\beta={beta:.3f}$")
    
    pdf = _st.pdf_s(beta, s)

    line = _plt.plot(s,pdf,**kwargs)
    _plt.xlabel(r"$s$")
    _plt.ylabel(r"$p_s(s)$")
    
    return line

def cdf_s(beta, s_min=0, s_max=4, n_points = 400, **kwargs):
    """
    Plot the cumulative distribution function for the Wigner-Dyson spacing distribution for a given value of beta.
    
    The distribution is plotted over the range s_min to s_max using n_points points.
    
    Additional keyword arguments are passed directly to pyplot.plot. They can be used for styling or to override default behavior.
    
    Parameters
    ----------
    beta : float
        Dyson index of the Wigner-Dyson distribution.
    s_min : float, optional
        Minimum spacing value to plot. Default is 0.
    s_max : float, optional
        Maximum spacing value to plot. Default is 4.
    n_points : int, optional
        Number of points used to evaluate the distribution. Default is 400.
    **kwargs
        Additional keyword arguments passed directly to pyplot.plot.
    
    Returns
    -------
    list
        List of Line2D objects returned by pyplot.plot.
    """

    s = _np.linspace(s_min,s_max,n_points)

    kwargs.setdefault("label", fr"$\beta={beta:.3f}$")
    
    pdf = _st.cdf_s(beta, s)

    line = _plt.plot(s,pdf,**kwargs)
    _plt.xlabel(r"$s$")
    _plt.ylabel(r"$P_s(s)$")
    
    return line
    
###########################################
# Plotting functions: spacing ratio distributions
###########################################

def pdf_r(beta, r_min=0, r_max=None, n_points=400, reduced=True, **kwargs):
    """
    Plot the Atas-Bogomolny-Giraud-Roux spacing ratio distribution for a given value of beta.
    
    The distribution is plotted over the range r_min to r_max using n_points points.

    By default plots the reduced spacing ratio distribution from 0 to 1. To plot the distribution for the un-reduced spacing ratios, specify reduced=False. In that case the distribution is by default plotted from 0 to 4.
    
    Additional keyword arguments are passed directly to pyplot.plot. They can be used for styling or to override default behavior.
    
    Parameters
    ----------
    beta : float
        Dyson index of the distribution.
    r_min : float, optional
        Minimum ratio value to plot.
    r_max : float, optional
        Maximum raio value to plot.
    n_points : int, optional
        Number of points used to evaluate the distribution.
    reduced : bool
        If True (default behavior) lots the distribution for the reduced spacing ratios, defined from 0 to 1. Else plot the unreduced ratio.
    **kwargs
        Additional keyword arguments passed directly to pyplot.plot.
    
    Returns
    -------
    list
        List of Line2D objects returned by pyplot.plot.
    """
    if r_max is None:
        r_max = 1 if reduced else 4

    kwargs.setdefault("label", fr"$\beta={beta:.3f}$")
    
    r = _np.linspace(0,r_max,n_points)
    
    pdf = _st.pdf_r(beta,r,reduced=reduced)

    line = _plt.plot(r,pdf,**kwargs)
    
    _plt.xlabel(r"$r$")
    _plt.ylabel(r"$p_r(r)$")
    _plt.xlim(r_min,r_max)
    
    return line
    
def cdf_r(beta, r_min=0, r_max=None, n_points=400, reduced=True, **kwargs):
    """
    Plot the cumulative distribution function of the Atas-Bogomolny-Giraud-Roux spacing ratio distribution for a given value of beta.

    The CDF is given by an analytic expression we derived, rather than a numerical integration.
    
    The distribution is plotted over the range r_min to r_max using n_points points.

    By default plots the reduced spacing ratio distribution from 0 to 1. To plot the distribution for the un-reduced spacing ratios, specify reduced=False and set the value of r_max.
    
    Additional keyword arguments are passed directly to pyplot.plot. They can be used for styling or to override default behavior.
    
    Parameters
    ----------
    beta : float
        Dyson index of the distribution.
    r_min : float, optional
        Minimum ratio value to plot.
    r_max : float, optional
        Maximum raio value to plot.
    n_points : int, optional
        Number of points used to evaluate the distribution.
    reduced : bool
        If True (default behavior) lots the distribution for the reduced spacing ratios, defined from 0 to 1. Else plot the unreduced ratio.
    **kwargs
        Additional keyword arguments passed directly to pyplot.plot.
    
    Returns
    -------
    list
        List of Line2D objects returned by pyplot.plot.
    """
    if r_max is None:
        r_max = 1 if reduced else 4
    
    r = _np.linspace(r_min, r_max, n_points)
    pdf = _st.cdf_r(beta,r,reduced)
    line = _plt.plot(r,pdf,**kwargs)
    _plt.xlabel(r'$r$')
    _plt.ylabel(r'$P_r(r)$')
    
    return line
    

###########################################
# Eigenvalue densities
###########################################

def semicircle(N, n_points=1000, **kwargs):
    """
    Plot the Wigner semicircle distribution for a given matrix dimension N.
    
    The semicircle has radius R = 2 * sqrt(N), such that N = (R / 2)**2.
    
    Additional keyword arguments are passed directly to pyplot.plot. They can be used for styling or to override default behavior.
    
    Parameters
    ----------
    N : int
        Matrix dimension, related to radius of semicircle by R = 2 * sqrt(N).
    n_points : int, optional
        Number of points used to evaluate the distribution.
    **kwargs
        Additional keyword arguments passed directly to pyplot.plot.
    
    Returns
    -------
    list
        List of Line2D objects returned by pyplot.plot.
    """
    R = 2*_np.sqrt(N)
    
    z = _np.linspace(-R+0.00001,R-0.00001,n_points)
    rho = 1 / 2 / _np.pi / N * _np.sqrt(4*N - z**2)

    line = _plt.plot(z,rho,**kwargs)

    return line


def MarchenkoPastur(beta, alpha, N, n_points=1000, **kwargs):
    """
    Plots the Marchenko-Pastur distribution with parameters matching our normalization convention, set in laguerre.spectra(...).
    
    The Marchenko-Pastur distribution is in the range x_m to x_p, where
    x_m = M (1 - sqrt(gamma))**2
    x_p = M (1 + sqrt(gamma))**2
    
    The parameters M and gamma are related to beta, alpha, and N by:
    M - N + 1 = 2 / beta * (alpha + 1)
    gamma = N / M

    Additional keyword arguments are passed directly to pyplot.plot. They can be used for styling or to override default behavior.
    
    Parameters
    ----------
    beta : float
        Dyson index of the Laguerre beta ensemble.
    alpha : float
        Alpha parameter of the Laguerre beta ensemble.
    N : int
        Number of levels in the spectrum.
    n_points : int, optional
        Number of points used to evaluate the distribution.
    **kwargs
        Additional keyword arguments passed directly to pyplot.plot.
    
    Returns
    -------
    list
        List of Line2D objects returned by pyplot.plot.
    """

    # The M parameter is given by (also defined in laguerre._M):
    M = N - 1 + 2/beta * (alpha+1)
    
    gamma = N / M
    
    xp = M * ((1+_np.sqrt(gamma))**2)
    xm = M * ((1-_np.sqrt(gamma))**2)
    
    x = _np.linspace(xm, xp,n_points)
    
    MP = 1 / 2 / _np.pi * _np.sqrt((xp - x) * (x - xm)) / x  / N

    line = _plt.plot(x,MP,**kwargs)

    return line

###########################################
# Plotting functions: SFF
###########################################

def predicted_sff(beta, t_min=None, t_max=None, n_points=1000, scale='linear', **kwargs):
    """
    Plot the predicted spectral form factor for a given value of beta, using our interpolation formulas for 1 <= beta <= 4.

    See sff.predicted_sff(...) for details.

    If t_min or t_max are not specified, their default values are 0 and 2 if scale is 'linear', or 1e-4 and 100 if scale is 'log'.

    Parameters
    ----------
    beta : float
        Dyson index of the ensemble. Must be between 1 and 4.
    t_min : float, optional
        Minimum value of the time variable t.
    t_max : float, optional
        Maximum value of the time variable t.
    n_points : int, optional
        Number of points used to evaluate the SFF.
    scale : {'linear', 'log'}, optional
        Scale used for the time axis. Default is 'linear'.
    **kwargs
        Additional keyword arguments passed directly to pyplot.plot.

    Returns
    -------
    list
        List of Line2D objects returned by pyplot.plot.
    """
    from .sff import predicted_sff

    # if scale is log, then t_min is in fact log10(t_min), likewise t_max below
    if t_min is None:
        t_min = 0 if scale == 'linear' else -4
    if t_max is None:
        t_max = 2 if scale == 'linear' else 2
    
    if scale == 'linear':
        ts = _np.linspace(t_min,t_max,n_points)
    elif scale == 'log':
        ts = _np.logspace(t_min,t_max,n_points)
    else:
        raise ValueError("Scale must be 'linear' or 'log'!")
    
    sff = predicted_sff(beta,ts)
    
    line = _plt.plot(ts,sff,**kwargs)
    _plt.xscale(scale)
    _plt.yscale(scale)
    
    return line