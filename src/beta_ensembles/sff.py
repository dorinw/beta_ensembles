"""
Spectral form factor calculations for random matrix ensembles.

This module provides functions for computing the spectral form factor from numerical spectra, as well as our predicted connected spectral form factor for beta ensembles using an interpolation valid for 1 <= beta <= 4.
"""

import numpy as _np
from tqdm import tqdm as _tqdm

###########################################
# Compute SFF
###########################################

def compute_sff(spectra, t_grid=None, n_points=1000, scale='linear', verbose=True):
    """
    Compute the averaged spectral form factor of a sample of spectra.

    The spectral form factor is computed for each spectrum in `spectra` and then averaged over the spectra. Both the full and connected spectral form factors are returned.
    
    If `t_grid` is not supplied, it is generated automatically according to `scale` and `n_points`.

    Parameters
    ----------
    spectra : array_like
        Collection of spectra for which to compute the spectral form factor. Each spectrum should contain the eigenvalues of one realization.
    t_grid : array_like, optional
        Values of the time coordinate at which to evaluate the spectral form factor. If `None`, a time grid is generated automatically.
    n_points : int, default=1000
        Number of points in the automatically generated time grid. Ignored if `t_grid` is supplied.
    scale : {"linear", "log", "both"}, default="linear"
        Scale used to construct the time grid when `t_grid` is `None`.
        - ``"linear"``: uniformly spaced points in linear time.
        - ``"log"``: logarithmically spaced points.
        - ``"both"``: a hybrid grid combining linear and logarithmic spacing, designed to provide adequate resolution when results are plotted on either a linear or logarithmic time axis.
    verbose : bool, default=True
        If True, display progress information during the computation.

    Returns
    -------
    t_grid : numpy.ndarray
        The time grid used for the computation. Returned only when `t_grid` was not supplied.
    sff : numpy.ndarray
        Averaged full spectral form factor evaluated on `t_grid`.
    connected_sff : numpy.ndarray
        Averaged connected spectral form factor evaluated on `t_grid`.

    Notes
    -----
    If `t_grid` is supplied, only `(sff, connected_sff)` are returned.
    Otherwise, `(t_grid, sff, connected_sff)` is returned.
    """
    spectra = _np.asarray(spectra)

    if t_grid is None:
        if scale == 'linear':
            ts = _np.linspace(0,2,n_points)
        elif scale == 'log':
            ts = _np.logspace(-4,2,n_points)
        elif scale == 'both':
            ts = _np.sort(_np.concatenate([_np.linspace(0,2,n_points),_np.logspace(-4,2,n_points)]))
        else:
            raise ValueError("Scale must be 'linear', 'log' or 'both'!")
    else:
        ts = t_grid

    r1 = _np.zeros_like(ts)
    r2 = _np.zeros_like(ts)

    n_spectra = len(spectra)
    N = len(spectra[0])

    iterator = range(n_spectra)
    if verbose:
        iterator = _tqdm(iterator,desc='Computing SFF',leave=False)

    try:
        count = 0
        for i in iterator:
            z_n = spectra[i]
            dat = _np.array([_np.sum(_np.exp(2 * _np.pi * 1j * z_n * t)) for t in ts])
            r1 = r1 + dat 
            r2 = r2 + dat * _np.conj(dat)
            count += 1
    except KeyboardInterrupt:
        if verbose and hasattr(iterator, "close"):
            iterator.close()
        print(
            f"\nProcess interrupted. Returning SFF for first {count} spectra."
        )
        
    r1 = r1 / count
    r2 = r2 / count
    
    sff = _np.real(r2) / N
    disc = _np.real(r1 * _np.conj(r1)) / N
    connected_sff = sff - disc
    
    if t_grid is None:
        return (ts,sff,connected_sff)
    else:
        return (sff,connected_sff)

###########################################
# Interpolation formula
###########################################

def predicted_sff(beta, t_grid):
    """
    Compute the predicted connected spectral form factor for a beta ensemble.

    The prediction is obtained by linearly interpolating between the standard random-matrix-theory expressions for beta = 1, 2, and 4 with beta-dependent weights. It is intended for 1 <= beta <= 4; at beta = 1, 2, and 4 it reproduces exactly the standard GOE, GUE, and GSE predictions for beta = 1, 2, and 4, respectively.

    The normalization is such that the plateau region of the SFF begins at t = 1, and the value at the plateau is 1.

    Parameters
    ----------
    beta : float
        Dyson index of the beta ensemble. Must satisfy 1 <= beta <= 4.
    t_grid : array_like
        Values of the rescaled time at which to evaluate the connected spectral form factor.

    Returns
    -------
    numpy.ndarray
        Connected spectral form factor evaluated at the points in `t_grid`.
        The returned array has the same shape as `t_grid`.
    """
    if beta >= 1 and beta <= 2:
        ramp = lambda t: 2/beta * t - (2/beta-1) * t * _np.log(1 + 2 * t)
        plateau = lambda t: 2/beta - (2/beta-1) * t * _np.log((1 + 2 * t)/(-1 + 2 * t))

        return _np.piecewise(t_grid, [t_grid <= 1, t_grid > 1], [ramp, plateau])

    if beta > 2 and beta <= 4:
        ramp = lambda t: 2/beta * t - 0.5 * (1-2/beta) * t * _np.log(1 - t)
        plateau_1 = lambda t: (4/beta-1) + (1-2/beta) * t - 1/2 * (1-2/beta) * t * _np.log(t-1)
        plateau_2 = lambda t: 1
        t1 = lambda t: _np.inf

        return _np.piecewise(t_grid, [t_grid < 1, t_grid == 1, (t_grid >= 1) & (t_grid < 2), t_grid >= 2], [ramp, t1, plateau_1, plateau_2])
    
    raise ValueError(f"Sorry! The value beta = {beta} is not supported! It's up to you to find the formula.")