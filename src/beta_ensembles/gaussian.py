"""
This module contains the generator functions for the Gaussian beta-ensemble, implementing the Dumitriu-Edelam construction.

Also implemented is a function for unfolding the spectra using the semicircle distribution's CDF.
"""

###########################################
# Imports
###########################################

import numpy as _np
import scipy as _scipy
from tqdm import tqdm as _tqdm

###########################################
# Random generators
###########################################

def spectra(beta, N, n_spectra=1, rng=None, verbose=True):
    """
    Generate spectra from the Gaussian beta ensemble (GBE).

    Uses the Dumitriu-Edelman tridiagonal matrix model to generate n_spectra independent spectra of length N from the GBE with Dyson index beta.

    The parameter beta is continuous and non-negative. The special cases beta = 1, 2, and 4 reproduce the classical Gaussian orthogonal, unitary, and symplectic ensembles (GOE, GUE, and GSE).

    The eigenvalues are normalized so that, in the large-N limit, their density approaches the semicircle law with radius 2*sqrt(N), independently of beta.

    Parameters
    ----------
    beta : float
        Dyson index of the ensemble. Must be non-negative (see Notes).
    N : int
        Number of levels in each spectrum.
    n_spectra : int, optional
        Number of spectra to generate.
    rng : numpy.random.Generator or None, optional
        Random number generator. If None, a new generator is created.
        Passing a seeded generator allows reproducible results.
    verbose : bool, optional
        If True, display a progress bar while generating spectra.

    Returns
    -------
    ndarray
        Array of shape (n_spectra, N) containing the generated spectra.

    Notes
    -----
    Setting beta = 0 returns Poisson spectra as a convenience option.

    If generation is interrupted with KeyboardInterrupt, any spectra computed up to that point are returned.
    """
    if beta == 0:
        return poisson_spectra(N,n_spectra,rng,verbose)
    
    if rng is None:
        rng = _np.random.default_rng()

    iterator = range(n_spectra)
    if verbose:
        iterator = _tqdm(iterator, desc="Generating GBE spectra")

    try:
        results = []
        for _ in iterator:
            results.append(_spectrum(beta,N,rng))
    except KeyboardInterrupt:
        if verbose and hasattr(iterator, "close"):
            iterator.close()
        print(
            f"\nProcess interrupted. Returning {len(results)} generated spectra."
        )
    
    return _np.array(results)

# Implementation of Dumitriu-Edelman
def _spectrum(beta, N, rng):
    """ 
    Generate a single GBE spectrum. See spectra for details.
    """
    
    # for convenience, in the special case beta = 0, return Poisson
    if beta == 0:
        return _poisson_spectrum(N,rng)

    # Uses _matrix(...) to get the diagonals
    di, xi = _matrix(beta,N,return_matrix=False,rng=rng)
    
    # the tridiagonal form lets us use an optimized algorithm for diagonalizing:
    return _np.sort(_scipy.linalg.eigvalsh_tridiagonal(di,xi))


    
def matrices(beta, N, n_matrices=1, rng=None):
    """
    Generate random matrices from the Gaussian beta-ensemble (GBE).

    The matrices are generated using the Dumitriu-Edelman tridiagonal model, whose eigenvalues follow the GBE distribution. The resulting matrices are tridiagonal symmetric rather than generic dense random matrices.

    Parameters
    ----------
    beta : float
        Dyson index of the ensemble. Must be positive.
    N : int
        Matrix dimension.
    n_matrices : int, optional
        Number of independent matrices to generate.
    rng : numpy.random.Generator or None, optional
        Random number generator. If None, a new generator is created. Passing a seeded generator allows reproducible results.

    Returns
    -------
    ndarray
        Array of shape (n_matrices, N, N) containing the generated tridiagonal matrices.
    """
    if rng is None:
        rng = _np.random.default_rng()
    
    return _np.array([_matrix(beta,N,return_matrix=True,rng=rng) for _ in range(n_matrices)])

def _matrix(beta, N, return_matrix=True, rng=None):
    """
    Generate a single random GBE matrix.
    
    Set return_matrix=False to return the diagonals instead of a dense matrix (for efficient diagonalization).
    
    See matrices for other details.
    """
     # Define the diagonal elements
    di = rng.normal(0,_np.sqrt(2),size=N) / _np.sqrt(beta)
    # Off-diagonals are from chi-distribution
    xi = [_np.sqrt(rng.chisquare((N-1-k)*beta)) for k in range(N-1)] / _np.sqrt(beta)

    if return_matrix:
        M = _np.zeros((N,N))

        _np.fill_diagonal(M, di)
        _np.fill_diagonal(M[:-1, 1:], xi) # Upper off-diagonal
        _np.fill_diagonal(M[1:, :-1], xi) # Lower off-diagonal

        return M
    else:
        return (di,xi)


###########################################
# Unfolding
###########################################

def unfold(spectra, radius=None):
    """
    Unfold Gaussian beta ensemble spectra using the semicircle density.

    Uses the cumulative distribution function of the semicircle law to unfold one or more spectra. The transformation maps eigenvalues to coordinates with approximately constant mean density and average spacing equal to one.

    This function by default assumes the normalization conventions used by spectra(...), such that the radius of the semicircle is 2*sqrt(N), where N is the dimension of each spectrum.

    Parameters
    ----------
    spectra : array_like, shape (N,) or (n_spectra, N)
        Eigenvalues of one or more spectra.
    radius : float, optional
        Radius of the semicircle distribution. If None, assumes our convention for the Gaussian beta ensemble normalization, with R = 2*sqrt(N).

    Returns
    -------
    array
        Unfolded spectra with the same shape as the input. The eigenvalues are mapped to the interval [0, N].

    Notes
    -----
    Eigenvalues outside the assumed spectral support are clipped: values below -R are mapped to 0 and values above R are mapped to N. Users should be aware that this can introduce unwanted edge effects.
    """
    if _np.asarray(spectra).ndim == 1:
        return _unfold(spectra, radius)
    else:
        return _np.array([_unfold(spectrum,radius) for spectrum in spectra])


def _unfold(l, radius=None):
    """
    Uses the CDF of the semicircle distribution to unfold a single spectrum of eigenvalues. See unfold.
    """
    # Uses the semicircle formula to unfold
    N = len(l)
    if radius == None:
        s = 2*_np.sqrt(N)
    else:
        s = radius

    # Define the piecewise function to avoid errors on values outside the support
    conds = [
        l <= -s,
        (l > -s) & (l < s),
        l >= s
    ]
    funcs = [
        0.0,                                   
        lambda x:  N / 2 + N / _np.pi * _np.arctan(x / _np.sqrt(s**2-x**2)) + x * _np.sqrt(s**2-x**2) / 4 / _np.pi,
        N
    ]
    
    return _np.piecewise(l, conds, funcs)