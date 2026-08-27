"""
This module contains the generator functions for the Circular beta-ensemble, implementing the Killip-Nenciu construction.
"""

###########################################
# Imports
###########################################

import numpy as _np
import scipy as _scipy
from tqdm import tqdm as _tqdm

###########################################
# Spectrum generation
###########################################

def spectra(beta, N, n_spectra=1, return_phases=True, normalize=True, rng=None, verbose=True):
    """
    Generate spectra from the Circular beta ensemble (CBE).

    Uses the Killip-Nenciu construction to generate n_spectra spectra of length N, corresponding to the eigenvalues of the CBE with parameter beta.

    The parameter beta is continuous and positive. The special cases beta = 1, 2, and 4 reproduce the classical circular ensembles (COE, CUE, and CSE).

    The function returns by default the eigenphases, shifted and rescaled to be in the range [0,N] with average spacing one. For typical uses, this means that no further unfolding is required. The optional parameters return_phases and normalize trigger other options. See below.
    
    Parameters
    ----------
    beta : float
        Dyson index of the ensemble. Must be greater or equal to zero (see note below).
    N : int
        Number of levels in each spectrum.
    n_spectra : int, optional
        Number of spectra to generate.
    return_phases : bool
        If True (default behavior), return the sorted eigenphases. If False, return the complex eigenvalues on the unit circle.
    normalize : bool
        This parameter affects behavior only when return_phases is set to True. If normalize is True (default behavior), normalize the eigenphases to be in the range [0,N]. If False, the eigenphases are returned in the range [-pi,pi].
    rng : numpy.random.Generator or None, optional
        Random number generator. If None, a new generator is created. Passing a seeded generator allows reproducible results.
    verbose : bool, optional
        If True, display a progress bar while generating spectra.

    Returns
    -------
    array, shape (n_spectra, N)
        Generated spectra.

    Notes
    -----
    Setting beta = 0 returns Poisson spectra as a convenience option.

    If generation is interrupted with KeyboardInterrupt, any spectra computed up to that point are returned.
    """
    if rng is None:
        rng = _np.random.default_rng()
    
    if beta == 0:
        from .classical import poisson_spectra
        return poisson_spectra(N,n_spectra,rng)

    iterator = range(n_spectra)
    if verbose:
        iterator = _tqdm(iterator, desc="Generating CBE spectra")

    try:
        results = []
        for _ in iterator:
            results.append(_spectrum(beta,N,return_phases,normalize,rng))
    except KeyboardInterrupt:
        if verbose and hasattr(iterator, "close"):
            iterator.close()
        print(
            f"\nProcess interrupted. Returning {len(results)} generated spectra."
        )
    
    return _np.array(results)

def _spectrum(beta, N, return_phases=True, normalize=True, rng=None):
    """ Returns a single CBE spectrum. See spectra for details.
    """
    
    M = _matrix(beta,N,rng)
    
    ls = _scipy.linalg.eigvals(M)

    if return_phases:
        phi = _np.sort(_np.real(_np.angle(ls)))
        if normalize:
            phi = N / 2 / _np.pi * phi + N/2
        return phi
    else:
        return ls

###########################################
# Matrix generation
###########################################

def matrices(beta, N, n_matrices=1, rng=None):
    """
    Generate random unitary matrices from the Circular beta-ensemble (CBE).

    The matrices are generated using the Killip-Nenciu model, whose eigenvalues follow the CBE distribution. The resulting matrices are five-diagonal rather than generic dense random matrices.
    
    For the classical circular ensembles, use the functions provided in the classical module.

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
        Array of shape (n_matrices, N, N) containing the generated matrices.
    """
    if rng is None:
        rng = _np.random.default_rng()

    return _np.array([_matrix(beta,N,rng) for _ in range(n_matrices)])

def _matrix(beta, N, rng):
    """
    Generate a single random unitary matrix from the CBE. See matrices().
    """
    
    # The random variables
    rho_k = [rng.beta(1,0.5*(N-k-1)*beta) for k in range(N-1)]
    rho_k.append(1)
    
    phi_k = 2*_np.pi*rng.random(N)
    
    alpha_k = _np.sqrt(rho_k)*_np.exp(1j * phi_k)

    # Debugging. At small beta, roundoff errors sometimes introduce alpha_k > 1. Verify by uncommenting the code below:
    if _np.any(_np.abs(alpha_k[:-1]) > 1):
        raise ValueError(
            "Numerical roundoff produced a Verblunsky coefficient with modulus greater than 1; try larger beta or lower N."
        )
    abs_alpha_k_sq = _np.abs(alpha_k)**2
    
    # Construct the Xi matrices
    Xi_k = [[-1]] # Note that the first element in Xi_k is \Xi_{-1} in our notation.
    for k in range(N-1):
        Xi_k.append([[_np.conj(alpha_k[k]), _np.sqrt(1-abs_alpha_k_sq[k])],[_np.sqrt(1-abs_alpha_k_sq[k]),-alpha_k[k]]])
    Xi_k.append([[_np.conj(alpha_k[N-1])]])
    
    # Combine into M_e and M_o in block diagonal matrices
    M_o = _scipy.linalg.block_diag(*Xi_k[0:N+1:2])
    M_e = _scipy.linalg.block_diag(*Xi_k[1:N+1:2])

    # And multiply them to get the random unitary matrix:
    return _np.matmul(M_e,M_o)


###########################################
# Unfolding
###########################################

def unfold(spectra):
    """
    Unfold Circular beta-ensemble spectra.

    By default spectra(...) generates spectra that were already normalized in this way. This function can be used if another option was passed to spectra(...) in the initial spectra generation.

    Therefore it assumes that the spectra are either complex eigenvalues on the unit circle, generated by spectra(...,return_phases=False), or phases in the range [-pi,pi], as returned by spectra(...,normalize=False).

    In both cases, returns the normalized eigenphases, in the range [0,N].
    
    Parameters
    ----------
    spectra : array_like, shape (N,) or (n_spectra, N)
        Eigenvalues of one or more spectra.

    Returns
    -------
    array
        Unfolded spectra with the same shape as the input. The eigenvalues are mapped to the interval [0, N].
    """
    if _np.asarray(spectra).ndim == 1:
        return _unfold(spectra)
    else:
        return _np.array([_unfold(spectrum) for spectrum in spectra])


def _unfold(l, radius=None):
    """
    Unfolds the spectrum l.
    
    Detects if l contains complex or real eigenvalues. If complex, assumes they are on the unit circle and takes the phases.

    Once it has the phases, rescales them to the interval [0,N].
    """
    l = _np.asarray(l)

    N = len(l)
    
    if isinstance(l[0],complex):
        l = _np.real(_np.angle(l))

    # We assume l is from [-pi,pi]
    return N / 2 * (l / _np.pi + 1)