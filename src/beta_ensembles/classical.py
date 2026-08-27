"""
This module contains the functions that generate random matrices or spectra from the classical ensembles, Gaussian, Circular, and Laguerre. Only the orthogonal and unitary ensembles of each type are implemented at present.
"""

###########################################
# Imports
###########################################

import numpy as _np
import scipy as _scipy
from tqdm import tqdm as _tqdm

###########################################
# Single random matrix implementations
###########################################

# In this section is the code for the functions generating a single matrix from each ensemble. In other words, this is the only actual math part in this file.

def _goe_matrix(N, rng):
    """Generate a single random matrix from the GOE. See goe_matrices."""
    
    X = rng.normal(0,_np.sqrt(2),size=(N,N))
    return (X + X.T)/2

def _gue_matrix(N, rng):
    """Generate a single random matrix from the GUE. See gue_matrices."""
    X = rng.normal(0,2,size=(N,N))
    Y = rng.normal(0,2,size=(N,N))

    M = (X + 1j * Y)/2
    return (M + _np.conj(M.T))/2

def _coe_matrix(N, rng):   
    """Generate a single random matrix from the COE. See coe_matrices."""
    
    W = _cue_matrix(N,rng)

    return _np.matmul(W,W.T)

def _cue_matrix(N, rng):
    """Generate a single random matrix from the CUE. See cue_matrices."""
    
    Z = rng.normal(0,1,size=(N,N)) + 1j * rng.normal(0,1,size=(N,N))

    Q, R = _scipy.linalg.qr(Z)

    diag_R = _np.diagonal(R)
    ph = diag_R / _np.abs(diag_R)

    return Q * ph

def _loe_matrix(M, N, rng):
    """Generate a single random matrix from the LOE. See loe_matrices."""
    X = rng.normal(0,1,size=(N,M))

    return _np.matmul(X,X.T)

def _lue_matrix(M, N, rng):
    """Generate a single random matrix from the LUE. See lUe_matrices."""
    X = rng.normal(0,1,size=(N,M)) + 1j * rng.normal(0,1,size=(N,M))

    return _np.matmul(X,X.conj().T) / 2


###########################################
# Generate matrices
###########################################

def goe_matrices(N, n_matrices=1, rng=None):
    """
    Generate random matrices from the Gaussian Orthogonal Ensemble (GOE).

    Each matrix is a real symmetric matrix with Gaussian-distributed entries. Normalization is consistent with the convention chosen for gbe_matrices.

    For more efficient generation of GOE spectra, use gaussian.spectra(beta=1, ...) instead.

    Parameters
    ----------
    N : int
        Matrix dimension.
    n_matrices : int, optional
        Number of independent matrices to generate.
    rng : numpy.random.Generator or None, optional
        Random number generator. If None, a new generator is created. Passing a seeded generator allows reproducible results.

    Returns
    -------
    ndarray
        Array of shape (n_matrices, N, N) containing the generated
        real symmetric matrices.
    """
    if rng is None:
        rng = _np.random.default_rng()

    return _np.array([_goe_matrix(N,rng) for _ in range(n_matrices)])

def gue_matrices(N, n_matrices=1, rng=None):
    """
    Generate random matrices from the Gaussian Unitary Ensemble (GUE).

    Each matrix is a complex Hermitian matrix with Gaussian-distributed entries. Normalization is consistent with the convention chosen for gbe_matrices.

    For more efficient generation of GUE spectra, use gaussian.spectra(beta=2, ...) instead.

    Parameters
    ----------
    N : int
        Matrix dimension.
    n_matrices : int, optional
        Number of independent matrices to generate.
    rng : numpy.random.Generator or None, optional
        Random number generator. If None, a new generator is created. Passing a seeded generator allows reproducible results.

    Returns
    -------
    ndarray
        Array of shape (n_matrices, N, N) containing the generated complex Hermitian matrices.
    """
    if rng is None:
        rng = _np.random.default_rng()

    return _np.array([_gue_matrix(N,rng) for _ in range(n_matrices)])

def coe_matrices(N, n_matrices=1, rng=None):
    """
    Generate random matrices from the Circular Orthogonal Ensemble (COE).

    Each matrix is a symmetric unitary matrix, obtained by generating a random matrix from the CUE and symmetrizing by multiplying it by its transpose.

    For more efficient generation of COE spectra, use circular.spectra(beta=1, ...) instead.

    Parameters
    ----------
    N : int
        Matrix dimension.
    n_matrices : int, optional
        Number of independent matrices to generate.
    rng : numpy.random.Generator or None, optional
        Random number generator. If None, a new generator is created. Passing a seeded generator allows reproducible results.

    Returns
    -------
    ndarray
        Array of shape (n_matrices, N, N) containing the generated unitary matrices.
    """
    if rng is None:
        rng = _np.random.default_rng()
        
    return _np.array([_coe_matrix(N,rng) for _ in range(n_matrices)])


def cue_matrices(N, n_matrices=1, rng=None):
    """
    Generate random matrices from the Circular Unitary Ensemble (CUE).

    Each matrix is a unitary matrix, generated from the QR decomposition of a complex matrix with Gaussian-distributed entries.

    For more efficient generation of CUE spectra, use circular.spectra(beta=2, ...) instead.

    Parameters
    ----------
    N : int
        Matrix dimension.
    n_matrices : int, optional
        Number of independent matrices to generate.
    rng : numpy.random.Generator or None, optional
        Random number generator. If None, a new generator is created. Passing a seeded generator allows reproducible results.

    Returns
    -------
    ndarray
        Array of shape (n_matrices, N, N) containing the generated unitary matrices.
    """
    if rng is None:
        rng = _np.random.default_rng()

    return _np.array([_cue_matrix(N,rng) for _ in range(n_matrices)])

def loe_matrices(M, N, n_matrices=1, rng=None):
    """
    Generate random matrices from the Laguerre Orthogonal (Wishart) ensemble.

    Each matrix is constructed as W = X X.T, where X is an (N, M) matrix with independent standard normal entries.

    For more efficient generation of LOE spectra, use laguerre.spectra(beta=1, ...) instead.

    Parameters
    ----------
    M : int
        Number of columns of the underlying Gaussian matrix.
    N : int
        Number of rows of the underlying Gaussian matrix, and dimension of the resulting symmetric matrices.
    n_matrices : int, optional
        Number of independent matrices to generate. Default is 1.
    rng : numpy.random.Generator or None, optional
        Random number generator. If None, a new generator is created. Passing a seeded generator allows reproducible results.

    Returns
    -------
    ndarray
        Array of shape (n_matrices, N, N) containing the generated real symmetric matrices.
    """
    if rng is None:
        rng = _np.random.default_rng()

    return _np.array([_loe_matrix(M,N,rng) for _ in range(n_matrices)])

def lue_matrices(M, N, n_matrices=1, rng=None):
    """
    Generate random matrices from the Laguerre Unitary (Wishart) ensemble.

    Each matrix is constructed as W = X X†, where X is an (N, M) matrix with independent standard complex normal entries.

    For more efficient generation of LUE spectra, use laguerre.spectra(beta=2, ...) instead.

    Parameters
    ----------
    M : int
        Number of columns of the underlying Gaussian matrix.
    N : int
        Number of rows of the underlying Gaussian matrix, and dimension of the resulting Hermitian matrices.
    n_matrices : int, optional
        Number of independent matrices to generate. Default is 1.
    rng : numpy.random.Generator or None, optional
        Random number generator. If None, a new generator is created. Passing a seeded generator allows reproducible results.

    Returns
    -------
    ndarray
        Array of shape (n_matrices, N, N) containing the generated complex Hermitian matrices.
    """
    if rng is None:
        rng = _np.random.default_rng()

    return _np.array([_lue_matrix(M,N,rng) for _ in range(n_matrices)])


###########################################
# Spectra generators
###########################################

def _generate_spectra(generator, n_spectra, verbose):
    """
    Helper function for the functions goe_spectra, gue_spectra, coe_spectra, cue_spectra, loe_spectra, lue_spectra. See documentation of those functions for behavior.

    generator() should be a function that returns a single spectrum with the desired parameters.
    """
    iterator = range(n_spectra)
    if verbose:
        iterator = _tqdm(iterator, desc="Generating spectra")
    
    results = []
    try:
        for _ in iterator:
            results.append(generator())

    except KeyboardInterrupt:
        if verbose and hasattr(iterator, "close"):
            iterator.close()
        print(f"\nProcess interrupted. Returning {len(results)} generated spectra.")

    return _np.array(results)

def goe_spectra(N, n_spectra=1, rng=None, verbose=False):
    """
    Generate spectra of the Gaussian orthogonal ensemble (GOE).

    Generates n_spectra independent GOE spectra of size N using dense random matrices. 
    
    This function is included only for completeness. Use gaussian.spectra(beta=1,...) for best performance.

    Parameters
    ----------
    N : int
        Number of levels in each spectrum.
    n_spectra : int, optional
        Number of spectra to generate.
    rng : numpy.random.Generator or None, optional
        Random number generator. If None, a new generator is created. Passing a seeded generator allows reproducible results.
    verbose : bool, default=True
        If True, display progress information during the computation.

    Returns
    -------
    ndarray
        Array of shape (n_spectra, N) containing the generated spectra.
    """
    if rng is None:
        rng = _np.random.default_rng()

    generator = lambda: _goe_spectrum(N, rng)

    return _generate_spectra(generator, n_spectra, verbose)

def gue_spectra(N, n_spectra=1, rng=None, verbose=False):
    """
    Generate spectra of the Gaussian unitary ensemble (GUE).

    Generates n_spectra independent GUE spectra of size N using dense random matrices.
    
    This function is included only for completeness. Use gaussian.spectra(beta=2,...) for best performance.

    Parameters
    ----------
    N : int
        Number of levels in each spectrum.
    n_spectra : int, optional
        Number of spectra to generate.
    rng : numpy.random.Generator or None, optional
        Random number generator. If None, a new generator is created. Passing a seeded generator allows reproducible results.
    verbose : bool, default=True
        If True, display progress information during the computation.

    Returns
    -------
    ndarray
        Array of shape (n_spectra, N) containing the generated spectra.
    """
    if rng is None:
        rng = _np.random.default_rng()

    generator = lambda: _gue_spectrum(N, rng)

    return _generate_spectra(generator, n_spectra, verbose)

def coe_spectra(N, n_spectra=1, return_phases=True, normalize=True, rng=None, verbose=False):
    """
    Generate spectra of the Circular orthogonal ensemble (COE).

    Generates n_spectra independent COE spectra of size N using dense random matrices.
    
    This function is included only for completeness. Use circular.spectra(beta=1,...) for best performance.

    The function returns by default the eigenphases, shifted and rescaled to be in the range [0,N] with average spacing one. For typical uses, this means that no further unfolding is required. The optional parameters return_phases and normalize trigger other options. See below.

    Parameters
    ----------
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
    verbose : bool, default=True
        If True, display progress information during the computation.
        
    Returns
    -------
    ndarray
        Array of shape (n_spectra, N) containing the generated spectra.
    """
    if rng is None:
        rng = _np.random.default_rng()

    generator = lambda: _coe_spectrum(N,return_phases,normalize,rng)

    return _generate_spectra(generator, n_spectra, verbose)

def cue_spectra(N, n_spectra=1, return_phases=True, normalize=True, rng=None, verbose=False):
    """
    Generate spectra of the Circular unitary ensemble (CUE).

    Generates n_spectra independent COE spectra of size N using dense random matrices.
    
    This function is included only for completeness. Use circular.spectra(beta=2,...) for best performance.

    The function returns by default the eigenphases, shifted and rescaled to be in the range [0,N] with average spacing one. For typical uses, this means that no further unfolding is required. The optional parameters return_phases and normalize trigger other options. See below.

    Parameters
    ----------
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
    verbose : bool, default=True
        If True, display progress information during the computation.
        
    Returns
    -------
    ndarray
        Array of shape (n_spectra, N) containing the generated spectra.
    """
    if rng is None:
        rng = _np.random.default_rng()

    generator = lambda: _cue_spectrum(N,return_phases,normalize,rng)

    return _generate_spectra(generator, n_spectra, verbose)

def loe_spectra(M, N, n_spectra=1, rng=None, verbose=False):
    """
    Generate spectra of the Laguerre Orthogonal (Wishart) ensemble (LOE).

    Generates n_spectra independent LOE spectra of size N using dense random matrices. Each matrix is constructed as W = X X.T, where X is an (N, M) matrix with independent standard normal entries.
    
    This function is included only for completeness. Use laguerre.spectra(beta=1,...) for best performance.

    Note: laguerre.spectra takes as a parameter the continuous parameter alpha instead of M. The two can be related using the helper functions laguerre.alpha(beta,M,N) and laguerre.M(beta,alpha,N).
    
    Parameters
    ----------
    M : int
        Number of columns of the underlying Gaussian matrix.
    N : int
        Number of rows of the underlying Gaussian matrix, and dimension of the resulting Hermitian matrices.
    n_spectra : int, optional
        Number of spectra to generate.
    rng : numpy.random.Generator or None, optional
        Random number generator. If None, a new generator is created. Passing a seeded generator allows reproducible results.
    verbose : bool, default=True
        If True, display progress information during the computation.
        
    Returns
    -------
    ndarray
        Array of shape (n_spectra, N) containing the generated spectra.
    """
    if rng is None:
        rng = _np.random.default_rng()

    generator = lambda: _loe_spectrum(M,N,rng)

    return _generate_spectra(generator, n_spectra, verbose)

def lue_spectra(M, N, n_spectra=1, rng=None, verbose=False):
    """
    Generate spectra of the Laguerre Unitary (Wishart) ensemble (LUE).

    Generates n_spectra independent LUE spectra of size N using dense random matrices. Each matrix is constructed as W = X X†, where X is an (N, M) matrix with independent standard complex normal entries.
    
    This function is included only for completeness. Use laguerre.spectra(beta=1,...) for best performance.

    Note: laguerre.spectra takes as a parameter the continuous parameter alpha instead of M. The two can be related using the helper functions laguerre.alpha(beta,M,N) and laguerre.M(beta,alpha,N).
    
    Parameters
    ----------
    M : int
        Number of columns of the underlying Gaussian matrix.
    N : int
        Number of rows of the underlying Gaussian matrix, and dimension of the resulting Hermitian matrices.
    n_spectra : int, optional
        Number of spectra to generate.
    rng : numpy.random.Generator or None, optional
        Random number generator. If None, a new generator is created. Passing a seeded generator allows reproducible results.
    verbose : bool, default=True
        If True, display progress information during the computation.
        
    Returns
    -------
    ndarray
        Array of shape (n_spectra, N) containing the generated spectra.
    """
    if rng is None:
        rng = _np.random.default_rng()

    generator = lambda: _lue_spectrum(M,N,rng)

    return _generate_spectra(generator, n_spectra, verbose)

def poisson_spectra(N, n_spectra=1, rng=None, verbose=False):
    """
    Generate Poisson spectra of size N.

    Each spectrum consists of N independent and identically distributed levels sampled from a uniform distribution on the interval [0, N), giving an average level spacing of one.

    Parameters
    ----------
    N : int
        Number of levels in each spectrum.
    n_spectra : int, optional
        Number of spectra to generate.
    rng : numpy.random.Generator or None, optional
        Random number generator. If None, a new generator is created.
        Passing a seeded generator allows reproducible results.

    Returns
    -------
    ndarray
        Array of shape (n_spectra, N) containing the generated spectra.
    """
    if rng is None:
        rng = _np.random.default_rng()

    generator = lambda: _poisson_spectrum(N,rng)
        
    return _generate_spectra(generator, n_spectra, verbose)

###########################################
# Helper functions: single spectrum generators
###########################################

def _goe_spectrum(N, rng):
    """
    Generates a single GOE spectrum. See goe_spectra.
    """
    M = _goe_matrix(N,rng)

    return _np.sort(_scipy.linalg.eigvalsh(M))


def _gue_spectrum(N, rng):
    """
    Generates a single GUE spectrum. See goe_spectra.
    """
    M = _gue_matrix(N,rng)

    return _np.sort(_scipy.linalg.eigvalsh(M))

    
def _coe_spectrum(N, return_phases=True, normalize=True, rng=None):
    """ 
    Generates a single COE spectrum. See coe_spectra.
    """
    M = _coe_matrix(N,rng)
    
    lambdas = _scipy.linalg.eigvals(M)

    if return_phases:
        phi = _np.sort(_np.real(_np.angle(lambdas)))
        if normalize:
            phi = N / 2 / _np.pi * phi + N/2
        return phi
    else:
        return lambdas
    
def _cue_spectrum(N, return_phases=True, normalize=True, rng=None):
    """ 
    Generates a single CUE spectrum. See cue_spectra.
    """
    M = _cue_matrix(N,rng)
    
    lambdas = _scipy.linalg.eigvals(M)

    if return_phases:
        phi = _np.sort(_np.real(_np.angle(lambdas)))
        if normalize:
            phi = N / 2 / _np.pi * phi + N/2
        return phi
    else:
        return lambdas

def _loe_spectrum(M, N, rng):
    """
    Generates a single LOE spectrum. See loe_spectra.
    """
    M = _loe_matrix(M,N,rng)

    return _np.sort(_scipy.linalg.eigvalsh(M))

def _lue_spectrum(M, N, rng):
    """
    Generates a single LUE spectrum. See lue_spectra.
    """
    M = _lue_matrix(M, N, rng)

    return _np.sort(_scipy.linalg.eigvalsh(M))


def _poisson_spectrum(N, rng):
    """
    Generate a single Poisson spectrum of size N. See poisson_spectra.
    """
    return _np.sort(N * rng.random(size=N))