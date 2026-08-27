"""
This module contains the generator functions for the Laguerre beta-ensemble, implementing the Dumitriu-Edelam construction.

Also implemented is a function for unfolding the spectra using the Marchenko-Pastur distribution's CDF.
"""

###########################################
# Imports
###########################################

import numpy as _np
import scipy as _scipy
from tqdm import tqdm as _tqdm
    
###############################################
# Laguerre beta ensembles
###############################################

def spectra(beta, alpha, N, n_spectra=1, rng=None, verbose=True):
    """
    Generate spectra from the Laguerre beta ensemble (LBE).

    Uses the Dumitriu-Edelman bidiagonal matrix model to generate n_spectra independent spectra of length N from the LBE with Dyson index beta.

    The parameter beta is continuous and non-negative. The special cases beta = 1, 2, and 4 reproduce the classical Laguerre (Wishart) orthogonal, unitary, and symplectic ensembles (LOE, LUE, and LSE).

We use the convention in which the joint probability density is proportional to prod_i lambda_i**alpha * exp(-beta * lambda_i / 2) * prod_{i<j} |lambda_i - lambda_j|**beta, with lambda_i >= 0.

    The eigenvalues are normalized such that, at large N, they follow the Marchenko-Pastur distribution in the range x_m to x_p, where
    x_m = M (1 - sqrt(gamma))**2
    x_p = M (1 + sqrt(gamma))**2

    The parameters M and gamma are related to beta, alpha, and N by:
    M - N + 1 = 2 / beta * (alpha + 1)
    gamma = N / M

    Parameters
    ----------
    beta : float
        Dyson index of the ensemble. Must be positive.
    alpha : float
        Alpha parameter of the ensemble.
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
    If generation is interrupted with KeyboardInterrupt, any spectra computed up to that point are returned.
    """
    if rng is None:
        rng = _np.random.default_rng()

    iterator = range(n_spectra)
    if verbose:
        iterator = _tqdm(iterator, desc="Generating LBE spectra")

    try:
        results = []
        for _ in iterator:
            results.append(_spectrum(beta,alpha,N,rng))
    except KeyboardInterrupt:
        if verbose and hasattr(iterator, "close"):
            iterator.close()
        print(
            f"\nProcess interrupted. Returning {len(results)} generated spectra."
        )
    
    return _np.array(results)

def _spectrum(beta, alpha, N, rng):
    """ 
    Generate a single LBE spectrum. See spectra for details.
    """

    # Uses gbe_matrices to get the diagonals
    di, xi = _matrix(beta,alpha,N,return_matrix=False,rng=rng)
    
    # the tridiagonal form lets us use an optimized algo for diagonalizing
    return _np.sort(_scipy.linalg.eigvalsh_tridiagonal(di,xi))

def matrices(beta, alpha, N, n_matrices=1, rng=None):
    """
    Generate random matrices from the Laguerre beta-ensemble (GBE).

    The matrices are generated using the Dumitriu-Edelman tridiagonal model, whose eigenvalues follow the LBE distribution, with the conventions of spectra(beta,alpha,N,...).
    
    The resulting matrices are tridiagonal symmetric rather than generic dense random matrices.

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
    
    return _np.array([_matrix(beta,alpha,N,return_matrix=True,rng=rng) for _ in range(n_matrices)])

def _matrix(beta, alpha, N, return_matrix=True, rng=None):
    """
    Generate a single random LBE matrix. See matrices(...).
    
    Set return_matrix=False to return the diagonals instead of a dense matrix.
    """
     # Draw a single variable from chi-distribution
    rand_chi = lambda k: _np.sqrt(rng.chisquare(k))

    # Map from our parameter alpha to a used by Dumitriu-Edeleman
    a = alpha + 1 + beta/2 *(N-1)

    # Diagonals of the bidiagonal (B) matrix
    bi = _np.array([rand_chi(2*a - beta*i) for i in range(N)])
    # Off-diagonals of B matrix. We define ci[0] = 0 for convenience
    ci = _np.zeros(N)
    ci[1:] = [rand_chi((N-1-i)*beta) for i in range(N-1)]

    # M = B @ B.T is a tridiagonal symmetric matrix with diagonals given by:
    # (the next line is the reason we defined ci[0]=0)
    di = bi**2 + ci**2
    # need to be a bit careful with the indices here:
    xi = bi[:-1] * ci[1:]

    if return_matrix:
        M = _np.zeros((N,N))

        _np.fill_diagonal(M, di)
        _np.fill_diagonal(M[:-1, 1:], xi[1:]) # Upper off-diagonal
        _np.fill_diagonal(M[1:, :-1], xi[1:]) # Lower off-diagonal

        return M / beta
    else:
        return (di / beta ,xi / beta)


def alpha(beta, M, N):
    """
    Helper function to convert the classical Laguerre ensemble parameters M and N to the corresponding alpha parameter defining the continuous LBE.

    For beta = 1, 2, and 4, this gives the alpha parameter corresponding to the classical LOE, LUE, and LSE respectively.

    The two parameterizations are related by:
    M - N + 1 = 2 / beta * (alpha + 1)

    Parameters
    ----------
    beta : int
        Dyson index of the ensemble. The classical Laguerre interpretation applies for beta = 1, 2, and 4.
    M : int
        Number of rows of the rectangular matrix defining the classical Laguerre ensemble.
    N : int
        Number of columns of the rectangular matrix defining the classical Laguerre ensemble.

    Returns
    -------
    float
        The corresponding LBE parameter alpha.

    Notes
    -------
    The constraint alpha > -1 implies M >= N
    """
    # return beta / 2 * M - (1 + beta / 2 * (N - 1))
    return beta/2 * (M - N + 1) - 1

def unfold(spectra, beta, alpha, N):
    """
    Unfold Laguerre beta ensemble spectra using the Marchenko-Pastur (MP) density.

    This function assumes the convention used by spectra(beta,alpha,N,...). See its documentation for details.

    Parameters
    ----------
    spectra : array_like, shape (N,) or (n_spectra, N)
        Eigenvalues of one or more spectra.
    beta : float
        Dyson index of the ensemble. Must be positive.
    alpha : float
        Alpha parameter of the ensemble.
    N : int
        Number of levels in each spectrum.
        
    Returns
    -------
    array
        Unfolded spectra with the same shape as the input. The eigenvalues are mapped to the interval [0, N].

    Notes
    -----
    Eigenvalues outside the assumed spectral support are clipped: values below x_m are mapped to 0 and values above x_p are mapped to N. Users should be aware that this can introduce unwanted edge effects.
    """
    if _np.asarray(spectra).ndim == 1:
        return _unfold(spectra, beta,alpha,N)
    else:
        return _np.array([_unfold(spectrum,beta,alpha,N) for spectrum in spectra])


def _unfold(l, beta, alpha, N):
    """
    Uses the Marchenko-Pastur CDF to unfold a single spectrum l. See unfold(...) for details.
    """
    M_val = _M(beta,alpha,N)
    gamma = N / M_val
    
    x_p = M_val * (1+_np.sqrt(gamma))**2
    x_m = M_val * (1-_np.sqrt(gamma))**2

    # Define a piecewise function to avoid errors on values outside the support
    conds = [
        l <= x_m,
        (l > x_m) & (l < x_p),
        l >= x_p
    ]
    
    funcs = [
        0.0,                                   
        lambda L:  N/2 + 1/(2*_np.pi)*_np.sqrt((x_p-L)*(L-x_m)) \
    + N*(1+gamma)/(2*_np.pi*gamma)*_np.arcsin((2*L-x_p-x_m)/(x_p-x_m)) \
    - N*(1-gamma)/(2*_np.pi*gamma)*_np.arcsin(((x_p+x_m)*L-2*x_p*x_m)/((x_p-x_m)*L)),
        N
    ]
    
    return _np.piecewise(l, conds, funcs)

def _M(beta,alpha,N):
    """
    Reverse function of alpha(beta, M, N), converting alpha to M. The main reason this function exists is that the Marchenko-Pastur law is more conveniently written in terms of M, rather than alpha.
    """
    return N - 1 + 2/beta * (alpha+1)