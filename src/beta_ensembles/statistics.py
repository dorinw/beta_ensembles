"""
This file contains the functions to define the spacings and spacing ratios, and formulas to compare their distributions to RMT predictions.
"""

import numpy as _np
import scipy as _scipy

###########################################
# Define spacings and ratios
###########################################

def spacings(spectra):
    """
    Compute the nearest-neighbor spacings of one or more spectra.

    Parameters
    ----------
    spectra : ndarray
        Array of ordered eigenvalues or eigenphases. The last axis is interpreted as the spectral dimension. The typical shape would be either (N,) or (n_spectra,N).

    Returns
    -------
    ndarray
        The nearest-neighbor spacings in array of shape (N-1,) or (...,N-1), depending on shape of input.
    """
    return k_spacings(spectra,k=1)

def k_spacings(spectra, k=1):
    """
    Compute the k-spacings of one or more spectra. The k-spacings are defined as
        s_k[i] = z[i+k] - z[i]

    Parameters
    ----------
    spectra : ndarray
        Array of ordered eigenvalues or eigenphases. The last axis is interpreted as the spectral dimension. The typical shape would be either (N,) or (n_spectra,N).
    k : int, default=1
        Spacing order.

    Returns
    -------
    ndarray
        The k-spacings in array of shape (N-k,) or (...,N-k), depending on shape of input.
    """
    return spectra[...,k:] - spectra[...,:-k]

def ratios(spectra, reduced=True):
    """
    Compute the nearest-neighbor spacing ratios of one or more spectra.

    If reduced=True (default behavior), each ratio r is replaced by r_tilde = min(r, 1/r), such that all values lie in the interval [0, 1].

    Parameters
    ----------
    spectra : ndarray
        Array of ordered eigenvalues or eigenphases. The last axis is interpreted as the spectral dimension. The typical shape would be either (N,) or (n_spectra,N).
    reduced : bool, optional
        Whether to return reduced ratios. If None, the package default set by set_reduced() is used.

    Returns
    -------
    ndarray
        The nearest-neighbor spacings in array of shape (N-2,) or (...,N-2), depending on shape of input.
    """
    return k_ratios(spectra,k=1,reduced=reduced)

def k_ratios(spectra, k=1, reduced=True):
    """
    Compute adjacent k-spacing ratios of one or more spectra.
    
    The ratios are defined as r_k[i] = (z[i+2k] - z[i+k]) / (z[i+k] - z[i]) = s_k[i+k]/s_k[i], where z[i] are the eigenvalues and s_k[i] their k-spacings.

    If reduced=True, each ratio r is replaced by min(r, 1/r), so that all values lie in the interval [0, 1].

    Parameters
    ----------
    spectra : ndarray
        Array of ordered eigenvalues or eigenphases. The last axis is interpreted as the spectral dimension. The typical shape would be either (N,) or (n_spectra,N).
    k : int, default=1
        Spacing order.
    reduced : bool, optional
        Whether to return reduced ratios. If None, the package default set by set_reduced() is used.

    Returns
    -------
    ndarray
        The nearest-neighbor spacings in array of shape (N-2*k,) or (...,N-2*k), depending on shape of input.
    """
    s = k_spacings(spectra,k)
    r = s[...,k:] / s[...,:-k]
    if reduced:
       return reduced_ratios(r)
    return r

def reduced_ratios(ratios):
    """
    Convert ratios to their reduced form, r_tilde = min(r, 1/r).
    
    Ratios equal to zero are mapped to zero.
    
    Returns
    -------
    ndarray
        Array of the same shape as the input.
    """
    # It's possible that some ratios are 0, in that case we want r_tilde(0) = 0, and without raising a division by zero warning.
    r = _np.asarray(ratios)
    with _np.errstate(divide='ignore'):
        r_tilde = _np.minimum(r, 1/r)
    r_tilde[r == 0] = 0
    return r_tilde

###########################################
# Distribution functions
###########################################

def pdf_s(beta, s):
    """
    Returns the Wigner-Dyson distribution for Dyson index beta, being the distribution of nearest-neighbor spacings in the beta ensembles.

    Parameters
    ----------
    beta : float
        Dyson index of the beta ensemble. Must be positive.
    s : float or ndarray
        Spacing values at which to evaluate the probability density.

    Returns
    -------
    ndarray
        Probability density evaluated at the supplied spacing values.
    """
    
    # Computing the logpdf and then exponentiating is a strategy we need for large beta.

    s = _np.asarray(s)
    lg1 = _scipy.special.loggamma((beta+1)/2)
    lg2 = _scipy.special.loggamma((beta+2)/2)

    lnb = _np.log(2) + (beta+1)*lg2 - (beta+2)*lg1
    lcb = 2*(lg2-lg1)
        
    
    cb = _np.exp(lcb)

    # This will overflow at large beta
    # nb = _np.exp(lnb)
    # pdf = nb * (s**beta) * _np.exp(-cb * s**2)

    # Ignore divide by zero warning in log(s) for s = 0
    with _np.errstate(divide='ignore'):
        logpdf = lnb + beta * _np.log(s) - cb * s**2
        
    pdf = _np.exp(logpdf)
    
    # Set pdf(0) = 0
    pdf[s==0] = 0
    
    return pdf

def cdf_s(beta, s):
    """
    Returns the cumulative distribution function of the Wigner-Dyson distribution for Dyson index beta.

    Parameters
    ----------
    beta : float
        Dyson index of the beta ensemble. Must be positive.
    s : float or ndarray
        Spacing values at which to evaluate the CDF.

    Returns
    -------
    ndarray
        Cumulative density function evaluated at the supplied spacing values.
    """
    # beta as array is for internal use, where we need to evaluate the CDF at various values of beta but a single value of s. This happens in fit_spacings. Note that beta and s can't be arrays simultaneously in this implementation, or we get broadcasting errors.
    beta = _np.asarray(beta)

    lg1 = _scipy.special.loggamma((beta+1)/2)
    lg2 = _scipy.special.loggamma((beta+2)/2)
    lcb = 2*(lg2-lg1)
    cb = _np.exp(lcb)

    return _scipy.special.gammainc((1+beta)/2,cb * s**2)
    

def pdf_r(beta, r, reduced=True):
    """
    Returns the probability density of consecutive spacing ratios in beta ensembles, given by the Atas-Bogomolny-Giraud-Roux surmise.
    
    Parameters
    ----------
    beta : float
        Dyson index of the beta ensemble.
    r : float or ndarray
        Spacing ratio values at which to evaluate the PDF.
    reduced : bool, optional
        Whether to return the PDF for reduced ratios. Default is True.

    Returns
    -------
    ndarray
        Probability density evaluated at the supplied ratio values.
    """
    
    r = _np.asarray(r)
    norm = 2 if reduced else 1

    # Simple implementation is fine for most use cases, but fails at large beta...
    # Cb = 3**((3+3*beta)/2) * (_scipy.special.gamma(1+beta/2)**2) / (2 * _np.pi * _scipy.special.gamma(1+beta))
    # pdf = norm * Cb * (r + r**2)**(beta) / (1 + r + r**2)**(1+3*beta/2)
    
    # The implementation with log works for large beta without issues
    lCb = 1.5 * (1+beta) * _np.log(3) + 2 * _scipy.special.loggamma(1+beta/2) - _np.log(2*_np.pi) -_scipy.special.loggamma(1+beta)

    with _np.errstate(divide='ignore'):
        lpdf = _np.log(norm) + lCb + beta*_np.log(r + r**2) - (1+3*beta/2)*_np.log(1+r+r**2)
    pdf = _np.exp(lpdf)
    pdf[r == 0] = 0

    return pdf


def cdf_r(beta, r, reduced=True):
    """
    Returns the cumulative distribution function of consecutive spacing ratios in beta ensembles.
    
    Parameters
    ----------
    beta : float
        Dyson index of the beta ensemble.
    r : float or ndarray
        Spacing ratio values at which to evaluate the CDF.
    reduced : bool, optional
        Whether to return the CDF for reduced ratios. Default is True.

    Returns
    -------
    ndarray
        Probability density evaluated at the supplied ratio values.
    """
    if reduced:
        return _cdf_r_tilde(beta,r)
        
    # Use the symmetry r->1/r for the un-reduced ratios
    else:
        func_1 = lambda r: 0.5 * _cdf_r_tilde(beta,r)
        func_2 = lambda r: 1 - 0.5 * _cdf_r_tilde(beta,1/r)
        return _np.piecewise(r, [r <= 1, r > 1],[func_1,func_2])
    
def _cdf_r_tilde(beta, r):
    """
    Return CDF of the reduced spacing ratios.
    
    """

    # The derivation of the CDF is due to this coordinate change:
    u = _np.cos(3 * _np.arctan((1 + 2 * r)/_np.sqrt(3)) - _np.pi / 2)

    lcb = (1+beta) * _np.log(2) + 2 * _scipy.special.loggamma(1+beta/2) - _np.log(_np.pi) - _scipy.special.loggamma(1+beta)

    cb = _np.exp(lcb)

    return (1 - cb * u * _scipy.special.hyp2f1(0.5-beta/2,0.5,1.5,u**2))
    
###########################################
# rv_continuous implementation of spacing ratios for fitting
###########################################

def fit_spacings(spacings, beta_init=2):
    """
    Fit the Wigner-Dyson distribution to a set of spacings.
    
    Parameters
    ----------
    spacings : array_like
        Array or collection of arrays of spacings.
    beta_init : float, optional
        Initial value of beta for the fit. Default is 2.
    
    Returns
    -------
    float
        Fitted value of beta.
    """
    spacings = _np.asarray(spacings)
    
    dist = _dist_s(beta_init, a=0.0, b = _np.inf)

    beta_fit, _, _ = dist.fit(spacings.ravel(), floc = 0, fscale = 1)

    return beta_fit

def fit_ratios(ratios,beta_init = 2):
    """
    Fit the Atas-Bogomolny-Giraud-Roux distribution to a set of spacing ratios.

    Since the distribution is symmetric under r -> 1/r, the fitting is always performed after converting to ratios to the reduced ratios r_tilde given by min(r, 1/r).

    Users can check separately whether their data is symmetric under r -> 1/r.
    
    Parameters
    ----------
    ratios : array_like
        Array or collection of arrays of spacing ratios.
    beta_init : float, optional
        Initial value of beta for the fit. Default is 2.
    
    Returns
    -------
    float
        Fitted value of beta.
    """
    
    # Since the distribution being fitted is symmetric under r <-> 1/r, we always work with reduced ratios only.
    ratios = reduced_ratios(ratios).ravel()
    
    dist = _dist_r(beta_init, a=0.0, b = 1.0)

    beta_fit, _, _ = dist.fit(ratios, floc = 0, fscale = 1)

    return beta_fit

class _dist_r(_scipy.stats.rv_continuous):
    """
    A _scipy.stats.rv_continuous implementation of the distribution of the reduced spacing ratios r-tilde. Used by fit_ratios.
    """
    def _argcheck(self, beta):
        # Return True for valid parameter values, False otherwise
        return beta > 0

    def _pdf(self, r, beta):
        return pdf_r(beta,r,reduced=True)
        
    def _cdf(self, r, beta):
        return cdf_r(beta,r,reduced=True)

class _dist_s(_scipy.stats.rv_continuous):
    """
    A _scipy.stats.rv_continuous implementation of the distribution of spacings used by fit_spacings.
    """
    def _argcheck(self, beta):
        # Return True for valid parameter values, False otherwise
        return beta > 0

    def _pdf(self, s, beta):
        return pdf_s(beta,s)

    def _cdf(self, s, beta):
        return cdf_s(beta,s)

###########################################
# Auxiliary functions
###########################################


def mean_ratio(beta, reduced=True):
    """
    Return the theoretical mean of the spacing ratio distribution for the given values of beta.
    
    Parameters
    ----------
    beta : array_like
        Dyson index of the ensemble. Can be a scalar or an array of non-negative values (see note).
    reduced : bool, optional
        If True, return the mean of the reduced ratio. Default is True.
    
    Returns
    -------
    float or ndarray
        Mean ratio for the given values of beta.

    Notes
    ------
    The value of mean_ratio(beta=0,reduced=False) is infinity (np.inf).
    """
    ndim = _np.asarray(beta).ndim

    if ndim == 0:
        beta_arr = _np.array([beta])
    else:
        beta_arr = _np.asarray(beta)

    if beta_arr.min() < 0:
        raise ValueError('Beta values must be non-negative.')

    if reduced:
        integrand = lambda r: r * pdf_r(beta_arr,r,reduced=True)
        res = _scipy.integrate.quad_vec(integrand,0,1)[0]
        
    else:
        # For the unreduced ratios, we need to take care of the value beta==0, we return _np.inf
        # This implementation assumes nothing about the shape of the array of betas
        res = _np.full(beta_arr.shape, _np.inf)
        non_zero_mask = beta_arr != 0
        beta_valid = beta_arr[non_zero_mask]
        integrand = lambda r: 0.5 * (r + 1/r) * pdf_r(beta_valid,r,reduced=True)
        res[non_zero_mask] = _scipy.integrate.quad_vec(integrand,0,1)[0]

    return res if ndim != 0 else res.item()

def mean_ratio_poisson(reduced=True):
    """
    Return the mean spacing-ratio for the Poisson distribution. 

    The values returned by this function are either 2 * log(2) - 1 ~ 0.368 for the reduced ratios (default), or infinity (np.inf) for the ordinary ratios.

    This function is purely for reference.
    """
    if reduced:
        return 2 * _np.log(2) - 1
    else:
        return _np.inf

def beta_of_mean_ratio(mean_r, beta_init=0.1, reduced=True):
    """
    Return the value of beta corresponding to a given mean ratio. This is the inverse function of mean_ratio(beta) and works for both reduced and ordinary spacing ratios.
    
    For reduced ratios, the mean lies between mean_ratio(0,reduced=True) (~0.408) and 1. For ordinary ratios, the mean is always greater than or equal to 1. If mean_r is outside the accepted range, the function returns None rather than raising an error. If mean_r = 1, the function returns inf.

    The initial value of beta is beta_init, set by default to a small value. This is to prevent the search from trying beta < 0, which will cause it to fail.
    
    Parameters
    ----------
    mean_r : float
        Mean ratio whose corresponding beta is to be determined.
    beta_init : float, optional
        Initial value of beta for the numerical inversion. Default is 0.1.
    reduced : bool, optional
        If True, invert the mean reduced ratio. If False, invert the mean ordinary ratio. Default is True.
    
    Returns
    -------
    float or None
        Value of beta corresponding to mean_r, or None if mean_r is outside the accepted range.
    """
    if reduced is None:
        reduced = _REDUCED

    if mean_r == 1:
        return _np.inf
    # Check value to avoid errors
    if reduced:
        if mean_r > 1:
            return None
        if mean_r < mean_ratio(0):
            return None
    else:
        if mean_r < 1:
            return None
    
    f = lambda beta: mean_ratio(beta,reduced) - mean_r

    return _scipy.optimize.fsolve(f,beta_init)[0]

def pdf_r_residual(beta, C_N, r):
    """
    Evaluate the finite-N residual PDF ansatz of Atas-Bogomolny-Giraud-Roux.

    The ansatz includes the N-dependent parameter C_N, which can be fitted alongside beta to account for finite-N corrections.

    Parameters
    ----------
    beta : float
        Dyson index of the ensemble.
    C_N : float
        N-dependent parameter C_N appearing in the finite-N residual PDF ansatz.
    r : array_like
        Spacing ratios at which to evaluate the PDF.

    Returns
    -------
    float or ndarray
        Value of the residual PDF at r.
    """
    # This function is not used elsewhere, but can be basis of a more complicated fitting procedure, where we should also fit the paramter C_N alongside beta
    
    # C_\pm(\beta):
    cm = _scipy.special.hyp2f1(-0.5,beta+1,beta+1.5,-1)
    cp = _scipy.special.hyp2f1(0.5,beta+1,beta+1.5,-1)
    # C_\pm(\beta+1):
    cm1 = _scipy.special.hyp2f1(-0.5,beta+2,beta+2.5,-1)
    cp1 = _scipy.special.hyp2f1(0.5,beta+2,beta+2.5,-1)

    # this parameter is chosen such that the integral of the residual vanishes
    cb = (2*beta+3) / (beta+1) * ((beta+1)*cm - (2*beta+1)*cp) / ((beta+2)*cm1 - (2*beta+3)*cp1)

    return C_N / (1+r**2) * ( (r / (1+r**2))**beta - cb * (r / (1+r**2))**(beta+1))