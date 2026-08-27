"""
beta_ensembles is a package for studying the continuous beta-ensembles of random matrix theory.

The main functions beta_ensembles provides are those for generating random spectra of matrices from the Gaussian, Circular, and Laguerre beta-ensembles at general values of beta > 0. These are found in the "gaussian", "circular", and "laguerre" modules. The Gaussian and Laguerre beta-ensembles use highly efficient algorithms [due to Dumitriu & Edelman] to generate spectra, using real tridiagonal matrices. Also included are functions to unfold the generated spectra using the appropriate density function for each type of ensemble.

In addition to generating spectra, with the "statistics" module we offer a modest suite of tools for their statistical analysis, mostly for examining the distributions of level spacings and spacing ratios and comparing with RMT predictions. Functions for computing the spectral form factor (SFF) are in the "sff" module.

The "plot" module has helper functions for plotting results and comparing them to RMT predictions.
"""

from . import laguerre
from . import circular
from . import gaussian
from . import classical

from . import sff
from . import statistics

from . import plot

# Selected top-level functions
from .statistics import (
    spacings,
    ratios,
    k_spacings,
    k_ratios,
    fit_spacings,
    fit_ratios,
)