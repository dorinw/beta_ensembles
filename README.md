# beta_ensembles

beta_ensembles is a Python package for numerical work with random matrix ensembles, with a particular focus on the Gaussian, Circular, and Laguerre beta ensembles. A central feature of the package is the ability to generate spectra for general positive values of the Dyson index $\\beta$, rather than restricting calculations to the classical values $\\beta=1$, 2, and 4.

For the Gaussian and Laguerre beta ensembles, the package implements the tridiagonal and bidiagonal matrix constructions introduced by Dumitriu and Edelman. These constructions reduce spectrum generation to the diagonalization of symmetric tridiagonal matrices, making it possible to efficiently generate spectra of large matrices at arbitrary $\\beta$ without constructing dense random matrices.

For the Circular beta ensemble, the package uses the Killip–Nenciu construction, which similarly provides an explicit matrix model for arbitrary positive $\\beta$. In the present implementation, however, the resulting structured matrices are diagonalized using standard dense-matrix routines, so spectrum generation is substantially less efficient than for the Gaussian and Laguerre ensembles at large $N$.

The beta-ensemble generators are designed to generate eigenvalue spectra rather than generic random matrices, and therefore return the spectra directly without constructing a full dense random matrix.

In addition to spectrum generation, the package provides tools for unfolding eigenvalues, computing spectral statistics, fitting random-matrix predictions to numerical data, and studying spectral form factors. The package is intended primarily as a research and educational tool, with an emphasis on transparent implementations of standard constructions and on making numerical experiments with random matrix theory straightforward to reproduce.

The package requires Python and standard scientific Python dependencies, including NumPy, SciPy, and Matplotlib.

## Features
The package currently implements:
* Spectrum generation for Gaussian beta ensembles (GBE) at general $\\beta$
* Spectrum generation for Circular beta ensembles (CBE) at general $\\beta$
* Spectrum generation for Laguerre beta ensembles (LBE) at general $\\beta$
* Classical Gaussian, Circular, and Laguerre ensembles as special cases
* Spectral unfolding utilities for generated spectra
* Nearest-neighbor spacings and spacing ratios
* $k$-spacings and $k$-spacing ratios
* Fits to spacing and ratio distributions
* Spectral form factor calculations
* Analytical predictions for the spectral form factor in the interpolating regime $1 \\leq \\beta \\leq 4$
* Plotting utilities for common random-matrix statistics

## Basic usage
The basic workflow is to generate many spectra from an ensemble, unfold the resulting eigenvalues, and then compute the desired spectral statistics. For example, the following code generates an array containing $2000$ spectra from the Gaussian beta ensemble with $\\beta=1.5$ and matrix size $N=100$:

```python
import beta_ensembles as be

beta = 1.5
N = 100
n_spectra = 2000

spectra = be.gaussian.spectra(beta, N, n_spectra)
```

The spectra can be unfolded and used to compute the nearest-neighbor spacings and spacing ratios:
```python
unfolded = be.gaussian.unfold(spectra)

spacings = be.spacings(unfolded)
ratios = be.ratios(unfolded)
```

The package also contains fitting routines for comparing numerical data with analytical distributions and surmises. For example, the Dyson index can be estimated from the spacing ratios and the resulting fit compared with the corresponding analytical PDF:
```python
beta_fit = be.fit_ratios(ratios)

be.plot.hist_r(ratios)
be.plot.pdf_r(beta_fit)
```

## Examples
The repository contains the following notebooks:

* `ex0_spacing_analysis_tutorial`: introduction to spacing statistics and the basic analysis workflow
* `ex1a_fitting_beta`: fitting numerical data to random-matrix predictions and estimating $\\beta$
* `ex1b_goodness_of_fit`: assessing the quality of fits using goodness-of-fit tests and estimating uncertainties in the fitted value of $\\beta$
* `ex2_k_spacing_ratios`: analysis of $k$-spacing ratios
* `ex3_spectral_form_factor`: numerical analysis of the spectral form factor

These notebooks are intended as practical examples of numerical studies that can be performed with the package. They can also serve as starting points for experimenting with other ensemble parameters, matrix sizes, and spectral observables.
