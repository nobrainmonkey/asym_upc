"""One-dimensional numerical Fourier transforms with explicit conventions.

Coordinates and frequencies must have reciprocal units. These generic helpers
do not convert fm to GeV^-1 or apply the nuclear-amplitude normalization.
"""

import numpy as np


def _validate_grid(grid):
    """Return an increasing, finite, real coordinate vector."""
    grid = np.asarray(grid)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("grid must be a 1D array with at least two points")
    if np.iscomplexobj(grid):
        raise ValueError("grid must contain real coordinates")
    grid = np.asarray(grid, dtype=float)
    if np.any(~np.isfinite(grid)):
        raise ValueError("grid must contain only finite values")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("grid must be strictly increasing")
    return grid


def _validate_samples(func, grid):
    """Return one sampled function and its increasing real coordinate grid."""
    func = np.asarray(func, dtype=np.complex128)
    grid = _validate_grid(grid)
    if func.ndim != 1 or func.shape != grid.shape:
        raise ValueError("func and grid must be 1D arrays with the same shape")
    if np.any(~np.isfinite(func)):
        raise ValueError("func must contain only finite values")
    return func, grid


def fft_frequency_grid(grid, angular=True):
    """Return the signed, shifted frequencies used by fast_fourier_transform_1D.

    The input is a uniform increasing coordinate grid of shape (N,). The
    output also has shape (N,), in reciprocal coordinate units: angular
    wave numbers for angular=True, or cycles per unit for angular=False.
    This needs only the coordinates, so grid-dependent inputs can be
    prepared before transforming any sampled function.
    """
    grid = _validate_grid(grid)
    if not isinstance(angular, (bool, np.bool_)):
        raise ValueError("angular must be a boolean")
    spacing = np.diff(grid)
    delta = spacing.mean()
    if not np.allclose(spacing, delta, rtol=1e-6, atol=0):
        raise ValueError("grid must be uniformly spaced")
    frequencies = np.fft.fftshift(np.fft.fftfreq(grid.size, d=delta))
    return 2 * np.pi * frequencies if angular else frequencies


def fourier_transform(func, function_grid, fourier_grid):
    """Return int func(x) exp(-i*k*x) dx / sqrt(2*pi) at requested k.

    func and function_grid have shape (N,); fourier_grid has shape (K,).
    The complex result has shape (K,). This uses numerical trapezoidal
    integration over [function_grid[0], function_grid[-1]], and supports
    nonuniform coordinate grids. It is not an analytic integration.
    """
    func, function_grid = _validate_samples(func, function_grid)
    fourier_grid = np.asarray(fourier_grid)
    if np.iscomplexobj(fourier_grid):
        raise ValueError("fourier_grid must contain real frequencies")
    fourier_grid = np.asarray(fourier_grid, dtype=float)
    if fourier_grid.ndim != 1 or fourier_grid.size == 0:
        raise ValueError("fourier_grid must be a nonempty 1D array")
    if np.any(~np.isfinite(fourier_grid)):
        raise ValueError("fourier_grid must contain only finite values")

    fourier_component = np.zeros(fourier_grid.shape, dtype=np.complex128)
    for i, k in enumerate(fourier_grid):
        integrand = func * np.exp(-1j * k * function_grid)
        fourier_component[i] = np.trapezoid(integrand, function_grid)
        fourier_component[i] /= np.sqrt(2 * np.pi)
    return fourier_component


def fast_fourier_transform_1D(func, grid, angular=True):
    """
    angular=True : X(w) = 1/sqrt(2pi) * int x(t) e^{-iwt} dt,  returns (X, w)
    angular=False: X(f) =              int x(t) e^{-2pi i f t} dt,  returns (X, f)

    func and grid have shape (N,); both returned arrays have shape (N,).
    The grid must be uniform and increasing. The FFT approximates the
    integral by a rectangle sum with period N*spacing. For a localized
    function, choose a box where its boundary values are negligible.
    For a periodic function, sample one period without repeating its endpoint.
    """
    func, grid = _validate_samples(func, grid)
    if not isinstance(angular, (bool, np.bool_)):
        raise ValueError("angular must be a boolean")
    f = fft_frequency_grid(grid, angular=False)
    delta = np.diff(grid).mean()
    # Correct for the physical location of the first sample, which need not be 0.
    F = np.fft.fftshift(np.fft.fft(func), axes=-1) * delta * np.exp(-2j*np.pi*f*grid[0])
    if angular:
        return F / np.sqrt(2*np.pi), 2*np.pi*f
    return F, f
