"""Complex amplitude means and population variances over target events."""

import numpy as np


def amplitude_moments(amplitudes):
    """Return the complex mean and real population variance over event axis 0.

    amplitudes has shape (E, K), or (E, ...) with additional non-event axes.
    All axes must be nonempty. Events must use the same kinematics and grids.
    A single event must retain its event axis: (1, K), not (K,).

    Both outputs retain every non-event axis. For the current photon-target
    amplitudes, mean has units GeV^-2 and variance has units GeV^-4.
    The centered variance is mean(abs(amplitudes - mean)**2), normalized by
    E (ddof=0). This function does not apply cross-section prefactors.
    """
    amplitudes = np.asarray(amplitudes, dtype=np.complex128)
    if amplitudes.ndim < 2 or amplitudes.size == 0:
        raise ValueError("amplitudes must have nonempty shape (E, K) or (E, ...)")
    if np.any(~np.isfinite(amplitudes)):
        raise ValueError("amplitudes must contain only finite values")

    mean = np.mean(amplitudes, axis=0)
    variance = np.mean(np.abs(amplitudes - mean)**2, axis=0)
    return mean, variance

def photon_target_spectra(mean_amplitude, variance, *, R_g=1.0):
    prefactor = R_g**2 / (16 * np.pi)
    coherent = prefactor * np.abs(mean_amplitude)**2
    incoherent = prefactor * variance
    return coherent, incoherent