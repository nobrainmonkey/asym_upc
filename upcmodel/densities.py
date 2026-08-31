"""Spherical two- and three-parameter Fermi densities."""

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.special import expit

from parameters import FermiParameters


def fermi_shape(r_fm, parameters: FermiParameters):
    """Return the unnormalized dimensionless Fermi shape.
    """
    r_fm = np.asarray(r_fm, dtype=float)

    if np.any(~np.isfinite(r_fm)):
        raise ValueError("Radius must be finite")

    if np.any(r_fm < 0):
        raise ValueError("Radius cannot be negative")

    numerator = (
        1.0
        + parameters.w
        * (r_fm / parameters.R_fm) ** 2
    )

    if np.any(numerator < 0):
        raise ValueError("3pF numerator is negative")

    # Equivalent to 1 / (1 + exp((r - R) / a)), but more stable.
    fermi_edge = expit(
        (parameters.R_fm - r_fm)
        / parameters.a_fm
    )

    return numerator * fermi_edge


@lru_cache(maxsize=None)
def fermi_density_normalization(
    parameters: FermiParameters,
) -> float:
    """Calculate rho_0 in fm^-3 such that integral rho(r) d^3r = A."""

    def integrand(r_fm: float) -> float:
        shape = float(fermi_shape(r_fm, parameters))
        return 4.0 * np.pi * r_fm**2 * shape

    integral, estimated_error = quad(
        integrand,
        0.0,
        parameters.r_max_fm,
        epsabs=1e-11,
        epsrel=1e-11,
        limit=200,
    )

    if not np.isfinite(integral) or integral <= 0:
        raise ValueError("Density normalization integral is invalid")

    if not np.isfinite(estimated_error):
        raise ValueError("Density normalization error estimate is invalid")

    return float(parameters.A / integral)


def fermi_density(r_fm, parameters: FermiParameters):
    """Return the nucleon number density rho(r) in fm^-3.
    The density is normalized according to
        4 pi integral r^2 rho(r) dr = A.
    """
    rho_0 = fermi_density_normalization(parameters)
    return rho_0 * fermi_shape(r_fm, parameters)


def fermi_radial_pdf(r_fm, parameters: FermiParameters):
    """Return the normalized radial probability density in fm^-1.
    This is the probability density for sampling the radial coordinate:
        P_r(r) = 4 pi r^2 rho(r) / A,
    with integral P_r(r) dr = 1.
    """
    r_fm = np.asarray(r_fm, dtype=float)
    density = fermi_density(r_fm, parameters)

    return (
        4.0
        * np.pi
        * r_fm**2
        * density
        / parameters.A
    )

