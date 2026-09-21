"""Transverse Gaus-LC meson wavefunction and photon-meson overlap.

Public inputs use r_fm in fm and a scalar Q_GeV in GeV. Internally,
s = r_fm / HBARC_GEV_FM is the dipole separation in GeV^-1.

For R dipole sizes and Z momentum fractions, the unintegrated functions
return (R, Z) arrays: phi_T is dimensionless, dphi_T/ds is in GeV, and the
transverse overlap is in GeV^2. The z-integrated overlap returns (R, K)
for K momentum-transfer magnitudes, also in GeV^2.
"""

import numpy as np
from scipy.special import k0, k1, j0

if __package__:
    from .parameters import (
        E_EM, HBARC_GEV_FM, GausLCParameters,
        RHO_GAUS_LC_PARAM, JPSI_GAUS_LC_PARAM,
    )
else:
    from parameters import (
        E_EM, HBARC_GEV_FM, GausLCParameters,
        RHO_GAUS_LC_PARAM, JPSI_GAUS_LC_PARAM,
    )

# Parameter names remain importable here for existing notebooks. Their
# definitions and literature presets now live in parameters.py.


N_C = 3


def _validate_coordinates(z, r_fm, *, positive_r=False):
    """Return one-dimensional input arrays; scalars become length-one arrays."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    r_fm = np.atleast_1d(np.asarray(r_fm, dtype=float))

    if z.ndim != 1 or z.size == 0:
        raise ValueError("z must be a scalar or a nonempty 1D array")
    if r_fm.ndim != 1 or r_fm.size == 0:
        raise ValueError("r_fm must be a scalar or a nonempty 1D array")
    if np.any(~np.isfinite(z)) or np.any((z <= 0) | (z >= 1)):
        raise ValueError("z must be finite and strictly between 0 and 1")
    if np.any(~np.isfinite(r_fm)) or np.any(r_fm < 0):
        raise ValueError("r_fm must be finite and nonnegative")
    if positive_r and np.any(r_fm == 0):
        raise ValueError(
            "The overlap requires r_fm > 0; use interior integration nodes"
        )

    return z, r_fm


def _photon_epsilon_GeV(z, Q_GeV, m_f_GeV):
    """Return epsilon(z) = sqrt(m_f^2 + z*(1-z)*Q^2), shape (Z,)."""
    Q_GeV = np.asarray(Q_GeV, dtype=float)
    if Q_GeV.ndim != 0 or not np.isfinite(Q_GeV) or Q_GeV < 0:
        raise ValueError("Q_GeV must be a finite nonnegative scalar in GeV")

    virtuality_scale_GeV = np.sqrt(z * (1 - z)) * Q_GeV
    # hypot evaluates the square root without overflowing when squaring Q.
    return np.hypot(m_f_GeV, virtuality_scale_GeV)


def gaus_lc_phi_T(parameter: GausLCParameters, z, r_fm):
    """Return dimensionless phi_T(s, z), shape (R, Z).

    phi_T = N_T * [z*(1-z)]^2 * exp[-s^2 / (2*R_T^2)].
    Inputs z and r_fm may be scalars or one-dimensional arrays.
    """
    z, r_fm = _validate_coordinates(z, r_fm)
    z_grid = z[None, :]                          # (1, Z)
    s_GeV_inv = r_fm[:, None] / HBARC_GEV_FM     # (R, 1)

    momentum_profile = (z_grid * (1 - z_grid))**2
    radial_profile = np.exp(
        -s_GeV_inv**2 / (2 * parameter.R_T_Squared_GeV_m2)
    )
    return parameter.N_T * momentum_profile * radial_profile


def gaus_lc_dphi_T_ds(parameter: GausLCParameters, z, r_fm):
    """Return dphi_T/ds in GeV, shape (R, Z), with s = r_fm / hbarc.

    dphi_T/ds = -s/R_T^2 * phi_T. The derivative is with respect to the
    separation in GeV^-1; a derivative with respect to r_fm differs by hbarc.
    """
    z, r_fm = _validate_coordinates(z, r_fm)
    s_GeV_inv = r_fm[:, None] / HBARC_GEV_FM
    phi_T = gaus_lc_phi_T(parameter, z, r_fm)

    derivative_scale_GeV = -s_GeV_inv / parameter.R_T_Squared_GeV_m2
    return derivative_scale_GeV * phi_T


def gaus_lc_overlap_T(
    parameter: GausLCParameters, z, r_fm, *, Q_GeV: float = 0.0
):
    """Return the transverse photon-meson overlap in GeV^2, shape (R, Z).

    Q_GeV is the nonnegative square root of photon virtuality Q^2 in GeV.
    Zero gives real photons. Use r_fm > 0 and 0 < z < 1.
    The result excludes integration measures, phases, and the dipole response.
    """
    z, r_fm = _validate_coordinates(z, r_fm, positive_r=True)
    z_grid = z[None, :]
    s_GeV_inv = r_fm[:, None] / HBARC_GEV_FM

    phi_T = gaus_lc_phi_T(parameter, z, r_fm)
    dphi_T_ds = gaus_lc_dphi_T_ds(parameter, z, r_fm)

    epsilon_GeV = _photon_epsilon_GeV(z, Q_GeV, parameter.m_f_GeV)[None, :]
    bessel_argument = epsilon_GeV * s_GeV_inv
    bessel_k0 = k0(bessel_argument)
    bessel_k1 = k1(bessel_argument)

    spin_weight = z_grid**2 + (1 - z_grid)**2
    charge_prefactor = (
        parameter.eff_charge * E_EM * N_C / (np.pi * z_grid * (1 - z_grid))
    )

    # m_f^2 stays in the mass term; the derivative term contains epsilon.
    mass_term = parameter.m_f_GeV**2 * bessel_k0 * phi_T
    derivative_term = spin_weight * epsilon_GeV * bessel_k1 * dphi_T_ds
    return charge_prefactor * (mass_term - derivative_term)

def gaus_lc_overlap_integral_z(
    parameter: GausLCParameters, r_fm, q_GeV, Q_GeV, nz
):
    """Estimate H_T(r, q; Q) = int dz overlap_T * J0 in GeV^2, shape (R, K).

    r_fm contains positive dipole sizes in fm; q_GeV contains nonnegative
    momentum-transfer magnitudes |Delta| in GeV, in the requested order.
    Both may be scalars or one-dimensional arrays. Q_GeV is the scalar
    square root of photon virtuality Q^2 in GeV. nz is a positive integer.

    J0 contains the dipole-angle average. The angular factor 2*pi, the
    paper's 1/(4*pi), and the radial measure belong to the later amplitude.
    Gauss-Legendre nodes and weights cover the full z interval [0, 1]
    while keeping overlap evaluations strictly inside the endpoints.
    """
    if np.iscomplexobj(r_fm) or np.iscomplexobj(q_GeV):
        raise ValueError("r_fm and q_GeV must contain real coordinates")
    r_fm = np.atleast_1d(np.asarray(r_fm, dtype=float))
    q_GeV = np.atleast_1d(np.asarray(q_GeV, dtype=float))
    if q_GeV.ndim != 1 or q_GeV.size == 0:
        raise ValueError("q_GeV must be a scalar or a nonempty 1D array")
    if np.any(~np.isfinite(q_GeV)) or np.any(q_GeV < 0):
        raise ValueError("q_GeV must be finite and nonnegative")
    if isinstance(nz, (bool, np.bool_)) or not isinstance(nz, (int, np.integer)) or nz < 1:
        raise ValueError("nz must be a positive integer")

    x, w = np.polynomial.legendre.leggauss(nz)
    z = 0.5 * (x + 1) # map [-1,1] to [0,1]
    w = 0.5 * w # scale weights for [0,1]
    overlap = gaus_lc_overlap_T(parameter, z, r_fm, Q_GeV=Q_GeV) # (R, nz)
    s_GeV_inv = r_fm / HBARC_GEV_FM
    bessel_argument = (
        s_GeV_inv[:, None, None]
        * q_GeV[None, :, None]
        * (0.5 - z[None, None, :])
    )  # (R, K, nz)
    bessel_j0 = j0(bessel_argument) # (R, K, nz)
    integrand = overlap[:, None, :] * bessel_j0 # (R, K, nz)
    integral = np.sum(integrand * w[None, None, :], axis=-1) # (R, K)
    return integral # (R, K)
