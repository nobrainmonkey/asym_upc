"""Equivalent-photon fields with public distances in fm and energies in GeV.

The transverse flux amplitude E is normalized so that
    |E|^2 = dN_gamma / (d ln(omega) d^2R), in fm^-2.
Divide this flux by omega_GeV for photons per GeV per fm^2.
Fields retain two Cartesian components for coherent source/path sums.

The point-charge expression is the transverse part of STARlight Eq. (1),
with beta retained explicitly: https://arxiv.org/html/1607.03838#S1.
See docs/epa_implementation_plan.md for conventions and the remaining UPC work.
"""

import numpy as np
from scipy.special import k1

if __package__:
    from .parameters import ALPHA_EM, HBARC_GEV_FM
else:
    from parameters import ALPHA_EM, HBARC_GEV_FM


def _real_scalar(value, name):
    """Read a finite real scalar without silently discarding a complex part."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real scalar")
    value = np.asarray(value, dtype=float)
    if value.ndim != 0 or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(value)


def point_charge_epa_field(omega_GeV, gamma, observation_xy_fm, Z):
    """Return the transverse point-source flux amplitude, complex (P, 2), fm^-1.

    omega_GeV > 0 and gamma > 1 are scalar photon energy and emitter boost
    in the same frame. Z >= 0 is the scalar source charge in proton-charge
    units, placed at the origin. observation_xy_fm is a finite real (P, 2)
    array of positions relative to that origin, with P >= 1 and R > 0.

    beta = sqrt(1 - gamma^-2), xi = omega*R/(gamma*beta*hbarc), and
        E = Z*sqrt(alpha_em)/(pi*beta*R) * xi*K1(xi) * R_hat.

    The global phase is chosen to make this single-source field real and
    outward. Returning complex values supports later coherent sums; reversing
    the observation vector reverses the field. Only transverse photons are
    included. This is a point-charge control, without a finite charge radius,
    hadronic survival, or a target amplitude. The singular origin is rejected.
    """
    omega_GeV = _real_scalar(omega_GeV, "omega_GeV")
    gamma = _real_scalar(gamma, "gamma")
    Z = _real_scalar(Z, "Z")
    if omega_GeV <= 0:
        raise ValueError("omega_GeV must be positive")
    if gamma <= 1:
        raise ValueError("gamma must be greater than one")
    if Z < 0:
        raise ValueError("Z must be nonnegative")

    if np.iscomplexobj(observation_xy_fm):
        raise ValueError("observation_xy_fm must contain real coordinates")
    points = np.asarray(observation_xy_fm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("observation_xy_fm must have nonempty shape (P, 2)")
    if np.any(~np.isfinite(points)):
        raise ValueError("observation_xy_fm must contain finite coordinates")
    with np.errstate(over="ignore"):
        radius_fm = np.hypot(points[:, 0], points[:, 1])
    if np.any(~np.isfinite(radius_fm)) or np.any(radius_fm == 0):
        raise ValueError("Observation distances must be finite and nonzero")

    inverse_gamma = 1.0 / gamma
    beta = np.sqrt((1.0 - inverse_gamma) * (1.0 + inverse_gamma))
    wave_number_fm_inv = omega_GeV / gamma / beta / HBARC_GEV_FM
    with np.errstate(over="ignore", invalid="ignore"):
        xi = wave_number_fm_inv * radius_fm
    if np.any(~np.isfinite(xi)):
        raise ValueError("The inputs produce an unrepresentable Bessel argument")

    # xi*K1(xi) -> 1; below 1e-8 the correction is at floating-point precision.
    # This also avoids 0*inf if an extremely soft photon's xi underflows.
    xi_k1 = np.ones_like(xi)
    resolved = xi >= 1e-8
    xi_k1[resolved] = xi[resolved] * k1(xi[resolved])
    direction = points / radius_fm[:, None]
    with np.errstate(over="ignore", invalid="ignore"):
        magnitude = (Z * np.sqrt(ALPHA_EM) / (np.pi * beta)) * xi_k1 / radius_fm
        field = magnitude[:, None] * direction
    if np.any(~np.isfinite(field)):
        raise ValueError("The inputs produce an unrepresentable point-source field")
    return field.astype(np.complex128)


def epa_flux_per_log_energy(field):
    """Return dN_gamma/(d ln(omega) d^2R) in fm^-2 from a field in fm^-1.

    Accept finite real or complex fields of shape (..., 2). Sum their squared
    magnitudes over the two polarization components, preserving all other
    axes. For multiple coherent sources, sum their fields before this call.
    Divide the result by omega_GeV for dN_gamma/(d omega_GeV d^2R).
    """
    field = np.asarray(field, dtype=np.complex128)
    if field.ndim < 1 or field.shape[-1] != 2 or field.size == 0:
        raise ValueError("field must have nonempty shape (..., 2)")
    if np.any(~np.isfinite(field)):
        raise ValueError("field must contain finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        flux = np.sum(np.abs(field)**2, axis=-1)
    if np.any(~np.isfinite(flux)):
        raise ValueError("The field produces an unrepresentable photon flux")
    return flux
