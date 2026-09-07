"""Dipole and photon-target amplitude building blocks.

The GBW saturation scale, dipole strength, and finite-A nuclear profile
are implemented below.
The derivation and subsequent nuclear-amplitude integration plan are in
docs/gbw_amplitude_plan.md and output/pdf/gbw_amplitude_plan.pdf.

The existing event interface stays unchanged:
    nucleon_positions_fm: (A, 3)
    proton_mask: (A,) -- reserved for the later electromagnetic emitter

For the target, use all nucleons. Either use their positions with unit
weights and an explicit nucleon width, or call sample_target_hotspots once
per event at fixed x. Both source choices feed the existing
geometry.position_to_gaussian_thickness function and give T_A(b) with
shape (P,), units fm^-2, and transverse integral A.
The model setup supplies sigma_0_fm2 separately from the source width.

Calculation flow, with GBW and the nuclear profile implemented:
    N_GBW(x, r)                         (R,), dimensionless
    nuclear dipole profile D(r, b)      (R, P), dimensionless
    impact-parameter transform F(r, q)  (R, K), complex GeV^-2
    photon-meson overlap integral H     (R, K), GeV^2
    one event amplitude                (K,), complex GeV^-2
    ensemble amplitudes                (E, K), complex GeV^-2

Public lengths are in fm. Use HBARC_GEV_FM to convert dipole lengths to
GeV^-1. K counts momentum-transfer vectors; it is not photon virtuality Q^2.
Hotspot sampling and event averaging belong outside the amplitude integrals.
"""

import numpy as np

if __package__:
    from .parameters import HBARC_GEV_FM, GBWParameters, GBW_PAPER
    from .geometry import position_to_gaussian_thickness
else:
    from parameters import HBARC_GEV_FM, GBWParameters, GBW_PAPER
    from geometry import position_to_gaussian_thickness


def gbw_saturation_scale_squared(
    bj_x, parameters: GBWParameters = GBW_PAPER
):
    """Return Q_s^2 in GeV^2, preserving the scalar or array shape of x.

    Q0_GeV is stored in GeV and squared here. The three model parameters
    must be finite positive scalars. The
    algebraic input domain is 0 < x <= 1; GBW is used at small x.
    """
    bj_x = np.asarray(bj_x, dtype=float)
    if np.any(~np.isfinite(bj_x)) or np.any((bj_x <= 0) | (bj_x > 1)):
        raise ValueError("bj_x must be finite and lie in (0, 1]")

    parameter_values = np.asarray(
        [parameters.Q0_GeV, parameters.x0, parameters.lambda_gbw], dtype=float
    )
    if (
        parameter_values.shape != (3,)
        or np.any(~np.isfinite(parameter_values))
        or np.any(parameter_values <= 0)
    ):
        raise ValueError("Q0_GeV, x0, and lambda_gbw must be finite positive scalars")
    Q0_GeV, x0, lambda_gbw = parameter_values
    Q0_squared_GeV2 = Q0_GeV ** 2

    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        Q_s_squared = Q0_squared_GeV2 * (bj_x / x0) ** (-lambda_gbw)
    if np.any(~np.isfinite(Q_s_squared)) or np.any(Q_s_squared <= 0):
        raise ValueError("Parameters produce a saturation scale outside the positive finite numerical range")
    return Q_s_squared


def gbw_dipole_amplitude(
    bj_x, r_fm, parameters: GBWParameters = GBW_PAPER
):

    """
    For the first implementation bj_x is one finite scalar in (0, 1].
    r_fm is a scalar or array of finite, nonnegative transverse dipole
    separations in fm. Preserve the shape of r_fm in the output.

    Reuse gbw_saturation_scale_squared and HBARC_GEV_FM. This function takes
    no event, density, hotspot flag, or random generator. It returns neither
    the dipole cross section sigma_0 * N nor the nuclear profile 2 * N_A.

    Checks: N(x, 0) = 0; quadratic growth at small r; N -> 1 at large r;
    N = 1 - exp(-1) when r_fm = 2 * HBARC_GEV_FM / Q_s.
    """
    bj_x = np.asarray(bj_x, dtype=float)
    if bj_x.ndim != 0:
        raise ValueError("bj_x must be a scalar for one target configuration")
    r_fm = np.asarray(r_fm, dtype=float)
    if np.any(~np.isfinite(r_fm)) or np.any(r_fm < 0):
        raise ValueError("r_fm must be finite and nonnegative")

    Q_s_squared = gbw_saturation_scale_squared(
        bj_x,
        parameters=parameters
    )
    # A very large positive exponent has the well-defined saturation limit 1.
    with np.errstate(over="ignore"):
        r_GeV_inv = r_fm / HBARC_GEV_FM
        u = r_GeV_inv ** 2 * Q_s_squared / 4.0
    # Preserve the small-r color-transparency limit without cancellation.
    return -np.expm1(-u)

def nuclear_dipole_profile(
    bj_x,
    r_fm,
    b_fm,
    centers,
    gbw_parameters: GBWParameters,
    *,
    weights,
    sigma_fm,
    sigma_0_fm2: float,
    A,
):
    """Return dimensionless D(r, b) = d sigma_dip^A / d^2b, shape (R, P).

    bj_x is scalar. r_fm is a scalar or nonempty 1D vector of dipole sizes
    in fm; a scalar is treated as R=1. b_fm is a signed Cartesian grid
    (P, 2) in fm, not a vector of impact-parameter magnitudes. The existing
    thickness helper also accepts (P, 3), using its transverse coordinates.

    centers has shape (M, 3), weights has shape (M,), and sigma_fm is the
    Gaussian standard deviation. For nucleons, use their original event
    positions and unit weights. For hotspots, use a configuration sampled
    once, with its existing weights and the hotspot width. The thickness
    helper integrates the longitudinal coordinate analytically.

    A is the positive integer nucleon count from the original event, not
    the hotspot count M. The source weights must be nonnegative and sum
    to A. sigma_0_fm2 is the finite positive scalar dipole cross-section
    normalization in fm^2, independent of the source width sigma_fm.
    For the paper's prescription, the caller supplies 4*pi*B_p_fm2.

    This is Eq. (9) of https://arxiv.org/html/2312.11320v2:
        kappa(r, b) = sigma_0 * N_GBW(x, r) * T_A(b) / (2*A)
        D(r, b) = 2 * [1 - (1 - kappa(r, b))**A]
    The intended absorptive branch requires 0 <= kappa <= 1. Violations
    raise rather than silently clipping or changing the event ensemble.
    Keep both axes here; overlap and amplitude integrations come later.
    """
    r_fm = np.atleast_1d(np.asarray(r_fm, dtype=float))
    if r_fm.ndim != 1 or r_fm.size == 0:
        raise ValueError("r_fm must be a scalar or a nonempty 1D array")
    # Reuse the existing scalar-x, radius, and GBW parameter checks.
    N_r = gbw_dipole_amplitude(bj_x, r_fm, parameters=gbw_parameters)
    if isinstance(A, (bool, np.bool_)) or not isinstance(A, (int, np.integer)) or A <= 0:
        raise ValueError("A must be a positive integer nucleon count")
    A = int(A)

    weights = np.asarray(weights, dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Source weights must be finite and nonnegative")
    if not np.isclose(weights.sum(), A, rtol=1e-12, atol=1e-12):
        raise ValueError("Source weights must sum to the nucleon count A")
    # This validates the grid, centers, weight shape, and source width.
    # It is evaluated once for all dipole radii in this frozen event.
    thickness = position_to_gaussian_thickness(b_fm, centers, sigma_fm, weights)

    sigma_0_fm2 = np.asarray(sigma_0_fm2, dtype=float)
    if sigma_0_fm2.ndim != 0 or not np.isfinite(sigma_0_fm2) or sigma_0_fm2 <= 0:
        raise ValueError("sigma_0_fm2 must be a finite positive scalar in fm^2")

    # (R, 1) * (1, P) gives every dipole-size/impact-point pair.
    with np.errstate(over="ignore"):
        kappa = (sigma_0_fm2 / (2 * A)) * N_r[:, None] * thickness[None, :]
    if np.any(~np.isfinite(kappa)) or np.any(kappa > 1):
        raise ValueError("Finite-A profile requires kappa <= 1; check the event thickness and sigma_0_fm2")

    # Your finite-A expression, evaluated without cancellation for small kappa.
    # At kappa=1, log1p(-1)=-inf and the expression correctly returns D=2.
    with np.errstate(divide="ignore"):
        dsigma_db = -2 * np.expm1(A * np.log1p(-kappa))
    return dsigma_db  # (R, P), dimensionless; no radius integral here.
