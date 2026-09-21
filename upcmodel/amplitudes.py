"""Dipole and photon-target amplitude building blocks.

The GBW saturation scale, dipole strength, factorized proton profile,
finite-A nuclear profile, projected FFT, and transverse photon-target
amplitude are implemented below.
The derivation and nuclear-amplitude integration plan are in
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

Calculation flow:
    N_GBW(x, r)                         (R,), dimensionless
    target dipole profile D(r, b)       (R, P), dimensionless
    impact-parameter transform F(r, q)  (R, K), complex GeV^-2
    photon-meson overlap integral H     (R, K), GeV^2
    one event amplitude                (K,), complex GeV^-2
    ensemble amplitudes                (E, K), complex GeV^-2

Public lengths are in fm. Use HBARC_GEV_FM to convert dipole lengths to
GeV^-1. K counts momentum-transfer vectors; it is not photon virtuality Q^2.
Hotspot sampling and event averaging belong outside the amplitude integrals.
event_amplitude assembles the profile, FFT, and radial integral for one
already-sampled event, with an explicitly selected profile and cached overlap.
"""

import numpy as np

if __package__:
    from .parameters import HBARC_GEV_FM, GBWParameters, GBW_PAPER
    from .geometry import position_to_gaussian_thickness
    from .fourier_transform import fast_fourier_transform_1D, fft_frequency_grid
else:
    from parameters import HBARC_GEV_FM, GBWParameters, GBW_PAPER
    from geometry import position_to_gaussian_thickness
    from fourier_transform import fast_fourier_transform_1D, fft_frequency_grid


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


def proton_dipole_profile(
    bj_x,
    r_fm,
    b_fm,
    centers,
    gbw_parameters: GBWParameters = GBW_PAPER,
    *,
    weights,
    sigma_fm,
    sigma_0_fm2,
):
    """Return the factorized proton profile D = sigma_0*N_GBW*T_p, (R, P).

    This is the proton prescription in Eq. (7) of
    https://arxiv.org/html/2312.11320v2. D is dimensionless; its transverse
    integral is sigma_0*N_GBW in fm^2. Unlike the finite-A nuclear opacity,
    this phenomenological factorized profile imposes no local D <= 2 bound.

    Use one parent nucleon at the origin. Supply that center with unit weight
    for a smooth proton, or its sampled hotspot centers and weights. centers
    has shape (M, 3), weights has shape (M,) and must sum to one. The existing
    thickness helper integrates out the longitudinal coordinates. r_fm is a
    scalar or nonempty 1D vector; b_fm is the transverse (P, 2) point array.
    Lengths are in fm, sigma_fm is a Gaussian standard deviation, and
    sigma_0_fm2 is the independent dipole cross-section normalization in fm^2.

    With fixed smooth geometry the event variance is zero. Fluctuating
    hotspots provide proton shape fluctuations for the dissociative channel.
    """
    r_fm = np.atleast_1d(np.asarray(r_fm, dtype=float))
    if r_fm.ndim != 1 or r_fm.size == 0:
        raise ValueError("r_fm must be a scalar or a nonempty 1D array")
    N_r = gbw_dipole_amplitude(bj_x, r_fm, parameters=gbw_parameters)

    weights = np.asarray(weights, dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Source weights must be finite and nonnegative")
    if not np.isclose(weights.sum(), 1.0, rtol=1e-12, atol=1e-12):
        raise ValueError("Proton source weights must sum to one")
    thickness = position_to_gaussian_thickness(b_fm, centers, sigma_fm, weights)

    sigma_0_fm2 = np.asarray(sigma_0_fm2, dtype=float)
    if sigma_0_fm2.ndim != 0 or not np.isfinite(sigma_0_fm2) or sigma_0_fm2 <= 0:
        raise ValueError("sigma_0_fm2 must be a finite positive scalar in fm^2")
    return sigma_0_fm2 * N_r[:, None] * thickness[None, :]


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

    For the Cartesian projection in dipole_profile_fourier_x, construct b_fm
    in this order (the nuclear profile itself also accepts arbitrary points):
    BX, BY = np.meshgrid(bx_fm, by_fm, indexing="ij")
    b_fm = np.column_stack((BX.ravel(), BY.ravel()))

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

def momentum_transfer_grid(bx_fm):
    """Return the signed FFT momentum transfers in GeV, shape (Nx,).

    bx_fm is a uniform increasing transverse coordinate grid in fm. This
    returns the same grid as dipole_profile_fourier_x without needing a
    profile, so H_T can be prepared once before the event loop. The values
    are 2*pi*h times the FFT frequencies and include negative momenta.
    Apply the same momentum selection to this grid and the columns of F.
    """
    return HBARC_GEV_FM * fft_frequency_grid(bx_fm, angular=True)


def dipole_profile_fourier_x(dipole_profile, bx_fm, by_fm):
    """Return (F, delta_x_GeV) for momentum transfer (delta_x, 0).

    dipole_profile is dimensionless with shape (R, Nx*Ny). Its points must
    follow meshgrid(bx_fm, by_fm, indexing="ij") flattened in C order.
    bx_fm and by_fm are increasing real vectors in fm, each with at least
    two points. The FFT requires bx_fm to be uniformly spaced; by_fm may
    be nonuniform. No azimuthal symmetry of the profile is assumed.

    First integrate D(r, bx, by) over by using trapezoidal weights, then
    Fourier-transform each radius row over bx with the FFT rectangle sum:
        F(r, delta_x) = int dbx dby D(r, bx, by) exp(-i*bx*delta_x/h) / h^2.
    Choose a box with negligible boundary values for a localized profile.

    F has shape (R, Nx), is complex, and has units GeV^-2. delta_x_GeV has
    shape (Nx,) and contains the signed FFT momenta in increasing order.
    The sqrt(2*pi) factor removes the 1D helper's Fourier normalization.
    The result excludes the leading i and the photon-meson overlap.
    Use momentum_transfer_grid(bx_fm) to obtain the same momenta in advance.
    """
    dipole_profile = np.asarray(dipole_profile)
    if np.iscomplexobj(bx_fm) or np.iscomplexobj(by_fm):
        raise ValueError("bx_fm and by_fm must contain real coordinates")
    bx_fm = np.asarray(bx_fm, dtype=float)
    by_fm = np.asarray(by_fm, dtype=float)
    for name, grid in (("bx_fm", bx_fm), ("by_fm", by_fm)):
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError(f"{name} must be a 1D array with at least two points")
        if np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
            raise ValueError(f"{name} must be finite and strictly increasing")

    Nx = bx_fm.size
    Ny = by_fm.size
    if (
        dipole_profile.ndim != 2
        or dipole_profile.shape[0] == 0
        or dipole_profile.shape[1] != Nx * Ny
    ):
        raise ValueError("dipole_profile must have shape (R, Nx*Ny) with R >= 1")
    if np.any(~np.isfinite(dipole_profile)):
        raise ValueError("dipole_profile must contain only finite values")

    R = dipole_profile.shape[0]
    profile_grid = dipole_profile.reshape(R, Nx, Ny)
    projected_profile = np.trapezoid(profile_grid, x=by_fm, axis=2)  # (R, Nx)

    F = np.empty((R, Nx), dtype=np.complex128)
    for i in range(R):
        spectrum, k_fm_inv = fast_fourier_transform_1D(
            projected_profile[i], bx_fm, angular=True
        )
        F[i] = spectrum * np.sqrt(2 * np.pi) / HBARC_GEV_FM**2

    delta_x_GeV = HBARC_GEV_FM * k_fm_inv
    #size (R,Nx) and (Nx)
    return F, delta_x_GeV


def photon_target_amplitude_T(r_fm, H_T, F):
    """Return one event's transverse amplitude in GeV^-2, shape (K,).

    With h = HBARC_GEV_FM, the amplitude is
        A_T(Delta) = i / (2*h^2) * int r dr H_T(r, |Delta|; Q) F(r, Delta).

    r_fm is a positive, strictly increasing 1D array with R >= 2 points.
    H_T is the real (R, K) overlap integral in GeV^2 returned by
    gaus_lc_overlap_integral_z. F is the (R, K) profile transform in GeV^-2
    returned by dipole_profile_fourier_x; its complex phase is preserved.
    Both arrays must share radius rows and momentum columns. For the x
    transform, evaluate H_T at abs(delta_x_GeV) without reordering columns.

    Trapezoidal integration covers the supplied interval [r_fm[0], r_fm[-1]]
    and supports nonuniform spacing. The caller controls radial cutoffs and
    resolution. The factor 1/2 is 2*pi from the dipole angle divided by 4*pi
    from the z measure. F already includes the b-area conversion h^-2;
    the remaining h^-2 converts r dr from fm^2 to GeV^-2.

    Target geometry and hotspot choices are already contained in F. This
    function returns the complex amplitude before any squaring or averaging.
    """
    if np.iscomplexobj(r_fm) or np.iscomplexobj(H_T):
        raise ValueError("r_fm and H_T must contain real values")
    r_fm = np.asarray(r_fm, dtype=float)
    H_T = np.asarray(H_T, dtype=float)
    F = np.asarray(F, dtype=np.complex128)

    if r_fm.ndim != 1 or r_fm.size < 2:
        raise ValueError("r_fm must be a 1D array with at least two points")
    if np.any(~np.isfinite(r_fm)) or np.any(r_fm <= 0):
        raise ValueError("r_fm must be finite and positive")
    if np.any(np.diff(r_fm) <= 0):
        raise ValueError("r_fm must be strictly increasing")
    if (
        H_T.ndim != 2
        or H_T.shape[0] != r_fm.size
        or H_T.shape[1] == 0
        or F.shape != H_T.shape
    ):
        raise ValueError("H_T and F must both have shape (R, K), with K >= 1")
    if np.any(~np.isfinite(H_T)) or np.any(~np.isfinite(F)):
        raise ValueError("H_T and F must contain finite values")

    radial_integrand = r_fm[:, None] * H_T * F
    radial_integral = np.trapezoid(radial_integrand, x=r_fm, axis=0)
    amplitude_prefactor = 1j / (2 * HBARC_GEV_FM**2)
    return amplitude_prefactor * radial_integral


def event_amplitude(
    bj_x,
    centers,
    weights,
    *,
    r_fm,
    bx_fm,
    by_fm,
    overlap_H,
    profile=proton_dipole_profile,
    profile_kwargs=None,
    momentum_mask=None,
):
    """Return one frozen event's complex transverse amplitude, (K,), GeV^-2.

    Assemble D(r,b) -> F(r,Delta_x) -> A(Delta_x) using the existing profile,
    projected FFT, and radial integral. The Cartesian b points are built
    from bx_fm and by_fm internally in the FFT's indexing="ij" order.

    centers (M,3) and weights (M,) describe either nucleons or hotspots;
    this function never samples or changes them. r_fm is the positive
    increasing (R,) dipole-radius grid. All input lengths are in fm.

    profile is called as
        profile(bj_x, r_fm, b_points, centers, weights=weights, **profile_kwargs)
    and must return dimensionless D with shape (R, Nx*Ny). The default is
    proton_dipole_profile. Pass nuclear_dipole_profile and A in profile_kwargs
    for the finite-A model, or another callable with the same interface.
    Model settings such as gbw_parameters, sigma_fm and sigma_0_fm2 belong
    in profile_kwargs; the orchestration itself makes no GBW assumption.

    overlap_H is a precomputed real (R,K) array in GeV^2. With no mask,
    prepare it at abs(momentum_transfer_grid(bx_fm)); then K=Nx. An optional
    boolean momentum_mask of shape (Nx,) selects FFT columns in their
    original order. In that case, prepare overlap_H at abs(delta[mask]),
    with K=mask.sum(). The helper never sorts or interpolates momenta.

    The result excludes skewness/real-part corrections, event averaging,
    and EPA factors. Those can act on the returned complex amplitude.
    """
    if momentum_mask is not None:
        momentum_mask = np.asarray(momentum_mask)
        if (
            momentum_mask.dtype != np.bool_
            or momentum_mask.ndim != 1
            or momentum_mask.shape != np.shape(bx_fm)
            or not np.any(momentum_mask)
        ):
            raise ValueError("momentum_mask must be a nonempty boolean selection of shape (Nx,)")

    bx_grid, by_grid = np.meshgrid(bx_fm, by_fm, indexing="ij")
    b_points = np.column_stack((bx_grid.ravel(), by_grid.ravel()))
    profile_kwargs = {} if profile_kwargs is None else dict(profile_kwargs)
    D = profile(bj_x, r_fm, b_points, centers, weights=weights, **profile_kwargs)
    F, _ = dipole_profile_fourier_x(D, bx_fm, by_fm)
    if momentum_mask is not None:
        F = F[:, momentum_mask]
    return photon_target_amplitude_T(r_fm, overlap_H, F)
