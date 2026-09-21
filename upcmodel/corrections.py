"""Model-independent logarithmic amplitude slopes and gluon skewness factors.

With Y = ln(1/x), use lambda_eff = d ln|A| / dY. For a complex amplitude
this is Re[(dA/dY)/A]; it does not include the derivative of its phase.
All results are real and dimensionless, with the input amplitude's shape.
No event sampling, averaging, or dipole-model parameters enter this module.

The R_g prescription assumes small-x, locally power-like behavior. Using an
amplitude slope in place of the gluon-density slope is a phenomenological
choice, not a general identity for arbitrary scattering amplitudes.
Reference: https://arxiv.org/html/hep-ph/9902410 (gluon ratio at x = xi).
"""

import numpy as np
from scipy.special import gammaln


def _positive_scalar(value, name):
    """Validate a real, finite positive scalar without discarding imaginary parts."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be a real, finite positive scalar")
    value = np.asarray(value, dtype=float)
    if value.ndim != 0 or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a real, finite positive scalar")
    return float(value)


def _amplitude_array(value, name, *, allow_zero=False):
    """Accept finite real or complex samples, preserving all array axes."""
    value = np.asarray(value, dtype=np.complex128)
    if value.size == 0 or np.any(~np.isfinite(value)):
        raise ValueError(f"{name} must contain finite values and be nonempty")
    if not allow_zero and np.any(value == 0):
        raise ValueError(f"{name} must be nonzero; the log slope is undefined at zeros")
    return value


def effective_lambda_from_samples(amplitude_low_x, amplitude_high_x, *, dlogx):
    """Estimate d ln|A| / d ln(1/x) from two samples centered in ln(1/x).

    Supply A(x*exp(-dlogx)) and A(x*exp(+dlogx)), in that order, using the
    same amplitude units and fixed non-x inputs. dlogx is the positive
    HALF-spacing in ln(1/x), not an additive step in x.

    The estimate is [ln|A_low_x| - ln|A_high_x|] / (2*dlogx), with error
    O(dlogx^2) for a smooth log magnitude. Scalars produce a scalar;
    arrays must have identical shapes, which are preserved in the result.
    No averaging or broadcasting between the two samples is performed.
    """
    dlogx = _positive_scalar(dlogx, "dlogx")
    low = _amplitude_array(amplitude_low_x, "amplitude_low_x")
    high = _amplitude_array(amplitude_high_x, "amplitude_high_x")
    if low.shape != high.shape:
        raise ValueError("The two amplitude samples must have identical shapes")

    # Log magnitudes avoid complex-log branch jumps and forming an A_low/A_high
    # ratio that could overflow. logaddexp also handles very large components.
    with np.errstate(divide="ignore"):
        log_low = 0.5 * np.logaddexp(
            2 * np.log(abs(low.real)), 2 * np.log(abs(low.imag))
        )
        log_high = 0.5 * np.logaddexp(
            2 * np.log(abs(high.real)), 2 * np.log(abs(high.imag))
        )
    with np.errstate(over="ignore", invalid="ignore"):
        slope = (log_low - log_high) / dlogx / 2
    if np.any(~np.isfinite(slope)):
        raise ValueError("The amplitude samples produce a nonfinite logarithmic slope")
    return slope


def effective_lambda(amplitude_at_x, bj_x, *, dlogx=0.05):
    """Estimate the real logarithmic slope of any callable amplitude A(x).

    amplitude_at_x takes one scalar x and returns a real or complex scalar
    or array. It is called twice, at x*exp(-dlogx) and x*exp(+dlogx). Both
    evaluations must return the same shape; lambda_eff preserves that shape.
    The central x and both neighboring x values must lie in (0, 1].

    For the frozen-event prescription, capture the SAME centers, weights,
    widths, Q and momentum grid in the callable. Do not resample hotspots
    inside it. For a common ensemble correction, have the callable return
    the complex mean forward amplitude, then differentiate that mean.

    Use the bare amplitude, before R_g. For the present correction evaluate
    at Delta=0; logarithmic slopes near diffraction zeros are unreliable.
    This two-sample estimate cannot detect a zero between its sample points.
    """
    if not callable(amplitude_at_x):
        raise ValueError("amplitude_at_x must be callable with one scalar x")
    bj_x = _positive_scalar(bj_x, "bj_x")
    dlogx = _positive_scalar(dlogx, "dlogx")
    if bj_x > 1:
        raise ValueError("bj_x must lie in (0, 1]")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        x_low = bj_x * np.exp(-dlogx)
        x_high = bj_x * np.exp(dlogx)
    if not (0 < x_low < bj_x < x_high <= 1):
        raise ValueError(
            "dlogx must give distinct neighboring x values within (0, 1]; "
            "choose a different step or supply an analytic derivative"
        )

    return effective_lambda_from_samples(
        amplitude_at_x(float(x_low)),
        amplitude_at_x(float(x_high)),
        dlogx=dlogx,
    )


def effective_lambda_from_derivative(amplitude, d_amplitude_dlog1x):
    """Return Re[(dA/dY)/A], where Y=ln(1/x), from an analytic derivative.

    amplitude and d_amplitude_dlog1x must have identical shapes and units.
    A derivative with respect to x must first be converted: dA/dY = -x*dA/dx.
    The amplitude must be nonzero; its derivative may be zero. The returned
    scalar or array preserves the amplitude shape and excludes phase growth.
    """
    amplitude = _amplitude_array(amplitude, "amplitude")
    derivative = _amplitude_array(
        d_amplitude_dlog1x, "d_amplitude_dlog1x", allow_zero=True
    )
    if amplitude.shape != derivative.shape:
        raise ValueError("The amplitude and its derivative must have identical shapes")

    # Re[A'/A] = (Re(A)*Re(A') + Im(A)*Im(A')) / |A|^2.
    # Scale the components first so changing amplitude units cannot overflow
    # the denominator or make a very small nonzero amplitude look like zero.
    scale = np.maximum(abs(amplitude.real), abs(amplitude.imag))
    real = amplitude.real / scale
    imag = amplitude.imag / scale
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        slope = (
            real * (derivative.real / scale) + imag * (derivative.imag / scale)
        ) / (real**2 + imag**2)
    if np.any(~np.isfinite(slope)):
        raise ValueError("The derivative produces a nonfinite logarithmic slope")
    return slope


def skewness_factor(lambda_eff):
    """Return R_g = 2^(2*lambda+3)/sqrt(pi) * Gamma(lambda+5/2)/Gamma(lambda+4).

    Accept a real scalar or array of slopes and preserve its shape. Compute
    the gamma ratio in log space. The positive-gamma branch requires finite
    lambda_eff > -2.5; this algebraic domain is not a physical validity claim.
    Negative slopes and slopes above the GBW exponent are not clipped.

    Multiply an amplitude by R_g, OR its cross section by R_g**2, once.
    This factor includes neither the real-part correction nor UPC effects.
    """
    if np.iscomplexobj(lambda_eff):
        raise ValueError("lambda_eff must be real")
    lambda_eff = np.asarray(lambda_eff, dtype=float)
    if (
        lambda_eff.size == 0
        or np.any(~np.isfinite(lambda_eff))
        or np.any(lambda_eff <= -2.5)
    ):
        raise ValueError("lambda_eff must be finite, nonempty, and greater than -2.5")

    with np.errstate(over="ignore", invalid="ignore"):
        log_factor = (
            (2 * lambda_eff + 3) * np.log(2)
            - 0.5 * np.log(np.pi)
            + gammaln(lambda_eff + 2.5)
            - gammaln(lambda_eff + 4)
        )
        factor = np.exp(log_factor)
    if np.any(~np.isfinite(factor)) or np.any(factor <= 0):
        raise ValueError("R_g is outside the positive finite numerical range")
    return factor
