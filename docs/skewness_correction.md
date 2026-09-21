# Amplitude slopes and the gluon skewness correction

`upcmodel/corrections.py` accepts any amplitude model. It imports no GBW,
geometry, or wavefunction code. With all other inputs held fixed, it computes

\[
Y=\ln(1/x),\qquad
\lambda_{\rm eff}=\frac{d\ln|\mathcal A|}{dY}
=\operatorname{Re}\left[\frac{1}{\mathcal A}\frac{d\mathcal A}{dY}\right],
\]

and then

\[
R_g(\lambda)=\frac{2^{2\lambda+3}}{\sqrt\pi}
\frac{\Gamma(\lambda+5/2)}{\Gamma(\lambda+4)}.
\]

The magnitude makes the slope real for complex amplitudes and removes any
constant normalization or phase. For the present positive imaginary forward
amplitude, this equals the logarithmic slope of its imaginary part. For a
general complex amplitude, it measures magnitude growth only.

The [small-x gluon GPD construction](https://arxiv.org/html/hep-ph/9902410)
gives this factor for a locally power-like gluon density at the crossover
point. Substituting an amplitude slope is a phenomenological prescription;
model-independent software does not make the correction exact for every model.

## Public functions and shapes

| Function | Inputs | Output |
| --- | --- | --- |
| `effective_lambda(amplitude_at_x, bj_x, dlogx=0.05)` | Callable `A(x)`, scalar central `x`, logarithmic half-step | Real slope with the shape returned by `A(x)` |
| `effective_lambda_from_samples(A_low_x, A_high_x, dlogx=...)` | Two amplitudes of identical shape | Real slope of that shape |
| `effective_lambda_from_derivative(A, dA_dlog1x)` | Amplitude and derivative of identical shape | Real slope of that shape |
| `skewness_factor(lambda_eff)` | Real scalar or array | Dimensionless `R_g` with the same shape |

The step is a keyword argument. The numerical helper evaluates only twice:

\[
x_{\rm low}=x e^{-\delta},\quad x_{\rm high}=x e^{+\delta},\qquad
\lambda_{\rm eff}\simeq
\frac{\ln|\mathcal A(x_{\rm low})|-\ln|\mathcal A(x_{\rm high})|}{2\delta}.
\]

The error is second order in `dlogx`; compare two step sizes when introducing
a new amplitude model. Neighboring x values must remain distinct and in
`(0, 1]`. For an existing derivative with respect to x, supply
`dA_dlog1x = -bj_x * dA_dx` to the analytic helper.

Scalar forward amplitudes give scalar slopes. Arrays `(K,)` or `(E, K)` give
slopes with those same shapes, evaluated elementwise. The helpers never
choose a momentum bin, average events, square amplitudes, or change geometry.
Zero and nonfinite amplitudes raise errors; no epsilon floor is inserted.
Zeros between the two sampled x values cannot be detected by the numerical
helper. Use the forward amplitude for the current correction prescription.

## A model-independent example

```python
from upcmodel.corrections import effective_lambda, skewness_factor

def amplitude_at_x(x):
    return 2j * x**(-0.4)

lambda_eff = effective_lambda(amplitude_at_x, bj_x=1e-3)  # 0.4
R_g = skewness_factor(lambda_eff)                       # 1.41674...
```

There is no GBW-specific bound on `lambda_eff`. The `R_g` implementation
accepts the positive-gamma branch `lambda_eff > -2.5`, including negative
slopes; this is an algebraic domain, not a statement of physical validity.
It uses log-gamma functions and rejects factors that exceed numerical range.

## Reuse the current amplitude notebook

After defining `run_case` in `02_dipole_amplitude.ipynb`, use:

```python
from upcmodel.corrections import effective_lambda, skewness_factor

def forward_for_event(sources, x):
    result = run_case(sources, bj_x=x)
    zero = int(np.argmin(np.abs(result["delta"])))
    return result["A"][zero]  # One complex scalar at Delta=0.

key = ("WS / 3pF", "hotspot")   # Also works with "nucleon" or alpha geometry.
sources = CASES[key]           # Reuse the same event at both x values.

lambda_eff = effective_lambda(
    lambda x: forward_for_event(sources, x), BJ_X, dlogx=0.05
)
R_g = skewness_factor(lambda_eff)
A_corrected = R_g * RESULTS[key]["A"]  # Still (K,), complex GeV^-2.
```

Keep centers, weights, widths, Q, and integration grids fixed inside the
callable. In particular, do not call `sample_target_hotspots` again at the
neighboring x values. This frozen-event prescription omits the x-dependence
of the event distribution, including the changing hotspot multiplicity.

For a common correction for a frozen ensemble, form its **complex mean
amplitude before differentiating**:

```python
# frozen_sources is your already-generated list of event source dictionaries.
def mean_forward_amplitude(x):
    return np.mean([forward_for_event(event, x) for event in frozen_sources])

lambda_eff = effective_lambda(mean_forward_amplitude, BJ_X)
R_g = skewness_factor(lambda_eff)
```

This is not the mean of individual slopes or a derivative of `mean(abs(A))`.
For a common scalar `R_g`, both coherent and incoherent baseline spectra
receive the same factor `R_g**2`. Alternatively multiply all bare amplitudes
by `R_g` before forming their mean and variance. Apply the correction once.
Real-part corrections and UPC photon fluxes are separate steps.

## Validation

Run `.venv/bin/python -m unittest discover -s tests -v` from the project root.
The tests use non-GBW power laws, amplitudes with nonlinear x dependence and
varying complex phase, analytic derivatives, ensemble means, exact gamma-ratio
values, and numerical-domain checks.
