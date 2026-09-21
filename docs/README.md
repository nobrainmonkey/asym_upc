# Geometry and amplitude diagnostics

The geometry checkpoint was saved on 7 September 2026. The amplitude
demonstration was added on 11 September 2026. The photon-target cross-section
checkpoint was prepared on 20 September 2026.

The implementation in `upcmodel/` now includes 3pF and alpha-cluster nucleon
events, weighted hotspot sampling, Gaussian densities and their analytic
thicknesses, the GBW saturation scale and dipole strength, the finite-A
nuclear dipole profile, the transverse Gaus-LC photon-meson overlap, its
z integral, the projected Fourier transform, and the single-event transverse
photon-target amplitude.

`nuclear_dipole_profile` accepts either nucleon or hotspot centers and weights.
Its required `sigma_0_fm2` keyword supplies the dipole cross-section normalization
independently of the Gaussian source width `sigma_fm`. For the paper's
prescription, calculate `sigma_0_fm2 = 4 * np.pi * B_p_fm2` in the model setup.
The function returns a dimensionless `(R, P)` array and does not integrate over
dipole size.

`amplitudes.event_amplitude` combines a selected dipole profile, the projected
FFT, and the radial integral for one frozen event. Supply the precomputed
`overlap_H`, coordinate grids, source centers and weights. `profile_kwargs`
holds the profile's parameters; select `nuclear_dipole_profile` and include
`A` there for nuclei, or use the default `proton_dipole_profile`. Both accept
nucleon or hotspot sources. An optional boolean `momentum_mask` selects FFT
columns, with the overlap prepared at the same selected momentum magnitudes.
Notebook 3 configures these fixed inputs once and reuses them for each event.

## Saved learning and diagnostic material

- [Photon-proton cross-section notebook](../notebooks/03_gamma_p_cross_sections.ipynb)
  and its [executed HTML](../notebooks/03_gamma_p_cross_sections.html): J/psi
  photoproduction at W = 75 GeV, with a fluctuating hotspot proton, GBW,
  Gaus-LC, skewness, and coherent/incoherent spectra. H1 2013 Table 5 supplies
  both channels. A short rho comparison at W = 44 GeV follows, using H1 2020
  Tables 12 and 13. Both use the actual experimental bin widths. The final
  note explains saturation-scale fluctuations; those are not enabled in
  these geometry-only calculations.
  [Local reference data and provenance](../notebooks/data/README.md) make the
  comparison runnable offline. The calculation uses the factorized proton
  profile `sigma_0 * N_GBW * T_p` through `proton_dipole_profile`.
- [Model-independent amplitude slopes and skewness](skewness_correction.md),
  including callable, sampled-amplitude, and analytic-derivative interfaces
  and examples using the existing event and ensemble amplitudes.
- [Published J/psi model and HERA data comparisons](heikki_figure1_comparison.md),
  with the [Figure 1 overlay PDF](../output/pdf/jpsi_heikki_figure1_overlay.pdf)
  and [H1/ZEUS comparison PDF](../output/pdf/jpsi_incoherent_h1_zeus.pdf).
  The notes distinguish the model ingredients and experimental breakup cuts.
- [EPA mathematics and implementation plan](epa_implementation_plan.md):
  point-charge and event-shaped proton-source fields, photon kinematics,
  two emission paths, and event-dependent hadronic survival. This is planned
  work; no EPA or UPC assembler is implemented yet.
- [GBW mathematics and integration plan](gbw_amplitude_plan.md), with a
  [PDF copy](../output/pdf/gbw_amplitude_plan.pdf). These were written before the
  GBW and nuclear-profile implementation; references to scaffolds and upcoming
  implementation describe that earlier stage.
- [Geometry notebook](../notebooks/01_spherical_geometry.ipynb), its
  [HTML export](../notebooks/01_spherical_geometry.html), and the figures,
  interactive plots, samples, and audit in
  [`geometry_debug_outputs`](../notebooks/geometry_debug_outputs/).
- [Dipole-amplitude notebook](../notebooks/02_dipole_amplitude.ipynb), its
  [HTML export](../notebooks/02_dipole_amplitude.html), and the figures,
  complex amplitudes, and audit in
  [`amplitude_debug_outputs`](../notebooks/amplitude_debug_outputs/).
  This notebook runs the current code from energy-to-x conversion through
  GBW, finite-A profiles, wavefunctions, the J0 overlap integral, FFTs, and
  radial amplitude integration. It includes analytic references, phase
  checks, energy and virtuality controls, and numerical convergence studies.

The saved geometry run used `B_p = 4.75 GeV^-2` under the then-current `RHO_HS`
label and `B_hs = 0.8 GeV^-2`. Its smooth comparison uses variance
`B_p + B_hs`, matching the ensemble mean of the hotspot density. The learning
plan's numerical width examples use that same earlier parameter choice.

The current presets use `B_p = 6.4 GeV^-2` for rho and `4.75 GeV^-2` for J/psi.
Both use `B_hs = 0.8 GeV^-2`. To reproduce the paper's smooth-nucleon profile,
use `sigma_fm = sqrt(B_p_fm2)`; for individual hotspot densities, use
`sigma_fm = sqrt(B_hs_fm2)`. The saved figures have not been regenerated after
the rho parameter change. Their exact inputs and source hashes are recorded in
[`geometry_audit.json`](../notebooks/geometry_debug_outputs/geometry_audit.json).

The amplitude notebook uses the current rho presets and the paper's smooth
width `sqrt(B_p_fm2)`. It records the meson mass and the approximation
`x = (Q^2 + M_V^2) / (Q^2 + W_gammaN^2)` explicitly. Its single-event
intensity plots are not coherent or incoherent ensemble spectra.

`upcmodel/ensemble.py` now supplies the ensemble mean amplitude, centered
population variance, and coherent/incoherent spectra used in notebook 3.
UPC photon fluxes, no-hadronic-interaction probabilities, and interference
between production pathways come later.

The checkpoint includes 41 passing unit tests covering Fourier conventions,
proton profiles, event amplitudes, skewness, and ensemble moments. Run them
from the project directory with:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

The saved amplitude notebook contains nuclear diagnostics; the current
cross-section data comparisons are proton benchmarks. Dedicated nuclear
cross-section benchmarks remain part of the validation work.
