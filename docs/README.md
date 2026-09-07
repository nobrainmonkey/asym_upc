# Geometry and GBW checkpoint

Saved on 7 September 2026.

The implementation in `upcmodel/` now includes 3pF and alpha-cluster nucleon
events, weighted hotspot sampling, Gaussian densities and their analytic
thicknesses, the GBW saturation scale and dipole strength, and the finite-A
nuclear dipole profile.

`nuclear_dipole_profile` accepts either nucleon or hotspot centers and weights.
Its required `sigma_0_fm2` keyword supplies the dipole cross-section normalization
independently of the Gaussian source width `sigma_fm`. For the paper's
prescription, calculate `sigma_0_fm2 = 4 * np.pi * B_p_fm2` in the model setup.
The function returns a dimensionless `(R, P)` array and does not integrate over
dipole size.

## Saved learning and diagnostic material

- [GBW mathematics and integration plan](gbw_amplitude_plan.md), with a
  [PDF copy](../output/pdf/gbw_amplitude_plan.pdf). These were written before the
  GBW and nuclear-profile implementation; references to scaffolds and upcoming
  implementation describe that earlier stage.
- [Geometry notebook](../notebooks/01_spherical_geometry.ipynb), its
  [HTML export](../notebooks/01_spherical_geometry.html), and the figures,
  interactive plots, samples, and audit in
  [`geometry_debug_outputs`](../notebooks/geometry_debug_outputs/).

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

The next physics implementation is the photon-meson wavefunction overlap,
followed by the impact-parameter transform and dipole-size integration in the
linked plan.
