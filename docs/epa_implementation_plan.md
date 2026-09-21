# EPA mathematics and implementation plan

This is a design document, not an EPA implementation. It connects the current
event geometry and photon–target amplitudes to a later UPC calculation.

The two requested controls are:

1. Put the entire emitter charge Z at its nuclear center.
2. Sum the fields of its sampled protons, doing the photon-momentum angular
   integral analytically for each proton, while retaining the event geometry.

The interpretation of “pre-averaged” matters. Case 2 below averages the momentum
angle in a source kernel; it does not average over nuclear events or rotate the
event into a spherical density. Section 5 explains how to recover the old
ensemble-averaged emitter if that is the intended prescription instead.

## 1. What is already available, and what EPA supplies

The current package calculates a photon–target amplitude

$$
\mathcal A_C(x,\boldsymbol\Delta),\qquad [\mathcal A]={\rm GeV}^{-2},
$$

for a target configuration C. `photon_target_amplitude_T` returns `(K,)`, one
complex number for each sampled momentum. `amplitude_moments` averages over
events; `photon_target_spectra` supplies the current $1/(16\pi)$ normalization.
The target can use nucleons or gluonic hotspots. Neither choice specifies its
electromagnetic charge density.

EPA supplies the photons emitted by the other ion. It is independent of the
choice of GBW or any future target interaction kernel.

Use these coordinates and units consistently:

| Symbol | Meaning | Unit |
|---|---|---|
| $\mathbf B$ | ion-center separation, from left to right | fm |
| $\mathbf b$ | interaction coordinate relative to target center | fm |
| $\mathbf R$ | observation point relative to emitter center | fm |
| $(\mathbf s_j,\zeta_j)$ | proton's rest-frame transverse position and longitudinal coordinate | fm |
| $\omega$ | emitted photon energy in the chosen collider frame | GeV |
| $k$ | transverse wave number used inside the EPA kernel | fm$^{-1}$ |
| $\boldsymbol\Delta$ | momentum transfer in the photon–target amplitude | GeV |
| $h=\hbar c$ | `HBARC_GEV_FM` | GeV fm |

For emission from the left ion, a target point has $\mathbf R=\mathbf B+\mathbf b$.
For a field evaluated just at the target center, $\mathbf R=\mathbf B$.
Do not identify the new ion separation B with the existing target coordinate b.

Define

$$
\beta=\sqrt{1-\gamma^{-2}},\qquad
a=\frac{\omega}{\gamma\beta h},\qquad C_\gamma=\frac{\sqrt{\alpha_{\rm em}}}{\pi\beta}.
$$

Here a is a wave number in fm$^{-1}$. It regulates the large-distance field
and represents the rest-frame longitudinal momentum transfer divided by h.

Return a *flux amplitude* $\boldsymbol{\mathcal E}$, normalized by

$$
\nu(\omega,\mathbf R)
\equiv \frac{dN_\gamma}{d\ln\omega\,d^2R}
=\omega\frac{dN_\gamma}{d\omega\,d^2R}
=\boldsymbol{\mathcal E}\cdot\boldsymbol{\mathcal E}^{*}.
$$

Thus $[\mathcal E]={\rm fm}^{-1}$, $[\nu]={\rm fm}^{-2}$, and the per-energy
flux is $n=\nu/\omega$. This convention absorbs the factor $|d\omega/dy|=\omega$
needed in a rapidity spectrum. It is a normalized photon amplitude, not a
claim that its numerical value equals an electric field in SI units.

The first implementation uses only transverse photons, matching the current
`photon_target_amplitude_T`. The standard point-source EPA and its transverse
dominance at large gamma are described in
[STARlight, Eq. (1)](https://arxiv.org/html/1607.03838).

## 2. Case 1: the entire nucleus as a point charge

Let $R=|\mathbf R|>0$ and $\xi=aR$. The transverse flux amplitude is

$$
\boxed{
\boldsymbol{\mathcal E}_{\rm point}(\omega,\mathbf R)
=Z C_\gamma\,a K_1(aR)\,\widehat{\mathbf R}
=\frac{Z C_\gamma}{R}\,\xi K_1(\xi)\,\widehat{\mathbf R}.
}
$$

Therefore

$$
\boxed{
\nu_{\rm point,T}(\omega,R)
=\frac{Z^2\alpha_{\rm em}}{\pi^2\beta^2 R^2}
\,\xi^2 K_1^2(\xi).
}
$$

For $aR\ll1$, $\xi K_1(\xi)\to1$: the field goes as $Z/R$ and the flux as
$Z^2/R^2$. For $aR\gg1$, it is exponentially suppressed. These K functions
are **modified** Bessel functions, distinct from the ordinary J functions below.

If a full point-charge flux is wanted as a separate check, its suppressed
longitudinal contribution is

$$
\nu_{\rm point,L}=
\frac{Z^2\alpha_{\rm em}}{\pi^2\beta^2}\frac{a^2}{\gamma^2}K_0^2(aR).
$$

Do not couple this extra component to the existing transverse target amplitude.
The leading high-energy calculation can consistently omit it.

For a sharp separation cut $B>B_{\min}$, an analytic transverse-flux control is

$$
\int_{B>B_{\min}}d^2B\,\nu_{\rm point,T}
=\frac{2Z^2\alpha_{\rm em}}{\pi\beta^2}
\left[\xi_0K_0(\xi_0)K_1(\xi_0)
-\frac{\xi_0^2}{2}\{K_1^2(\xi_0)-K_0^2(\xi_0)\}\right],
\quad \xi_0=aB_{\min}.
$$

The integrated result is dimensionless: photons per logarithmic energy interval.
The point field is singular at R=0; that is outside this control's domain,
not a place to silently clip the field.

## 3. Case 2: proton sources with their event geometry retained

Select the sources directly from the current event:

```python
protons_fm = nucleon_positions_fm[proton_mask]  # (Z, 3)
```

Use all protons and no neutrons in this leading charge model. Retain the nuclear
center-of-mass origin; do not separately recenter the proton subset. Its
displacement relative to the matter distribution is part of the sampled event.
Gluonic hotspots, their multiplicity weights, B_p and B_hs do not enter EPA.

Let $f_p(Q^2)$ be the normalized electromagnetic form factor of one spherical
proton, with $f_p(0)=1$. It is separate from the gluonic widths. Taking $f_p=1$
means point protons at the sampled positions. An optional Gaussian charge model
would have

$$
f_p(Q^2)=\exp[-\sigma_{\rm ch}^2 Q^2/(2h^2)],
\qquad \langle r_{\rm ch}^2\rangle=3\sigma_{\rm ch}^2.
$$

No numerical charge radius is selected by this plan. A precision proton EPA
can require additional magnetic/recoil terms; the present source is the
leading electric-charge model appropriate to the planned nuclear calculation.

For beam direction $\eta=+1$ or $-1$, the normalized event charge form factor is

$$
F_C(\mathbf k,\eta a)
=\frac{f_p(h^2(k^2+a^2))}{Z}
\sum_{j=1}^{Z}e^{-i\mathbf k\cdot\mathbf s_j-i\eta a\zeta_j}.
$$

Its zero-three-momentum normalization is one. At nonzero a,
$F_C(\mathbf k=0,\eta a)$ need not equal one. The longitudinal factor is retained
because the input events are three dimensional. Setting it to one is a
long-coherence approximation requiring $a|\zeta_j|\ll1$.

The general transverse transform, with k in fm$^{-1}$, is

$$
\boldsymbol{\mathcal E}_C(\omega,\mathbf R)
=-i Z C_\gamma
\int\frac{d^2k}{2\pi}\,
\frac{\mathbf k}{k^2+a^2}
F_C(\mathbf k,\eta a)e^{i\mathbf k\cdot\mathbf R}.
$$

The charge-form-factor representation of the EPA field and its conversion to
photon flux are given in
[Schäfer, Eqs. (2)–(3)](https://link.springer.com/article/10.1140/epja/s10050-020-00231-8).
The per-source reduction below follows from inserting the discrete charge sum.

### Exactly where J0 enters

For proton j, define

$$
\mathbf d_j=\mathbf R-\mathbf s_j,\qquad d_j=|\mathbf d_j|.
$$

The angular identity is

$$
\frac{1}{2\pi}\int_0^{2\pi}d\phi_k\,
e^{i\mathbf k\cdot(\mathbf R-\mathbf s_j)}=J_0(kd_j).
$$

Define the dimensionless scalar kernel

$$
G_a(d)=\int_0^\infty dk\,
\frac{k}{k^2+a^2}f_p(h^2(k^2+a^2))J_0(kd).
$$

The vector field comes from its transverse gradient. Since
$dJ_0(kd)/dd=-kJ_1(kd)$,

$$
\mathcal K_a(d)=-\frac{dG_a}{dd}
=\int_0^\infty dk\,
\frac{k^2}{k^2+a^2}f_p(h^2(k^2+a^2))J_1(kd),
$$

and

$$
\boxed{
\boldsymbol{\mathcal E}_C(\omega,\mathbf R)
=C_\gamma\sum_{j=1}^{Z}e^{-i\eta a\zeta_j}
\widehat{\mathbf d}_j\,\mathcal K_a(d_j).
}
$$

This is the desired source-by-source angular reduction. The angular integration
is exact for spherical proton charge profiles. The **nucleus need not be
azimuthally symmetric**: both d_j and its direction depend on every proton's
actual transverse position. Do not discard the vector directions or replace
d_j by the proton's distance from the nuclear origin.

For point protons the integrals have analytic answers:

$$
G_a(d)=K_0(ad),\qquad \mathcal K_a(d)=aK_1(ad),\qquad d>0.
$$

Consequently the first implementation of case 2 can be a direct sum of shifted
point-proton fields; numerical integration of J0 is unnecessary. A finite
proton form factor later requires only a one-dimensional radial kernel table,
reused for every source and observation point at the same energy.

The electric J1 kernel is also the form used in the radial EPA formula in
[Dyndal and Schoeffel, Eq. (2)](https://scoap3-prod-backend.s3.cern.ch/media/files/5142/10.1016/j.physletb.2014.12.019_a.pdf).

### Sum fields before calculating flux

$$
\nu_C=\left|\sum_j\boldsymbol{\mathcal E}_j\right|^2
=\sum_j|\boldsymbol{\mathcal E}_j|^2
+2\operatorname{Re}\sum_{i<j}\boldsymbol{\mathcal E}_i\cdot
\boldsymbol{\mathcal E}_j^*.
$$

The cross terms supply coherent charge enhancement. Summing single-proton
fluxes would discard them and generally give Z scaling instead of the coherent
$Z^2$ limit. Because each source contributes unit charge, do not multiply the
source sum by another Z; the Z in the normalized-form-factor representation
has already canceled its 1/Z.

## 4. Shapes and numerical implementation

For one event:

| Array | Shape | Meaning |
|---|---|---|
| `nucleon_positions_fm` | `(A, 3)` | original nucleon event |
| `proton_mask` | `(A,)` | boolean charge selection |
| `observation_xy_fm` | `(P, 2)` | emitter-relative observation points |
| `protons_fm` | `(Z, 3)` | selected charge centers |
| `displacement_fm` | `(P, Z, 2)` | observation minus source position |
| `distance_fm` | `(P, Z)` | transverse distance |
| `longitudinal_phase` | `(Z,)` | complex rest-frame coherence phase |
| `field` | `(P, 2)` | complex transverse flux amplitude, fm$^{-1}$ |
| `flux_per_log_energy` | `(P,)` | real norm squared, fm$^{-2}$ |

An ensemble of fields has `(E, P, 2)`. The final axis is a transverse vector
index. In contrast, the existing photon–target amplitude `(K,)` has no
transverse-component axis: it is a complex scalar at each momentum point.

For a finite proton form factor, tabulate the J1 integral directly. Do not
differentiate a noisy numerical J0 transform. Chunk observation points/source
distances so no unnecessary `(P, Z, N_k)` array is retained. The analytic
point-proton path requires no k-grid or FFT. A general 2D FFT field provider
can be added later without changing downstream APIs.

## 5. Three distinct meanings of averaging

**Kernel angular integration:** $J_0(k|\mathbf R-\mathbf s_j|)$ above retains
the complete transverse event shape. This is the interpretation used for case 2.

**Azimuthal averaging of the emitter itself:** replacing each source phase by
$J_0(k|\mathbf s_j|)$ turns each proton into a ring around the nuclear origin.
This retains event-dependent radial structure, but erases its directional
shape. It is not the same operation as integrating the shifted source kernel.

**Ensemble preaveraging:** use

$$
\overline{\boldsymbol{\mathcal E}}=\langle\boldsymbol{\mathcal E}_C\rangle_C,
\qquad
\nu_{\rm elastic\ emitter}=|\overline{\boldsymbol{\mathcal E}}|^2.
$$

Linearity makes this equivalent to calculating the field from the ensemble
mean charge density. In general it differs from
$\langle|\boldsymbol{\mathcal E}_C|^2\rangle_C$. A rotationally invariant
ensemble has a smooth radial mean, although alpha and 3pF ensembles can retain
different mean radial profiles.

The old reference's `oxygen_upc_helpers.ensemble_mean_form_factor` takes this
last route and analytically averages **three-dimensional** orientations using

$$
j_0(qr)=\frac{\sin(qr)}{qr},\qquad
\overline F(q)=f_p(h^2q^2)
\left\langle\frac1Z\sum_jj_0(q|\mathbf r_j|)\right\rangle_C.
$$

Here q is in fm$^{-1}$ and lower-case spherical j0 differs from cylindrical J0.
Its `radial_epa_field` uses J1 for transverse fields and J0 for the suppressed
longitudinal component. The new per-proton field extends its averaging choice
while adopting the current explicit proton mask; its overall normalization
uses flux per logarithmic energy rather than the old per-energy amplitude.

For an explicitly intact emitter, use the coherent emitter projection. If
fluctuating source fields enter joint Good–Walker moments, the variance can
include emitter dissociation as well as target dissociation. Label that
observable accordingly; it is not automatically the target-only incoherent
cross section calculated in notebook 3.

## 6. Kinematics and the first connection to the existing amplitude

For high-energy, small-pT photoproduction in the nucleon–nucleon CM frame,

$$
\omega_+=\frac{M_V}{2}e^{+y},\qquad
\omega_-=\frac{M_V}{2}e^{-y},\qquad
W_{\gamma N,\pm}^2\simeq2\omega_\pm\sqrt{s_{NN}},
$$

$$
x_\pm\simeq\frac{M_V^2}{W_{\gamma N,\pm}^2}
=\frac{M_V}{\sqrt{s_{NN}}}e^{\mp y}.
$$

The plus photon is emitted by the ion moving along +z and scatters from the
other ion. Both omega and gamma must be defined in the same frame. For equal
per-nucleon CM beam energies, $\gamma\simeq\sqrt{s_{NN}}/(2m_N)$; keep emitter
gamma explicit so asymmetric beam configurations are possible. EPA photon
energy/fraction and target Bjorken x are different inputs connected here.

The first UPC control treats the photon field as constant over the target:

$$
\boldsymbol{\mathcal M}_{L\to R}(y,\mathbf B,\boldsymbol\Delta)
=\boldsymbol{\mathcal E}_L(\omega_+,\mathbf B)
\mathcal A_{R}(x_+,\boldsymbol\Delta).
$$

This directly reuses the existing complex `(K,)` amplitudes. With field samples
`(P, 2)`, broadcasting produces `(P, K, 2)`. Its units are
fm$^{-1}$ GeV$^{-2}$. Any target amplitude correction such as R_g belongs in
$\mathcal A_R$ before the two paths are combined, once only.

For a deterministic/preaveraged emitter and a configuration-independent
survival probability, one direction gives

$$
\frac{d\sigma_{AA}^{\rm coh/inc}}{dy\,d|t|}
=\int d^2B\,P_{\rm nohad}(B)\nu_L(\omega_+,\mathbf B)
\frac{d\sigma_{\gamma R}^{\rm coh/inc}}{d|t|}(x_+).
$$

The fm$^2$ integration cancels the flux's fm$^{-2}$ units. No extra factor of
omega is needed with the chosen flux definition. This flux folding is the
first integration control, before two-emitter interference.

The approximation is physical, not just numerical: the field can vary across
the target near grazing separations. Also the measured meson momentum obeys
$\mathbf p_T=\mathbf k_{\gamma T}+\boldsymbol\Delta$. Identifying pT with
Delta neglects the photon transverse momentum. An accurate pT distribution
later needs an amplitude-level photon-momentum convolution (or its consistent
coordinate-space equivalent), retaining the target transfer in its nonforward
dipole phase. Do not call the target's Delta-grid a full UPC pT calculation.

## 7. Two emission paths and survival: interfaces to preserve now

Place the ion centers at $-\mathbf B/2$ and $+\mathbf B/2$. In the preceding
small-photon-kT, polarization-preserving approximation, choose a common lab
transverse polarization basis and write

$$
\boldsymbol{\mathcal M}=
e^{-i\mathbf p_T\cdot\mathbf B/(2h)}
\boldsymbol{\mathcal E}_L(\omega_+,\mathbf B)\mathcal A_R(x_+,\mathbf p_T)
+e^{+i\mathbf p_T\cdot\mathbf B/(2h)}
\boldsymbol{\mathcal E}_R(\omega_-,-\mathbf B)\mathcal A_L(x_-,\mathbf p_T).
$$

For identical radial emitters the two signed fields point in opposite
directions. Thus this vector convention already gives the familiar destructive
interference. In the alternative scalar convention with positive field
magnitudes, write an explicit minus between the paths. Do not do both.
This matches the low-pT interference discussed by
[Klein and Nystrand](https://arxiv.org/abs/hep-ph/9909237).

The integration over B is a cross-section integration: combine the two
histories and square at fixed B, then integrate. Do not add amplitudes for
different classical ion separations before squaring.

The interference phase can reduce to $J_0(p_TB/h)$ only when the remaining
coefficient is independent of the relative azimuth. Event-shaped fields,
event-shaped target amplitudes, and event-dependent survival generally
prevent that replacement at fixed configurations. One may retain the current
Delta-x slice if the global ensemble and B-angle integration are sampled
consistently; fixing both B and Delta parallel would lose their relative-angle
dependence. This J0 is separate from the per-proton EPA J0 and from H_T's
internal dipole J0.

For later event-dependent survival, retain the original nucleons in each event:

$$
P_{\rm nohad}(\mathbf B|C_L,C_R)
=\prod_{i\in L,j\in R}
\left[1-p_{NN}\!\left(|\mathbf B+\mathbf s_j^R-\mathbf s_i^L|\right)\right],
$$

with $0\le p_{NN}\le1$ and
$\int d^2d\,p_{NN}(d)=\sigma_{NN}^{\rm inel}$ in consistent fm units. This uses
all nucleons, not just protons and not gluonic hotspots. For
$p_{NN}=1-e^{-\Omega_{NN}}$, it is $\exp[-\sum_{ij}\Omega_{NN}(d_{ij})]$.

This is the mathematical structure of the legacy `pnohad.py` and its
`SurvivalCalculator`. Reuse its physical idea, not its implicit GeV$^{-1}$
coordinate convention; the new public geometry remains in fm.

For a specified real absorptive survival-amplitude model, take
$S_{\rm nohad}=\sqrt{P_{\rm nohad}}$ and form

$$\widetilde{\boldsymbol{\mathcal M}}=S_{\rm nohad}\boldsymbol{\mathcal M}.$$

The same S multiplies both histories of the same event pair. Their diagonal
and cross terms then acquire the same P. At each B and momentum, joint
Good–Walker moments are

$$
C=\sum_{a=x,y}|\langle\widetilde{\mathcal M}_a\rangle|^2,
\qquad
I=\sum_{a=x,y}\left(\langle|\widetilde{\mathcal M}_a|^2\rangle
-|\langle\widetilde{\mathcal M}_a\rangle|^2\right).
$$

For a rotationally invariant completed ensemble, their B integrals divided by
$16\pi$ give $d\sigma/(dy\,d|t|)$ in the current amplitude convention and
small-photon-kT approximation. Before azimuthal averaging, the corresponding
$d\sigma/(dy\,d^2p_T)$ normalization is $1/(16\pi^2)$.

These are unconditioned scattering moments with a modeled survival amplitude.
They are not the same as redefining the ensemble with normalized weights
$P/\langle P\rangle$. A no-inelastic probability alone does not determine a
possible phase of the survival amplitude; the real square root is the stated
absorptive model choice. Define the desired breakup channels before labeling
the joint variance as target incoherence.

Keep survival outside the EPA module so the same fields support a sharp-cut
control, optical survival, and the later event-pair prescription.

## 8. Code sequence

Add `upcmodel/epa.py` with these planned functions; no target-model imports:

```python
point_charge_epa_field(omega_GeV, gamma, observation_xy_fm, Z)
# -> complex (P, 2), fm^-1; entire nucleus at origin

epa_flux_per_log_energy(field)
# -> (...,), fm^-2; sum abs(field)**2 over its last axis of length 2

proton_source_epa_field(
    omega_GeV, gamma, observation_xy_fm,
    nucleon_positions_fm, proton_mask, *, beam_direction=1,
)
# -> complex (P, 2), fm^-1; analytic point-proton kernel initially
```

A later finite-proton option can supply an independent electromagnetic form
factor and a reusable radial-kernel table. Centralize its parameters in
`parameters.py`; do not infer them from `HotspotParameters`.

Then add a small kinematics helper for $(y,M_V,\sqrt{s_{NN}})$ to
$(\omega_+,\omega_-,W_+,W_-,x_+,x_-)$. A separate `upc.py` can assemble the two
field–target histories and their translation phases. `ensemble.py` already
accepts extra non-event axes, so its moment calculation can be reused on
`(E, P, K, 2)` arrays, summing polarization components after the moments.
Stream/chunk B samples instead of retaining the whole event tensor when large.

Implement in this order:

1. Point-charge field and its flux; validate against its analytic limits and
   integrated sharp-cut expression.
2. Sum sampled proton fields with the same output convention. Co-located point
   protons must reproduce case 1 exactly. Check transverse rotation covariance
   and coherent field summation. Keep a non-spherical event to verify that
   different observation directions give different fields.
3. Select emitter treatment explicitly: fixed event, coherent ensemble-mean
   emitter, or azimuthally averaged event. Never average flux when an average
   amplitude is intended.
4. Fold one direction with existing gamma–target spectra and a sharp B cut.
   This is the first end-to-end normalization control.
5. Combine two histories at amplitude level with a common polarization and
   phase convention; require identical smooth ions at y=0 to cancel in the
   coherent pT-to-zero limit. Integrate the relative B angle where required.
6. Add event-pair survival and a clearly specified breakup observable; only
   then pursue finite-proton refinements and the resolved photon-kT calculation.

Use a B grid that resolves the lower cut and extends past the relevant
$1/a=\gamma\beta h/\omega$ scale. Check convergence in B range and quadrature,
especially at the lowest photon energy. A nuclear-sized FFT box is usually
far too small for the long-range EPA tail.

**Immediate task:** implement `point_charge_epa_field` first. It has one
closed-form expression, fixes the units and polarization convention, and
becomes the single-proton building block for the event-shaped calculation.
