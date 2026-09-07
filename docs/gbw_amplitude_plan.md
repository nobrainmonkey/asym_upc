# From your geometry events to a dipole amplitude

GBW learning notes and integration plan | asym_upc | 7 September 2026

## 1. What you are building

The next object is one complex photon-target amplitude for one frozen target event and one momentum transfer. Your geometry already provides the spatial distribution. GBW supplies a dipole-size-dependent interaction; a photon-meson overlap says how strongly each dipole contributes. You will implement these pieces yourself. The new amplitude module provides two GBW entry points for you to fill in.

<!-- figure: pipeline -->

In the dipole picture a high-energy photon fluctuates into a quark-antiquark pair, the pair scatters from the target, and the outgoing pair is projected onto a vector meson. The pair has a transverse separation and a longitudinal momentum sharing. Integrating over these unobserved variables adds their amplitudes. Squaring comes after this sum. See [Kowalski, Motyka and Watt, Section 2](https://arxiv.org/pdf/hep-ph/0606272).

## The coordinates have different jobs

| Symbol | Meaning | Units |
| --- | --- | --- |
| $\mathbf R_k=(\mathbf b_k,z_{\rm geo,k})$ | A nucleon or hotspot center from your event | fm |
| $\mathbf b$ | Dipole midpoint relative to the target origin | fm |
| $\mathbf r$ | Transverse quark-antiquark separation | fm |
| $\zeta$ | Quark light-cone momentum fraction, between 0 and 1 | dimensionless |
| $\boldsymbol\Delta$, $q=|\boldsymbol\Delta|$ | Momentum transferred to the target; $|t|\simeq q^2$ | GeV |
| $Q^2$ | Incoming photon virtuality; initially zero | GeV$^2$ |

Your geometric $z_{\rm geo}$ has already been integrated to make the thickness. The later integral over $\zeta$ is a different physical operation. A dipole separation is also independent of the position of any nucleon. In later UPC work the ion-ion separation will be called $\mathbf B$, distinct from the target coordinate $\mathbf b$.

This plan starts with the usual scalar transverse overlap and a response depending on $r=|\mathbf r|$, not its orientation. Additional polarization or dipole-orientation dependence would require retaining more angular structure.

<!-- page -->

# 2. Feed both event modes into one thickness

Both geometry samplers return positions with shape `(E, A, 3)` and proton masks with shape `(E, A)`. Select one event to obtain `(A, 3)`. For oxygen, `A = 16`. The strong target response uses every nucleon; the proton mask is retained for the later electromagnetic emitter.

| Target mode | Inputs to the existing thickness function | Integral |
| --- | --- | --- |
| Nucleons | `centers = positions[e]`, unit weights `(A,)`, explicit nucleon standard deviation | $A$ |
| Hotspots | Sample once from `positions[e]` at fixed `bj_x`; centers `(M,3)`, weights `(M,)`, hotspot standard deviation | $A$ |

Use `upcmodel.geometry.position_to_gaussian_thickness(b_grid_fm, centers_fm, sigma_fm, weights)`. The evaluation grid is `(P,2)` and the result is `(P,)`, in inverse square femtometers. No new event class or separate GBW mode is needed.

$$
T_{\cal C}(\mathbf b)=\int_{-\infty}^{\infty} dz_{\rm geo}\,\rho_{\cal C}(\mathbf b,z_{\rm geo})
$$

$$
T_{\cal C}(\mathbf b)=\sum_{k=1}^{M}\frac{w_k}{2\pi\sigma^2}\exp\!\left[-\frac{|\mathbf b-\mathbf b_k|^2}{2\sigma^2}\right],\qquad \int d^2b\,T_{\cal C}=\sum_k w_k=A
$$

In hotspot mode, each parent's weights sum to one because each of its hotspots has weight $1/N_{{\rm hs},i}$. Increasing the number of hotspots changes the distribution of strength, rather than increasing the total nucleon count. Keep the sampled centers and weights fixed for every integration point in that event. Resampling inside an integral would replace a fixed configuration by noise and change its fluctuations.

## Choose the smooth width explicitly

With the current parameters, hotspot centers have standard deviation 0.430064 fm and each hotspot profile has standard deviation 0.176495 fm. Averaging many hotspot samples at fixed parent positions convolves these Gaussians: the resulting nucleon variance is $B_p+B_{\rm hs}$, giving 0.464872 fm. This is the smooth comparison used in your geometry diagnostics.

The smooth profile of Eq. (13) in the [hotspot reference](https://arxiv.org/html/2312.11320v2) instead uses variance $B_p$. Either comparison can be studied, but label the choice. For the first hotspot versus non-hotspot debug comparison, use the matched mean width so width changes do not obscure the effect of fluctuations. Keep the GBW normalization parameter separate from this debug choice.

<!-- page -->

# 3. GBW: your immediate implementation

Write a function for the saturation scale and then one for the dimensionless dipole interaction. Let $h=\hbar c=0.1973269804$ GeV fm and $s=r_{\rm fm}/h$, so $s$ is in GeV$^{-1}$.

$$
Q_s^2(x)=Q_0^2\left(\frac{x_0}{x}\right)^{\lambda_{\rm GBW}},\qquad u=\frac{s^2 Q_s^2(x)}{4}
$$

$$
N_{\rm GBW}(x,r)=1-e^{-u},\qquad \sigma_{\rm dip}^{p}(x,r)=\sigma_0 N_{\rm GBW}(x,r)
$$

This is the saturation ansatz of [Golec-Biernat and Wuesthoff, Section 2](https://arxiv.org/pdf/hep-ph/9807513). A small color-neutral pair interacts weakly: expanding the exponential gives $N\simeq u\propto r^2$, called color transparency. For a large pair, $N$ tends to one and the proton cross section tends to $\sigma_0$. Decreasing $x$ increases $Q_s^2$ for positive $\lambda_{\rm GBW}$, moving the turnover to smaller dipoles. GBW alone has no event geometry or impact parameter.

<!-- figure: saturation -->

Use explicit parameters initially. The hotspot reference's baseline is $Q_0^2=1$ GeV$^2$, $x_0=2\times10^{-4}$, $\lambda_{\rm GBW}=0.21$; this is not the original GBW fit. At $x=x_0$, the dipole size $r=2h=0.394654$ fm gives $u=1$ and $N=0.63212056$. Treat these as learning checks, not a completed rho fit. [Reference, Section 4](https://arxiv.org/html/2312.11320v2).

The scaffold's `gbw_saturation_scale_squared` accepts scalar or array `bj_x` and preserves its shape. `gbw_dipole_amplitude` starts with scalar `bj_x` and scalar or array `r_fm`, returning the shape of `r_fm`. It takes no event. Check finite positive scales and $0<x\leq1$, nonnegative radii, $N(x,0)=0$, small-radius quadratic growth, and saturation. Small $x$ is the physical application range; the algebraic domain is wider.

Numerically, `expm1` avoids subtracting nearly equal numbers at very small $u$. Use the existing `HBARC_GEV_FM`. Do not add a second unit constant.

<!-- page -->

# 4. Combine GBW with the finite nucleus

GBW provides an area, $\sigma_{\rm dip}^{p}$. Your thickness provides nucleons per area. Their product measures local interaction strength. For this project, use the finite-$A$ thickness prescription of [Eq. (9) in the hotspot reference](https://arxiv.org/html/2312.11320v2):

$$
\kappa_{\cal C}(r,\mathbf b)=\frac{\sigma_{0,{\rm fm}^2} N_{\rm GBW}(x,r)T_{\cal C}(\mathbf b)}{2A}
$$

$$
D_{\cal C}(x,r,\mathbf b)\equiv\frac{d\sigma_{\rm dip}^{A}}{d^2b}=2\left[1-(1-\kappa_{\cal C})^A\right]=2N_{A,\cal C}
$$

Here $D$ is dimensionless and includes the factor two. Use $A=16$, not the variable total number of hotspots $M$. Do not divide the current thickness by $A$ and also retain the $A$ in the denominator: the formula already accounts for its normalization.

## What the power means, and its limitation

In a binomial thickness approximation, $1-\kappa$ is a single effective survival factor and $A$ such factors multiply. Subtract the survival amplitude from one and multiply by two to obtain the dipole cross-section density. Applying this expression to your fluctuating event thickness is the chosen phenomenological prescription. It is not an exact derivation for a correlated alpha-cluster configuration or an explicit product over its individual nucleon profiles. The latter would be a different model and must be compared separately.

$$
D=\sigma_0 NT-\frac{A-1}{4A}(\sigma_0 NT)^2+\cdots
$$

The first term is additive scattering. The following terms screen overlapping strength. Thus a hotspot's larger density does not translate linearly into a proportionally larger response. For $A\to\infty$ at fixed $\sigma_0NT$, the mathematical limit is $2[1-\exp(-\sigma_0NT/2)]$. Keep the finite power for the oxygen baseline.

The intended absorptive branch has $0\leq\kappa\leq1$. Check the maximum for the sampled profiles and dipole grid. Narrow peaks can violate it; an integer power still returns a number, but that does not validate the model there. Report a violation and revisit the prescription or parameters. Clipping or rejecting those events would change the model or ensemble.

For the reference choice $\sigma_0=4\pi B_p$, the current `B_p_GeV_m2 = 4.75` gives $\sigma_0=2.324216$ fm$^2$ = 23.24216 mb. A variance converts as $B_{p,{\rm fm}^2}=h^2 B_{p,{\rm GeV}^{-2}}$. Both sigma-zero and the thickness must use compatible area units.

<!-- page -->

# 5. Weight dipoles with the photon-meson overlap

Define $\Omega_T=(\Psi_V^*\Psi_\gamma)_T$. Use the transverse Gaus-LC convention of [Kowalski, Motyka and Watt, Eqs. (21), (29), Table 1](https://arxiv.org/pdf/hep-ph/0606272). All radial arguments and derivatives on this page use $s=r_{\rm fm}/h$, in GeV$^{-1}$.

$$
\epsilon^2=\zeta(1-\zeta)Q^2+m_f^2,\qquad \phi_T(s,\zeta)=N_T[\zeta(1-\zeta)]^2 e^{-s^2/(2R_T^2)}
$$

$$
\partial_s\phi_T=-\frac{s}{R_T^2}\phi_T
$$

$$
\Omega_T=\frac{e\widehat e_f N_c}{\pi\zeta(1-\zeta)}\left[m_f^2 K_0(\epsilon s)\phi_T-[\zeta^2+(1-\zeta)^2]\epsilon K_1(\epsilon s)\partial_s\phi_T\right]
$$

Here $K_0,K_1$ are modified Bessel functions, $e=\sqrt{4\pi\alpha_{\rm em}}$, and $N_c=3$. For rho, the tabulated set is $m_f=0.14$ GeV, $N_T=4.47$, $R_T^2=21.9$ GeV$^{-2}$ and $\widehat e_f=1/\sqrt{2}$. The overlap has units GeV$^2$. The scalar profile is dimensionless in this convention.

Our starting real-photon calculation sets $Q^2=0$, hence $\epsilon=m_f$. This light-quark mass is an effective wavefunction parameter. The model's rho predictions remain sensitive to long-distance dipoles; convergence of an integral does not remove that phenomenological dependence.

## How to read the expression

For a selected $(s,\zeta)$, the photon and meson wavefunctions describe the same pair. Their product tells how much that pair contributes to the transition. The mass and derivative terms encode its spin structure. It is an amplitude overlap, not a probability density to normalize to one on a numerical grid.

The derivative term is especially easy to mishandle during unit conversion. If you differentiate a profile written in femtometers, then $\partial_s\phi=h\,\partial_{r_{\rm fm}}\phi$. A factor of $h$ belongs in the derivative as well as in Bessel arguments and integration measures.

Implement the scalar profile, its derivative, then the overlap. Compare the derivative with a finite difference away from zero. Use interior quadrature nodes for $0<\zeta<1$ and $s>0$. Evaluating separate $K_0(0)$ or $K_1(0)$ terms gives singular intermediate values even though the amplitude's weighted small-$s$ limit is integrable. The integration convention on the next pages keeps $d\zeta/(4\pi)$ outside this overlap.

<!-- page -->

# 6. Where the event amplitude comes from

Write this page in natural units, with $\mathbf s=\mathbf r_{\rm fm}/h$ and $\boldsymbol\beta=\mathbf b_{\rm fm}/h$. The event amplitude is

$$
\mathcal A_{\cal C}(\boldsymbol\Delta)=i\int d^2s\int_0^1\frac{d\zeta}{4\pi}\,\Omega_T(s,\zeta,Q^2)\int d^2\beta\,e^{-i[\boldsymbol\beta-(1/2-\zeta)\mathbf s]\cdot\boldsymbol\Delta}D_{\cal C}(x,hs,h\boldsymbol\beta).
$$

This adopts the normalization of [Kowalski, Motyka and Watt, Section 2](https://arxiv.org/pdf/hep-ph/0606272), with the midpoint phase used in [the hotspot reference, Eq. (2)](https://arxiv.org/html/2312.11320v2). The arguments $hs$ and $h\boldsymbol\beta$ tell you where the existing femtometer geometry enters.

## Three ingredients in this sum

The overlap weights the unobserved dipole. The profile $D=2(1-S)$ is its interaction with the frozen target. The exponential carries the phase of scattering at different transverse positions into a specified final momentum transfer. The leading $i$ is the convention for the dominantly absorptive scattering amplitude. No absolute square is taken anywhere inside this expression.

## Derive the midpoint phase

Put the quark at $\mathbf b+\mathbf r/2$ and the antiquark at $\mathbf b-\mathbf r/2$. Their longitudinal-momentum-weighted transverse coordinate is

$$
\mathbf R_\zeta=\zeta(\mathbf b+\mathbf r/2)+(1-\zeta)(\mathbf b-\mathbf r/2)=\mathbf b-(1/2-\zeta)\mathbf r.
$$

The Fourier factor is $\exp[-i\mathbf R_\zeta\cdot\boldsymbol\Delta/h]$. This also respects simultaneous interchange $\zeta\to1-\zeta$ and $\mathbf r\to-\mathbf r$. The midpoint shift is discussed in [Hatta, Xiao and Yuan, Section II](https://arxiv.org/html/1703.02085). Literature using a different definition of the impact coordinate may display a different shift; carry the coordinate definition and phase together.

## Unit audit

The two transverse measures each have units GeV$^{-2}$, the overlap has GeV$^2$, and $D$ and the phase are dimensionless. Therefore $\mathcal A$ has GeV$^{-2}$. If both numerical integration grids are in fm, each area measure contributes its own factor $h^{-2}$. Converting only the phase is insufficient.

GBW saturation makes $D(r,\mathbf b)$ nonlinear in $N(r)T(\mathbf b)$. It cannot generally be separated into one function of $r$ times one function of $\mathbf b$. This is why the legacy scalar overlap factor cannot simply multiply its old geometry transform.

<!-- page -->

# 7. Reduce the integrals without erasing the event

For our scalar overlap and orientation-independent dipole response, the dipole angle can be integrated exactly, even when the target event is not circular. Factor the phase into an impact-coordinate part and a dipole part. With $q=|\boldsymbol\Delta|$,

$$
\int_0^{2\pi}d\theta_r\,e^{i(1/2-\zeta)r q\cos\theta_r/h}=2\pi J_0\!\left((1/2-\zeta)r q/h\right).
$$

Define the event's impact-parameter transform using your femtometer grid:

$$
F_{\cal C}(r,\boldsymbol\Delta)=\frac{1}{h^2}\int d^2b\,e^{-i\mathbf b\cdot\boldsymbol\Delta/h}D_{\cal C}(x,r,\mathbf b).
$$

The overlap and the remaining dipole phase give an event-independent function:

$$
H(r,q)=\int_0^1 d\zeta\,\Omega_T(r/h,\zeta,Q^2)J_0\!\left((1/2-\zeta)r q/h\right).
$$

Combining $2\pi$ from the angle with $1/(4\pi)$ in the original measure leaves

$$
\boxed{\mathcal A_{\cal C}(\boldsymbol\Delta)=\frac{i}{2h^2}\int_0^\infty r\,dr\,H(r,q)F_{\cal C}(r,\boldsymbol\Delta).}
$$

These equations are a direct reduction of the full expression on page 6. The radial factor $r$ is the Jacobian of $d^2r=r\,dr\,d\theta_r$; it is not an optional weighting choice. The factor two is already in $D$, so do not insert another one here.

At $q=0$, $J_0=1$ and every impact-coordinate phase is one. The transform $F$ is then real and nonnegative on the intended branch. The event amplitude is purely imaginary in this convention. At nonzero momentum transfer, an asymmetric event generally has a complex transform and must keep both components.

This removes the angle of the dipole, not the angle of the target. Replacing $F$ by a radial Hankel transform of an azimuthally averaged event thickness would erase orientation information and alter the fluctuations. A circular Gaussian control may use that shortcut as an analytic test; a sampled oxygen event should retain its two-dimensional profile.

<!-- page -->

# 8. A concrete array and integration plan

Use scalar $x$ and $Q^2$ for one run. Let $R$ count radial nodes, $Z$ count momentum-sharing nodes, $P$ count impact-coordinate points, and $K$ count requested momentum-transfer vectors. These letters describe array dimensions; $Z$ here does not mean proton number.

| Object | Shape | Units |
| --- | --- | --- |
| `b_grid_fm`, `b_area_weights_fm2` | `(P,2)`, `(P,)` | fm, fm$^2$ |
| `r_fm`, `r_weights_fm` | `(R,)`, `(R,)` | fm, fm |
| `zeta`, `zeta_weights` | `(Z,)`, `(Z,)` | dimensionless |
| `delta_GeV`, `q_GeV` | `(K,2)`, `(K,)` | GeV |
| `thickness`, `N_GBW` | `(P,)`, `(R,)` | fm$^{-2}$, dimensionless |
| `D`, `phase` | `(R,P)`, `(P,K)` | dimensionless |
| `Omega`, `H`, `F` | `(R,Z)`, `(R,K)`, `(R,K)` | GeV$^2$, GeV$^2$, GeV$^{-2}$ |
| `event_amplitude`, `samples` | `(K,)`, `(E,K)` | complex GeV$^{-2}$ |

The broadcast `N[:,None] * thickness[None,:]` builds all $(r,b)$ combinations. It is not elementwise pairing of equally numbered points. The following discrete sums define an auditable first implementation:

$$
F_{a k}\simeq h^{-2}\sum_{p=1}^{P}v_p D_{a p}\exp[-i\mathbf b_p\cdot\boldsymbol\Delta_k/h]
$$

$$
H_{a k}\simeq\sum_{j=1}^{Z}w_j\Omega_{a j}J_0\!\left[(1/2-\zeta_j)r_a q_k/h\right]
$$

$$
\mathcal A_k\simeq\frac{i}{2h^2}\sum_{a=1}^{R}u_a r_a H_{a k}F_{a k}
$$

Here $v_p$, $w_j$, $u_a$ are area, momentum-fraction, and radial quadrature weights. Precompute $N$, $H$, and the Fourier phases outside the event loop. For each event, build sources once, evaluate thickness once, then form and transform $D$. Store only the final complex amplitude array if diagnostic fields are unnecessary. Chunk the radial or momentum dimensions when the intermediate arrays become large.

Use a direct weighted Fourier sum as the reference implementation. It accepts arbitrary momentum vectors and makes origin and area factors explicit. FFT or projected transforms can be checked against it later.

<!-- page -->

# 9. Numerical checks that test the physics

## Start with controlled integrals

Integrate the thickness on the actual impact grid and check that it recovers $A$. Choose a window covering all sampled transverse centers plus Gaussian tails, then enlarge it. Resolve the smallest width; a spacing around $\sigma_{\rm hs}/4$ is a starting point to refine, not an accuracy guarantee. Finite grids can otherwise mimic changes in normalization or high-momentum structure.

For the radial and momentum-sharing integrals, interior Gauss-Legendre nodes on $(0,r_{\max})$ and $(0,1)$ are a straightforward first reference. Refine the node counts and extend $r_{\max}$ independently. If later using a logarithmic radial variable $\ell=\log r$, the measure is $r\,dr=r^2d\ell$, not $r\,d\ell$. Light mesons require an explicit large-dipole tail check.

## An analytic geometry test using your own Gaussians

In the dilute limit, $D\simeq\sigma_0 N T$. For your weighted Gaussian sources, direct integration gives

$$
F_{\rm dilute}(r,\boldsymbol\Delta)=\frac{\sigma_{0,{\rm fm}^2}}{h^2}N(x,r)\sum_k w_k e^{-\sigma^2q^2/(2h^2)}e^{-i\mathbf b_k\cdot\boldsymbol\Delta/h}.
$$

At zero transfer this becomes $\sigma_{0,{\rm fm}^2}NA/h^2$. This single check probes the source weights, Gaussian normalization, Fourier sign and units. Translate all centers by a known transverse vector $\mathbf d$; the transform must gain the phase $e^{-i\mathbf d\cdot\boldsymbol\Delta/h}$. For real profiles, $F(-\boldsymbol\Delta)=F(\boldsymbol\Delta)^*$ and, with the leading $i$, $\mathcal A(-\boldsymbol\Delta)=-\mathcal A(\boldsymbol\Delta)^*$.

## Reuse the old acceleration idea carefully

If every requested vector is $(q,0)$, first sum each $D(r,b_x,b_y)$ over $b_y$, then Fourier-transform the resulting function of $b_x$. This is a Cartesian projection and retains the event's asymmetry. It generalizes the old helper, but now the projected profile depends on $r$.

For an FFT on a uniform fm grid, the momentum nodes are $2\pi h$ times `fftfreq`; the area factor is $db^2/h^2$. Verify array ordering, the origin phase and shifts against the direct sum. A circular average of amplitudes is not a substitute for averaging angle-dependent cross sections.

Record convergence of the complex amplitude before forming observables. Near diffraction zeros, use absolute differences as well as relative errors. Increase event statistics separately from quadrature resolution so sampling noise and integration error are distinguishable.

<!-- page -->

# 10. From one event to observable spectra

Calculate an amplitude for each complete nucleon-plus-hotspot configuration before averaging. For equal event probabilities, let $\mu=\langle\mathcal A\rangle$ and $m_2=\langle|\mathcal A|^2\rangle$. The coherent and incoherent definitions are [Eqs. (1) and (3) of the hotspot reference](https://arxiv.org/html/2312.11320v2):

$$
\frac{d\sigma_{\rm coh}}{dt}=\frac{|\mu|^2}{16\pi},\qquad \frac{d\sigma_{\rm inc}}{dt}=\frac{m_2-|\mu|^2}{16\pi}.
$$

For the numerical sample, use the stable population-moment expression:

$$
m_2-|\mu|^2=\frac{1}{E}\sum_{e=1}^{E}|\mathcal A_e-\mu|^2.
$$

An unbiased finite-sample variance uses $E-1$ instead, so report which estimator is used and study event-count dependence. Duplicating one event gives zero population variance; averaging its absolute amplitude first does not compute the coherent amplitude.

The result has units GeV$^{-4}$. Multiplying by 0.38937937 converts it to mb/GeV$^2$. At a fixed magnitude $q$, an isotropically sampled ensemble permits a fixed direction for $\boldsymbol\Delta$. If analyzing orientations, average the appropriate cross sections over angle, preserving amplitudes until the required moments have been formed.

## Why hotspot resolution matters after geometry is normalized

With the matched smooth width, the average hotspot thickness at fixed parents equals the smooth thickness. But averaging a nonlinear response generally gives a different result from responding to the average thickness. The same applies to its Fourier amplitude and variance. Both target modes therefore share an interface while remaining distinct physics ensembles.

## Continue to UPC only after the target calculation converges

First validate one photon-target channel at fixed $x$. Later use $x\simeq(Q^2+M_V^2)/(Q^2+W^2)$, where $W$ is the photon-nucleon energy used to define the nucleon Bjorken variable, with the stated high-energy, small-transfer approximation. Do not substitute the total photon-nucleus energy. The target Bjorken variable, the EPA photon momentum fraction, and the photon virtuality are separate inputs. Corrections involving $t$ need an explicit kinematic convention.

Then build the preaveraged electromagnetic emitter from protons, combine the two emission paths with consistent translation and parity phases, and square their sum at the required event-pair level. Any later no-hadronic-interaction prescription needs a documented ensemble measure: the handoff proposes applying the same survival amplitude $\sqrt{P_{\rm nohad}}$ to both paths before moments. That is not automatically the same as conditioning and renormalizing a probability-weighted ensemble. Derive that choice before coding it; never scale only the final incoherent spectrum.

Real-part and skewness corrections, photon fluxes, and event-pair survival remain subsequent physics choices, not hidden factors in the GBW function.

<!-- page -->

# 11. What to reuse, and what the old code changes

This audit refers to the current `upcmodel` files and the legacy code under `handoff-please git ignore this- for claude/Asymmetric-UPC-Pheno-Paper-main/`. The existing geometry and notebooks were not edited for this plan.

| Existing component | How it enters the new calculation |
| --- | --- |
| `sample_3pf_events`, `sample_tetrahedron_nucleon_events` | Keep their `(E,A,3)` positions and `(E,A)` masks. Geometry model selection stays outside the amplitude. |
| `sample_target_hotspots` | Reuse positions, weights and widths; draw once at the run's target x for each configuration. |
| `position_to_gaussian_thickness` | Reuse directly in both modes. The sampled event needs this discrete sum; do not replace it with a smooth Fermi density. |
| `HBARC_GEV_FM`, hotspot width properties | Reuse for all conversions. The current `B_p` fields are variances, whereas old `constants.B_p` is a length. |
| Old `EventAmplitudeCalculator` | Reuse the idea of Cartesian projection and direct phases. Replace its radius-independent opacity and scalar overlap prefactor. |
| Old ensemble moments | Reuse the complex mean and mean squared magnitude idea with the current arrays. Preserve the full configuration's fluctuations. |

## The old overlap is not the tabulated transverse Gaus-LC form

Legacy `gausLC` uses a single power of $\zeta(1-\zeta)$ and $\exp(-s^2/R^2)$. Its derivative $-2s\phi/R^2$ is mathematically correct for that expression. The mismatch arises when using the tabulated transverse parameters with that scalar form, and when omitting the external $1/[\zeta(1-\zeta)]$ in the corresponding overlap. Adopt the complete convention on page 5 together; changing only the derivative would leave an inconsistent combination.

Legacy `e_f` combines `k_perp` and `x` into an effective virtuality. Our real-photon starting point specifies $Q^2=0$ independently. Transfer momentum is not a replacement for photon virtuality. The old production profile also lacks the GBW radius dependence, so its completed overlap integral cannot be carried over as one number.

## A parameter choice to keep visible

Current `RHO_HS.B_p_GeV_m2` is 4.75. The hotspot paper uses 6.4 for rho in Section 4; changing it also changes the center distribution and, if linked, $\sigma_0=4\pi B_p$. Its calculations use a boosted Gaussian overlap and $Q^2=0.05$ GeV$^2$. Our selected Gaus-LC, $Q^2=0$ starting point is therefore a deliberate model combination, not a direct reproduction of all that paper's rho predictions. Keep 4.75 for continuity in initial debug runs and compare the rho-specific set as a separate documented choice before phenomenology. [Reference](https://arxiv.org/html/2312.11320v2).

<!-- page -->

# 12. What to implement next

## Your next task: only the two GBW functions

Open `upcmodel/amplitudes.py`. Implement `gbw_saturation_scale_squared`, then `gbw_dipole_amplitude`, using the formulas on page 3. Complete the function bodies yourself; the imports and unit constant are ready. Begin with scalar x and a vector of dipole radii. No target event is required for this first step.

Check $Q_s^2(x_0)=Q_0^2$, $N(x,0)=0$, the analytic $u=1$ value, the small-$r$ ratio $N/u\to1$, and approach to one at large radius. Verify shape preservation and invalid inputs in the same pass. Once these are right, the next physics task is the finite-$A$ profile, not the whole amplitude integrator.

| Stage | New responsibility | Completion evidence |
| --- | --- | --- |
| 1. GBW | Saturation scale and dipole strength | Limits, units, numeric anchor and shapes |
| 2. Nuclear response | `(R,)` and `(P,)` to `(R,P)` | Dilute expansion, factor two, finite-A power and valid domain |
| 3. Gaus-LC overlap | Scalar profile, derivative, overlap | Finite difference, selected normalization convention and endpoint behavior |
| 4. One-event integral | Direct Fourier sum, H, radial sum | Analytic Gaussian control, phase tests and quadrature convergence |
| 5. Ensembles | Complex samples to coherent/incoherent spectra | Repeated-event control and event-count convergence |
| 6. UPC extensions | Emitter, two paths, survival convention | Channel and phase checks before comparing physical spectra |

After these stages, alpha versus 3pF is an event-source switch. A classical mixture samples the chosen geometry with its probability and uses the resulting ensemble moments. It does not add coherent nuclear state wavefunctions. The same array interface can later accommodate a different mass number.

<!-- page -->

# 13. References and conventions

These primary sources supply complementary parts of the calculation. The amplitude measure, overlap normalization, transverse-coordinate convention and parameter set must agree with each other when combining them.

## Primary references

[Golec-Biernat and Wuesthoff, hep-ph/9807513](https://arxiv.org/pdf/hep-ph/9807513): saturation ansatz, Section 2. The original fit parameters should not be silently mixed with a later hotspot fit.

[Kowalski, Motyka and Watt, hep-ph/0606272](https://arxiv.org/pdf/hep-ph/0606272): amplitude convention and wavefunction measure, Section 2; transverse overlap Eq. (21); Gaus-LC Eq. (29), Table 1.

[Cepila, Contreras, Matas and Ridzikova, 2312.11320v2](https://arxiv.org/html/2312.11320v2): finite-A response Eq. (9), event profile Eqs. (10)-(14), normalization and parameter choices in Section 4.

[Hatta, Xiao and Yuan, 1703.02085](https://arxiv.org/html/1703.02085): midpoint nonforward phase, Section II.

The source-to-array mapping, unit audit, discrete sums, Gaussian transform check and implementation sequence in these notes are derived for the current repository. This document specifies the calculation to build; it does not report a completed GBW or amplitude implementation.
