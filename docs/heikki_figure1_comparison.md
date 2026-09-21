# GBW proton comparison with Mäntysaari and Schenke Figure 1

The overlay compares the existing GBW proton implementation with the actual
published IPsat curves in [arXiv:1603.04349v2, Figure 1](https://arxiv.org/html/1603.04349v2).
It uses **J/psi photoproduction at W=100 GeV, Q²=0** on both sides. Notebook
03 now calculates J/psi photoproduction at W=75 GeV, followed by a rho
comparison at W=44 GeV. This separate Figure 1 overlay matches the published
J/psi energy explicitly.

Run from the project directory:

```sh
.venv/bin/python notebooks/compare_heikki_figure1.py
```

Outputs are `notebooks/gamma_p_outputs/jpsi_heikki_figure1_overlay.png`,
`output/pdf/jpsi_heikki_figure1_overlay.pdf`, a CSV of the GBW spectra and a
JSON file recording the calculation settings. The extraction provenance is
in `notebooks/data/heikki_figure1_README.md`. No upcmodel implementation is
changed by this comparison.

## What is matched, and what remains different

The GBW calculation reuses the existing amplitude, overlap, skewness and
spectrum functions, with the existing `JPSI_HS` preset. For J/psi, the
existing Gaus-LC class is instantiated using [Kowalski, Motyka and Watt,
hep-ph/0606272 Table 1](https://arxiv.org/html/hep-ph/0606272):
mf=1.4 GeV, NT=1.23, RT²=6.5 GeV^-2, effective charge=2/3.

| Ingredient | This repo's calculation | Published Figure 1 |
|---|---|---|
| Dipole profile | sigma0 N_GBW(r,x) T(b) | 2[1-exp(-r² F(r,x) T(b))] |
| Geometry | existing JPSI_HS, normalized random hotspot count | three normalized Gaussian constituent sources |
| Widths, GeV^-2 | Bp=4.75, Bhs=0.8 | Bqc=3.5, Bq=0.5 or 1; Bqc=1, Bq=3; smooth Bp=4 |
| Meson overlap | existing Gaus-LC | boosted Gaussian |
| Dipole phase as implemented/published | midpoint (1/2-z) | paper uses (1-z) |
| Corrections | common forward-amplitude Rg; no real-part factor | paper's gluon-density skewness prescription and real-part correction |

The comparison retains each model's own choices. It is not a controlled
replacement of only the dipole kernel. In particular, shifting the dipole
coordinate convention requires transforming the profile consistently; the
phase is not an independent interchangeable prefactor for a generic model.
No normalization or width was fitted to the extracted curves.

The repo uses x=M²/W²=0.0009591409, with M=3.097 GeV. The count-distribution
parameter is 8.69075, E[Nhs]=8.69221, E[1/Nhs]=0.133126. The frozen forward
slope gives lambda=0.151818 and Rg=1.137472.

## Results

The GBW incoherent spectrum peaks at |t| approximately 0.385 GeV² with
dσ/dt approximately 37.75 nb/GeV². The published blue and red curves peak
at the plotted vertices 0.45 and 0.35 GeV², at approximately 36.76 and
36.07 nb/GeV². Their initial rise is therefore shared by this calculation.

| abs(t), GeV² | GBW incoherent | Figure 1 blue, Bq=0.5 | Figure 1 red, Bq=1 |
|---:|---:|---:|---:|
| 0.10 | 21.97 | 19.56 | 22.27 |
| 0.50 | 36.78 | 36.56 | 33.81 |
| 1.00 | 25.69 | 27.97 | 19.83 |
| 2.00 | 10.61 | 12.35 | 5.14 |

All cross sections in this table are nb/GeV². Values between published
vertices use interpolation on the logarithmic ordinate.

The coherent curves do not agree equally well: at |t|=0.05 GeV², GBW gives
359.97 nb/GeV² versus 175.30 for the blue IPsat curve and 206.22 for the red
curve. The GBW coherent spectrum subsequently falls faster. Similarity of
the incoherent curves alone does not establish a simultaneous fit, and it
does not resolve the earlier rho-data discrepancy.

## How the geometry average was evaluated and checked

For the current *factorized* Gaussian proton, the event amplitude is

\[
 A_\omega(\Delta)=C(\Delta)e^{-B_{hs}|t|/2}
 \frac1{n_\omega}\sum_i e^{-i b_{i,x}\Delta/\hbar c}.
\]

Here C is computed using the existing GBW, Gaus-LC and radial-amplitude
helpers. For independent source centers and normalized weights, its exact
ensemble moments are

\[
 \langle A\rangle=C e^{-(B_p+B_{hs})|t|/2},\qquad
 \operatorname{Var} A=|C|^2 e^{-B_{hs}|t|}
 \left\langle\frac1n\right\rangle(1-e^{-B_p|t|}).
\]

Using these moments removes a finite-event noise floor in the coherent
tail. It is mathematically equivalent to averaging the current sampled
proton events; it is not a new IPsat implementation and does not apply to
the nonlinear nuclear profile.

Executed verification:

- Eight events were evaluated through `sample_target_hotspots`,
  `proton_dipole_profile`, `dipole_profile_fourier_x` and
  `photon_target_amplitude_T`. The analytic event expression agreed with
  the FFT result within 2.89e-13 of the peak amplitude.
- A separate ensemble of 32,768 events from the actual sampler agreed
  with the exact mean and variance. The largest tested variance deviation
  was 1.49 Monte Carlo standard errors, at |t|=0.05 GeV².
- Doubling the r-grid resolution and z-quadrature order changed the
  tested amplitudes by at most 4.07e-5 relative (about 0.0041%).
- All upcmodel Python source hashes remained unchanged.

Detailed validation numbers are in
`notebooks/gamma_p_outputs/jpsi_heikki_validation.json`.

## Elastic, proton dissociative, coherent and incoherent

These names have a precise relation within the Good-Walker framework.
For a proton target, elastic vector-meson production means gamma+p -> V+p
(elastic at the target vertex), and coherent diffraction refers to that
same intact-target channel. Diffractive proton dissociation gamma+p -> V+Y,
Y != p, is the target-incoherent channel when summed over the relevant
diffractive final states. See [Mäntysaari and Schenke, arXiv:1607.01711,
Section II](https://arxiv.org/html/1607.01711#S2).

The distinction comes from quantum final states, not from whether an
interaction occurs at one point or whether an individual event looks
symmetric. Write the incoming proton as a superposition of scattering
eigenstates, |p> = sum_a c_a |a>, with A|a> = A_a|a>. Identifying the model's
configuration weights with |c_a|² gives

\[
 A_{p\to p}=\langle p|\widehat A|p\rangle=\langle A\rangle.
\]

Completeness of the target final states in the diffractive description
then gives

\[
 \sum_Y|\langle Y|\widehat A|p\rangle|^2
 =\langle p|\widehat A^\dagger\widehat A|p\rangle
 =\langle|A|^2\rangle.
\]

Subtracting the intact-proton final state gives

\[
 \sum_{Y\ne p}|A_{p\to Y}|^2
 =\langle|A|^2\rangle-|\langle A\rangle|^2.
\]

The usual differential cross sections follow after division by 16pi
(with any common phenomenological correction applied consistently).
This is why the amplitude variance is related to dissociation; it is not
merely Monte Carlo statistical uncertainty.

There are two qualifications:

1. **Experimental final-state restrictions.** For example, the [H1 rho
   data used in notebook 03](https://link.springer.com/article/10.1140/epjc/s10052-020-08587-3)
   restrict dissociation to mp < MY < 10 GeV. The restricted sum involves
   a projector P_cut onto those final states,
   <p|A† P_cut A|p>, rather than completeness over all Y. The current
   events contain source positions and weights, with no target mass,
   fragmentation or final-state transition amplitudes. They therefore
   cannot implement an MY cut just by selecting geometric events.
   Spatial fluctuations also do not automatically represent all radiative
   or higher-Fock-state contributions to proton dissociation. Agreement
   with mass-restricted data requires an additional physical modeling
   assumption or a mass-resolved treatment.
2. **Which target is intact matters.** In a nucleus, coherent means the
   *whole nucleus* remains in the same internal state. A process that is
   elastic on an individual nucleon can still excite or break the nucleus
   and contribute to nuclear incoherent diffraction. Thus nucleon elastic
   and nuclear coherent are not interchangeable labels.

The older HERA points shown in the requested Figure 1 also have differing
definitions. [ZEUS hep-ex/0205081, Table 3](https://arxiv.org/html/hep-ex/0205081)
uses x_J>0.01, with x_J=|t|/(MY²-mp²+|t|). [H1 hep-ex/0306013](https://arxiv.org/html/hep-ex/0306013)
measures total diffraction with an event-elasticity cut z>0.95; its large-t
contribution is dominated by proton dissociation. That z is a final-state
energy fraction, not the quark's light-cone fraction in the overlap.
The extracted theory overlay avoids mixing these data definitions.

Therefore `amplitude_moments` has the appropriate Good-Walker structure.
The model's fluctuation content and the experimental phase space determine
whether its numerical variance describes a particular dissociation dataset.
The terminology itself does not explain away the rho shape disagreement.

## H1 and ZEUS data added to the incoherent comparison

Run:

```sh
.venv/bin/python notebooks/compare_incoherent_h1_zeus.py
```

The new figure is `notebooks/gamma_p_outputs/jpsi_incoherent_h1_zeus.png`
and `output/pdf/jpsi_incoherent_h1_zeus.pdf`. The left panel adds the
original ZEUS 2003 dissociative and H1 2003 total-diffractive points to the
Figure 1 comparison at W=100 GeV. The right panel calculates the same GBW
model at W=75 GeV and adds the H1 2013 proton-dissociative data, including
the small-|t| bins absent from ZEUS. Both panels use J/psi photoproduction.
The original notebook's H1 rho measurement is from 2020 and is a different
process; it is not overlaid on the J/psi prediction.

| Dataset | Published W interval, GeV | Published abs(t) interval, GeV² | Final-state restriction |
|---|---:|---:|---|
| ZEUS 2003 | 80-120 | 1.2-6.5 | proton dissociation, x_J>0.01 |
| H1 2003 | 50-150 | 2-30 | total diffraction, event elasticity z>0.95 |
| H1 2013 | 40-110 | 0-8 | proton dissociation, mp<MY<10 GeV |

For ZEUS, x_J=|t|/(MY²-mp²+|t|), so its cut corresponds to
MY²<mp²+99|t|. This is a t-dependent upper mass limit, unlike H1 2013's
fixed 10 GeV. H1 2003's elasticity cut corresponds approximately to
MY²<0.05 W²-|t|. Its high-t elastic contribution is negligible, as the
theory paper's footnote explicitly explains. These selections probe the
same diffractive breakup process in different phase space. H1 2003 reports
agreement with ZEUS when the H1 analysis is repeated with the ZEUS cuts.

Sources, original data, error conventions and the automated cross-check
against H1 Table 5 are documented in
`notebooks/data/jpsi_dissociation_README.md`. H1 2013 points are taken from
the collaboration's official numerical table, not digitized. Original
H1 2003 values are used in the left panel; the older values rescaled to
the 2013 phase space in the H1 supplemental file are not substituted.

The model has no MY or target fragmentation prediction, so it cannot
reproduce these mass selections directly. Its representative W values
follow the theory literature but do not replace integration over the
experiments' full W intervals. Data are neither energy-rescaled nor fitted.

Some resulting bin averages at W=75 GeV, in nb/GeV²:

| abs(t) bin, GeV² | H1 2013, total error | Current GBW, bin average |
|---:|---:|---:|
| 0-0.20 | 47.3 ± 6.7 | 18.43 |
| 0.20-0.40 | 43.8 ± 6.0 | 33.29 |
| 0.40-0.64 | 36.7 ± 5.1 | 33.25 |
| 0.64-0.93 | 27.8 ± 4.2 | 28.09 |
| 1.31-1.83 | 10.05 ± 1.56 | 14.36 |

All nine bins are retained in
`notebooks/gamma_p_outputs/jpsi_h1_2013_bin_comparison.csv`. The model's
low-t deficit reflects its fixed forward amplitude: its factorized
profile and unit-normalized thickness make A(t=0) identical in every
geometric event, even when the normalized hotspot count fluctuates.
Consequently its forward incoherent spectrum vanishes. The mass-cut
difference alone does not supply the missing amplitude fluctuations.

The H1 J/psi data were already available in 2013. Indeed,
[the longer 2016 paper, arXiv:1607.01711](https://arxiv.org/html/1607.01711),
compares with them in Figure 7 and discusses the small-t deficit from
geometry alone. Figure 10 shows improved agreement after adding
saturation-scale fluctuations. That is evidence for testing additional
fluctuation content; it is not a numerical reproduction of that IPsat
calculation by the present GBW model.

Verification for this addition: the W=100 GeV results at six selected
momenta reproduce the earlier saved spectra exactly; doubling the r-grid
and z-quadrature resolution at W=75 GeV changes the tested amplitudes by
at most 4.07e-5 relative and spectra by 8.13e-5. All upcmodel source hashes
are unchanged. Results are saved in
`notebooks/gamma_p_outputs/jpsi_h1_zeus_validation.json`.
