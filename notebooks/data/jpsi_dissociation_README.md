# HERA proton-dissociative J/psi photoproduction

The current comparison uses three distinct datasets. All entries in
`jpsi_dissociation_hera.csv` are **photon-proton** cross sections in nb/GeV²,
already divided by the photon flux and the J/psi decay branching fraction.
The plotting script does not divide by either factor again.

## H1 2013: the data reaching small |t|

- Paper: [arXiv:1304.5162](https://arxiv.org/html/1304.5162), Eur. Phys. J. C
  73 (2013) 2466, Table 5, high-energy dataset, proton-dissociative rows.
- Numerical source: [H1 official table, version 2](https://www-h1.desy.de/psfiles/figures/d13-058.table_tHE_allH1_v2.txt).
  An unchanged copy is stored as `h1_2013_tHE_official.txt`.
- Use only rows 9-17 of that file. They were checked independently against
  the paper's nine proton-dissociative entries in Table 5.
- Measured range: 40<W<110 GeV, Q²<2.5 GeV², 0<|t|<8 GeV²,
  mp<MY<10 GeV. The figure uses a real-photon calculation at W=75 GeV,
  the representative energy used in arXiv:1607.01711 Figures 7 and 10.
- `err_minus` and `err_plus` both contain H1's total uncertainty. The full
  correlated-error decomposition is retained in the original text file.
  It is not appropriate to infer a correlated chi-square from the excerpt
  alone without constructing that covariance matrix.
- Rows 18-27 of the official file are the older high-t H1 measurement
  **after adjustment to the 2013 phase space**. They are not the original
  points in Heikki's 2016 Figure 1 and are not reused as such here.

## ZEUS 2003: the points labeled incoherent in Figure 1

- Paper: [hep-ex/0205081](https://arxiv.org/html/hep-ex/0205081), Eur. Phys.
  J. C 26 (2003) 389. Preprint appeared in 2002; data were collected in
  1996-97. All six J/psi rows are taken from Table 3.
- Range: 80<W<120 GeV, tagged photoproduction Q²<0.02 GeV²,
  1.2<|t|<6.5 GeV², x_J>0.01. The paper calls x_J simply x; it is the
  struck-parton fraction and is **not** the small-x argument in GBW.
- The asymmetric bars combine statistical, systematic and dissociation-
  model uncertainties in quadrature. The additional 10% normalization
  uncertainty is stored separately as `extra_norm_fraction=0.10`.
- No extrapolation of these data into |t|<1.2 is made.

## H1 2003: the points labeled total in Figure 1

- Paper: [hep-ex/0306013](https://arxiv.org/html/hep-ex/0306013), Phys. Lett.
  B 568 (2003) 205. All ten rows are from the original Table 1.
- Range: 50<W<150 GeV, photoproduction Q² approximately below 1 GeV²,
  2<|t|<30 GeV², event elasticity z_event>0.95.
- This measurement is total diffraction. The elastic contribution is
  negligible in the displayed high-|t| region, which is why the theory
  paper compares it with the incoherent prediction. The plot retains the
  label **total diffraction** rather than silently relabeling the data.
- Bars combine the two published uncertainties, statistical and systematic,
  in quadrature. At |t|=2.43 the original value is 5.10, whereas the adjusted
  2013 supplement gives 4.753. The Figure 1 comparison uses 5.10.

## What the definitions imply

ZEUS applies

\[
x_J=\frac{|t|}{M_Y^2-m_p^2+|t|}>0.01
\quad\Longrightarrow\quad
M_Y^2<m_p^2+99|t|.
\]

It therefore integrates over a t-dependent dissociation-mass range: for
example, approximately MY<11.6 GeV at |t|=1.34, and MY<14.9 GeV at |t|=2.24.
H1 2013 integrates up to a fixed 10 GeV. H1 2003's elasticity selection
instead gives approximately MY²<0.05 W²-|t|. In that expression z_event is
the vector meson's energy fraction in the proton rest frame, not the
quark's momentum fraction used inside the wavefunction overlap.

All three are diffractive photoproduction measurements, but their energy
averages, mass ranges, and (for H1 2003) inclusion of the elastic channel
differ. A universal multiplicative conversion between these datasets is
not justified without the W, t and MY dependences. H1 2003, Section 4.1,
explicitly reports good agreement with ZEUS when its analysis is performed
in the ZEUS kinematic region. H1 2013 also reports consistency with the
older H1 high-t data after a phase-space correction of about 7%.

The current GBW event format does not predict MY or fragmentation. The
plots are therefore comparisons of the geometric Good-Walker prediction
with mass-selected data, not a simulation of the experimental selections.
The representative W values also do not replace a flux-weighted integration
over the full experimental W bins.

The H1 data previously used in **notebook 03** are a different measurement:
rho photoproduction, [arXiv:2005.14471](https://arxiv.org/abs/2005.14471),
published in 2020, using mp<MY<10 GeV. They are newer than Heikki's 2016
paper but are not J/psi data. The H1 J/psi data added here are from 2013.

Regenerate the excerpts with:

```sh
.venv/bin/python notebooks/data/fetch_jpsi_dissociation_data.py
```

Source URL SHA256 hashes are in `jpsi_dissociation_sources.json`. The
cross-section excerpt retains all available t bins; the main figure shows
the first seven H1 2013 bins, the first three ZEUS bins, and the first
original H1 2003 bin.
