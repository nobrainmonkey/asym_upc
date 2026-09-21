# H1 photon-proton benchmarks for notebook 03

The main calculation is J/psi; rho is a short comparison afterward. Both
datasets are photon-proton cross sections, already corrected for photon
flux and decay branching fractions. The notebook reads these local files
offline and averages the model over each published t bin.

## J/psi: H1 2013, reference W = 75 GeV

`h1_jpsi_W75.csv` contains the **8 elastic and 9 proton-dissociative**
entries from Table 5 of H1, *Elastic and Proton-Dissociative Photoproduction
of J/psi Mesons at HERA*, Eur. Phys. J. C **73** (2013) 2466:
[paper](https://arxiv.org/html/1304.5162),
[DOI](https://doi.org/10.1140/epjc/s10052-013-2466-y),
[official numerical table](https://www-h1.desy.de/psfiles/figures/d13-058.table_tHE_allH1_v2.txt).

- Units: **nb/GeV²** for `dsigma_dt`, `err_minus`, and `err_plus`.
- `channel`: `el` = intact proton; `pd` = proton dissociation.
- `t_min`, `t_max`, `t_center`: GeV². The centers are the published
  effective centers; the measured cross sections represent bin averages.
- The data cover **40<W<110 GeV**, photoproduction Q²<2.5 GeV²,
  |t|<1.2 GeV² for elastic and |t|<8 GeV² for dissociation, mp<MY<10 GeV.
- W=75 GeV in the filename denotes the model's representative energy,
  following [Mäntysaari and Schenke, Figure 7](https://arxiv.org/html/1607.01711).
  The data have not been rescaled to a different energy. The notebook
  uses Q²=0 and does not integrate the model over the experimental W range.
- Rows 1-17 of `h1_2013_tHE_official.txt` supply the values and total
  errors. All were checked against Table 5, allowing for the article's
  rounded error entries (18 vs 18.4 and 2.7 vs 2.74). Rows 18-27 are older,
  rescaled high-t points and are excluded from this notebook dataset.
- `err_minus` = `err_plus` = the official total uncertainty. The original
  file preserves the correlated-error information for any future fit.

Regenerate both this excerpt and the separate historical comparison with:

```sh
.venv/bin/python notebooks/data/fetch_jpsi_dissociation_data.py
```

The source hashes are in `jpsi_dissociation_sources.json`. For the different
H1/ZEUS breakup-mass cuts, see [the selection notes](jpsi_dissociation_README.md).
The geometric events themselves do not assign a dissociation mass MY.

## Rho: H1 2020, reference W = 44 GeV

`h1_rho_W44.csv` contains the 12 rows at the tabulated mean energy
W = 44 GeV (38 < W < 50 GeV) from Tables 12 and 13 of:

H1 Collaboration, *Measurement of exclusive pi+ pi- and rho0 meson
photoproduction at HERA*, Eur. Phys. J. C **80** (2020) 1189,
[doi:10.1140/epjc/s10052-020-08587-3](https://doi.org/10.1140/epjc/s10052-020-08587-3),
[arXiv:2005.14471](https://arxiv.org/abs/2005.14471).

Downloaded 2026-09-20 from the
[official H1 numerical table](https://www-h1.desy.de/psfiles/figures/d20-080.t12t13-rho-Wt_syst.csv).
The [H1 README](https://www-h1.desy.de/psfiles/figures/d20-080.README)
defines the original columns. No digitization or rounding was performed.
The original table uses x for |t| and y for W; selection was y_ctr = 44.

- `channel`: el = elastic; pd = proton dissociation, m_p < M_Y < 10 GeV.
- `t_min`, `t_max`, `t_center`: |t| in GeV^2; the centers are effective centers.
- `dsigma_dt`: bin-averaged cross section in microbarn / GeV^2.
- `err_stat`: statistical uncertainty in the same units.
- `err_minus`, `err_plus`: positive magnitudes of the total lower and upper
  uncertainties, already including the statistical contribution.

These are photon-proton cross sections quoted at Q^2 = 0, extracted using
an effective photon flux. No additional flux factor is needed. The notebook
uses one energy, W = 44 GeV, rather than integrating over the W bin.

For comparison the model is averaged over each experimental |t| bin. The
integrated data values printed by the notebook are sums of these bins, not
an independent integrated measurement. Correlations are not retained in this
small excerpt; do not use it alone for a correlated fit or integrated errors.

Original file SHA256: `cc39d1a8b52505b65df09c92a61973e86c73c3744365ddeeb2dc0fa5a8762e7b`.

## Model parameter reference

Both Gaus-LC wavefunctions and the meson masses use Kowalski, Motyka and
Watt, [hep-ph/0606272, Table 1](https://arxiv.org/html/hep-ph/0606272).
`upcmodel/parameters.py` stores `JPSI_GAUS_LC_PARAM`, `RHO_GAUS_LC_PARAM`,
`JPSI_MASS_GEV`, and `RHO_MASS_GEV`; the existing `JPSI_HS` and `RHO_HS`
presets supply the respective hotspot geometries. This is an untuned
comparison using Gaus-LC, not a reproduction of a boosted-Gaussian fit.
