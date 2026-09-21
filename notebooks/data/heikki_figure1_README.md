# Published Figure 1 curves

`heikki_1603_figure1_curves.csv` contains the seven theory curves and the
statistical-band boundaries drawn in Figure 1 of H. Mäntysaari and B. Schenke,
[arXiv:1603.04349v2](https://arxiv.org/html/1603.04349v2), *Evidence of strong
proton shape fluctuations from incoherent diffraction*, Phys. Rev. Lett. 117,
052301 (2016).

The process is J/psi photoproduction, Q² = 0, W = 100 GeV. Cross sections are
in nb/GeV²; the horizontal variable is |t| in GeV². Figure 1 is the IPsat
calculation. Figure 2 is IP-Glasma and is not included here.

The curves were extracted from the published
[SVG](https://arxiv.org/html/1603.04349v2/spectra_q2_0_bp_35_10.svg), downloaded
on 2026-09-20. Source SHA256:

```text
c1b946b38496f1b4646a58f332345256ec1ba2e8af3a35065ffb86b6932bd403
```

These are numerical coordinates recovered from the **plotted vector paths**,
not the authors' original simulation tables. No curve was fitted or rescaled.
The SVG transform is inverted using its axis ticks: x = 72 + 178.56 |t|,
and the y coordinates 103.85281, 172.0858, 240.31879, 308.5518, 376.78477
correspond to cross sections 0.1, 1, 10, 100, 1000. All five log-axis ticks
are used to calibrate the mapping. Interpolation between stored vertices
should be linear in log(cross section) to reproduce the displayed paths.

Column definitions:

- `model`: smooth Bp=4; or the pair Bqc, Bq in GeV^-2.
- `channel`: coherent or incoherent.
- `t_GeV2`: |t|.
- `dsigma_dt_nb_per_GeV2`: central plotted curve.
- `band_low_nb_per_GeV2`, `band_high_nb_per_GeV2`: plotted statistical band.

The smooth curve has identical central and boundary values. The paper's
curves start at |t| = 0.05: the CSV does not invent their forward limit.
One vertex beyond the visible right boundary is retained when present, so
clipping at |t| = 2.5 reproduces the published line segment. Portions below
the figure's lower boundary should not be used to infer high-t accuracy.
Several coherent lower-band values reach the log plot's effectively-zero
sentinel; these are represented by 1e-300. They do not represent resolved
positive predictions at that scale.

Regenerate with:

```sh
.venv/bin/python notebooks/data/extract_heikki_figure1.py
```

The overlay deliberately contains **theory curves only**. It does not treat
the original figure's different HERA datasets as one common measurement.
In particular, its H1 large-|t| points are total diffraction, whereas its
ZEUS proton-dissociative points use x_J > 0.01 (x_J is not the small-x GBW
argument). Their final-state cuts differ from the H1 rho dataset used in
notebook 03. See the linked paper's references 52, 53, 59 and 60.
