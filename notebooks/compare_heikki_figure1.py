"""Overlay the existing GBW proton model and published IPsat Fig. 1 curves.

Run from the repository root:
    .venv/bin/python notebooks/compare_heikki_figure1.py

This comparison changes no upcmodel implementation. It uses the existing
Gaus-LC functions with the J/psi parameters from KMW, hep-ph/0606272 Table 1.
For this factorized Gaussian proton only, the ensemble moments are analytic;
using them avoids a finite-event noise floor in the coherent high-t tail.
"""

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from upcmodel.amplitudes import gbw_dipole_amplitude, photon_target_amplitude_T
from upcmodel.corrections import effective_lambda_from_samples, skewness_factor
from upcmodel.ensemble import photon_target_spectra
from upcmodel.geometry import mean_hotspot_count
from upcmodel.parameters import (
    JPSI_HS, JPSI_GAUS_LC_PARAM as JPSI_GAUS_LC, JPSI_MASS_GEV,
)
from upcmodel.wavefunctions import gaus_lc_overlap_integral_z


W_GEV = 100.0
M_GEV = JPSI_MASS_GEV
X = M_GEV**2 / W_GEV**2
GEV_M2_TO_NB = 389379.3721
OUTPUT = ROOT / "notebooks/gamma_p_outputs"
PAPER_CURVES = ROOT / "notebooks/data/heikki_1603_figure1_curves.csv"


def calculate_gbw_jpsi(q_GeV, *, W_GeV=W_GEV, nr=601, nz=96):
    """Return exact geometry-ensemble moments with numerical r,z integrals.

    For each event, A=C(q)*exp(-Bhs*q^2/2)*sum_i exp(-i*b_i*q)/n.
    The Gaussian centers have component variance Bp, and n is zero-truncated
    Poisson. Thus E[A]=C*exp(-(Bp+Bhs)*q^2/2), while
    Var(A)=|C|^2*exp(-Bhs*q^2)*E[1/n]*(1-exp(-Bp*q^2)).
    This identity is specific to the existing linear proton profile.
    """
    q_GeV = np.atleast_1d(q_GeV)
    bj_x = M_GEV**2 / W_GeV**2
    r_fm = np.geomspace(0.0001, 5.0, nr)
    # Chunk only the overlap calculation to keep its (R,K,Z) allocation small.
    H = np.concatenate([
        gaus_lc_overlap_integral_z(JPSI_GAUS_LC, r_fm, q_GeV[i:i + 32], 0.0, nz)
        for i in range(0, len(q_GeV), 32)
    ], axis=1)
    sigma0_GeV_m2 = 4 * np.pi * JPSI_HS.B_p_GeV_m2

    def common_amplitude(x_value):
        F = np.broadcast_to(
            sigma0_GeV_m2 * gbw_dipole_amplitude(x_value, r_fm)[:, None], H.shape
        )
        return photon_target_amplitude_T(r_fm, H, F)

    common = common_amplitude(bj_x)
    # Evaluate the same frozen-profile forward slope as notebook 03.
    H0 = gaus_lc_overlap_integral_z(JPSI_GAUS_LC, r_fm, [0.0], 0.0, nz)
    forward = []
    for x_value in (bj_x * np.exp(-0.05), bj_x * np.exp(0.05)):
        F0 = sigma0_GeV_m2 * gbw_dipole_amplitude(x_value, r_fm)[:, None]
        forward.append(photon_target_amplitude_T(r_fm, H0, F0)[0])
    lam = float(effective_lambda_from_samples(*forward, dlogx=0.05))
    Rg = float(skewness_factor(lam))

    mu = float(mean_hotspot_count(bj_x, JPSI_HS))
    inverse_count_mean = (expi(mu) - np.euler_gamma - np.log(mu)) / np.expm1(mu)
    t = q_GeV**2
    Bp, Bhs = JPSI_HS.B_p_GeV_m2, JPSI_HS.B_hs_GeV_m2
    mean = common * np.exp(-(Bp + Bhs) * t / 2)
    variance = abs(common)**2 * np.exp(-Bhs * t) * inverse_count_mean * (-np.expm1(-Bp * t))
    coherent, incoherent = photon_target_spectra(mean, variance, R_g=Rg)
    metadata = {
        "process": "J/psi photoproduction", "W_GeV": W_GeV, "Q2_GeV2": 0.0,
        "M_GeV": M_GEV, "x": bj_x, "Bp_GeV_m2": Bp, "Bhs_GeV_m2": Bhs,
        "sigma0_GeV_m2": sigma0_GeV_m2, "poisson_parameter": mu,
        "mean_Nhs": mu / (-np.expm1(-mu)), "mean_inverse_Nhs": inverse_count_mean,
        "lambda_forward": lam, "Rg": Rg, "real_part_correction_applied": False,
        "wavefunction": "KMW Table 1 J/psi Gaus-LC: mf=1.4, NT=1.23, RT2=6.5, ehat=2/3",
        "phase": "J0((1/2-z)*r*Delta/hbarc), unchanged repo convention",
        "ensemble": "Exact Gaussian and zero-truncated-Poisson moments; no Monte Carlo noise",
        "r_fm_min": 0.0001, "r_fm_max": 5.0, "nr": nr, "nz": nz,
    }
    return {
        "t": t, "q": q_GeV, "common": common, "mean": mean, "variance": variance,
        "coherent": coherent * GEV_M2_TO_NB,
        "incoherent": incoherent * GEV_M2_TO_NB,
        "metadata": metadata,
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pdf_output = ROOT / "output/pdf"
    pdf_output.mkdir(parents=True, exist_ok=True)
    result = calculate_gbw_jpsi(np.sqrt(np.linspace(0, 2.5, 501)))
    paper = np.genfromtxt(PAPER_CURVES, delimiter=",", names=True, dtype=None, encoding="utf8")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.6))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.24, top=0.73, wspace=0.20)
    styles = [
        ("Bqc3.5_Bq0.5", "#2864d6", "--", r"IPsat: $B_{qc}=3.5,\ B_q=0.5$"),
        ("Bqc3.5_Bq1", "#d94a42", "-.", r"IPsat: $B_{qc}=3.5,\ B_q=1.0$"),
        ("Bqc1_Bq3", "#777777", ":", r"IPsat: $B_{qc}=1.0,\ B_q=3.0$"),
        ("smooth_Bp4", "#ad946d", "-", r"IPsat: smooth, $B_p=4.0$"),
    ]
    for ax, channel, title in zip(axes, ("coherent", "incoherent"),
                                 ("Coherent / intact proton", "Incoherent / diffractive breakup")):
        ax.plot(result["t"], result[channel], color="#172a34", linewidth=2.8,
                label="Your GBW + Gaus-LC", zorder=5)
        for model, color, linestyle, label in styles:
            rows = paper[(paper["model"] == model) & (paper["channel"] == channel)]
            if not len(rows):
                continue
            ax.fill_between(rows["t_GeV2"], rows["band_low_nb_per_GeV2"],
                            rows["band_high_nb_per_GeV2"], color=color, alpha=0.13, linewidth=0)
            ax.plot(rows["t_GeV2"], rows["dsigma_dt_nb_per_GeV2"],
                    color=color, linestyle=linestyle, linewidth=1.9, label=label)
        ax.set(yscale="log", xlim=(0, 2.5), xlabel=r"$|t|\ [\mathrm{GeV}^2]$", title=title)
        ax.grid(alpha=0.16, which="major")
    axes[0].set(ylim=(0.02, 800), ylabel=r"$d\sigma/dt\ [\mathrm{nb}/\mathrm{GeV}^2]$")
    axes[1].set(ylim=(0.02, 80))
    axes[1].text(0.97, 0.05, "Smooth IPsat: zero incoherent cross section",
                 transform=axes[1].transAxes, ha="right", color="#77674d", fontsize=9)
    # The paper paths begin at |t|=0.05. No extrapolation to t=0 is made.
    for ax in axes:
        ax.axvspan(0, 0.05, color="#777777", alpha=0.06, linewidth=0)
    fig.suptitle("Your GBW model overlaid with Mäntysaari and Schenke Figure 1",
                 fontsize=17, fontweight="bold", y=0.98)
    fig.text(0.5, 0.92, r"$\gamma p\to J/\psi+p\ \mathrm{or}\ Y$  |  $W=100$ GeV  |  $Q^2=0$  |  absolute normalization, no fit",
             ha="center", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.865),
               ncol=3, frameon=False, fontsize=10, columnspacing=1.6)
    fig.text(0.075, 0.085,
             "GBW: existing JPSI_HS, Gaus-LC overlap, midpoint phase, forward Rg; no real-part factor.\n"
             "Paper: published IPsat curves with boosted Gaussian overlap and its corrections; bands are paper statistics.",
             fontsize=9, color="#444444", linespacing=1.6)
    fig.text(0.075, 0.025,
             "Source: arXiv:1603.04349v2, Fig. 1 (vector-path extraction). Theory comparison only; no HERA points redigitized.",
             fontsize=8.5, color="#555555")
    png = OUTPUT / "jpsi_heikki_figure1_overlay.png"
    pdf = pdf_output / "jpsi_heikki_figure1_overlay.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    np.savetxt(OUTPUT / "jpsi_heikki_gbw_spectra.csv",
               np.column_stack([result["t"], result["coherent"], result["incoherent"]]),
               delimiter=",", header="t_GeV2,coherent_nb_per_GeV2,incoherent_nb_per_GeV2", comments="")
    (OUTPUT / "jpsi_heikki_metadata.json").write_text(json.dumps(result["metadata"], indent=2) + "\n")
    print(json.dumps(result["metadata"], indent=2))
    print("Saved:", png, pdf, sep="\n")


if __name__ == "__main__":
    main()
