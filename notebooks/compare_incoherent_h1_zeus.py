"""Add primary H1/ZEUS data to the GBW incoherent comparison.

Run: .venv/bin/python notebooks/compare_incoherent_h1_zeus.py

The panels use the reference energies used by Mantysaari and Schenke for
each dataset. Data retain their published values, energy ranges and cuts.
No breakup-mass selection is imposed on the geometry-only GBW prediction.
"""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

from compare_heikki_figure1 import ROOT, OUTPUT, PAPER_CURVES, calculate_gbw_jpsi


DATA_FILE = ROOT / "notebooks/data/jpsi_dissociation_hera.csv"


def plot_measurement(ax, rows, *, color, marker, label, horizontal=True):
    center = rows["t_center_GeV2"]
    xerr = None
    if horizontal:
        xerr = np.vstack([center - rows["t_min_GeV2"], rows["t_max_GeV2"] - center])
    return ax.errorbar(
        center, rows["dsigma_dt_nb_per_GeV2"],
        yerr=np.vstack([rows["err_minus"], rows["err_plus"]]), xerr=xerr,
        fmt=marker, color=color, markerfacecolor="white", markersize=6,
        markeredgewidth=1.3, elinewidth=1.1, capsize=2.5, label=label, zorder=7,
    )


def main():
    data = np.genfromtxt(DATA_FILE, names=True, delimiter=",", dtype=None, encoding="utf8")
    paper = np.genfromtxt(PAPER_CURVES, names=True, delimiter=",", dtype=None, encoding="utf8")
    h1 = data[data["dataset"] == "H1_2013"]
    zeus = data[data["dataset"] == "ZEUS_2003"]
    h1_old = data[data["dataset"] == "H1_2003_total"]

    # Include all H1 bins in the saved predictions, even though the plot
    # emphasizes the small-|t| region missing from the ZEUS measurement.
    t = np.unique(np.r_[np.linspace(0, 2.7, 901), np.linspace(2.7, 8, 401),
                        h1["t_min_GeV2"], h1["t_max_GeV2"]])
    results = {W: calculate_gbw_jpsi(np.sqrt(t), W_GeV=float(W)) for W in (75, 100)}
    h1_interpolator = PchipInterpolator(t, results[75]["incoherent"])
    bin_means = np.array([
        h1_interpolator.integrate(row["t_min_GeV2"], row["t_max_GeV2"])
        / (row["t_max_GeV2"] - row["t_min_GeV2"])
        for row in h1
    ])

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.265, top=0.79, wspace=0.12)
    for ax, W in zip(axes, (100, 75)):
        ax.plot(t, results[W]["incoherent"], color="#193d46", linewidth=2.6,
                label=f"Your GBW, W={W} GeV", zorder=4)
        ax.set(yscale="log", ylim=(0.15, 110), xlabel=r"$|t|\ [\mathrm{GeV}^2]$")
        ax.grid(which="major", alpha=0.17)
    axes[0].set(xlim=(0, 2.5), title=r"Figure 1 comparison: $W=100$ GeV",
                ylabel=r"$d\sigma/dt\ [\mathrm{nb}/\mathrm{GeV}^2]$")
    axes[1].set(xlim=(0, 2.65), title=r"H1 data reaching small $|t|$: $W=75$ GeV")

    styles = [
        ("Bqc3.5_Bq0.5", "#2864d6", "--", r"IPsat: $B_{qc}=3.5,\ B_q=0.5$"),
        ("Bqc3.5_Bq1", "#d94a42", "-.", r"IPsat: $B_{qc}=3.5,\ B_q=1.0$"),
        ("Bqc1_Bq3", "#888888", ":", r"IPsat: $B_{qc}=1.0,\ B_q=3.0$"),
    ]
    for model, color, linestyle, label in styles:
        rows = paper[(paper["model"] == model) & (paper["channel"] == "incoherent")]
        axes[0].fill_between(rows["t_GeV2"], rows["band_low_nb_per_GeV2"],
                             rows["band_high_nb_per_GeV2"], color=color, alpha=0.12)
        axes[0].plot(rows["t_GeV2"], rows["dsigma_dt_nb_per_GeV2"],
                     color=color, linestyle=linestyle, linewidth=1.7, label=label)

    plot_measurement(axes[0], zeus[zeus["t_center_GeV2"] <= 2.5],
                     color="#623c83", marker="o", label=r"ZEUS 2003: $x_J>0.01$")
    # This is the original total-diffractive datum in Heikki's Figure 1.
    # Only one H1 2003 point falls in the displayed |t| range.
    plot_measurement(axes[0], h1_old[h1_old["t_center_GeV2"] <= 2.5],
                     color="#a4781e", marker="^", horizontal=False,
                     label="H1 2003: total diffraction")
    visible = h1["t_center_GeV2"] <= 2.65
    plot_measurement(axes[1], h1[visible], color="#b5531c", marker="s",
                     label=r"H1 2013: dissociation, $M_Y<10$ GeV")
    axes[1].plot(h1["t_center_GeV2"][visible], bin_means[visible],
                 linestyle="none", marker="_", markersize=13, markeredgewidth=2,
                 color="#193d46", label="GBW averaged over each H1 t bin", zorder=6)
    axes[0].legend(loc="lower left", fontsize=8.6, framealpha=0.92, borderpad=0.6)
    axes[1].legend(loc="lower left", fontsize=9, framealpha=0.92, borderpad=0.6)
    axes[1].annotate("H1 adds the low-t constraint",
                      xy=(0.1, 47.3), xytext=(0.65, 77), fontsize=10,
                      color="#8f4319", arrowprops={"arrowstyle": "->", "color": "#8f4319"})

    fig.suptitle(r"$J/\psi$ proton dissociation: H1 and ZEUS test different ranges",
                 fontsize=17, fontweight="bold", y=0.97)
    fig.text(0.5, 0.90, "Published photon-level cross sections; data values and energy ranges are not rescaled",
             ha="center", fontsize=11)
    fig.text(0.075, 0.15,
             "Left data: ZEUS 80<W<120 GeV; H1 total 50<W<150 GeV. Right data: H1 40<W<110 GeV.\n"
             "Model curves use Q²=0 and representative W; GBW includes geometry fluctuations with no breakup-mass selection.\n"
             "Bars show published uncertainties (ZEUS: stat + syst + dissociation model; its extra 10% normalization is separate).",
             fontsize=9, color="#444444", linespacing=1.55, va="top")
    fig.text(0.075, 0.032,
             "Sources: H1 arXiv:1304.5162 Table 5; ZEUS hep-ex/0205081 Table 3; H1 hep-ex/0306013 Table 1; theory arXiv:1603.04349 Fig. 1.",
             fontsize=8.4, color="#555555")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    png = OUTPUT / "jpsi_incoherent_h1_zeus.png"
    pdf = ROOT / "output/pdf/jpsi_incoherent_h1_zeus.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)

    # Save bin averages without presenting an uncorrelated chi-square fit.
    with (OUTPUT / "jpsi_h1_2013_bin_comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["t_min_GeV2", "t_max_GeV2", "t_center_GeV2", "H1_nb_per_GeV2",
                         "H1_total_error", "GBW_W75_bin_average", "model_over_data"])
        for row, prediction in zip(h1, bin_means):
            writer.writerow([row["t_min_GeV2"], row["t_max_GeV2"], row["t_center_GeV2"],
                             row["dsigma_dt_nb_per_GeV2"], row["err_plus"], prediction,
                             prediction / row["dsigma_dt_nb_per_GeV2"]])
    for W, result in results.items():
        np.savetxt(OUTPUT / f"jpsi_incoherent_W{W}_hera.csv",
                   np.column_stack([t, result["incoherent"]]), delimiter=",",
                   header="t_GeV2,incoherent_nb_per_GeV2", comments="")
    (OUTPUT / "jpsi_h1_zeus_metadata.json").write_text(json.dumps({
        "calculations": {str(W): result["metadata"] for W, result in results.items()},
        "energy_treatment": "Representative W=100 and 75 GeV, as in the literature; no energy rescaling of data",
        "data": str(DATA_FILE.relative_to(ROOT)),
        "model_M_Y_cut": None,
    }, indent=2) + "\n")
    print("Saved:", png, pdf, sep="\n")
    for row, pred in zip(h1, bin_means):
        print(f"H1 bin {row['t_min_GeV2']:.2f}-{row['t_max_GeV2']:.2f}: "
              f"data={row['dsigma_dt_nb_per_GeV2']:.4g} +/- {row['err_plus']:.4g}, "
              f"GBW at W75 bin mean={pred:.5g} nb/GeV^2")


if __name__ == "__main__":
    main()
