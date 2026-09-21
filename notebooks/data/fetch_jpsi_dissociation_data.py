"""Recover HERA photon-level cross sections from primary numerical tables.

Run from the repository root with .venv/bin/python. The plotting script uses
the resulting local CSV and needs no network connection.
"""

import csv
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request

from bs4 import BeautifulSoup
import numpy as np


HERE = Path(__file__).resolve().parent
H1_NUMERICAL = "https://www-h1.desy.de/psfiles/figures/d13-058.table_tHE_allH1_v2.txt"
SOURCES = {
    "H1_2013": "https://arxiv.org/html/1304.5162",
    "ZEUS_2003": "https://arxiv.org/html/hep-ex/0205081",
    "H1_2003_total": "https://arxiv.org/html/hep-ex/0306013",
}


def html_table(raw, table_id):
    soup = BeautifulSoup(raw, "html.parser")
    for math in soup.find_all("math"):
        math.replace_with(math.get("alttext", ""))
    table = soup.find(id=table_id)
    return [[cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
            for row in table.find_all("tr")]


def main():
    raw = urllib.request.urlopen(H1_NUMERICAL, timeout=30).read()
    raw_path = HERE / "h1_2013_tHE_official.txt"
    raw_path.write_bytes(raw)
    hashes = {H1_NUMERICAL: hashlib.sha256(raw).hexdigest()}
    table = np.loadtxt(io.BytesIO(raw))
    assert table.shape == (27, 23)
    notebook_rows = table[(table[:, 0] >= 1) & (table[:, 0] <= 17)]
    selected = table[(table[:, 0] >= 9) & (table[:, 0] <= 17)]
    assert len(selected) == 9 and np.all(selected[:, 1] == 1)
    records = []
    for row in selected:
        # Rows 18-27 are older H1 data adjusted to the newer phase space.
        # They MUST NOT be used as the original 2003 points in Figure 1.
        records.append(["H1_2013", *row[2:5], row[6], row[7], row[7],
                        "total", 0.0, 40.0, 110.0, 75.0])

    pages = {}
    for name, url in SOURCES.items():
        pages[name] = urllib.request.urlopen(url, timeout=30).read()
        hashes[url] = hashlib.sha256(pages[name]).hexdigest()

    # Verify the official excerpt against the independently published Table 5.
    h1_rows = html_table(pages["H1_2013"], "S5.T5")
    published = []
    error_decimals = []
    for cells in h1_rows:
        if len(cells) == 17 and re.fullmatch(r"[\d.]+\s*-\s*[\d.]+", cells[0]):
            bounds = [float(v) for v in cells[0].split("-")]
            published.append([*bounds, *map(float, cells[1:4])])
            error_decimals.append(len(cells[3].partition(".")[2]))
    assert len(published) == 17
    assert np.allclose(np.asarray(published)[:, :4], notebook_rows[:, [2, 3, 4, 6]])
    # The official elastic errors retain more digits (18.4 vs 18 and
    # 2.74 vs 2.7). Compare at the precision printed in the article.
    for printed, row, decimals in zip(published, notebook_rows, error_decimals):
        assert round(float(row[7]), decimals) == printed[4]

    # Notebook 03 compares both channels, using the same columns as the
    # rho excerpt. J/psi cross sections and uncertainties are in nb/GeV^2.
    with (HERE / "h1_jpsi_W75.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "t_min", "t_max", "t_center", "dsigma_dt",
                         "err_minus", "err_plus"])
        for row in notebook_rows:
            writer.writerow(["pd" if row[1] else "el", *row[2:5],
                             row[6], row[7], row[7]])

    for cells in html_table(pages["ZEUS_2003"], "S11.T3"):
        if len(cells) != 3 or not re.fullmatch(r"[\d.]+-[\d.]+", cells[0]):
            continue
        low, high = map(float, cells[0].split("-"))
        values = [float(v) for v in re.findall(r"[-+]?\d+(?:\.\d+)?", cells[2])]
        assert len(values) == 6
        central, stat, sys_minus, sys_plus, model_minus, model_plus = values
        err_minus = np.sqrt(stat**2 + sys_minus**2 + model_minus**2)
        err_plus = np.sqrt(stat**2 + sys_plus**2 + model_plus**2)
        records.append(["ZEUS_2003", low, high, float(cells[1]), central,
                        err_minus, err_plus, "stat_sys_dissociation_model", 0.10,
                        80.0, 120.0, 100.0])

    for cells in html_table(pages["H1_2003_total"], "Sx1.T1"):
        if len(cells) != 3 or not re.fullmatch(r"[\d.]+-[\d.]+", cells[0]):
            continue
        low, high = map(float, cells[0].split("-"))
        values = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", cells[2])]
        assert len(values) == 3
        central, stat, syst = values
        error = np.hypot(stat, syst)
        records.append(["H1_2003_total", low, high, float(cells[1]), central,
                        error, error, "stat_sys", 0.0, 50.0, 150.0, 100.0])

    assert len(records) == 25
    out = HERE / "jpsi_dissociation_hera.csv"
    with out.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "t_min_GeV2", "t_max_GeV2", "t_center_GeV2",
                         "dsigma_dt_nb_per_GeV2", "err_minus", "err_plus",
                         "error_contents", "extra_norm_fraction", "W_min_GeV",
                         "W_max_GeV", "W_reference_GeV"])
        writer.writerows(records)
    (HERE / "jpsi_dissociation_sources.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"Saved {len(records)} rows to {out}")
    print("H1 2013: all 17 official numerical rows independently match Table 5.")
    print("Saved 8 elastic and 9 dissociative rows to h1_jpsi_W75.csv (nb/GeV^2).")
    print("ZEUS: 6 rows from Table 3; extra 10% normalization kept separate.")
    print("H1 2003: 10 ORIGINAL Table 1 rows, not the rescaled 2013 supplement.")


if __name__ == "__main__":
    main()
