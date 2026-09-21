"""Extract plotted theory paths, not underlying simulation data, from Fig. 1.

Source: Mantysaari and Schenke, arXiv:1603.04349v2.
Only Python's standard library is required. The original SVG is fetched into
memory; the CSV retains its central curves and statistical-band boundaries.
"""

import csv
import hashlib
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET


SOURCE = "https://arxiv.org/html/1603.04349v2/spectra_q2_0_bp_35_10.svg"


def vertices(path):
    """Decode the straight-line SVG commands used by this particular figure."""
    tokens = re.findall(r"[MLHVZ]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", path)
    points = []
    x = y = 0.0
    command = None
    i = 0
    while i < len(tokens):
        if tokens[i] in ("M", "L", "H", "V", "Z"):
            command = tokens[i]
            i += 1
        if command == "Z":
            break
        if command in ("M", "L"):
            x, y = map(float, tokens[i:i + 2])
            i += 2
            command = "L"  # Subsequent coordinates after M are implicit L.
        elif command == "H":
            x = float(tokens[i])
            i += 1
        elif command == "V":
            y = float(tokens[i])
            i += 1
        else:
            raise ValueError(f"Unexpected SVG command: {command}")
        points.append((x, y))
    return points


def physical_coordinates(x, y):
    # SVG path coordinates precede its y -> 432-y display transformation.
    # x ticks: t=0 at 72, t=2.5 at 518.4.
    # y ticks: 0.1,1,10,100,1000 at the following vector coordinates.
    y_ticks = (103.85281, 172.0858, 240.31879, 308.5518, 376.78477)
    # Least-squares calibration over all five ticks, avoiding rounding bias.
    # Their integer logarithms are -1,0,1,2,3; centered values are -2..2.
    slope = sum((j - 2) * v for j, v in enumerate(y_ticks)) / 10.0
    intercept = sum(y_ticks) / 5.0 - slope
    return (x - 72.0) / 178.56, 10.0 ** max(-300.0, (y - intercept) / slope)


def main():
    raw = urllib.request.urlopen(SOURCE, timeout=30).read()
    root = ET.fromstring(raw)
    ns = "{http://www.w3.org/2000/svg}"
    paths = []

    def visit(element, in_defs=False):
        in_defs = in_defs or element.tag == ns + "defs"
        if element.tag == ns + "path" and not in_defs:
            paths.append(element.attrib)
        for child in element:
            visit(child, in_defs)

    visit(root)
    definitions = [
        ("smooth_Bp4", "coherent", 38, None),
        ("Bqc3.5_Bq0.5", "coherent", 39, 2),
        ("Bqc3.5_Bq1", "coherent", 40, 3),
        ("Bqc1_Bq3", "coherent", 41, 4),
        ("Bqc3.5_Bq0.5", "incoherent", 42, 5),
        ("Bqc3.5_Bq1", "incoherent", 43, 6),
        ("Bqc1_Bq3", "incoherent", 44, 7),
    ]
    destination = Path(__file__).with_name("heikki_1603_figure1_curves.csv")
    with destination.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "channel", "t_GeV2", "dsigma_dt_nb_per_GeV2",
                         "band_low_nb_per_GeV2", "band_high_nb_per_GeV2"])
        for model, channel, line_index, band_index in definitions:
            line = paths[line_index]
            assert line.get("fill") == "none" and line.get("stroke-width") in ("2.5", ".9")
            bands = {}
            if band_index is not None:
                assert paths[band_index].get("fill-opacity") == ".2"
                for x, y in vertices(paths[band_index]["d"]):
                    bands.setdefault(round(x, 5), []).append(y)
            for x, y in vertices(line["d"]):
                t, central = physical_coordinates(x, y)
                # Retain one vertex beyond t=2.5 so clipping reproduces the
                # visible endpoint. All plotted axes stop at t=2.5.
                if not 0 <= t <= 2.55 + 1e-6:
                    continue
                boundary = bands.get(round(x, 5), [y])
                low = physical_coordinates(x, min(boundary))[1]
                high = physical_coordinates(x, max(boundary))[1]
                assert low <= central * (1 + 1e-6) and central <= high * (1 + 1e-6)
                writer.writerow([model, channel, f"{t:.9g}", f"{central:.9g}",
                                 f"{low:.9g}", f"{high:.9g}"])
    print(destination)
    print("Source SVG SHA256:", hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    main()
