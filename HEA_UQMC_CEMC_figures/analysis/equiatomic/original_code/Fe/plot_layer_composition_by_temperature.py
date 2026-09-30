from __future__ import annotations

import argparse
import csv
import re
import shlex
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path("/home/jinsookim/HEA_MC/FeCoNiPtPd_HER/equi-atomic")
ELEMENTS = ["Pd", "Pt", "Fe", "Co", "Ni"]
COLORS = {
    "Pd": "#FFD700",
    "Pt": "#1F77B4",
    "Fe": "#E31A1C",
    "Co": "#FF9900",
    "Ni": "#2CA02C",
}
PAIR_LABELS = ["Surface", "Subsurface", "3rd layer", "4th layer", "5th layer"]
SITE_ROOTS = [
    ("100", "bridge"),
    ("100", "hollow"),
    ("110", "bridge"),
    ("111", "hollow"),
    ("111", "top"),
]


def read_cif_symbols_and_z(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    for start, line in enumerate(lines):
        if line.strip() != "loop_":
            continue
        headers: list[str] = []
        cursor = start + 1
        while cursor < len(lines) and lines[cursor].lstrip().startswith("_"):
            headers.append(lines[cursor].strip().split()[0])
            cursor += 1
        if "_atom_site_fract_z" not in headers:
            continue
        symbol_key = (
            "_atom_site_type_symbol"
            if "_atom_site_type_symbol" in headers
            else "_atom_site_label"
        )
        symbol_col = headers.index(symbol_key)
        z_col = headers.index("_atom_site_fract_z")
        symbols: list[str] = []
        z_values: list[float] = []
        while cursor < len(lines):
            raw = lines[cursor].strip()
            if not raw:
                if symbols:
                    break
                cursor += 1
                continue
            if raw == "loop_" or raw.startswith("_") or raw.startswith("data_"):
                break
            fields = shlex.split(raw)
            if len(fields) < len(headers):
                raise ValueError(f"{path}: incomplete atom row: {raw}")
            symbol = fields[symbol_col]
            if symbol_key == "_atom_site_label":
                symbol = "".join(c for c in symbol if c.isalpha())
            symbols.append(symbol)
            z_values.append(float(fields[z_col]))
            cursor += 1
        return np.asarray(symbols), np.asarray(z_values, dtype=float)
    raise ValueError(f"{path}: atom-site fractional-z loop not found")


def split_layers(z: np.ndarray, tolerance: float = 1.0e-5) -> list[np.ndarray]:
    order = np.argsort(z)
    groups: list[list[int]] = []
    centers: list[float] = []
    for idx in order:
        value = float(z[idx])
        if not groups or abs(value - centers[-1]) > tolerance:
            groups.append([int(idx)])
            centers.append(value)
        else:
            groups[-1].append(int(idx))
            centers[-1] = float(np.mean(z[groups[-1]]))
    return [np.asarray(group, dtype=int) for group in groups]


def temperature_key(filename: str) -> tuple[int, str]:
    if filename == "final_0298K_cemc.cif":
        return 298, "0298K"
    match = re.fullmatch(r"T_(\d{4})K_cemc\.cif", filename)
    if not match:
        raise ValueError(filename)
    value = int(match.group(1))
    return value, f"{value:04d}K"


def find_temperature_files(slabs_root: Path) -> list[tuple[int, str, str]]:
    names: set[str] = set()
    for slab_dir in sorted(slabs_root.glob("slab_*")):
        for path in slab_dir.glob("*cemc.cif"):
            if path.name == "final_0298K_cemc.cif" or re.fullmatch(
                r"T_\d{4}K_cemc\.cif", path.name
            ):
                names.add(path.name)
    return sorted((temperature_key(name)[0], temperature_key(name)[1], name) for name in names)


def average_layer_pairs(
    slabs_root: Path, filename: str
) -> tuple[np.ndarray, list[int], list[list[int]]]:
    slab_dirs = sorted(path for path in slabs_root.glob("slab_*") if path.is_dir())
    if len(slab_dirs) != 20:
        raise ValueError(f"{slabs_root}: expected 20 runs, found {len(slab_dirs)}")

    counts = [{element: 0 for element in ELEMENTS} for _ in range(5)]
    totals = [0] * 5
    atom_counts: list[int] = []
    layer_sizes_all: list[list[int]] = []

    for slab_dir in slab_dirs:
        cif_path = slab_dir / filename
        if not cif_path.is_file():
            raise FileNotFoundError(cif_path)
        symbols, z = read_cif_symbols_and_z(cif_path)
        layers = split_layers(z)
        if len(layers) != 10:
            raise ValueError(f"{cif_path}: expected 10 layers, found {len(layers)}")
        unexpected = set(symbols) - set(ELEMENTS)
        if unexpected:
            raise ValueError(f"{cif_path}: unexpected elements {sorted(unexpected)}")
        atom_counts.append(len(symbols))
        layer_sizes_all.append([len(layer) for layer in layers])
        for pair_index in range(5):
            paired = np.concatenate([layers[pair_index], layers[9 - pair_index]])
            pair_counter = Counter(symbols[paired])
            for element in ELEMENTS:
                counts[pair_index][element] += pair_counter[element]
            totals[pair_index] += len(paired)

    fractions = np.asarray(
        [
            [100.0 * counts[row][element] / totals[row] for element in ELEMENTS]
            for row in range(5)
        ],
        dtype=float,
    )
    return fractions, atom_counts, layer_sizes_all


def draw_chart(fractions: np.ndarray, png_path: Path, pdf_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 10), constrained_layout=True)
    y = np.arange(5)
    left = np.zeros(5)
    for column, element in enumerate(ELEMENTS):
        values = fractions[:, column]
        ax.barh(
            y - 0.15,
            values,
            left=left,
            height=0.34,
            color=COLORS[element],
            edgecolor="none",
            label=element,
        )
        left += values

    # Place all values below the bars. Adjacent small-segment labels are spread
    # horizontally while preserving element order.
    for row in range(5):
        starts = np.r_[0.0, np.cumsum(fractions[row, :-1])]
        centers = starts + fractions[row] / 2.0
        label_x = np.clip(centers.copy(), 2.1, 97.9)
        minimum_separation = 4.4
        for _ in range(8):
            for column in range(1, len(label_x)):
                label_x[column] = max(
                    label_x[column], label_x[column - 1] + minimum_separation
                )
            if label_x[-1] > 97.9:
                label_x -= label_x[-1] - 97.9
            for column in range(len(label_x) - 2, -1, -1):
                label_x[column] = min(
                    label_x[column], label_x[column + 1] - minimum_separation
                )
            if label_x[0] < 2.1:
                label_x += 2.1 - label_x[0]

        for column, value in enumerate(fractions[row]):
            if value <= 0.0:
                continue
            target_x = float(label_x[column])
            label_y = y[row] + 0.17
            ax.text(
                target_x,
                label_y,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=24,
                color="black",
            )

    ax.set_yticks(y, PAIR_LABELS)
    ax.invert_yaxis()
    ax.set_ylim(4.58, -0.55)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Stoichiometric composition (%)", fontsize=28)
    ax.tick_params(axis="both", labelsize=24, width=1.5, length=7)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        frameon=False,
        columnspacing=1.4,
        handlelength=1.2,
        fontsize=32,
    )
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def write_temperature_csv(path: Path, fractions: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Layer pair", *[f"{e} (%)" for e in ELEMENTS], "Total (%)"])
        for label, values in zip(PAIR_LABELS, fractions):
            writer.writerow(
                [label, *[f"{value:.4f}" for value in values], f"{values.sum():.4f}"]
            )


def process_site(base: Path, facet: str, site: str) -> tuple[int, Path]:
    slabs_root = base / facet / site / "slabs"
    output_root = base / facet / "layer_composition_analysis" / site
    plot_dir = output_root / "plots_by_temperature"
    data_dir = output_root / "data_by_temperature"
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    temperatures = find_temperature_files(slabs_root)
    summary_path = output_root / f"fcc{facet}_{site}_layer_pair_element_fractions_all_temperatures.csv"
    validation_path = output_root / f"fcc{facet}_{site}_validation.csv"

    summary_rows: list[list[str]] = []
    validation_rows: list[list[str]] = []
    for temperature, tag, filename in temperatures:
        fractions, atom_counts, layer_sizes_all = average_layer_pairs(slabs_root, filename)
        stem = f"fcc{facet}_{site}_{tag}_layer_pair_element_fractions_20runs"
        draw_chart(fractions, plot_dir / f"{stem}.png", plot_dir / f"{stem}.pdf")
        write_temperature_csv(data_dir / f"{stem}.csv", fractions)
        for label, values in zip(PAIR_LABELS, fractions):
            summary_rows.append(
                [str(temperature), tag, label, *[f"{value:.4f}" for value in values]]
            )
        unique_atom_counts = sorted(set(atom_counts))
        unique_layer_sizes = sorted({tuple(sizes) for sizes in layer_sizes_all})
        validation_rows.append(
            [
                str(temperature),
                tag,
                filename,
                "20",
                ";".join(map(str, unique_atom_counts)),
                ";".join("/".join(map(str, sizes)) for sizes in unique_layer_sizes),
            ]
        )
        print(f"DONE fcc{facet}_{site} {tag}", flush=True)

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Temperature (K)", "Temperature tag", "Layer pair", *[f"{e} (%)" for e in ELEMENTS]])
        writer.writerows(summary_rows)
    with validation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Temperature (K)", "Temperature tag", "CIF filename", "Run count", "Atom counts", "Layer sizes"])
        writer.writerows(validation_rows)
    return len(temperatures), output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    args = parser.parse_args()
    total = 0
    for facet, site in SITE_ROOTS:
        count, output_root = process_site(args.base, facet, site)
        total += count
        print(f"OUTPUT {output_root} ({count} temperatures)")
    print(f"TOTAL_PLOTS={total}")


if __name__ == "__main__":
    main()
