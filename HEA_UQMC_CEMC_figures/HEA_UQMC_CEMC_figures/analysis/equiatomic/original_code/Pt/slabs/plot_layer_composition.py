from __future__ import annotations

import csv
import argparse
import shlex
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = Path("/home/jinsookim/HEA_MC/PtPdRhRuIr/equi-atomic/slabs")
ELEMENTS = ["Pd", "Pt", "Rh", "Ru", "Ir"]
COLORS = {
    "Pd": "#FFD700",
    "Pt": "#1F77B4",
    "Rh": "#FF9900",
    "Ru": "#E31A1C",
    "Ir": "#2CA02C",
}
PAIR_LABELS = ["Surface", "Subsurface", "3rd layer", "4th layer", "5th layer"]


def read_cif_symbols_and_z(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the atom-site symbol and fractional z columns from a CIF."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for start, line in enumerate(lines):
        if line.strip() != "loop_":
            continue
        headers = []
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
        symbols = []
        z_values = []
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
                symbol = "".join(character for character in symbol if character.isalpha())
            symbols.append(symbol)
            z_values.append(float(fields[z_col]))
            cursor += 1
        return np.asarray(symbols), np.asarray(z_values, dtype=float)
    raise ValueError(f"{path}: atom-site fractional-z loop not found")


def split_layers(z: np.ndarray, tolerance: float = 1.0e-5) -> list[np.ndarray]:
    """Return atom-index arrays for z-ordered layers (bottom to top)."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate counts across all 20 slabs. Because every layer has the same
    # number of sites, this is also the arithmetic mean of slab-level fractions.
    pair_counts = [{element: 0 for element in ELEMENTS} for _ in range(5)]
    pair_totals = [0] * 5
    slab_summaries = []

    for slab_index in range(1, 21):
        cif_path = root / f"slab_{slab_index:02d}" / "final_0298K_cemc.cif"
        if not cif_path.is_file():
            raise FileNotFoundError(cif_path)

        symbols, z = read_cif_symbols_and_z(cif_path)
        layers = split_layers(z)
        if len(layers) != 10:
            raise ValueError(f"{cif_path}: expected 10 layers, found {len(layers)}")

        layer_sizes = [len(layer) for layer in layers]
        slab_summaries.append((slab_index, len(symbols), layer_sizes))

        for pair_index in range(5):
            paired_indices = np.concatenate(
                [layers[pair_index], layers[9 - pair_index]]
            )
            counts = Counter(symbols[paired_indices])
            unexpected = set(counts) - set(ELEMENTS)
            if unexpected:
                raise ValueError(f"{cif_path}: unexpected elements {unexpected}")
            for element in ELEMENTS:
                pair_counts[pair_index][element] += counts[element]
            pair_totals[pair_index] += len(paired_indices)

    fractions = np.asarray(
        [
            [100.0 * pair_counts[row][element] / pair_totals[row] for element in ELEMENTS]
            for row in range(5)
        ]
    )

    csv_path = output_dir / "layer_pair_element_fractions_20slabs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Layer pair", *[f"{e} (%)" for e in ELEMENTS], "Total (%)"])
        for label, values in zip(PAIR_LABELS, fractions):
            writer.writerow([label, *[f"{value:.4f}" for value in values], f"{values.sum():.4f}"])

    fig, ax = plt.subplots(figsize=(18, 10), constrained_layout=True)
    y = np.arange(5)
    left = np.zeros(5)
    for column, element in enumerate(ELEMENTS):
        values = fractions[:, column]
        bars = ax.barh(
            y - 0.15,
            values,
            left=left,
            height=0.34,
            color=COLORS[element],
            edgecolor="none",
            label=element,
        )
        for row, (bar, value) in enumerate(zip(bars, values)):
            if value > 0.0:
                x = left[row] + value / 2.0
                label = f"{value:.1f}"
                horizontal_alignment = "center"
                if x < 3.5:
                    x = left[row] + 0.35
                    horizontal_alignment = "left"
                elif 0.0 < left[row] < 3.0 and value < 10.0:
                    x = left[row] + value - 0.35
                    horizontal_alignment = "right"
                ax.text(
                    x,
                    y[row] + 0.20,
                    label,
                    ha=horizontal_alignment,
                    va="center",
                    fontsize=24,
                    color="black",
                )
        left += values

    ax.set_yticks(y, PAIR_LABELS)
    ax.invert_yaxis()
    ax.set_ylim(4.55, -0.55)
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

    png_path = output_dir / "layer_pair_element_fractions_20slabs.png"
    pdf_path = output_dir / "layer_pair_element_fractions_20slabs.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    for slab_index, atom_count, layer_sizes in slab_summaries:
        print(f"slab_{slab_index:02d}: atoms={atom_count}, layer_sizes={layer_sizes}")
    print("\nAveraged layer-pair fractions (%)")
    for label, values in zip(PAIR_LABELS, fractions):
        print(label, ", ".join(f"{e}={v:.4f}" for e, v in zip(ELEMENTS, values)))
    print(f"\nPNG={png_path}")
    print(f"PDF={pdf_path}")
    print(f"CSV={csv_path}")


if __name__ == "__main__":
    main()
