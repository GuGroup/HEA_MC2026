from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from plot_layer_composition_by_temperature import ELEMENTS, PAIR_LABELS, draw_chart


BASE = Path("/home/jinsookim/HEA_MC/FeCoNiPtPd_HER/equi-atomic")
OUTPUT_DIR = BASE / "layer_composition_analysis" / "final_annealed"

# Values transcribed from the supplied final-annealed reference plot. The
# original equi-atomic CIF directories are no longer present at BASE, so only
# the one-decimal precision visible in that plot is available here.
FRACTIONS = np.asarray(
    [
        # The Fe segment is visibly nonzero in the reference but rounds to 0.0;
        # 0.025% is the smallest nonzero increment for 4,000 paired-layer sites.
        [86.1, 6.2, 1.3, 0.025, 6.3],
        [5.7, 13.5, 33.9, 26.3, 20.6],
        [0.6, 27.7, 17.9, 34.6, 19.2],
        [3.8, 26.8, 23.2, 20.3, 25.9],
        [3.9, 25.9, 23.7, 18.7, 27.9],
    ],
    dtype=float,
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "final_0298K_layer_pair_element_fractions_reference"
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    csv_path = OUTPUT_DIR / f"{stem}.csv"

    draw_chart(FRACTIONS, png_path, pdf_path)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Layer", *[f"{element} (%)" for element in ELEMENTS]])
        for label, values in zip(PAIR_LABELS, FRACTIONS):
            writer.writerow([label, *[f"{value:.1f}" for value in values]])

    note_path = OUTPUT_DIR / "README.txt"
    note_path.write_text(
        "This figure was replotted from the one-decimal values shown in the supplied "
        "final-annealed reference plot. The original equi-atomic facet/site CIF "
        "directories were not present under the project base on 2026-08-31.\n",
        encoding="utf-8",
    )
    print(f"PNG={png_path}")
    print(f"PDF={pdf_path}")
    print(f"CSV={csv_path}")
    print(f"NOTE={note_path}")


if __name__ == "__main__":
    main()
