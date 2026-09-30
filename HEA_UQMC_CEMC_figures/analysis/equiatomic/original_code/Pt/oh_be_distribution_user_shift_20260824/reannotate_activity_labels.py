from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "oh_be_distribution_1x3_random_cemc_layer_shuffled.png"
METADATA_PATH = Path(
    r"Z:\HEA_MC\PtPdRhRuIr\equi-atomic\oh_be_distribution_user_shift_20260824"
) / "oh_be_distribution_1x3_random_cemc_layer_shuffled_metadata.json"


def format_sig_figs(value: float, digits: int = 2) -> str:
    """Format with exactly ``digits`` significant figures, including zeros."""
    if value == 0:
        return f"{value:.{digits - 1}f}"
    decimal_places = digits - math.floor(math.log10(abs(value))) - 1
    if decimal_places > 0:
        return f"{value:.{decimal_places}f}"
    return f"{value:.0f}"


def draw_condensed_text(
    image: Image.Image,
    position: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    width_scale: float = 0.93,
) -> None:
    """Draw 24 pt text with a subtle horizontal condensation."""
    layer = Image.new("RGBA", (1400, 180), (255, 255, 255, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((0, 0), text, fill="black", font=font, anchor="lt")
    bbox = layer.getbbox()
    if bbox is None:
        return
    glyphs = layer.crop(bbox)
    glyphs = glyphs.resize(
        (round(glyphs.width * width_scale), glyphs.height),
        resample=Image.Resampling.LANCZOS,
    )
    image.paste(glyphs, position, glyphs)


def main() -> None:
    image = Image.open(IMAGE_PATH).convert("RGB")
    if image.size != (5553, 1821):
        raise ValueError(f"Unexpected image size: {image.size}")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    activities = metadata["activities"]
    draw = ImageDraw.Draw(image)

    # Remove the current 28 pt annotations while leaving the curves untouched.
    old_boxes = (
        (270, 585, 1230, 720),
        (2100, 175, 3010, 320),
        (3885, 175, 4845, 320),
    )
    for box in old_boxes:
        draw.rectangle(box, fill="white")

    # Restore the short portion of the Homogeneous optimum line covered by the
    # old annotation box.  Its 300-dpi Matplotlib dash pattern is 32 px on and
    # 14–15 px off.
    for y0 in (588, 634):
        draw.line((1290, y0, 1290, y0 + 31), fill="black", width=7)

    dpi = image.info.get("dpi", (300.0, 300.0))[0]
    font_px = round(24 * dpi / 72)
    font_path = (
        Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    )
    font = ImageFont.truetype(str(font_path), font_px)

    labels = (
        ("Homogeneous", (282, 603)),
        ("CEMC", (2118, 190)),
        ("CEMC + layer shuffled", (3902, 190)),
    )
    for method, position in labels:
        text = f"Activity = {format_sig_figs(float(activities[method]))}"
        draw_condensed_text(image, position, text, font)

    image.save(IMAGE_PATH, dpi=(dpi, dpi))
    print(f"Updated {IMAGE_PATH}")
    for method, _ in labels:
        print(f"{method}: Activity = {format_sig_figs(float(activities[method]))}")


if __name__ == "__main__":
    main()
