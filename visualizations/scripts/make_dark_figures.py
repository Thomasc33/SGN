"""Generate dark-mode variants of the project-page figures.

Matplotlib writes figures on a white canvas, which reads as a glaring white
slab on the dark theme of the project page. Rather than re-running the whole
figure pipeline with a dark style (which needs the NTU data and the trained
checkpoints), this recolours the exported images directly.

Two treatments, chosen per figure:

``invert``
    Full lightness inversion in HLS space, preserving hue and saturation.
    Used for the schematic diagrams, where black text sits inside pastel
    boxes — both the box and the text have to flip for the text to stay
    readable.

``neutral``
    Invert only the low-saturation pixels: background, axes, gridlines and
    text. Saturated pixels are data marks (plasma colormaps, skeleton
    joints) and keep their colour, so "bright = high score" still holds.

Usage::

    python visualizations/scripts/make_dark_figures.py

Reads ``fig/`` and writes ``fig/dark/``.
"""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageSequence

# Lightness of the dark theme's surface and primary text, from index.html.
# White (L=1) maps to the surface, black (L=0) maps to the text colour.
SURFACE_L = 0.075
TEXT_L = 0.925

# Below this saturation a pixel is treated as chrome (background, text, axes)
# rather than data.
CHROME_SATURATION = 0.18

# Saturated colours are floored to this lightness so the dark end of the
# plasma colormap stays visible against a dark background.
DATA_MIN_L = 0.22

# Treatment per figure stem.
TREATMENTS = {
    "process": "invert",
    "pipeline": "invert",
    "teaser": "neutral",
    "sensitivity": "neutral",
    "attribution": "neutral",
    "heatmap": "neutral",
    "masking": "neutral",
    "noise": "neutral",
}
DEFAULT_TREATMENT = "neutral"


def _to_hls(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised RGB->HLS. ``rgb`` is float in [0, 1] with shape (..., 3)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    lightness = (mx + mn) / 2.0
    delta = mx - mn

    # Saturation
    denom = np.where(lightness < 0.5, mx + mn, 2.0 - mx - mn)
    saturation = np.divide(delta, denom, out=np.zeros_like(delta), where=denom > 1e-9)

    # Hue
    safe = np.where(delta > 1e-9, delta, 1.0)
    rc = (mx - r) / safe
    gc = (mx - g) / safe
    bc = (mx - b) / safe
    hue = np.where(mx == r, bc - gc, np.where(mx == g, 2.0 + rc - bc, 4.0 + gc - rc))
    hue = np.where(delta > 1e-9, (hue / 6.0) % 1.0, 0.0)

    return hue, lightness, saturation


def _from_hls(hue: np.ndarray, lightness: np.ndarray, saturation: np.ndarray) -> np.ndarray:
    """Vectorised HLS->RGB, mirroring ``colorsys.hls_to_rgb``."""
    m2 = np.where(lightness <= 0.5, lightness * (1.0 + saturation),
                  lightness + saturation - lightness * saturation)
    m1 = 2.0 * lightness - m2

    def channel(offset: float) -> np.ndarray:
        h = (hue + offset) % 1.0
        out = np.where(
            h < 1 / 6, m1 + (m2 - m1) * h * 6.0,
            np.where(
                h < 0.5, m2,
                np.where(h < 2 / 3, m1 + (m2 - m1) * (2 / 3 - h) * 6.0, m1),
            ),
        )
        return np.where(saturation < 1e-9, lightness, out)

    return np.stack([channel(1 / 3), channel(0.0), channel(-1 / 3)], axis=-1)


def _flip_lightness(lightness: np.ndarray) -> np.ndarray:
    """Map white -> dark surface and black -> light text, linearly."""
    return TEXT_L - (TEXT_L - SURFACE_L) * lightness


def recolour(rgb_u8: np.ndarray, treatment: str) -> np.ndarray:
    """Return a dark-theme version of an (H, W, 3) uint8 array."""
    rgb = rgb_u8.astype(np.float32) / 255.0
    hue, lightness, saturation = _to_hls(rgb)

    flipped = _flip_lightness(lightness)

    if treatment == "invert":
        new_l = flipped
        new_s = saturation
    else:
        chrome = saturation < CHROME_SATURATION
        new_l = np.where(chrome, flipped, np.maximum(lightness, DATA_MIN_L))
        # Fully desaturate chrome so faint colour casts in anti-aliased text
        # do not tint the inverted background.
        new_s = np.where(chrome, 0.0, saturation)

    out = _from_hls(hue, np.clip(new_l, 0.0, 1.0), np.clip(new_s, 0.0, 1.0))
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def convert_png(src: Path, dst: Path, treatment: str) -> None:
    img = Image.open(src)
    has_alpha = img.mode in ("RGBA", "LA") or "transparency" in img.info
    alpha = img.convert("RGBA").getchannel("A") if has_alpha else None

    out = Image.fromarray(recolour(np.array(img.convert("RGB")), treatment), "RGB")
    if alpha is not None:
        out = out.convert("RGBA")
        out.putalpha(alpha)

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst, optimize=True)


def _gif_frames(src: Image.Image, treatment: str) -> Iterator[Image.Image]:
    for frame in ImageSequence.Iterator(src):
        rgb = np.array(frame.convert("RGB"))
        yield Image.fromarray(recolour(rgb, treatment), "RGB").convert(
            "P", palette=Image.ADAPTIVE, colors=128
        )


def convert_gif(src: Path, dst: Path, treatment: str) -> None:
    with Image.open(src) as img:
        duration = img.info.get("duration", 80)
        loop = img.info.get("loop", 0)
        frames = list(_gif_frames(img, treatment))

    dst.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        dst, save_all=True, append_images=frames[1:],
        duration=duration, loop=loop, optimize=True, disposal=2,
    )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    fig_dir = root / "fig"
    dark_dir = fig_dir / "dark"

    if not fig_dir.is_dir():
        raise SystemExit(f"No fig/ directory at {fig_dir}")

    for src in sorted(fig_dir.glob("*.png")):
        treatment = TREATMENTS.get(src.stem, DEFAULT_TREATMENT)
        dst = dark_dir / src.name
        convert_png(src, dst, treatment)
        print(f"{src.relative_to(root)} -> {dst.relative_to(root)}  [{treatment}]")

    for src in sorted((fig_dir / "gif").glob("*.gif")):
        dst = dark_dir / "gif" / src.name
        convert_gif(src, dst, DEFAULT_TREATMENT)
        print(f"{src.relative_to(root)} -> {dst.relative_to(root)}  [{DEFAULT_TREATMENT}]")


if __name__ == "__main__":
    main()
