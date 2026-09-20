"""
WSI tile processor — sequential tiled processing with pyramidal OME-TIFF output.

Backends
--------
* ``"openslide"`` (default for SVS/NDPI/...) — uses openslide.
* ``"tifffile"`` — for OME-TIFF files saved with sub-IFD pyramid levels
  (e.g. output of valis ``warp_and_save_slides(..., pyramid=True)``),
  which openslide may expose as a single level.
* ``"auto"`` — opens with openslide first; if the file has more tifffile
  pyramid levels than openslide levels, switches to tifffile.

Architecture
------------
The primary processing contract is defined by the abstract base class:
* :class:`WSIProcessor`

And implemented by two concrete classes:
* :class:`FastWSIProcessor` — Direct streaming for non-overlapping tiles
  (stride == tile_size), using a uint8 memmap and pyvips pyramidization.
  Highly efficient for RGB virtual staining or any per-tile inference.
* :class:`GaussianWSIProcessor` — Overlapping tiles with smooth Gaussian
  blending across boundaries, using a float16 memmap accumulator before
  normalizing and pyramidizing with pyvips.

Background Filtering
--------------------
Tile-level background detection is handled by :class:`BackgroundFilter`
subclasses (or any callable with the same signature). Built-in filters:
* :class:`BrightnessBackgroundFilter` — brightness/saturation heuristic
  tuned for H&E and IHC slides (default).
* :class:`OtsuBackgroundFilter` — grayscale Otsu thresholding via scikit-image.
* :class:`SaturationBackgroundFilter` — purely saturation-based filter.

Tissue Mask Detection
---------------------
Coarse-level background skipping is handled by :class:`TissueMaskDetector`
subclasses. Built-in:
* :class:`BrightnessTissueMaskDetector` — low-res thumbnail thresholding
  + morphological cleanup.

A convenience factory :func:`process_wsi_to_ome_tiff` is provided for
backward compatibility.
"""

from __future__ import annotations

__version__ = "0.1.0"

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

__all__ = [
    # Abstract base classes
    "BackgroundFilter",
    "TissueMaskDetector",
    # Built-in background filters
    "BrightnessBackgroundFilter",
    "OtsuBackgroundFilter",
    "SaturationBackgroundFilter",
    # Built-in tissue mask detectors
    "BrightnessTissueMaskDetector",
    # Processors
    "WSIProcessor",
    "FastWSIProcessor",
    "GaussianWSIProcessor",
    # Pyramid utility
    "pyramidize_with_pyvips",
]


# ---------------------------------------------------------------------------
# Background Filter ABCs and built-in implementations
# ---------------------------------------------------------------------------



def _tile_origins(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= 0:
        return []
    origins = list(range(0, length, stride))
    last = max(0, length - tile_size)
    if not origins or origins[-1] < last:
        origins.append(last)
    return origins


def _make_weight_kernel(tile_size: int, sigma: float | None) -> np.ndarray:
    if sigma is None:
        return np.ones((tile_size, tile_size), dtype=np.float32)
    impulse = np.zeros((tile_size, tile_size), dtype=np.float64)
    cy, cx = tile_size // 2, tile_size // 2
    impulse[cy, cx] = 1.0
    kernel = gaussian_filter(impulse, sigma=sigma, mode="constant", cval=0.0)
    return np.clip(kernel, 1e-8, None).astype(np.float32)


def _pad_to(tile_rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = tile_rgb.shape[:2]
    out = np.full((size, size, 3), 255, dtype=np.uint8)
    out[:h, :w] = tile_rgb
    return out


def _to_output(raw: np.ndarray, output_channels: int) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    elif arr.ndim == 3 and arr.shape[-1] != output_channels:
        raise ValueError(
            f"generate returned {arr.shape[-1]} channels, expected {output_channels}"
        )
    return arr


# ---------------------------------------------------------------------------
# Pyvips pyramid builder
# ---------------------------------------------------------------------------


def pyramidize_with_pyvips(
    tiff_path: str | Path,
    Q: int = 95,
    tile_size: int = 512,
    verbose: bool = False,
) -> None:
    """Re-encode an existing flat TIFF as a full JPEG-compressed pyramid in-place.

    Parameters
    ----------
    tiff_path:
        Path to the BigTIFF file to pyramidize. Modified in-place.
    Q:
        JPEG quality (1-100). Default 95.
    tile_size:
        Internal tile size for the output pyramid. Default 512.
    verbose:
        Print progress messages. Default False.
    """
    import pyvips

    tiff_path = Path(tiff_path)
    tmp_path = tiff_path.with_suffix(".tmp.ome.tif")

    if verbose:
        print(f"  [pyvips] reading base level from {tiff_path} ...")
    img = pyvips.Image.new_from_file(str(tiff_path), access="sequential")

    if verbose:
        print(f"  [pyvips] writing pyramid (Q={Q}, tile={tile_size}x{tile_size}) ...")
    img.tiffsave(
        str(tmp_path),
        tile=True,
        tile_width=tile_size,
        tile_height=tile_size,
        pyramid=True,
        compression="jpeg",
        Q=Q,
        bigtiff=True,
        subifd=True,
    )
    tmp_path.replace(tiff_path)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

