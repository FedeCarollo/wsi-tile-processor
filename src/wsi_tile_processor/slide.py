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


from pathlib import Path

import numpy as np
import openslide
import tifffile



# ---------------------------------------------------------------------------
# Background Filter ABCs and built-in implementations
# ---------------------------------------------------------------------------



class _TifffileSlide:
    """Minimal openslide-compatible adapter around a tifffile pyramidal TIFF."""

    def __init__(self, path: str | Path):
        import zarr

        self._path = Path(path)
        self._tf = tifffile.TiffFile(str(path))
        self._series = self._tf.series[0]
        self._levels = self._series.levels

        z_root = zarr.open(self._series.aszarr(), mode="r")
        # zarr v3 uses zarr.Group; zarr v2 used zarr.hierarchy.Group
        _ZarrGroup = getattr(zarr, "Group", None) or getattr(
            getattr(zarr, "hierarchy", None), "Group", type(None)
        )
        if isinstance(z_root, _ZarrGroup):
            self._zarr_levels = [z_root[str(i)] for i in range(len(self._levels))]
        else:
            self._zarr_levels = [z_root]

    @property
    def level_count(self) -> int:
        return len(self._levels)

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        return tuple((lvl.shape[1], lvl.shape[0]) for lvl in self._levels)

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        w0, h0 = self.level_dimensions[0]
        return tuple(
            ((w0 / w) + (h0 / h)) / 2.0
            for w, h in self.level_dimensions
        )

    @property
    def properties(self) -> dict:
        props: dict = {}
        # 1. Prefer OME-XML PhysicalSizeX (correct in VALIS sub-IFD files)
        if self._tf.ome_metadata:
            import re

            match = re.search(
                r'PhysicalSizeX=[\'""]([0-9\\.]+)[\'""]', self._tf.ome_metadata
            )
            if match:
                try:
                    props["openslide.mpp-x"] = match.group(1)
                    return props
                except ValueError:
                    pass
        # 2. Fall back to TIFF XResolution tag
        page0 = self._levels[0].keyframe
        tags = page0.tags
        if "XResolution" in tags and "ResolutionUnit" in tags:
            try:
                xres = tags["XResolution"].value
                unit = tags["ResolutionUnit"].value
                px_per_unit = xres[0] / xres[1] if isinstance(xres, tuple) else float(xres)
                if unit == 3 and px_per_unit > 0:  # cm
                    props["openslide.mpp-x"] = str(10000.0 / px_per_unit)
                elif unit == 2 and px_per_unit > 0:  # inch
                    props["openslide.mpp-x"] = str(25400.0 / px_per_unit)
            except Exception:
                pass
        return props

    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ):
        from PIL import Image

        x0, y0 = location
        w, h = size
        ds = self.level_downsamples[level]
        lx0 = int(round(x0 / ds))
        ly0 = int(round(y0 / ds))

        z_level = self._zarr_levels[level]
        max_y, max_x = z_level.shape[:2]

        crop = z_level[ly0 : min(ly0 + h, max_y), lx0 : min(lx0 + w, max_x)]
        if crop.ndim == 2:
            crop = np.stack([crop] * 3, axis=-1)
        elif crop.shape[-1] == 4:
            crop = crop[..., :3]

        pad_y = max(0, (ly0 + h) - max_y)
        pad_x = max(0, (lx0 + w) - max_x)
        if pad_y > 0 or pad_x > 0:
            crop = np.pad(
                crop,
                ((0, pad_y), (0, pad_x), (0, 0)),
                mode="constant",
                constant_values=255,
            )
        return Image.fromarray(crop.astype(np.uint8), mode="RGB")

    def close(self):
        self._tf.close()


def _open_slide(
    path: str | Path, backend: str
) -> openslide.OpenSlide | _TifffileSlide:
    path_str = str(path)
    if backend == "openslide":
        return openslide.OpenSlide(path_str)
    elif backend == "tifffile":
        return _TifffileSlide(path_str)
    elif backend == "auto":
        os_slide = openslide.OpenSlide(path_str)
        try:
            tf_slide = _TifffileSlide(path_str)
            if tf_slide.level_count > os_slide.level_count:
                os_slide.close()
                return tf_slide
            tf_slide.close()
            return os_slide
        except Exception:
            return os_slide
    else:
        raise ValueError(
            f"Unknown backend {backend!r}; choose 'openslide', 'tifffile', or 'auto'."
        )


def _get_mpp_with_source(
    slide: openslide.OpenSlide | _TifffileSlide,
) -> tuple[float, str]:
    """Return level-0 microns/pixel and the metadata source used.

    TIFF XResolution is pixels per inch/cm, not microns per pixel. It must be
    converted using ResolutionUnit instead of being returned directly.
    """
    props = slide.properties
    for key in ("openslide.mpp-x", "openslide.mpp-y"):
        val = props.get(key)
        if val is not None:
            try:
                mpp = float(val)
                if np.isfinite(mpp) and mpp > 0:
                    return mpp, key
            except (TypeError, ValueError):
                pass

    unit = str(props.get("tiff.ResolutionUnit", "")).strip().lower()
    microns_per_unit = None
    if unit in {"2", "inch", "inches"}:
        microns_per_unit = 25400.0
    elif unit in {"3", "centimeter", "centimeters", "cm"}:
        microns_per_unit = 10000.0

    if microns_per_unit is not None:
        for key in ("tiff.XResolution", "tiff.YResolution"):
            try:
                pixels_per_unit = float(props.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(pixels_per_unit) and pixels_per_unit > 0:
                return microns_per_unit / pixels_per_unit, f"{key}+tiff.ResolutionUnit"

    raise ValueError(
        "No valid microns-per-pixel metadata found. Refusing to silently fall back "
        "to a default value because physical tile sizes would be unverified."
    )


def _get_mpp(slide: openslide.OpenSlide | _TifffileSlide) -> float:
    return _get_mpp_with_source(slide)[0]

