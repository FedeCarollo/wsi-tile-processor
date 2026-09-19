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

import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np
import openslide
import tifffile
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


class BackgroundFilter(ABC):
    """Abstract base class for tile-level background detection.

    Subclass this to implement custom background rejection logic.
    The ``__call__`` method receives an RGB uint8 tile and returns
    ``True`` if the tile should be treated as background (skipped).
    """

    @abstractmethod
    def __call__(self, tile_rgb: np.ndarray) -> bool:
        """Return ``True`` if *tile_rgb* is background.

        Parameters
        ----------
        tile_rgb:
            HxWx3 uint8 numpy array in RGB order.

        Returns
        -------
        bool
            ``True``  → tile is background and will be filled with
            ``background_value``.
            ``False`` → tile is tissue and will be passed to ``generate``.
        """


class BrightnessBackgroundFilter(BackgroundFilter):
    """Brightness + optional saturation heuristic for H&E / IHC slides.

    This is the default filter used by :class:`WSIProcessor` when no
    explicit ``background_filter`` is provided.  It was tuned for adrenal
    gland H&E and IHC slides but works well for most brightfield histology.

    A tile is classified as background when **any** of the following holds:

    1. **Brightness** — the fraction of pixels where all three channels
       exceed *threshold_intensity* (default 215) is greater than
       *threshold* (default 0.70).  This rejects both pure white background
       (>95 % bright) and adipose tissue vacuoles (75-90 % bright).
    2. **Low variance** (optional, off by default) — the grayscale variance
       is below *min_variance*.
    3. **Low saturation** (optional, off by default) — the fraction of pixels
       with HSV saturation above *min_sat* is below *tissue_threshold*.

    Parameters
    ----------
    threshold:
        Fraction of pixels that must be "bright" to call the tile background.
        Default 0.70 (70 %).
    threshold_intensity:
        Per-channel intensity cutoff to call a pixel "bright". Default 215.
    min_variance:
        Minimum grayscale variance.  0 disables the check.
    min_sat:
        Minimum per-pixel saturation (0-255 scale) for the saturation check.
        0 disables it.
    tissue_threshold:
        Minimum fraction of saturated pixels required to keep the tile.
        Only used when *min_sat* > 0.
    """

    def __init__(
        self,
        threshold: float = 0.70,
        threshold_intensity: int = 215,
        min_variance: float = 0.0,
        min_sat: float = 0.0,
        tissue_threshold: float = 0.2,
    ) -> None:
        self.threshold = threshold
        self.threshold_intensity = threshold_intensity
        self.min_variance = min_variance
        self.min_sat = min_sat
        self.tissue_threshold = tissue_threshold

    def __call__(self, tile_rgb: np.ndarray) -> bool:
        if tile_rgb.size == 0:
            return True

        # 1. Brightness check
        bright_mask = np.all(tile_rgb > self.threshold_intensity, axis=-1)
        if float(bright_mask.mean()) > self.threshold:
            return True

        # 2. Variance check (optional)
        if self.min_variance > 0:
            gray = np.dot(tile_rgb[..., :3], [0.299, 0.587, 0.114])
            if float(np.var(gray)) < self.min_variance:
                return True

        # 3. Saturation check (optional)
        if self.min_sat > 0 and self.tissue_threshold > 0:
            rgb = tile_rgb[..., :3]
            cmax = rgb.max(axis=-1)
            cmin = rgb.min(axis=-1)
            delta = cmax - cmin
            sat = np.where(cmax > 0, (delta.astype(np.float32) / cmax) * 255.0, 0.0)
            sat_ratio = float((sat > self.min_sat).mean())
            if sat_ratio < self.tissue_threshold:
                return True

        return False


class OtsuBackgroundFilter(BackgroundFilter):
    """Background detection via Otsu thresholding on the grayscale image.

    Requires ``scikit-image`` (``pip install scikit-image`` or
    ``pip install wsi-tile-processor[full]``).

    A tile is classified as background if the fraction of pixels above the
    Otsu threshold exceeds *background_fraction* (default 0.70).

    Parameters
    ----------
    background_fraction:
        Fraction of pixels classified as "light" (above Otsu threshold) for
        the tile to be considered background.  Default 0.70.
    """

    def __init__(self, background_fraction: float = 0.70) -> None:
        self.background_fraction = background_fraction

    def __call__(self, tile_rgb: np.ndarray) -> bool:
        if tile_rgb.size == 0:
            return True
        try:
            from skimage.filters import threshold_otsu
        except ImportError as exc:
            raise ImportError(
                "OtsuBackgroundFilter requires scikit-image. "
                "Install it with: pip install scikit-image"
            ) from exc

        gray = np.dot(tile_rgb[..., :3].astype(np.float32), [0.299, 0.587, 0.114])
        # Degenerate case: uniform tile (zero variance) — Otsu is undefined.
        # Fall back to a simple mean brightness check.
        if gray.std() < 1.0:
            return float(gray.mean()) > 200.0
        thresh = threshold_otsu(gray)
        return float((gray > thresh).mean()) > self.background_fraction


class SaturationBackgroundFilter(BackgroundFilter):
    """Background detection based purely on HSV saturation.

    A tile is background when the mean saturation (0-255 scale) is below
    *min_mean_saturation*.  Fast and simple; works best on stained tissue
    where background is nearly achromatic.

    Parameters
    ----------
    min_mean_saturation:
        Mean saturation threshold (0-255 scale). Tiles with mean saturation
        below this value are classified as background.  Default 15.
    """

    def __init__(self, min_mean_saturation: float = 15.0) -> None:
        self.min_mean_saturation = min_mean_saturation

    def __call__(self, tile_rgb: np.ndarray) -> bool:
        if tile_rgb.size == 0:
            return True
        rgb = tile_rgb[..., :3].astype(np.float32)
        cmax = rgb.max(axis=-1)
        cmin = rgb.min(axis=-1)
        delta = cmax - cmin
        sat = np.where(cmax > 0, (delta / cmax) * 255.0, 0.0)
        return float(sat.mean()) < self.min_mean_saturation


# ---------------------------------------------------------------------------
# Tissue Mask Detector ABCs and built-in implementations
# ---------------------------------------------------------------------------


class TissueMaskDetector(ABC):
    """Abstract base class for coarse-level tissue mask computation.

    The tissue mask is computed once on a low-resolution thumbnail of the
    slide and used to skip entire tiles that fall entirely within background
    regions before reading the full-resolution tile.

    Subclass this and implement ``__call__`` to provide custom tissue
    detection logic (e.g. deep learning-based segmentation).
    """

    @abstractmethod
    def __call__(self, slide: object) -> tuple[np.ndarray, float]:
        """Compute a boolean tissue mask from a slide object.

        Parameters
        ----------
        slide:
            An openslide-compatible slide object (has ``level_count``,
            ``level_dimensions``, ``level_downsamples``, ``read_region``).

        Returns
        -------
        mask : np.ndarray
            2-D boolean array (H × W) at a reduced resolution.
            ``True`` → tissue present, ``False`` → background.
        downsample : float
            The downsample factor of the mask relative to level-0 pixel
            coordinates.  Used to map level-0 coordinates into mask
            coordinates.
        """


class BrightnessTissueMaskDetector(TissueMaskDetector):
    """Tissue mask via brightness thresholding on a low-res thumbnail.

    Reads the lowest pyramid level that fits within *max_side* × *max_side*
    pixels, converts to grayscale, thresholds at *intensity_threshold*, and
    applies morphological opening + closing to remove noise and close gaps.

    Requires ``scikit-image`` for the morphological operations
    (``pip install scikit-image`` or ``pip install wsi-tile-processor[full]``).

    Parameters
    ----------
    intensity_threshold:
        Grayscale intensity below which a pixel is considered tissue.
        Default 215 (pixels brighter than this are background).
    max_side:
        Maximum side length (px) for the thumbnail used for mask computation.
        Default 4000.
    open_radius:
        Disk radius for morphological opening (noise removal). Default 2.
    close_radius:
        Disk radius for morphological closing (gap filling). Default 4.
    verbose:
        Print progress information. Default False.
    """

    def __init__(
        self,
        intensity_threshold: int = 215,
        max_side: int = 4000,
        open_radius: int = 2,
        close_radius: int = 4,
        verbose: bool = False,
    ) -> None:
        self.intensity_threshold = intensity_threshold
        self.max_side = max_side
        self.open_radius = open_radius
        self.close_radius = close_radius
        self.verbose = verbose

    def __call__(self, slide: object) -> tuple[np.ndarray, float]:
        try:
            from skimage import morphology
        except ImportError as exc:
            raise ImportError(
                "BrightnessTissueMaskDetector requires scikit-image. "
                "Install it with: pip install scikit-image"
            ) from exc

        target_level = slide.level_count - 1
        for i, (w, h) in enumerate(slide.level_dimensions):
            if w <= self.max_side and h <= self.max_side:
                target_level = i
                break

        downsample = slide.level_downsamples[target_level]
        dims = slide.level_dimensions[target_level]

        if self.verbose:
            print(
                f"  computing tissue mask at level {target_level} "
                f"(dims={dims}, downsample={downsample:.2f}) ..."
            )

        thumb = slide.read_region((0, 0), target_level, dims).convert("RGB")
        thumb_np = np.array(thumb)

        gray = np.dot(thumb_np[..., :3], [0.299, 0.587, 0.114])
        mask = gray < self.intensity_threshold

        mask = morphology.opening(mask, morphology.disk(self.open_radius))
        mask = morphology.closing(mask, morphology.disk(self.close_radius))

        return mask, downsample


# ---------------------------------------------------------------------------
# Slide backend abstraction
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


# ---------------------------------------------------------------------------
# Internal helpers
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


class WSIProcessor(ABC):
    """Abstract base class for tile-by-tile WSI processing into pyramidal OME-TIFF.

    Parameters
    ----------
    wsi_path:
        Path to the input whole slide image.
    tiff_path:
        Destination path for the output OME-TIFF.
    level:
        Pyramid level of *wsi_path* to process (0 = full resolution).
    tile_size:
        Side length (px) of each square tile fed to *generate*.
    generate:
        Callable that maps an HxWx3 uint8 numpy array (or a batch BxHxWx3)
        to an output array.  Float outputs in [0, 1] are scaled to uint8.
    output_channels:
        Number of channels in the output (e.g. 3 for RGB, 1 for grayscale).
    background_filter:
        A :class:`BackgroundFilter` instance, a plain ``callable(tile_rgb)
        -> bool``, or ``None``.  When ``None``, a
        :class:`BrightnessBackgroundFilter` is constructed from the
        *background_threshold*, *min_variance*, *min_sat*, and
        *tissue_threshold* keyword arguments (backward-compatible defaults).
    background_threshold:
        Passed to :class:`BrightnessBackgroundFilter` when
        *background_filter* is ``None``.  Ignored otherwise.
    background_value:
        Fill value for background tiles.  Float in [0, 1] is scaled to
        uint8; integer is used directly. Default 1.0 (white).
    min_variance:
        Passed to :class:`BrightnessBackgroundFilter` when
        *background_filter* is ``None``.
    min_sat:
        Passed to :class:`BrightnessBackgroundFilter` when
        *background_filter* is ``None``.
    tissue_threshold:
        Passed to :class:`BrightnessBackgroundFilter` when
        *background_filter* is ``None``.
    tissue_mask_detector:
        A :class:`TissueMaskDetector` instance for coarse background
        skipping, or ``None``.  When not ``None``, the mask is computed
        once before tile iteration and used to skip tiles that fall
        entirely within masked-out regions.  Supersedes *use_tissue_mask*.
    use_tissue_mask:
        Convenience flag: when ``True`` and *tissue_mask_detector* is
        ``None``, a :class:`BrightnessTissueMaskDetector` is used.
    jpeg_quality:
        JPEG compression quality for the output OME-TIFF. Default 95.
    tiff_chunk:
        Internal tile size for the TIFF writer and pyvips pyramid.
        Default 512.
    batch_size:
        Number of tiles per *generate* call.  1 disables batching.
    tmp_dir:
        Directory for temporary memmap files.  Uses the system temp dir
        when ``None``.
    backend:
        Slide reading backend: ``"openslide"``, ``"tifffile"``, or
        ``"auto"`` (default).
    verbose:
        Print progress messages. Default False.
    """

    def __init__(
        self,
        wsi_path: str | Path,
        tiff_path: str | Path,
        level: int,
        tile_size: int,
        generate: Callable[[np.ndarray], np.ndarray],
        output_channels: int = 3,
        background_filter: BackgroundFilter | Callable[[np.ndarray], bool] | None = None,
        background_threshold: float = 0.70,
        background_value: float = 1.0,
        min_variance: float = 0.0,
        min_sat: float = 0.0,
        tissue_threshold: float = 0.2,
        tissue_mask_detector: TissueMaskDetector | None = None,
        use_tissue_mask: bool = False,
        jpeg_quality: int = 95,
        tiff_chunk: int = 512,
        batch_size: int = 1,
        tmp_dir: str | Path | None = None,
        backend: str = "auto",
        verbose: bool = False,
    ):
        self.wsi_path = Path(wsi_path)
        self.tiff_path = Path(tiff_path)
        self.level = level
        self.tile_size = tile_size
        self.generate = generate
        self.output_channels = output_channels

        # Resolve background filter
        if background_filter is not None:
            self.background_filter: Callable[[np.ndarray], bool] = background_filter
        else:
            self.background_filter = BrightnessBackgroundFilter(
                threshold=background_threshold,
                min_variance=min_variance,
                min_sat=min_sat,
                tissue_threshold=tissue_threshold,
            )

        # Keep legacy params accessible for subclass use / backward compat
        self.background_threshold = background_threshold
        self.background_value = background_value
        self.min_variance = min_variance
        self.min_sat = min_sat
        self.tissue_threshold = tissue_threshold

        # Resolve tissue mask detector
        if tissue_mask_detector is not None:
            self._tissue_mask_detector: TissueMaskDetector | None = tissue_mask_detector
        elif use_tissue_mask:
            self._tissue_mask_detector = BrightnessTissueMaskDetector(verbose=verbose)
        else:
            self._tissue_mask_detector = None

        self.use_tissue_mask = use_tissue_mask
        self.jpeg_quality = jpeg_quality
        self.tiff_chunk = tiff_chunk
        self.batch_size = batch_size
        self.tmp_dir = tmp_dir
        self.backend = backend
        self.verbose = verbose

    @abstractmethod
    def process(self) -> None:
        """Execute tile-by-tile processing and write the output OME-TIFF."""

    def _is_tile_background(self, tile_rgb: np.ndarray) -> bool:
        """Return ``True`` if *tile_rgb* should be treated as background."""
        return self.background_filter(tile_rgb)

    def _compute_tissue_mask(self, slide: object) -> tuple[np.ndarray, float]:
        """Compute and return the global tissue mask using the configured detector."""
        if self._tissue_mask_detector is None:
            raise RuntimeError(
                "_compute_tissue_mask called but no tissue_mask_detector is configured."
            )
        return self._tissue_mask_detector(slide)

    def _build_pyramid_pyvips(self) -> None:
        """Build the JPEG OME-TIFF pyramid in-place using pyvips."""
        if self.verbose:
            print("  building OME-TIFF pyramid with pyvips ...")
        pyramidize_with_pyvips(
            self.tiff_path,
            Q=self.jpeg_quality,
            tile_size=self.tiff_chunk,
            verbose=self.verbose,
        )


# ---------------------------------------------------------------------------
# Concrete processor: fast (non-overlapping)
# ---------------------------------------------------------------------------


class FastWSIProcessor(WSIProcessor):
    """Processor for non-overlapping tiles (stride == tile_size).

    Streams tiles directly into a uint8 binary memmap, writes the base
    BigTIFF, and builds the pyramid in-place via pyvips.  Requires zero
    floating-point accumulators or overlapping buffers.

    This is the recommended processor for most use cases (virtual staining,
    classification heatmaps) where strict tile alignment is acceptable.
    """

    def process(self) -> None:
        slide = _open_slide(self.wsi_path, self.backend)
        level_w, level_h = slide.level_dimensions[self.level]
        downsample = slide.level_downsamples[self.level]
        mpp = _get_mpp(slide) * downsample

        if self.verbose:
            print(
                f"[FastWSIProcessor] level={self.level}, dims=({level_w}x{level_h}), "
                f"downsample={downsample:.2f}, mpp@level={mpp:.4f} um/px"
                + (f", batch_size={self.batch_size}" if self.batch_size > 1 else "")
            )

        tmp_dir_path = Path(tempfile.mkdtemp(prefix="_wsi_fast_", dir=self.tmp_dir))
        tmp_bin = tmp_dir_path / "base_level.bin"

        try:
            shape = (
                (level_h, level_w, self.output_channels)
                if self.output_channels > 1
                else (level_h, level_w)
            )
            mm = np.memmap(str(tmp_bin), dtype=np.uint8, mode="w+", shape=shape)

            bg_val_uint8 = (
                int(self.background_value * 255)
                if isinstance(self.background_value, float)
                and self.background_value <= 1.0
                else int(self.background_value)
            )

            origins_y = list(range(0, level_h, self.tile_size))
            origins_x = list(range(0, level_w, self.tile_size))
            total = len(origins_y) * len(origins_x)

            _buf_tiles: list[np.ndarray] = []
            _buf_meta: list[tuple] = []

            def _flush_batch() -> None:
                if not _buf_tiles:
                    return
                batch = np.stack(_buf_tiles)
                batch_out = self.generate(batch)
                arr = np.asarray(batch_out)
                if np.issubdtype(arr.dtype, np.floating):
                    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
                if self.output_channels == 1:
                    if arr.ndim == 4:
                        arr = arr[:, :, :, 0]
                else:
                    if arr.ndim == 3:
                        arr = arr[:, :, :, np.newaxis]
                if self.output_channels > 1 and arr.shape[-1] != self.output_channels:
                    raise ValueError(
                        f"generate batch returned {arr.shape[-1]} channel(s), "
                        f"expected {self.output_channels}"
                    )
                for i, (bty, btx, brh, brw) in enumerate(_buf_meta):
                    tile_out = arr[i]
                    if self.output_channels == 1:
                        mm[bty : bty + brh, btx : btx + brw] = tile_out[:brh, :brw]
                    else:
                        mm[bty : bty + brh, btx : btx + brw, :] = tile_out[
                            :brh, :brw, :
                        ]
                _buf_tiles.clear()
                _buf_meta.clear()

            # Compute coarse tissue mask once (if a detector is configured)
            tissue_mask_np, mask_downsample = None, 1.0
            if self._tissue_mask_detector is not None:
                tissue_mask_np, mask_downsample = self._compute_tissue_mask(slide)

            for tile_idx, (ty, tx) in enumerate(
                (y, x) for y in origins_y for x in origins_x
            ):
                if self.verbose and tile_idx % max(1, total // 20) == 0:
                    print(
                        f"  tile {tile_idx + 1}/{total}  "
                        f"({100 * (tile_idx + 1) // total}%)",
                        flush=True,
                    )

                read_w = min(self.tile_size, level_w - tx)
                read_h = min(self.tile_size, level_h - ty)
                if read_w <= 0 or read_h <= 0:
                    continue
                x0 = int(tx * downsample)
                y0 = int(ty * downsample)

                # Coarse mask check: skip tiles fully outside tissue
                is_bg = False
                if tissue_mask_np is not None:
                    mx0 = int(x0 / mask_downsample)
                    my0 = int(y0 / mask_downsample)
                    mw = max(1, int((read_w * downsample) / mask_downsample))
                    mh = max(1, int((read_h * downsample) / mask_downsample))
                    mask_patch = tissue_mask_np[my0 : my0 + mh, mx0 : mx0 + mw]
                    if mask_patch.size == 0 or not np.any(mask_patch):
                        is_bg = True

                tile_rgb = None
                if not is_bg:
                    pil_tile = slide.read_region(
                        (x0, y0), self.level, (read_w, read_h)
                    ).convert("RGB")
                    tile_rgb = np.array(pil_tile, dtype=np.uint8)

                    # Fine-grained high-resolution background check on tiles
                    # that passed the coarse mask
                    is_bg = self._is_tile_background(tile_rgb)

                if is_bg:
                    _flush_batch()
                    if self.output_channels == 1:
                        mm[ty : ty + read_h, tx : tx + read_w] = bg_val_uint8
                    else:
                        mm[ty : ty + read_h, tx : tx + read_w, :] = bg_val_uint8
                else:
                    padded = (
                        _pad_to(tile_rgb, self.tile_size)
                        if (read_w < self.tile_size or read_h < self.tile_size)
                        else tile_rgb
                    )
                    if self.batch_size <= 1:
                        out = self.generate(padded)
                        arr = np.asarray(out)
                        if np.issubdtype(arr.dtype, np.floating):
                            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                        else:
                            arr = arr.astype(np.uint8)
                        if self.output_channels == 1:
                            if arr.ndim == 3:
                                arr = arr[:, :, 0]
                            mm[ty : ty + read_h, tx : tx + read_w] = arr[
                                :read_h, :read_w
                            ]
                        else:
                            if arr.ndim == 2:
                                arr = arr[:, :, np.newaxis]
                            mm[ty : ty + read_h, tx : tx + read_w, :] = arr[
                                :read_h, :read_w, :
                            ]
                    else:
                        _buf_tiles.append(padded)
                        _buf_meta.append((ty, tx, read_h, read_w))
                        if len(_buf_tiles) >= self.batch_size:
                            _flush_batch()

            _flush_batch()
            slide.close()
            mm.flush()

            if self.verbose:
                print("  writing base level to TIFF from fast memmap ...")

            resolution = (1e4 / mpp, 1e4 / mpp)
            photometric = "minisblack" if self.output_channels == 1 else "rgb"

            tmp_tiff_path = self.tiff_path.with_suffix(".tmp.ome.tiff")

            with tifffile.TiffWriter(str(tmp_tiff_path), bigtiff=True) as tif:
                tif.write(
                    mm,
                    tile=(self.tiff_chunk, self.tiff_chunk),
                    compression="jpeg",
                    compressionargs={"level": self.jpeg_quality},
                    photometric=photometric,
                    resolution=resolution,
                    resolutionunit=tifffile.RESUNIT.CENTIMETER,
                    subfiletype=0,
                    metadata=None,
                )

            del mm
            tmp_tiff_path.rename(self.tiff_path)
            self._build_pyramid_pyvips()

            if self.verbose:
                print(f"  OME-TIFF written to: {self.tiff_path}", flush=True)

        finally:
            shutil.rmtree(tmp_dir_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Concrete processor: Gaussian blending (overlapping tiles)
# ---------------------------------------------------------------------------


class GaussianWSIProcessor(WSIProcessor):
    """Processor for overlapping tiles (stride < tile_size) with Gaussian blending.

    Accumulates weighted tile predictions into temporary float16 memmaps on
    disk, normalizes block-by-block into uint8, streams the base BigTIFF, and
    builds the pyramid in-place via pyvips.

    The Gaussian weight kernel ensures smooth transitions between overlapping
    tiles, eliminating the grid artifacts that would otherwise appear with
    naive averaging or overwriting.

    Use this processor when:

    * Your model produces boundary artifacts that need to be averaged away.
    * You want maximum output quality at the cost of higher memory/disk usage
      and longer processing time.

    Parameters
    ----------
    stride:
        Step between consecutive tile origins (px). Must be ≤ *tile_size*.
        Smaller strides produce smoother blending but increase computation.
    blur_sigma:
        Standard deviation of the Gaussian weight kernel (px).  ``None``
        uses a uniform (box) kernel — equivalent to plain averaging.
        Default 64.0.

    All other parameters are inherited from :class:`WSIProcessor`.
    """

    def __init__(
        self,
        wsi_path: str | Path,
        tiff_path: str | Path,
        level: int,
        tile_size: int,
        stride: int,
        generate: Callable[[np.ndarray], np.ndarray],
        blur_sigma: float | None = 64.0,
        output_channels: int = 3,
        background_filter: BackgroundFilter | Callable[[np.ndarray], bool] | None = None,
        background_threshold: float = 0.70,
        background_value: float = 1.0,
        min_variance: float = 0.0,
        min_sat: float = 0.0,
        tissue_threshold: float = 0.2,
        tissue_mask_detector: TissueMaskDetector | None = None,
        use_tissue_mask: bool = False,
        jpeg_quality: int = 95,
        tiff_chunk: int = 512,
        batch_size: int = 1,
        tmp_dir: str | Path | None = None,
        backend: str = "auto",
        verbose: bool = False,
    ):
        super().__init__(
            wsi_path=wsi_path,
            tiff_path=tiff_path,
            level=level,
            tile_size=tile_size,
            generate=generate,
            output_channels=output_channels,
            background_filter=background_filter,
            background_threshold=background_threshold,
            background_value=background_value,
            min_variance=min_variance,
            min_sat=min_sat,
            tissue_threshold=tissue_threshold,
            tissue_mask_detector=tissue_mask_detector,
            use_tissue_mask=use_tissue_mask,
            jpeg_quality=jpeg_quality,
            tiff_chunk=tiff_chunk,
            batch_size=batch_size,
            tmp_dir=tmp_dir,
            backend=backend,
            verbose=verbose,
        )
        if stride > tile_size:
            raise ValueError(
                f"stride ({stride}) must be <= tile_size ({tile_size})"
            )
        self.stride = stride
        self.blur_sigma = blur_sigma

    def process(self) -> None:
        slide = _open_slide(self.wsi_path, self.backend)
        level_w, level_h = slide.level_dimensions[self.level]
        downsample = slide.level_downsamples[self.level]
        mpp = _get_mpp(slide) * downsample

        if self.verbose:
            print(
                f"[GaussianWSIProcessor] level={self.level}, "
                f"dims=({level_w}x{level_h}), stride={self.stride}, "
                f"downsample={downsample:.2f}, mpp@level={mpp:.4f} um/px"
                + (f", batch_size={self.batch_size}" if self.batch_size > 1 else "")
            )

        tmp_dir_path = Path(
            tempfile.mkdtemp(prefix="_wsi_gaussian_", dir=self.tmp_dir)
        )
        tmp_acc = tmp_dir_path / "acc.bin"
        tmp_wgt = tmp_dir_path / "wgt.bin"
        tmp_out = tmp_dir_path / "out.bin"

        try:
            shape = (
                (level_h, level_w, self.output_channels)
                if self.output_channels > 1
                else (level_h, level_w)
            )
            acc = np.memmap(str(tmp_acc), dtype=np.float16, mode="w+", shape=shape)
            wgt = np.memmap(
                str(tmp_wgt), dtype=np.float16, mode="w+", shape=(level_h, level_w)
            )

            weight_kernel = _make_weight_kernel(self.tile_size, self.blur_sigma)
            origins_y = _tile_origins(level_h, self.tile_size, self.stride)
            origins_x = _tile_origins(level_w, self.tile_size, self.stride)
            total = len(origins_y) * len(origins_x)

            _buf_tiles: list[np.ndarray] = []
            _buf_meta: list[tuple] = []

            def _flush_batch() -> None:
                if not _buf_tiles:
                    return
                batch = np.stack(_buf_tiles)
                batch_out = self.generate(batch)
                arr = np.asarray(batch_out, dtype=np.float32)
                if self.output_channels == 1:
                    if arr.ndim == 4:
                        arr = arr[:, :, :, 0]
                else:
                    if arr.ndim == 3:
                        arr = arr[:, :, :, np.newaxis]
                if self.output_channels > 1 and arr.shape[-1] != self.output_channels:
                    raise ValueError(
                        f"generate batch returned {arr.shape[-1]} channel(s), "
                        f"expected {self.output_channels}"
                    )
                for i, (bty, btx, brh, brw) in enumerate(_buf_meta):
                    tile_out = arr[i]
                    k = weight_kernel[:brh, :brw]
                    k16 = k.astype(np.float16)
                    if self.output_channels == 1:
                        weighted = (tile_out[:brh, :brw] * k16).astype(np.float16)
                        acc[bty : bty + brh, btx : btx + brw] += weighted
                    else:
                        weighted = (
                            tile_out[:brh, :brw, :] * k16[:, :, np.newaxis]
                        ).astype(np.float16)
                        acc[bty : bty + brh, btx : btx + brw, :] += weighted
                    wgt[bty : bty + brh, btx : btx + brw] += k16
                _buf_tiles.clear()
                _buf_meta.clear()

            # Compute coarse tissue mask once (if a detector is configured)
            tissue_mask_np, mask_downsample = None, 1.0
            if self._tissue_mask_detector is not None:
                tissue_mask_np, mask_downsample = self._compute_tissue_mask(slide)

            for tile_idx, (ty, tx) in enumerate(
                (y, x) for y in origins_y for x in origins_x
            ):
                if self.verbose and tile_idx % max(1, total // 20) == 0:
                    print(
                        f"  tile {tile_idx + 1}/{total}  "
                        f"({100 * (tile_idx + 1) // total}%)",
                        flush=True,
                    )

                read_w = min(self.tile_size, level_w - tx)
                read_h = min(self.tile_size, level_h - ty)
                if read_w <= 0 or read_h <= 0:
                    continue
                x0 = int(tx * downsample)
                y0 = int(ty * downsample)

                # Coarse mask check: skip tiles fully outside tissue
                is_bg = False
                if tissue_mask_np is not None:
                    mx0 = int(x0 / mask_downsample)
                    my0 = int(y0 / mask_downsample)
                    mw = max(1, int((read_w * downsample) / mask_downsample))
                    mh = max(1, int((read_h * downsample) / mask_downsample))
                    mask_patch = tissue_mask_np[my0 : my0 + mh, mx0 : mx0 + mw]
                    if mask_patch.size == 0 or not np.any(mask_patch):
                        is_bg = True

                tile_rgb = None
                if not is_bg:
                    pil_tile = slide.read_region(
                        (x0, y0), self.level, (read_w, read_h)
                    ).convert("RGB")
                    tile_rgb = np.array(pil_tile, dtype=np.uint8)

                    # Fine-grained high-resolution background check on tiles
                    # that passed the coarse mask
                    is_bg = self._is_tile_background(tile_rgb)

                if is_bg:
                    _flush_batch()
                    k = weight_kernel[:read_h, :read_w]
                    k16 = k.astype(np.float16)
                    if self.output_channels == 1:
                        weighted = (self.background_value * k16).astype(np.float16)
                        acc[ty : ty + read_h, tx : tx + read_w] += weighted
                    else:
                        weighted = (
                            self.background_value * k16[:, :, np.newaxis]
                        ).astype(np.float16)
                        acc[ty : ty + read_h, tx : tx + read_w, :] += weighted
                    wgt[ty : ty + read_h, tx : tx + read_w] += k16
                else:
                    padded = (
                        _pad_to(tile_rgb, self.tile_size)
                        if (read_w < self.tile_size or read_h < self.tile_size)
                        else tile_rgb
                    )
                    if self.batch_size <= 1:
                        out = _to_output(
                            self.generate(padded), self.output_channels
                        )
                        k = weight_kernel[:read_h, :read_w]
                        k16 = k.astype(np.float16)
                        if self.output_channels == 1:
                            weighted = (out[:read_h, :read_w, 0] * k16).astype(
                                np.float16
                            )
                            acc[ty : ty + read_h, tx : tx + read_w] += weighted
                        else:
                            weighted = (
                                out[:read_h, :read_w, :] * k16[:, :, np.newaxis]
                            ).astype(np.float16)
                            acc[ty : ty + read_h, tx : tx + read_w, :] += weighted
                        wgt[ty : ty + read_h, tx : tx + read_w] += k16
                    else:
                        _buf_tiles.append(padded)
                        _buf_meta.append((ty, tx, read_h, read_w))
                        if len(_buf_tiles) >= self.batch_size:
                            _flush_batch()

            _flush_batch()
            slide.close()
            acc.flush()
            wgt.flush()

            if self.verbose:
                print("  normalizing accumulated tiles into base level memmap ...")

            mm_out = np.memmap(
                str(tmp_out), dtype=np.uint8, mode="w+", shape=shape
            )
            for y0 in range(0, level_h, self.tiff_chunk):
                y1 = min(y0 + self.tiff_chunk, level_h)
                wgt_slice = wgt[y0:y1, :]
                wgt_safe = np.where(wgt_slice == 0, 1.0, wgt_slice)
                if self.output_channels == 1:
                    norm = (
                        acc[y0:y1, :].astype(np.float32) / wgt_safe
                    ).clip(0, 255)
                else:
                    norm = (
                        acc[y0:y1, :, :].astype(np.float32)
                        / wgt_safe[:, :, np.newaxis]
                    ).clip(0, 255)
                mm_out[y0:y1] = norm.astype(np.uint8)
            mm_out.flush()

            del acc, wgt

            if self.verbose:
                print("  writing base level to TIFF ...")

            resolution = (1e4 / mpp, 1e4 / mpp)
            photometric = "minisblack" if self.output_channels == 1 else "rgb"

            tmp_tiff_path = self.tiff_path.with_suffix(".tmp.ome.tiff")

            with tifffile.TiffWriter(str(tmp_tiff_path), bigtiff=True) as tif:
                tif.write(
                    mm_out,
                    tile=(self.tiff_chunk, self.tiff_chunk),
                    compression="jpeg",
                    compressionargs={"level": self.jpeg_quality},
                    photometric=photometric,
                    resolution=resolution,
                    resolutionunit=tifffile.RESUNIT.CENTIMETER,
                    subfiletype=0,
                    metadata=None,
                )

            del mm_out
            tmp_tiff_path.rename(self.tiff_path)
            self._build_pyramid_pyvips()

            if self.verbose:
                print(f"  OME-TIFF written to: {self.tiff_path}", flush=True)

        finally:
            shutil.rmtree(tmp_dir_path, ignore_errors=True)

