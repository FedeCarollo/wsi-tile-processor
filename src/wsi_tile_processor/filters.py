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


from abc import ABC, abstractmethod

import numpy as np



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

