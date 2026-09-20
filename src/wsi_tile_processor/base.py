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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np

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


from .filters import (BackgroundFilter, BrightnessBackgroundFilter,
                      TissueMaskDetector)
from .utils import pyramidize_with_pyvips


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

