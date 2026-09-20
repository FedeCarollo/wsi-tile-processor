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
  
"""

from __future__ import annotations


import shutil
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import tifffile


from .base import WSIProcessor
from .filters import BackgroundFilter, TissueMaskDetector
from .slide import _get_mpp, _open_slide
from .utils import _make_weight_kernel, _pad_to, _tile_origins, _to_output


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

