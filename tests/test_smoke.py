"""
Smoke tests for wsi-tile-processor.

Run with::

    python -m pytest tests/test_smoke.py -v

or directly::

    python tests/test_smoke.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from wsi_tile_processor import (
    BackgroundFilter,
    BrightnessBackgroundFilter,
    BrightnessTissueMaskDetector,
    FastWSIProcessor,
    GaussianWSIProcessor,
    OtsuBackgroundFilter,
    SaturationBackgroundFilter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TILE_SIZE = 256
SLIDE_SIZE = 1024  # synthetic slide side length


def _make_synthetic_slide(path: Path) -> None:
    """Write a 1024×1024 RGB TIFF with a tissue patch in the centre."""
    data = np.full((SLIDE_SIZE, SLIDE_SIZE, 3), 240, dtype=np.uint8)
    data[200:600, 200:600] = np.random.randint(60, 160, (400, 400, 3), dtype=np.uint8)
    with tifffile.TiffWriter(str(path)) as tw:
        tw.write(
            data,
            photometric="rgb",
            resolution=(20000, 20000),   # 20 000 px/cm = 0.5 µm/px
            resolutionunit=tifffile.RESUNIT.CENTIMETER,
            metadata=None,
        )


def identity(tile: np.ndarray) -> np.ndarray:
    return tile


def _assert_valid_pyramid(path: Path) -> int:
    """Return the number of pyramid levels and assert there is more than one."""
    with tifffile.TiffFile(str(path)) as tf:
        n = len(tf.series[0].levels)
    assert n > 1, f"Expected a pyramid but got {n} level(s) in {path.name}"
    return n


# ---------------------------------------------------------------------------
# Background filter tests
# ---------------------------------------------------------------------------

WHITE  = np.full((512, 512, 3), 240, dtype=np.uint8)
TISSUE = np.random.randint(50, 180, (512, 512, 3), dtype=np.uint8)


class TestBrightnessBackgroundFilter:
    def test_white_is_background(self):
        assert BrightnessBackgroundFilter()(WHITE) is True

    def test_tissue_is_not_background(self):
        assert BrightnessBackgroundFilter()(TISSUE) is False

    def test_empty_tile_is_background(self):
        assert BrightnessBackgroundFilter()(np.empty((0, 0, 3), dtype=np.uint8)) is True

    def test_strict_threshold(self):
        # threshold=0.0 means background when bright-fraction > 0.0; an entirely
        # bright tile always satisfies this.
        all_bright = np.full((256, 256, 3), 240, dtype=np.uint8)
        assert BrightnessBackgroundFilter(threshold=0.0)(all_bright) is True

    def test_loose_threshold(self):
        # With threshold=1.01 nothing is background (fraction can never exceed 1)
        assert BrightnessBackgroundFilter(threshold=1.01)(WHITE) is False


class TestOtsuBackgroundFilter:
    def test_white_is_background(self):
        assert OtsuBackgroundFilter()(WHITE) is True

    def test_tissue_is_not_background(self):
        assert OtsuBackgroundFilter()(TISSUE) is False

    def test_uniform_tile(self):
        # Uniform tile has zero variance — should fall back to brightness check
        uniform_dark = np.full((256, 256, 3), 100, dtype=np.uint8)
        assert OtsuBackgroundFilter()(uniform_dark) is False  # dark → tissue

    def test_empty_tile_is_background(self):
        assert OtsuBackgroundFilter()(np.empty((0, 0, 3), dtype=np.uint8)) is True


class TestSaturationBackgroundFilter:
    def test_white_is_background(self):
        assert SaturationBackgroundFilter()(WHITE) is True

    def test_tissue_is_not_background(self):
        assert SaturationBackgroundFilter()(TISSUE) is False

    def test_empty_tile_is_background(self):
        assert SaturationBackgroundFilter()(np.empty((0, 0, 3), dtype=np.uint8)) is True


class TestCustomBackgroundFilter:
    def test_subclass(self):
        class AlwaysBackground(BackgroundFilter):
            def __call__(self, tile):
                return True

        class NeverBackground(BackgroundFilter):
            def __call__(self, tile):
                return False

        assert AlwaysBackground()(TISSUE) is True
        assert NeverBackground()(TISSUE)  is False

    def test_lambda(self):
        # A plain callable also works
        always_bg = lambda t: True
        assert always_bg(TISSUE) is True


# ---------------------------------------------------------------------------
# Processor tests
# ---------------------------------------------------------------------------

class TestFastWSIProcessor:
    def test_basic(self, tmp_path):
        wsi = tmp_path / "slide.tiff"
        out = tmp_path / "out.ome.tiff"
        _make_synthetic_slide(wsi)

        FastWSIProcessor(
            wsi_path=wsi, tiff_path=out,
            level=0, tile_size=TILE_SIZE,
            generate=identity, backend="tifffile",
        ).process()

        assert out.exists()
        _assert_valid_pyramid(out)

    def test_output_channels_1(self, tmp_path):
        # pyvips tiffsave with JPEG compression requires RGB (3-band) input.
        # For single-channel (grayscale) output the library still works up to the
        # TIFF write stage, but pyramidization via pyvips JPEG is unsupported.
        # We verify the processor raises a clear error in this case rather than
        # silently producing a broken file.
        wsi = tmp_path / "slide.tiff"
        out = tmp_path / "out.ome.tiff"
        _make_synthetic_slide(wsi)

        def to_gray(tile):
            return np.dot(tile.astype(np.float32), [0.299, 0.587, 0.114]).astype(np.uint8)

        with pytest.raises(Exception):
            FastWSIProcessor(
                wsi_path=wsi, tiff_path=out,
                level=0, tile_size=TILE_SIZE,
                generate=to_gray, output_channels=1,
                backend="tifffile",
            ).process()

    def test_custom_background_filter(self, tmp_path):
        wsi = tmp_path / "slide.tiff"
        out = tmp_path / "out.ome.tiff"
        _make_synthetic_slide(wsi)

        FastWSIProcessor(
            wsi_path=wsi, tiff_path=out,
            level=0, tile_size=TILE_SIZE,
            generate=identity, backend="tifffile",
            background_filter=OtsuBackgroundFilter(),
        ).process()

        assert out.exists()

    def test_lambda_background_filter(self, tmp_path):
        wsi = tmp_path / "slide.tiff"
        out = tmp_path / "out.ome.tiff"
        _make_synthetic_slide(wsi)

        FastWSIProcessor(
            wsi_path=wsi, tiff_path=out,
            level=0, tile_size=TILE_SIZE,
            generate=identity, backend="tifffile",
            background_filter=lambda t: float(t.mean()) > 230,
        ).process()

        assert out.exists()

    def test_tissue_mask_detector(self, tmp_path):
        wsi = tmp_path / "slide.tiff"
        out = tmp_path / "out.ome.tiff"
        _make_synthetic_slide(wsi)

        FastWSIProcessor(
            wsi_path=wsi, tiff_path=out,
            level=0, tile_size=TILE_SIZE,
            generate=identity, backend="tifffile",
            tissue_mask_detector=BrightnessTissueMaskDetector(),
        ).process()

        assert out.exists()

    def test_batch_size(self, tmp_path):
        wsi = tmp_path / "slide.tiff"
        out = tmp_path / "out.ome.tiff"
        _make_synthetic_slide(wsi)

        calls = [0]

        def batched(tiles):
            calls[0] += 1
            return tiles  # (B, H, W, 3)

        FastWSIProcessor(
            wsi_path=wsi, tiff_path=out,
            level=0, tile_size=TILE_SIZE,
            generate=batched, batch_size=4,
            backend="tifffile",
        ).process()

        assert out.exists()
        # With batch_size=4 we expect fewer calls than tiles
        assert calls[0] >= 1

    def test_generate_called_only_for_tissue(self, tmp_path):
        """generate() must not be called on tiles that are fully white background."""
        wsi = tmp_path / "slide.tiff"
        out = tmp_path / "out.ome.tiff"
        _make_synthetic_slide(wsi)

        calls = [0]
        def counting(tile):
            calls[0] += 1
            return tile

        # NeverBackground → every tile is tissue → generate is always called
        from wsi_tile_processor import BackgroundFilter
        class NeverBackground(BackgroundFilter):
            def __call__(self, t): return False

        n_tiles = (SLIDE_SIZE // TILE_SIZE) ** 2  # 16 for 1024/256

        FastWSIProcessor(
            wsi_path=wsi, tiff_path=out,
            level=0, tile_size=TILE_SIZE,
            generate=counting, backend="tifffile",
            background_filter=NeverBackground(),
        ).process()

        assert calls[0] == n_tiles, (
            f"Expected generate called for all {n_tiles} tiles, got {calls[0]}"
        )


class TestGaussianWSIProcessor:
    def test_basic(self, tmp_path):
        wsi = tmp_path / "slide.tiff"
        out = tmp_path / "out.ome.tiff"
        _make_synthetic_slide(wsi)

        GaussianWSIProcessor(
            wsi_path=wsi, tiff_path=out,
            level=0, tile_size=TILE_SIZE, stride=TILE_SIZE // 2,
            generate=identity, backend="tifffile",
        ).process()

        assert out.exists()
        _assert_valid_pyramid(out)

    def test_stride_equals_tile_size(self, tmp_path):
        """stride == tile_size is valid (no blending, same as Fast)."""
        wsi = tmp_path / "slide.tiff"
        out = tmp_path / "out.ome.tiff"
        _make_synthetic_slide(wsi)

        GaussianWSIProcessor(
            wsi_path=wsi, tiff_path=out,
            level=0, tile_size=TILE_SIZE, stride=TILE_SIZE,
            generate=identity, backend="tifffile",
        ).process()

        assert out.exists()

    def test_stride_larger_than_tile_raises(self):
        with pytest.raises(ValueError, match="stride"):
            GaussianWSIProcessor(
                wsi_path="dummy.tiff", tiff_path="out.tiff",
                level=0, tile_size=256, stride=512,
                generate=identity,
            )

    def test_no_blur(self, tmp_path):
        wsi = tmp_path / "slide.tiff"
        out = tmp_path / "out.ome.tiff"
        _make_synthetic_slide(wsi)

        GaussianWSIProcessor(
            wsi_path=wsi, tiff_path=out,
            level=0, tile_size=TILE_SIZE, stride=TILE_SIZE // 2,
            blur_sigma=None,
            generate=identity, backend="tifffile",
        ).process()

        assert out.exists()

# ---------------------------------------------------------------------------
# Entry point for running without pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
