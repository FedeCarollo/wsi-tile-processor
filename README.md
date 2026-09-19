# wsi-tile-processor

[![PyPI version](https://badge.fury.io/py/wsi-tile-processor.svg)](https://pypi.org/project/wsi-tile-processor/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Tile-by-tile inference on Whole Slide Images (WSI) with pyramidal OME-TIFF output.

`wsi-tile-processor` is a Python library for running any deep learning or image-processing model on whole slide images (WSI) in a memory-efficient, tiled manner, and saving the result as a multi-resolution pyramidal OME-TIFF ready for viewing in QuPath, OMERO, or any TIFF-compatible viewer.

**Key features:**

- **Two processors** — `FastWSIProcessor` (non-overlapping, fast) and `GaussianWSIProcessor` (overlapping tiles with smooth Gaussian blending, no seam artifacts)
- **Pluggable background filters** — swap or subclass `BackgroundFilter` to implement custom tissue/background detection
- **Pluggable tissue mask detectors** — coarse-level background skipping with `TissueMaskDetector`
- **Multi-backend slide reading** — `openslide`, `tifffile` (for OME-TIFF sub-IFD pyramids), or `auto`
- **Batch inference** — feed multiple tiles per forward pass
- **BigTIFF + JPEG pyramid output** via pyvips

---

## Installation

**From PyPI (Recommended):**
```bash
pip install wsi-tile-processor
```

For full support (zarr-backed tifffile reads + scikit-image tissue detection):
```bash
pip install wsi-tile-processor[full]
```

**From GitHub (Latest development version):**
```bash
pip install git+https://github.com/FedeCarollo/wsi-tile-processor.git
```

### System dependencies

#### OpenSlide

[OpenSlide](https://openslide.org/download/) must be installed on your system:

```bash
# Ubuntu / Debian
sudo apt install openslide-tools

# macOS
brew install openslide
```

#### libvips (for pyramid generation)

`pyvips` is a Python binding for [libvips](https://www.libvips.org/) and **does not bundle it** by default.
You have two options:

**Option A — system package (recommended for servers/HPC):**

```bash
# Ubuntu / Debian
sudo apt install libvips

# macOS
brew install vips
```

**Option B — bundled binary via pip (no system install needed):**

```bash
pip install "pyvips[binary]"
```

This pulls in `pyvips-binary`, which ships pre-compiled libvips binaries and runs in faster CFFI API mode. Useful for Docker containers or environments where you can't install system packages.

> **ℹ️ Note:** Without libvips installed (system or binary), `pyvips` falls back to ABI mode which may be ~20% slower. Pyramidization will still work correctly.

---

## Quick Start

### Non-overlapping tiles — `FastWSIProcessor`

Use this for most use cases: virtual staining, classification heatmaps, segmentation masks.

```python
import numpy as np
from wsi_tile_processor import FastWSIProcessor

# 'generate' maps an HxWx3 uint8 tile to an HxWx3 output.
# Here we use an identity function as a placeholder.
def my_model(tile: np.ndarray) -> np.ndarray:
    return tile  # replace with your model

processor = FastWSIProcessor(
    wsi_path="input.svs",
    tiff_path="output.ome.tiff",
    level=1,          # pyramid level to process (0 = full res)
    tile_size=512,    # tile side length in pixels
    generate=my_model,
    output_channels=3,
    verbose=True,
)
processor.process()
```

### Overlapping tiles with blending — `GaussianWSIProcessor`

Use this when your model produces boundary artifacts. Overlapping tiles are averaged with a Gaussian weight kernel, eliminating grid patterns.

```python
from wsi_tile_processor import GaussianWSIProcessor

processor = GaussianWSIProcessor(
    wsi_path="input.svs",
    tiff_path="output.ome.tiff",
    level=1,
    tile_size=512,
    stride=256,         # 50% overlap
    blur_sigma=64.0,    # Gaussian kernel sigma in pixels
    generate=my_model,
    output_channels=3,
    verbose=True,
)
processor.process()
```


### Batched inference

```python
import numpy as np

def batched_model(batch: np.ndarray) -> np.ndarray:
    # batch shape: (B, H, W, 3) uint8
    # return shape: (B, H, W, 3)
    return batch

processor = FastWSIProcessor(
    wsi_path="input.svs",
    tiff_path="output.ome.tiff",
    level=1,
    tile_size=512,
    generate=batched_model,
    batch_size=8,   # tiles per forward pass
    verbose=True,
)
processor.process()
```

---

## Architecture

```
WSIProcessor  (abstract)
├── FastWSIProcessor      — stride == tile_size, uint8 memmap
└── GaussianWSIProcessor  — stride < tile_size, float16 acc + wgt memmaps
```

### `FastWSIProcessor`

- Iterates tiles in a regular grid (no overlap).
- Each non-background tile is passed to `generate`, and the uint8 result is written directly into a memory-mapped binary file.
- After all tiles, writes a BigTIFF and pyramidizes in-place with pyvips.
- **Disk usage:** 1× the output image size.
- **Best for:** virtual staining, classification maps, any task where tile alignment is fine.

### `GaussianWSIProcessor`

- Iterates tiles with a configurable stride (overlap).
- Each tile's output is multiplied by a Gaussian weight kernel and accumulated into `float16` memmaps (`acc` and `wgt`).
- After all tiles, every pixel is normalized by its total accumulated weight, then converted to uint8.
- **Disk usage:** ~5–6× the output image size during processing.
- **Best for:** models that produce boundary artifacts, or when smooth spatial continuity is critical.

### Custom Processor

Both processors share the same abstract contract defined by `WSIProcessor`.
You can subclass it to implement a completely different accumulation strategy — for example, writing tiles to HDF5, streaming to a remote store, or applying multi-scale blending.

The only method you must implement is `process()`:

```python
import numpy as np
from wsi_tile_processor import WSIProcessor, _open_slide, _get_mpp

class MyProcessor(WSIProcessor):
    def process(self) -> None:
        slide = _open_slide(self.wsi_path, self.backend)
        level_w, level_h = slide.level_dimensions[self.level]
        downsample = slide.level_downsamples[self.level]

        for ty in range(0, level_h, self.tile_size):
            for tx in range(0, level_w, self.tile_size):
                read_w = min(self.tile_size, level_w - tx)
                read_h = min(self.tile_size, level_h - ty)
                x0, y0 = int(tx * downsample), int(ty * downsample)

                pil_tile = slide.read_region((x0, y0), self.level, (read_w, read_h))
                tile = np.array(pil_tile.convert("RGB"), dtype=np.uint8)

                if self._is_tile_background(tile):
                    continue  # uses whatever BackgroundFilter was configured

                result = self.generate(tile)
                # ... write result to your custom output format ...

        slide.close()

# Use it like any other processor
proc = MyProcessor(
    wsi_path="input.svs",
    tiff_path="output.ome.tiff",
    level=1,
    tile_size=512,
    generate=my_model,
)
proc.process()
```

> **💡 Tip:** Your custom processor automatically inherits `_is_tile_background()`, `_compute_tissue_mask()`, and `_build_pyramid_pyvips()` from the base class, so you can reuse all the built-in filtering and pyramidization logic.

---

## Background Filters

Background tiles are skipped by default: `generate` is never called on them,
and they are filled with `background_value` (default: white, 1.0).

### Built-in filters

#### `BrightnessBackgroundFilter` *(default)*

Brightness + optional saturation heuristic. Tuned for H&E and IHC slides.
A tile is background when more than `threshold` (default 70%) of its pixels
have all three channels > `threshold_intensity` (default 215).

```python
from wsi_tile_processor import BrightnessBackgroundFilter, FastWSIProcessor

bg_filter = BrightnessBackgroundFilter(
    threshold=0.75,           # stricter: 75% bright pixels
    threshold_intensity=220,
    min_variance=100.0,       # also skip low-variance tiles
)

processor = FastWSIProcessor(
    wsi_path="input.svs",
    tiff_path="output.ome.tiff",
    level=1,
    tile_size=512,
    generate=my_model,
    background_filter=bg_filter,
)
processor.process()
```

#### `OtsuBackgroundFilter`

Uses scikit-image Otsu thresholding on the grayscale image.
Requires `pip install scikit-image` (or `pip install wsi-tile-processor[full]`).

```python
from wsi_tile_processor import OtsuBackgroundFilter

bg_filter = OtsuBackgroundFilter(background_fraction=0.65)
```

#### `SaturationBackgroundFilter`

Classifies a tile as background when its mean HSV saturation is below a threshold.
Fast and robust for highly stained tissue.

```python
from wsi_tile_processor import SaturationBackgroundFilter

bg_filter = SaturationBackgroundFilter(min_mean_saturation=20.0)
```

### Custom background filter

Subclass `BackgroundFilter` and implement `__call__`:

```python
import numpy as np
from wsi_tile_processor import BackgroundFilter, FastWSIProcessor

class MyPenMarkFilter(BackgroundFilter):
    """Skip tiles dominated by blue pen marks."""

    def __call__(self, tile_rgb: np.ndarray) -> bool:
        r, g, b = tile_rgb[..., 0], tile_rgb[..., 1], tile_rgb[..., 2]
        # Blue pen: b >> r and b >> g
        blue_mask = (b.astype(int) - r.astype(int) > 50) & \
                    (b.astype(int) - g.astype(int) > 50)
        return float(blue_mask.mean()) > 0.30  # >30% blue → skip

processor = FastWSIProcessor(
    wsi_path="input.svs",
    tiff_path="output.ome.tiff",
    level=1,
    tile_size=512,
    generate=my_model,
    background_filter=MyPenMarkFilter(),
)
processor.process()
```

You can also pass any plain callable with signature `(tile_rgb: np.ndarray) -> bool`:

```python
processor = FastWSIProcessor(
    ...,
    background_filter=lambda t: t.mean() > 230,
)
```

---

## Tissue Mask Detection

For very large slides, computing the background filter on every tile can be
slow. A `TissueMaskDetector` provides a coarse, one-time mask computed on a
low-resolution thumbnail. Tiles that fall entirely outside the tissue mask are
skipped *without* reading the full-resolution data.

### Built-in detector

```python
from wsi_tile_processor import BrightnessTissueMaskDetector, FastWSIProcessor

detector = BrightnessTissueMaskDetector(
    intensity_threshold=215,  # pixels below this are tissue
    max_side=4000,            # thumbnail max dimension
    open_radius=2,
    close_radius=4,
    verbose=True,
)

processor = FastWSIProcessor(
    wsi_path="input.svs",
    tiff_path="output.ome.tiff",
    level=1,
    tile_size=512,
    generate=my_model,
    tissue_mask_detector=detector,  # supersedes use_tissue_mask=True
)
processor.process()
```

Requires `scikit-image` (`pip install wsi-tile-processor[full]`).

### Custom tissue mask detector

```python
import numpy as np
from wsi_tile_processor import TissueMaskDetector

class MyDLTissueMask(TissueMaskDetector):
    def __init__(self, model):
        self.model = model

    def __call__(self, slide) -> tuple[np.ndarray, float]:
        # Use the smallest level for speed
        level = slide.level_count - 1
        dims = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]
        thumb = np.array(slide.read_region((0, 0), level, dims).convert("RGB"))
        mask = self.model.predict(thumb)  # → bool H×W
        return mask, downsample
```

---

## Slide Backends

| Backend | When to use |
|---------|-------------|
| `"auto"` *(default)* | Tries openslide first; falls back to tifffile if tifffile exposes more pyramid levels. Best for mixed environments. |
| `"openslide"` | SVS, NDPI, SCN, and other formats natively supported by OpenSlide. |
| `"tifffile"` | OME-TIFF files with sub-IFD pyramid levels (e.g. output of [VALIS](https://github.com/MathOnco/valis)). OpenSlide may only see the base level in these files. |

```python
processor = FastWSIProcessor(
    ...,
    backend="tifffile",
)
```

---

## Output Format

The output is a **BigTIFF OME-TIFF** with:

- A full JPEG-compressed pyramid (sub-IFD layout, readable by QuPath, OMERO, pyvips, tifffile, …)
- Physical resolution metadata (`XResolution` / `YResolution` in pixels/cm, derived from the input slide's `mpp-x`)
- Configurable JPEG quality (default 95) and internal tile size (default 512 px)

---

## License

MIT © 2026 Federico Carollo — see [LICENSE](LICENSE) for details.
