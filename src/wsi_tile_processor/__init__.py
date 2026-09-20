"""
wsi-tile-processor
"""
from .base import WSIProcessor
from .filters import (BackgroundFilter, BrightnessBackgroundFilter,
                      BrightnessTissueMaskDetector, OtsuBackgroundFilter,
                      SaturationBackgroundFilter, TissueMaskDetector)
from .processors import FastWSIProcessor, GaussianWSIProcessor

__version__ = "0.1.2"

__all__ = [
    "BackgroundFilter",
    "BrightnessBackgroundFilter",
    "OtsuBackgroundFilter",
    "SaturationBackgroundFilter",
    "TissueMaskDetector",
    "BrightnessTissueMaskDetector",
    "WSIProcessor",
    "FastWSIProcessor",
    "GaussianWSIProcessor",
]
