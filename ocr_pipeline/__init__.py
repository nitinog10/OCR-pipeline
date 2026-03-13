"""
OCR Pipeline Package

A complete OCR pipeline using Python and Gemini Vision for document text extraction.

Features:
- Multi-format support (JPG, PNG, PDF)
- Advanced image preprocessing with OpenCV
- Text region detection using contour analysis
- OCR using Google Gemini Vision API
- Post-processing with regex and spell correction
- Structured JSON output

Usage:
    from ocr_pipeline import OCRPipeline

    pipeline = OCRPipeline()
    result = pipeline.process_file("document.pdf", "output.json")
"""

from.config import get_config, PipelineConfig
from.preprocess import ImagePreprocessor
from.text_detector import TextDetector, TextRegion
from.gemini_ocr import GeminiOCR, OCRResult
from.postprocess import TextPostProcessor, StructuredOutput
from.utils import shared_utility_function

__version__ = "1.0.0"
__author__ = "OCR Pipeline Team"

__all__ = [
    "get_config",
    "PipelineConfig",
    "ImagePreprocessor",
    "TextDetector",
    "TextRegion",
    "GeminiOCR",
    "OCRResult",
    "TextPostProcessor",
    "StructuredOutput",
    "shared_utility_function",
]