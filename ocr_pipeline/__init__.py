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

from ocr_pipeline.config import get_config, PipelineConfig
from ocr_pipeline.preprocess import ImagePreprocessor
from ocr_pipeline.text_detector import TextDetector, TextRegion
from ocr_pipeline.gemini_ocr import GeminiOCR, OCRResult
from ocr_pipeline.postprocess import TextPostProcessor, StructuredOutput
from ocr_pipeline.utils import shared_utility_function

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