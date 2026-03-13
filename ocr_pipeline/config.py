```python
"""
Configuration file for OCR Pipeline.

Contains all configurable settings for the OCR system.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Load environment variables from.env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

from ocr_pipeline.utils import create_directory

@dataclass
class GeminiConfig:
    """Configuration for Google Gemini API."""
    api_key: str = ""
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.1
    max_output_tokens: int = 2048
    timeout: int = 60

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY", "")

@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing."""
    # Grayscale conversion
    use_grayscale: bool = True

    # Noise removal
    denoise_method: str = "bilateral"  # gaussian, bilateral, median
    denoise_strength: int = 5

    # Thresholding
    threshold_method: str = "adaptive"  # simple, adaptive, otsu
    threshold_value: int = 127

    # Deskewing
    enable_deskew: bool = True
    deskew_angle_threshold: float = 0.5

    # Contrast enhancement
    enhance_contrast: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8

    # Resizing
    target_dpi: int = 300
    max_dimension: int = 4000

@dataclass
class TextDetectorConfig:
    """Configuration for text detection."""
    method: str = "contour"  # east, contour
    min_text_height: int = 10
    min_text_width: int = 5
    max_text_height: int = 1000
    aspect_ratio_min: float = 0.1
    aspect_ratio_max: float = 50.0
    confidence_threshold: float = 0.5

    # Contour detection parameters
    contour_mode: int = 2  # RETR_EXTERNAL
    contour_method: int = 3  # CHAIN_APPROX_SIMPLE

    # EAST model path (if using EAST)
    east_model_path: str = "frozen_east_text_detection.pb"

@dataclass
class OCRConfig:
    """Configuration for OCR processing."""
    # Batching
    batch_size: int = 10
    use_batching: bool = True

    # Caching
    enable_caching: bool = True
    cache_dir: str = ".cache/ocr"

    # Parallel processing
    use_multiprocessing: bool = True
    max_workers: int = 4

    # Quality settings
    image_quality: int = 95
    min_confidence: float = 0.5

@dataclass
class PostProcessConfig:
    """Configuration for post-processing."""
    # Regex patterns for cleaning
    remove_extra_spaces: bool = True
    normalize_whitespace: bool = True

    # Spell correction
    enable_spell_check: bool = False
    spellcheck_language: str = "en"

    # Line reconstruction
    reconstruct_lines: bool = True
    paragraph_detection: bool = True

    # Artifacts removal
    remove_artifacts: bool = True
    min_word_length: int = 1

    # Date/pattern extraction
    extract_dates: bool = True
    extract_emails: bool = True
    extract_phones: bool = True

@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    project_root: Path = Path(__file__).parent
    output_dir: Path = Path("output")

    # Component configurations
    gemini: GeminiConfig = GeminiConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    text_detector: TextDetectorConfig = TextDetectorConfig()
    ocr: OCRConfig = OCRConfig()
    postprocess: PostProcessConfig = PostProcessConfig()

    # Supported formats
    supported_image_formats: tuple = ("jpg", "jpeg", "png", "bmp", "tiff", "webp")
    supported_document_formats: tuple = ("pdf",)

    def __post_init__(self):
        create_directory(self.output_dir)

# Global configuration instance
config = PipelineConfig()

def get_config() -> PipelineConfig:
    """Get the global pipeline configuration."""
    return config

def validate_config() -> bool:
    """Validate the configuration settings."""
    cfg = get_config()

    # Check Gemini API key
    if not cfg.gemini.api_key:
        print("Warning: GEMINI_API_KEY not set. Set via config or environment variable.")

    # Check output directory
    try:
        create_directory(cfg.output_dir)
    except Exception as e:
        print(f"Warning: Could not create output directory: {e}")

    return True
```