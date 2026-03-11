"""
Utility functions for OCR Pipeline.

Provides common utilities for:
- File handling
- Image loading and conversion
- PDF processing
- Logging setup
"""

import os
import io
import json
import logging
import hashlib
from pathlib import Path
from typing import Union, List, Optional, Tuple
from dataclasses import asdict

import cv2
import numpy as np
from PIL import Image


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path for logging.
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = [logging.StreamHandler()]

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


class ImageUtils:
    """Utility class for image operations."""

    @staticmethod
    def load_image(path: Union[str, Path]) -> np.ndarray:
        """
        Load an image from file.

        Args:
            path: Path to image file.

        Returns:
            Image as numpy array (BGR format).

        Raises:
            ValueError: If image cannot be loaded.
        """
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Image not found: {path}")

        # Try different loading methods
        # First with OpenCV
        image = cv2.imread(str(path))
        if image is not None:
            return image

        # Fallback to PIL
        try:
            pil_image = Image.open(path)
            if pil_image.mode == 'RGBA':
                # Convert RGBA to RGB
                pil_image = pil_image.convert('RGB')
            # Convert RGB to BGR for OpenCV
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")

    @staticmethod
    def save_image(image: np.ndarray, path: Union[str, Path], quality: int = 95) -> bool:
        """
        Save an image to file.

        Args:
            image: Image as numpy array.
            path: Output path.
            quality: JPEG quality (1-100).

        Returns:
            True if successful.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Determine format from extension
        ext = path.suffix.lower()

        if ext in ['.jpg', '.jpeg']:
            return cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif ext == '.png':
            # PNG doesn't support quality parameter
            return cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        else:
            return cv2.imwrite(str(path), image)

    @staticmethod
    def image_to_bytes(image: np.ndarray, format: str = 'png') -> bytes:
        """
        Convert image to bytes.

        Args:
            image: Image as numpy array.
            format: Image format (png, jpg).

        Returns:
            Image as bytes.
        """
        if format.lower() == 'png':
            _, buffer = cv2.imencode('.png', image)
        else:
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return buffer.tobytes()

    @staticmethod
    def bytes_to_image(data: bytes) -> np.ndarray:
        """
        Convert bytes to image.

        Args:
            data: Image as bytes.

        Returns:
            Image as numpy array.
        """
        nparr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    @staticmethod
    def resize_image(
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        scale: Optional[float] = None,
        keep_aspect: bool = True
    ) -> np.ndarray:
        """
        Resize an image.

        Args:
            image: Input image.
            target_size: Target (width, height).
            scale: Scale factor.
            keep_aspect: Keep aspect ratio.

        Returns:
            Resized image.
        """
        h, w = image.shape[:2]

        if target_size:
            new_w, new_h = target_size
            if keep_aspect:
                # Calculate scale to fit within target while preserving aspect
                scale = min(new_w / w, new_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
        elif scale:
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            return image

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(image.shape) == 2:
            return image

        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def display_image(image: np.ndarray, window_name: str = "Image", wait_key: bool = True):
        """
        Display image in window (for debugging).

        Args:
            image: Image to display.
            window_name: Window title.
            wait_key: Wait for key press.
        """
        cv2.imshow(window_name, image)
        if wait_key:
            cv2.waitKey(0)
            cv2.destroyAllWindows()


class PDFUtils:
    """Utility class for PDF operations."""

    @staticmethod
    def pdf_to_images(
        pdf_path: Union[str, Path],
        dpi: int = 300,
        first_page: Optional[int] = None,
        last_page: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Convert PDF pages to images.

        Args:
            pdf_path: Path to PDF file.
            dpi: Output DPI.
            first_page: First page to convert (1-based).
            last_page: Last page to convert (1-based).

        Returns:
            List of images (one per page).

        Raises:
            ImportError: If required packages are not installed.
        """
        try:
            from pdf2image import convert_from_path
            from pdf2image.exceptions import PDFInfoError, PDFPageCountError
        except ImportError:
            raise ImportError(
                "pdf2image is required for PDF processing. "
                "Install with: pip install pdf2image"
            )

        # Convert
        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=first_page,
            last_page=last_page
        )

        # Convert PIL images to OpenCV format
        result = []
        for pil_image in images:
            # Convert RGB to BGR for OpenCV
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            result.append(cv_image)

        return result

    @staticmethod
    def get_pdf_page_count(pdf_path: Union[str, Path]) -> int:
        """
        Get the number of pages in a PDF.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Number of pages.
        """
        try:
            from pdf2image import pdfinfo_from_path
        except ImportError:
            raise ImportError("pdf2image is required for PDF processing")

        info = pdfinfo_from_path(str(pdf_path))
        return info.get('Pages', 0)


class FileUtils:
    """Utility class for file operations."""

    @staticmethod
    def get_file_extension(path: Union[str, Path]) -> str:
        """Get file extension without dot."""
        return Path(path).suffix.lstrip('.').lower()

    @staticmethod
    def is_image(path: Union[str, Path]) -> bool:
        """Check if file is an image."""
        image_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp', 'gif'}
        return FileUtils.get_file_extension(path) in image_extensions

    @staticmethod
    def is_pdf(path: Union[str, Path]) -> bool:
        """Check if file is a PDF."""
        return FileUtils.get_file_extension(path) == 'pdf'

    @staticmethod
    def get_output_path(
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        suffix: str = "_output",
        extension: Optional[str] = None
    ) -> Path:
        """
        Generate output file path.

        Args:
            input_path: Input file path.
            output_dir: Output directory.
            suffix: Suffix to add to filename.
            extension: New extension (if different from input).

        Returns:
            Output path.
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        stem = input_path.stem + suffix
        ext = extension or input_path.suffix

        return output_dir / f"{stem}{ext}"


class JSONUtils:
    """Utility class for JSON operations."""

    @staticmethod
    def save_json(data: dict, path: Union[str, Path], indent: int = 2):
        """Save data to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

    @staticmethod
    def load_json(path: Union[str, Path]) -> dict:
        """Load data from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


def load_document(path: Union[str, Path]) -> List[np.ndarray]:
    """
    Load a document (image or PDF) and return list of images.

    Args:
        path: Path to document.

    Returns:
        List of images (for PDF, one per page; for images, one image).
    """
    path = Path(path)

    if not path.exists():
        raise ValueError(f"File not found: {path}")

    if FileUtils.is_pdf(path):
        return PDFUtils.pdf_to_images(path)
    elif FileUtils.is_image(path):
        return [ImageUtils.load_image(path)]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def save_ocr_results(
    results,
    output_path: Union[str, Path],
    format: str = "json"
):
    """
    Save OCR results to file.

    Args:
        results: OCR results (StructuredOutput or dict).
        output_path: Output file path.
        format: Output format (json, txt).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        if hasattr(results, 'to_dict'):
            data = results.to_dict()
        else:
            data = results

        JSONUtils.save_json(data, output_path)

    elif format == "txt":
        if hasattr(results, 'cleaned_text'):
            text = results.cleaned_text
        elif isinstance(results, dict):
            text = results.get('cleaned_text', results.get('raw_text', ''))
        else:
            text = str(results)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)


def calculate_file_hash(path: Union[str, Path]) -> str:
    """Calculate MD5 hash of a file."""
    path = Path(path)
    hasher = hashlib.md5()

    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)

    return hasher.hexdigest()