"""
Image Preprocessing Module for OCR Pipeline.

Handles all image preprocessing operations using OpenCV:
- Grayscale conversion
- Noise removal
- Thresholding
- Deskewing
- Contrast enhancement
- Image resizing
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Union
from pathlib import Path
import logging

from config import PreprocessingConfig, get_config

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Handles all image preprocessing operations for OCR.

    This class provides a comprehensive suite of image enhancement
    techniques optimized for text recognition in documents.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessor with configuration.

        Args:
            config: Preprocessing configuration. Uses default if not provided.
        """
        self.config = config or get_config().preprocessing

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline to an image.

        Args:
            image: Input image as numpy array (BGR format from OpenCV).

        Returns:
            Preprocessed image ready for OCR.
        """
        logger.info("Starting preprocessing pipeline")

        # Step 1: Convert to grayscale if configured
        if self.config.use_grayscale:
            image = self.to_grayscale(image)
            logger.debug("Converted to grayscale")

        # Step 2: Remove noise
        image = self.remove_noise(image)
        logger.debug("Noise removal complete")

        # Step 3: Enhance contrast
        if self.config.enhance_contrast:
            image = self.enhance_contrast_clahe(image)
            logger.debug("Contrast enhancement complete")

        # Step 4: Apply thresholding
        image = self.apply_thresholding(image)
        logger.debug("Thresholding complete")

        # Step 5: Deskew if enabled
        if self.config.enable_deskew:
            image = self.deskew(image)
            logger.debug("Deskewing complete")

        # Step 6: Resize if needed
        image = self.resize_for_ocr(image)
        logger.debug("Resizing complete")

        logger.info("Preprocessing pipeline completed")
        return image

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.

        Args:
            image: Input image (BGR or RGB).

        Returns:
            Grayscale image.
        """
        if len(image.shape) == 2:
            return image  # Already grayscale

        if image.shape[2] == 4:  # RGBA
            # Convert RGBA to RGB first, then to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from the image using configured method.

        Args:
            image: Input grayscale image.

        Returns:
            Denoised image.
        """
        method = self.config.denoise_method
        strength = self.config.denoise_strength

        if len(image.shape) == 2:
            # Grayscale image
            if method == "gaussian":
                return cv2.GaussianBlur(image, (strength * 2 + 1, strength * 2 + 1), 0)
            elif method == "bilateral":
                return cv2.bilateralFilter(image, strength, strength * 2, strength // 2)
            elif method == "median":
                return cv2.medianBlur(image, strength * 2 + 1)
            else:
                return cv2.bilateralFilter(image, 5, 50, 50)

        # Color image
        if method == "gaussian":
            return cv2.GaussianBlur(image, (strength * 2 + 1, strength * 2 + 1), 0)
        elif method == "bilateral":
            return cv2.bilateralFilter(image, strength, strength * 2, strength // 2)
        elif method == "median":
            return cv2.medianBlur(image, strength * 2 + 1)
        else:
            return cv2.bilateralFilter(image, 5, 50, 50)

    def apply_thresholding(self, image: np.ndarray) -> np.ndarray:
        """
        Apply thresholding to binarize the image.

        Args:
            image: Input grayscale image.

        Returns:
            Binarized image.
        """
        method = self.config.threshold_method

        if method == "simple":
            _, result = cv2.threshold(
                image,
                self.config.threshold_value,
                255,
                cv2.THRESH_BINARY
            )
            return result

        elif method == "adaptive":
            # Adaptive Gaussian thresholding
            return cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )

        elif method == "otsu":
            # Otsu's thresholding with Gaussian blur
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            _, result = cv2.threshold(
                blurred,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return result

        else:
            # Default to adaptive
            return cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )

    def enhance_contrast_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

        This is particularly effective for document images as it enhances
        text visibility without amplifying noise.

        Args:
            image: Input grayscale image.

        Returns:
            Contrast-enhanced image.
        """
        # Ensure image is grayscale
        if len(image.shape) == 3:
            image = self.to_grayscale(image)

        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(self.config.clahe_grid_size, self.config.clahe_grid_size)
        )

        # Apply CLAHE
        return clahe.apply(image)

    def detect_skew_angle(self, image: np.ndarray) -> float:
        """
        Detect the skew angle of the document.

        Args:
            image: Input binarized image.

        Returns:
            Skew angle in degrees.
        """
        # Get non-zero pixels coordinates
        coords = np.column_stack(np.where(image > 0))

        if len(coords) == 0:
            return 0.0

        # Compute minimum area bounding box
        angle = cv2.minAreaRect(coords)[-1]

        # Adjust angle
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        return angle

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew (straighten) the document image.

        Args:
            image: Input image.

        Returns:
            Deskewed image.
        """
        # Get current dimensions
        h, w = image.shape[:2]

        # Detect skew angle
        angle = self.detect_skew_angle(image)

        # Only deskew if angle exceeds threshold
        if abs(angle) < self.config.deskew_angle_threshold:
            logger.debug(f"Skew angle {angle:.2f} below threshold, skipping")
            return image

        logger.info(f"Detected skew angle: {angle:.2f} degrees")

        # Compute rotation matrix
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding dimensions
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Adjust rotation matrix
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        # Apply rotation
        return cv2.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    def resize_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to optimal size for OCR.

        Args:
            image: Input image.

        Returns:
            Resized image.
        """
        h, w = image.shape[:2]
        max_dim = self.config.max_dimension

        # Check if resize is needed
        if w <= max_dim and h <= max_dim:
            return image

        # Calculate scaling factor
        if w > h:
            scale = max_dim / w
        else:
            scale = max_dim / h

        # Resize
        new_w = int(w * scale)
        new_h = int(h * scale)

        logger.info(f"Resizing from ({w}, {h}) to ({new_w}, {new_h})")

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def preprocess_image(
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[PreprocessingConfig] = None
) -> np.ndarray:
    """
    Convenience function to preprocess an image from file.

    Args:
        image_path: Path to input image.
        output_path: Optional path to save preprocessed image.
        config: Optional preprocessing configuration.

    Returns:
        Preprocessed image as numpy array.
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Preprocess
    preprocessor = ImagePreprocessor(config)
    processed = preprocessor.preprocess(image)

    # Save if output path provided
    if output_path:
        cv2.imwrite(str(output_path), processed)
        logger.info(f"Saved preprocessed image to {output_path}")

    return processed


def preprocess_image_bytes(
    image_bytes: bytes,
    config: Optional[PreprocessingConfig] = None
) -> np.ndarray:
    """
    Preprocess an image from bytes.

    Args:
        image_bytes: Image data as bytes.
        config: Optional preprocessing configuration.

    Returns:
        Preprocessed image as numpy array.
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode image from bytes")

    # Preprocess
    preprocessor = ImagePreprocessor(config)
    return preprocessor.preprocess(image)