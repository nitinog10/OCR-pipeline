```
"""
Text Detection Module for OCR Pipeline.

Detects text regions in document images using:
- Contour-based detection (primary method)
- EAST text detector (optional, requires model file)

Returns bounding boxes of text regions for OCR processing.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging
import os

from ocr_pipeline.config import TextDetectorConfig, get_config
from ocr_pipeline.utils import merge_regions, draw_regions, crop_region

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """Represents a detected text region."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float = 1.0
    text: Optional[str] = None
    is_handwriting: bool = False
    is_table: bool = False

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def w(self) -> int:
        return self.bbox[2]

    @property
    def h(self) -> int:
        return self.bbox[3]

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def aspect_ratio(self) -> float:
        return self.w / self.h if self.h > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "text": self.text,
            "is_handwriting": self.is_handwriting,
            "is_table": self.is_table
        }


class TextDetector:
    """
    Detects text regions in document images.

    Uses contour-based detection as the primary method, with optional
    support for EAST text detector.
    """

    def __init__(self, config: Optional[TextDetectorConfig] = None):
        """
        Initialize the text detector.

        Args:
            config: Text detector configuration.
        """
        self.config = config or get_config().text_detector
        self.east_model = None

    def detect(self, image: np.ndarray) -> List[TextRegion]:
        """
        Detect text regions in the image.

        Args:
            image: Input image (grayscale or color).

        Returns:
            List of detected text regions.
        """
        if self.config.method == "east":
            return self._detect_with_east(image)
        else:
            return self._detect_with_contours(image)

    def _detect_with_contours(self, image: np.ndarray) -> List[TextRegion]:
        """
        Detect text regions using contour analysis.

        This method works by:
        1. Applying morphological operations to enhance text regions
        2. Finding contours that likely contain text
        3. Filtering contours based on size and aspect ratio

        Args:
            image: Input image (grayscale preferred).

        Returns:
            List of detected text regions.
        """
        logger.info("Detecting text regions using contour method")

        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply morphological operations to find text regions
        # Using horizontal then vertical kernels to find text lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

        # Apply tophat to isolate text
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, horizontal_kernel)

        # Apply threshold
        _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(
            binary,
            self.config.contour_mode,
            self.config.contour_method
        )

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter based on minimum size
            if h < self.config.min_text_height or w < self.config.min_text_width:
                continue

            # Filter based on maximum size
            if h > self.config.max_text_height:
                continue

            # Filter based on aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < self.config.aspect_ratio_min:
                continue
            if aspect_ratio > self.config.aspect_ratio_max:
                continue

            regions.append(TextRegion(
                bbox=(x, y, w, h),
                confidence=1.0
            ))

        logger.info(f"Detected {len(regions)} text regions using contours")

        # Also try direct contour detection on inverted image
        inverted = 255 - gray
        contours2, _ = cv2.findContours(
            inverted,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours2:
            x, y, w, h = cv2.boundingRect(contour)

            # Apply same filters
            if h < self.config.min_text_height or w < self.config.min_text_width:
                continue
            if h > self.config.max_text_height:
                continue

            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < self.config.aspect_ratio_min:
                continue

            # Check if this region overlaps with existing ones
            is_duplicate = False
            for existing in regions:
                ex, ey, ew, eh = existing.bbox
                # Check overlap
                if (x < ex + ew and x + w > ex and y < ey + eh and y + h > ey):
                    is_duplicate = True
                    break

            if not is_duplicate:
                regions.append(TextRegion(
                    bbox=(x, y, w, h),
                    confidence=0.8
                ))

        # Merge overlapping regions
        regions = merge_regions(regions)

        logger.info(f"Final text regions after merging: {len(regions)}")
        return regions

    def _detect_with_east(self, image: np.ndarray) -> List[TextRegion]:
        """
        Detect text regions using EAST (Efficient and Accurate Scene Text Detector).

        Note: Requires the EAST model file to be downloaded and configured.
        This is a more advanced method but requires additional setup.

        Args:
            image: Input image.

        Returns:
            List of detected text regions.
        """
        logger.info("Detecting text regions using EAST")

        # Lazy load EAST model
        if self.east_model is None:
            model_path = self.config.east_model_path
            if not os.path.exists(model_path):
                logger.warning(f"EAST model not found at {model_path}. Using contours instead.")
                return self._detect_with_contours(image)

            self.east_model = cv2.dnn.readNet(model_path)

        # Prepare image for EAST
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (123.68, 116.78, 103.94), swapRB=True)

        # Forward pass
        self.east_model.setInput(blob)
        scores, geometry = self.east_model.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fronfusion/concat_3"])

        # Decode detections
        regions = self._decode_east_detections(scores, geometry)

        return regions

    def _decode_east_detections(
        self,
        scores: np.ndarray,
        geometry: np.ndarray
    ) -> List[TextRegion]:
        """
        Decode EAST detections into text regions.

        Args:
            scores: Detection scores from EAST.
            geometry: Geometry information from EAST.

        Returns:
            List of text regions.
        """
        regions = []
        rows, cols = scores.shape[2:4]

        for y in range(rows):
            for x in range(cols):
                score = scores[0, 0, y, x]
                if score < self.config.confidence_threshold:
                    continue

                # Get geometry
                offset_x = x * 4.0
                offset_y = y * 4.0

                angle = geometry[0, 4, y, x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = geometry[0, 0, y, x] + geometry[0, 2, y, x]
                w = geometry[0, 1, y, x] + geometry[0, 3, y, x]

                end_x = int(offset_x + cos * geometry[0, 1, y, x] + sin * geometry[0, 3, y, x])
                end_y = int(offset_y - sin * geometry[0, 1, y, x] + cos * geometry[0, 3, y, x])

                start_x = int(end_x - w)
                start_y = int(end_y - h)

                # Ensure valid coordinates
                if start_x < 0 or start_y < 0:
                    continue

                regions.append(TextRegion(
                    bbox=(start_x, start_y, w, h),
                    confidence=float(score)
                ))

        return regions


def detect_text_regions(
    image: np.ndarray,
    method: str = "contour",
    config: Optional[TextDetectorConfig] = None
) -> List[TextRegion]:
    """
    Convenience function to detect text regions.

    Args:
        image: Input image.
        method: Detection method ("contour" or "east").
        config: Optional configuration.

    Returns:
        List of detected text regions.
    """
    if config is None:
        config = get_config().text_detector

    # Update method if specified
    if method:
        config.method = method

    detector = TextDetector(config)
    return detector.detect(image)
```