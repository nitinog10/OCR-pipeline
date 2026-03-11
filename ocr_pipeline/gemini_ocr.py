"""
Gemini OCR Module for OCR Pipeline.

Uses Google Gemini Vision API to extract text from detected regions.
Handles batching, caching, and error handling.
"""

import base64
import io
import hashlib
import json
import os
import time
import logging
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from config import GeminiConfig, OCRConfig, get_config
from text_detector import TextRegion

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result of OCR on a text region."""
    region_id: int
    text: str
    confidence: float
    error: Optional[str] = None


class GeminiOCR:
    """
    Handles OCR using Google Gemini Vision API.

    Features:
    - Batching of requests for efficiency
    - Caching of results to avoid redundant API calls
    - Multiprocessing for parallel processing
    - Automatic retry with exponential backoff
    """

    def __init__(
        self,
        gemini_config: Optional[GeminiConfig] = None,
        ocr_config: Optional[OCRConfig] = None
    ):
        """
        Initialize the Gemini OCR handler.

        Args:
            gemini_config: Gemini API configuration.
            ocr_config: OCR processing configuration.
        """
        self.gemini_config = gemini_config or get_config().gemini
        self.ocr_config = ocr_config or get_config().ocr

        # Initialize API
        self._setup_api()

        # Setup caching
        self.cache = {}
        if self.ocr_config.enable_caching:
            self._setup_cache()

        # Setup model
        self.model = None

    def _setup_api(self):
        """Setup Gemini API client."""
        if not self.gemini_config.api_key:
            # Try to get from environment
            import os
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Gemini API key not configured. "
                    "Set GEMINI_API_KEY environment variable or pass api_key to config."
                )
            self.gemini_config.api_key = api_key

        genai.configure(api_key=self.gemini_config.api_key)

        # Create generation config
        self.generation_config = {
            "temperature": self.gemini_config.temperature,
            "max_output_tokens": self.gemini_config.max_output_tokens,
        }

        logger.info("Gemini API configured successfully")

    def _setup_cache(self):
        """Setup caching directory."""
        cache_dir = self.ocr_config.cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"OCR cache enabled at {cache_dir}")

    def _get_cache_key(self, image: np.ndarray) -> str:
        """
        Generate cache key for an image.

        Args:
            image: Image as numpy array.

        Returns:
            Cache key string.
        """
        # Encode image as bytes
        _, buffer = cv2.imencode('.png', image)
        image_bytes = buffer.tobytes()

        # Generate hash
        return hashlib.md5(image_bytes).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """
        Get cached result.

        Args:
            cache_key: Cache key.

        Returns:
            Cached text or None.
        """
        if not self.ocr_config.enable_caching:
            return None

        cache_file = os.path.join(self.ocr_config.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                logger.debug(f"Cache hit for key {cache_key}")
                return data.get("text")
            except Exception as e:
                logger.warning(f"Failed to read cache: {e}")

        return None

    def _save_to_cache(self, cache_key: str, text: str):
        """
        Save result to cache.

        Args:
            cache_key: Cache key.
            text: Extracted text.
        """
        if not self.ocr_config.enable_caching:
            return

        cache_file = os.path.join(self.ocr_config.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump({"text": text, "timestamp": time.time()}, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _encode_image_base64(self, image: np.ndarray) -> str:
        """
        Encode image as base64 string.

        Args:
            image: Image as numpy array.

        Returns:
            Base64 encoded string.
        """
        # Encode as PNG
        _, buffer = cv2.imencode('.png', image)
        image_bytes = buffer.tobytes()
        return base64.b64encode(image_bytes).decode('utf-8')

    def _extract_text_from_image(
        self,
        image: np.ndarray,
        retry_count: int = 3
    ) -> str:
        """
        Extract text from a single image using Gemini.

        Args:
            image: Image to process.
            retry_count: Number of retries on failure.

        Returns:
            Extracted text.
        """
        # Check cache
        cache_key = self._get_cache_key(image)
        cached_text = self._get_from_cache(cache_key)
        if cached_text is not None:
            return cached_text

        # Lazily initialize model
        if self.model is None:
            self.model = genai.GenerativeModel(
                model_name=self.gemini_config.model_name,
                generation_config=self.generation_config
            )

        # Encode image
        image_base64 = self._encode_image_base64(image)

        # Create prompt for OCR
        prompt = """You are an OCR system. Extract ALL text from this image accurately.
        - Preserve the exact text content
        - Maintain line breaks as they appear in the image
        - Do not add explanations or commentary
        - If no text is found, return an empty string
        - Handle mixed content including print and handwritten text
        """

        for attempt in range(retry_count):
            try:
                # Create content with image
                content = [
                    {"text": prompt},
                    {
                        "mime_type": "image/png",
                        "data": image_base64
                    }
                ]

                # Generate response
                response = self.model.generate_content(content)
                text = response.text.strip()

                # Save to cache
                self._save_to_cache(cache_key, text)

                return text

            except google_exceptions.ResourceExhausted:
                # Rate limited - wait and retry with exponential backoff
                wait_time = 2 ** attempt
                logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                time.sleep(wait_time)

            except google_exceptions.GoogleAPIError as e:
                logger.error(f"Google API error: {e}")
                if attempt < retry_count - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise

            except Exception as e:
                logger.error(f"Error extracting text: {e}")
                if attempt < retry_count - 1:
                    time.sleep(1)
                else:
                    raise

        return ""

    def extract_text_from_regions(
        self,
        image: np.ndarray,
        regions: List[TextRegion]
    ) -> List[OCRResult]:
        """
        Extract text from multiple detected regions.

        Args:
            image: Source image.
            regions: List of detected text regions.

        Returns:
            List of OCR results.
        """
        logger.info(f"Starting OCR on {len(regions)} regions")

        results = []

        if self.ocr_config.use_batching and len(regions) > self.ocr_config.batch_size:
            # Use batch processing
            results = self._process_batched(image, regions)
        elif self.ocr_config.use_multiprocessing:
            # Use multiprocessing
            results = self._process_parallel(image, regions)
        else:
            # Sequential processing
            results = self._process_sequential(image, regions)

        # Filter out empty results
        results = [r for r in results if r.text and r.text.strip()]

        logger.info(f"OCR completed: {len(results)} regions with text")
        return results

    def _process_sequential(
        self,
        image: np.ndarray,
        regions: List[TextRegion]
    ) -> List[OCRResult]:
        """Process regions sequentially."""
        results = []

        for i, region in enumerate(regions):
            logger.debug(f"Processing region {i+1}/{len(regions)}")

            try:
                # Crop region
                crop = self._crop_region(image, region)

                # Extract text
                text = self._extract_text_from_image(crop)

                results.append(OCRResult(
                    region_id=i,
                    text=text,
                    confidence=1.0
                ))

            except Exception as e:
                logger.error(f"Error processing region {i}: {e}")
                results.append(OCRResult(
                    region_id=i,
                    text="",
                    confidence=0.0,
                    error=str(e)
                ))

        return results

    def _process_parallel(
        self,
        image: np.ndarray,
        regions: List[TextRegion]
    ) -> List[OCRResult]:
        """Process regions in parallel using ThreadPoolExecutor."""
        results = []
        max_workers = self.ocr_config.max_workers

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_region = {}
            for i, region in enumerate(regions):
                future = executor.submit(self._process_single_region, image, region, i)
                future_to_region[future] = i

            # Collect results
            for future in as_completed(future_to_region):
                i = future_to_region[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in parallel processing for region {i}: {e}")
                    results.append(OCRResult(
                        region_id=i,
                        text="",
                        confidence=0.0,
                        error=str(e)
                    ))

        # Sort by region_id
        results.sort(key=lambda x: x.region_id)
        return results

    def _process_batched(
        self,
        image: np.ndarray,
        regions: List[TextRegion]
    ) -> List[OCRResult]:
        """Process regions in batches for efficiency."""
        batch_size = self.ocr_config.batch_size
        results = []

        logger.info(f"Processing {len(regions)} regions in batches of {batch_size}")

        for i in range(0, len(regions), batch_size):
            batch = regions[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(regions) + batch_size - 1) // batch_size

            logger.debug(f"Processing batch {batch_num}/{total_batches}")

            # Process batch in parallel
            batch_results = self._process_parallel(image, batch)

            # Adjust region IDs
            for result in batch_results:
                result.region_id += i
                results.append(result)

            # Small delay between batches to avoid rate limiting
            if total_batches > 1:
                time.sleep(0.5)

        return results

    def _process_single_region(
        self,
        image: np.ndarray,
        region: TextRegion,
        region_id: int
    ) -> OCRResult:
        """Process a single region (for parallel execution)."""
        try:
            crop = self._crop_region(image, region)
            text = self._extract_text_from_image(crop)

            return OCRResult(
                region_id=region_id,
                text=text,
                confidence=1.0
            )
        except Exception as e:
            return OCRResult(
                region_id=region_id,
                text="",
                confidence=0.0,
                error=str(e)
            )

    def _crop_region(
        self,
        image: np.ndarray,
        region: TextRegion,
        padding: int = 10
    ) -> np.ndarray:
        """
        Crop a region from the image with padding.

        Args:
            image: Source image.
            region: Text region to crop.
            padding: Padding around the region.

        Returns:
            Cropped image.
        """
        h, w = image.shape[:2]

        x1 = max(0, region.x - padding)
        y1 = max(0, region.y - padding)
        x2 = min(w, region.x + region.w + padding)
        y2 = min(h, region.y + region.h + padding)

        return image[y1:y2, x1:x2]

    def extract_full_image_text(self, image: np.ndarray) -> str:
        """
        Extract text from the entire image without region detection.

        Useful for simple documents or when regions are not needed.

        Args:
            image: Input image.

        Returns:
            Extracted text from entire image.
        """
        logger.info("Extracting text from full image")
        return self._extract_text_from_image(image)


def extract_text_from_image(
    image: np.ndarray,
    regions: Optional[List[TextRegion]] = None,
    use_regions: bool = True
) -> Union[List[OCRResult], str]:
    """
    Convenience function to extract text from an image.

    Args:
        image: Input image.
        regions: Optional list of text regions. If None, processes entire image.
        use_regions: Whether to use region detection (if regions is None).

    Returns:
        List of OCR results or full text.
    """
    ocr = GeminiOCR()

    if regions:
        # Process known regions
        return ocr.extract_text_from_regions(image, regions)
    elif use_regions:
        # Need to detect regions first
        from text_detector import TextDetector
        detector = TextDetector()
        detected_regions = detector.detect(image)
        return ocr.extract_text_from_regions(image, detected_regions)
    else:
        # Process entire image
        return ocr.extract_full_image_text(image)