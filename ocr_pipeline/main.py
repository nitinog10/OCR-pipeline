"""
OCR Pipeline - Main Entry Point

Complete OCR pipeline using Gemini Vision for document text extraction.

Usage:
    python main.py --file input.pdf
    python main.py --file input.jpg --output results.json
    python main.py --file document.pdf --preview

The pipeline works in stages:
1. INPUT: Load document (image or PDF)
2. PREPROCESS: Enhance image quality for better OCR
3. DETECT: Find text regions in the image
4. OCR: Extract text using Gemini Vision
5. POSTPROCESS: Clean and structure the output

Author: OCR Pipeline Team
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, List
import json

from config import (
    get_config,
    PipelineConfig,
    GeminiConfig,
    PreprocessingConfig,
    TextDetectorConfig,
    OCRConfig,
    PostProcessConfig
)

# Import pipeline components
from preprocess import ImagePreprocessor
from text_detector import TextDetector, TextRegion
from gemini_ocr import GeminiOCR, OCRResult
from postprocess import TextPostProcessor, StructuredOutput
from utils import (
    setup_logging,
    get_logger,
    load_document,
    save_ocr_results,
    FileUtils,
    ImageUtils
)

# Setup logger
logger = get_logger(__name__)


class OCRPipeline:
    """
    Main OCR Pipeline class that orchestrates all stages.

    This class manages the flow from document input to structured output,
    handling all the intermediate processing steps.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the OCR pipeline.

        Args:
            config: Pipeline configuration. Uses default if not provided.
        """
        self.config = config or get_config()

        # Initialize components
        self.preprocessor = ImagePreprocessor(self.config.preprocessing)
        self.text_detector = TextDetector(self.config.text_detector)
        self.gemini_ocr = GeminiOCR(self.config.gemini, self.config.ocr)
        self.postprocessor = TextPostProcessor(self.config.postprocess)

        logger.info("OCR Pipeline initialized")

    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        save_preprocessed: bool = False,
        show_preview: bool = False
    ) -> StructuredOutput:
        """
        Process a document file through the OCR pipeline.

        Args:
            input_path: Path to input document (image or PDF).
            output_path: Optional path to save results.
            save_preprocessed: Whether to save preprocessed images.
            show_preview: Whether to show preview windows.

        Returns:
            Structured output with extracted text and metadata.
        """
        start_time = time.time()
        logger.info(f"Processing file: {input_path}")

        # Stage 1: Load document
        logger.info("Stage 1/5: Loading document...")
        images = load_document(input_path)
        logger.info(f"Loaded {len(images)} page(s)")

        # Process each page
        all_results = []
        all_regions = []

        for page_num, image in enumerate(images):
            logger.info(f"\n--- Processing Page {page_num + 1}/{len(images)} ---")

            # Stage 2: Preprocess
            logger.info("Stage 2/5: Preprocessing image...")
            preprocessed = self.preprocessor.preprocess(image)

            if save_preprocessed:
                preprocessed_path = f"preprocessed_page_{page_num + 1}.png"
                ImageUtils.save_image(preprocessed, preprocessed_path)
                logger.info(f"Saved preprocessed image to {preprocessed_path}")

            if show_preview:
                ImageUtils.display_image(preprocessed, f"Preprocessed Page {page_num + 1}")

            # Stage 3: Detect text regions
            logger.info("Stage 3/5: Detecting text regions...")
            regions = self.text_detector.detect(preprocessed)
            logger.info(f"Detected {len(regions)} text regions")

            if show_preview:
                # Draw regions on original image for preview
                preview = self.text_detector.draw_regions(preprocessed, regions)
                ImageUtils.display_image(preview, f"Detected Regions - Page {page_num + 1}")

            all_regions.extend(regions)

            # Stage 4: OCR with Gemini
            logger.info("Stage 4/5: Extracting text with Gemini Vision...")
            ocr_results = self.gemini_ocr.extract_text_from_regions(preprocessed, regions)
            logger.info(f"Extracted text from {len(ocr_results)} regions")

            all_results.extend(ocr_results)

        # Stage 5: Post-process
        logger.info("\nStage 5/5: Post-processing results...")
        structured_output = self.postprocessor.create_structured_output(all_results)

        # Add page information
        structured_output.entities["page_count"] = len(images)
        structured_output.entities["region_count"] = len(all_regions)

        # Calculate processing time
        elapsed_time = time.time() - start_time
        logger.info(f"\n✓ Processing completed in {elapsed_time:.2f} seconds")

        # Save output if requested
        if output_path:
            save_ocr_results(structured_output, output_path, format="json")
            logger.info(f"Results saved to: {output_path}")

            # Also save raw text
            txt_path = Path(output_path).with_suffix('.txt')
            save_ocr_results(structured_output, str(txt_path), format="txt")
            logger.info(f"Raw text saved to: {txt_path}")

        return structured_output

    def process_image_only(
        self,
        image_path: str,
        full_image: bool = True
    ) -> str:
        """
        Process a single image without region detection.

        Args:
            image_path: Path to image.
            full_image: If True, process entire image. If False, detect regions first.

        Returns:
            Extracted text.
        """
        logger.info(f"Processing image: {image_path}")

        # Load image
        image = ImageUtils.load_image(image_path)

        # Preprocess
        preprocessed = self.preprocessor.preprocess(image)

        if full_image:
            # Process entire image
            text = self.gemini_ocr.extract_full_image_text(preprocessed)
        else:
            # Detect and process regions
            regions = self.text_detector.detect(preprocessed)
            ocr_results = self.gemini_ocr.extract_text_from_regions(preprocessed, regions)
            text = "\n".join([r.text for r in ocr_results])

        # Post-process
        cleaned = self.postprocessor.process_single(text)

        return cleaned


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="OCR Pipeline - Extract text from documents using Gemini Vision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --file invoice.pdf
  python main.py --file receipt.jpg --output results.json
  python main.py --file document.pdf --preview
  python main.py --file scan.png --full-image

For more information, see the README.md file.
        """
    )

    parser.add_argument(
        '--file', '-f',
        required=True,
        help='Input file (image or PDF)'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output file path (JSON format)'
    )

    parser.add_argument(
        '--preview', '-p',
        action='store_true',
        help='Show preview windows at each stage'
    )

    parser.add_argument(
        '--save-preprocessed',
        action='store_true',
        help='Save preprocessed images'
    )

    parser.add_argument(
        '--full-image',
        action='store_true',
        help='Process entire image without region detection (faster but less accurate)'
    )

    parser.add_argument(
        '--method',
        choices=['contour', 'east'],
        default='contour',
        help='Text detection method (default: contour)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for OCR processing'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--api-key',
        help='Gemini API key (or set GEMINI_API_KEY environment variable)'
    )

    return parser.parse_args()


def main():
    """Main entry point for the OCR pipeline."""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(level=args.log_level)

    # Validate input file
    input_path = Path(args.file)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.file}")
        sys.exit(1)

    # Validate file type
    if not FileUtils.is_image(input_path) and not FileUtils.is_pdf(input_path):
        logger.error(f"Unsupported file type: {input_path.suffix}")
        logger.error("Supported formats: JPG, PNG, PDF")
        sys.exit(1)

    # Configure pipeline
    config = get_config()

    # Override config with command-line arguments
    if args.api_key:
        config.gemini.api_key = args.api_key

    if args.batch_size:
        config.ocr.batch_size = args.batch_size

    config.text_detector.method = args.method

    # Validate API key
    if not config.gemini.api_key:
        logger.error("Gemini API key not configured!")
        logger.error("Please set GEMINI_API_KEY environment variable or use --api-key")
        logger.error("\nGet your API key from: https://aistudio.google.com/app/apikey")
        sys.exit(1)

    # Create pipeline
    pipeline = OCRPipeline(config)

    try:
        # Process file
        if args.full_image:
            # Simple mode - process entire image
            logger.info("Running in full-image mode (no region detection)")
            result = pipeline.process_image_only(args.file, full_image=True)
            print("\n" + "=" * 50)
            print("EXTRACTED TEXT:")
            print("=" * 50)
            print(result)

            # Save if output specified
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump({"text": result}, f, indent=2)
                print(f"\nSaved to: {args.output}")

        else:
            # Full pipeline with region detection
            result = pipeline.process_file(
                args.file,
                output_path=args.output,
                save_preprocessed=args.save_preprocessed,
                show_preview=args.preview
            )

            # Display results
            print("\n" + "=" * 50)
            print("OCR RESULTS")
            print("=" * 50)

            if result.title:
                print(f"\nTitle: {result.title}")

            if result.date:
                print(f"Date: {result.date}")

            print("\n--- Extracted Text ---")
            print(result.cleaned_text)

            if result.entities:
                print("\n--- Extracted Entities ---")
                for key, value in result.entities.items():
                    if value:
                        print(f"  {key}: {value}")

            print(f"\n--- Statistics ---")
            print(f"  Pages processed: {result.entities.get('page_count', 1)}")
            print(f"  Text regions: {result.entities.get('region_count', 'N/A')}")

            print(f"\n✓ Results saved to: {args.output or 'stdout'}")

    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()