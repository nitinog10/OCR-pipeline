# OCR Pipeline

A production-quality OCR (Optical Character Recognition) pipeline built from scratch using Python and Google Gemini Vision API.

## Overview

This OCR pipeline takes documents (images or PDFs) and extracts structured text using a multi-stage process:

```
┌─────────┐    ┌──────────────┐    ┌────────────┐    ┌─────────┐    ┌────────────┐
│  INPUT  │───▶│ PREPROCESS  │───▶│   DETECT   │───▶│   OCR   │───▶│  OUTPUT    │
│  LAYER  │    │    STAGE     │    │   STAGE    │    │   STAGE │    │   LAYER    │
└─────────┘    └──────────────┘    └────────────┘    └─────────┘    └────────────┘
```

## Pipeline Stages

### 1. Input Layer
- **Accepts**: JPG, PNG, JPEG, BMP, TIFF, WEBP images
- **Accepts**: PDF files (converts to images using pdf2image)
- Handles multi-page PDFs by processing each page separately
- Uses Pillow and OpenCV for flexible image loading

### 2. Preprocessing Stage
Improves image quality for better OCR accuracy:

- **Grayscale Conversion**: Converts color images to grayscale for faster processing
- **Noise Removal**: Uses Gaussian, Bilateral, or Median filtering
- **Thresholding**: Adaptive, Otsu, or simple thresholding for binarization
- **Deskewing**: Detects and corrects document rotation
- **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Resizing**: Scales images to optimal DPI for OCR

### 3. Text Detection Stage
Identifies where text is located in the image:

- **Contour-Based Detection** (Primary):
  1. Applies morphological operations to enhance text regions
  2. Finds contours using OpenCV
  3. Filters by size, aspect ratio, and position

- **EAST Text Detector** (Optional):
  1. Uses deep learning model for scene text detection
  2. More accurate for complex layouts
  3. Requires model file download

### 4. OCR Stage
Extracts text from detected regions using Gemini Vision:

- Crops each detected region from the image
- Sends to Google Gemini 1.5 Flash API
- Supports batching for efficiency
- Includes caching to avoid redundant API calls
- Parallel processing with multiprocessing

### 5. Post-Processing Stage
Cleans and structures the extracted text:

- **Artifact Removal**: Removes OCR misreads and control characters
- **Whitespace Normalization**: Fixes spacing issues
- **Regex Pattern Matching**: Extracts dates, emails, phones, URLs
- **Spell Correction** (Optional): Uses pyspellchecker
- **Line Reconstruction**: Reconstructs paragraphs and formatting
- **Structure Extraction**: Identifies title, date, and other metadata

## Installation

### Prerequisites
- Python 3.8+
- Poppler for PDF processing (Windows: download and add to PATH)

### Install Dependencies

```bash
# Navigate to project directory
cd ocr_pipeline

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Poppler Installation (Required for PDF support)

**Windows:**
1. Download from https://github.com/oschwartz10612/poppler-windows/releases/
2. Extract to a folder (e.g., C:\poppler)
3. Add `C:\poppler\Library\bin` to PATH

**macOS:**
```bash
brew install poppler
```

**Linux:**
```bash
sudo apt-get install poppler-utils
```

### Set Up Gemini API Key

Get your API key from: https://aistudio.google.com/app/apikey

```bash
# Option 1: Set environment variable
export GEMINI_API_KEY="your-api-key-here"

# Option 2: Pass as command-line argument
python main.py --file document.pdf --api-key "your-api-key-here"
```

## Usage

### Basic Usage

```bash
# Process a PDF
python main.py --file invoice.pdf

# Process an image
python main.py --file receipt.jpg

# Save results to file
python main.py --file document.pdf --output results.json
```

### Advanced Options

```bash
# Process full image without region detection (faster)
python main.py --file scan.png --full-image

# Show preview windows
python main.py --file document.pdf --preview

# Save preprocessed images for debugging
python main.py --file document.pdf --save-preprocessed

# Adjust batch size for API calls
python main.py --file document.pdf --batch-size 5

# Use different text detection method
python main.py --file document.pdf --method east
```

### Python API

```python
from ocr_pipeline import OCRPipeline

# Create pipeline
pipeline = OCRPipeline()

# Process document
result = pipeline.process_file("invoice.pdf", "output.json")

# Access results
print(result.title)         # Extracted title
print(result.date)          # Extracted date
print(result.cleaned_text)  # Cleaned text
print(result.to_json())     # Full JSON output
```

### Component Usage

```python
from ocr_pipeline import ImagePreprocessor, TextDetector, GeminiOCR

# Preprocessing only
preprocessor = ImagePreprocessor()
processed = preprocessor.preprocess(image)

# Text detection only
detector = TextDetector()
regions = detector.detect(processed)

# OCR only
ocr = GeminiOCR()
results = ocr.extract_text_from_regions(processed, regions)
```

## Project Structure

```
ocr_pipeline/
├── __init__.py          # Package initialization
├── main.py              # Main entry point
├── config.py            # Configuration settings
├── preprocess.py        # Image preprocessing
├── text_detector.py     # Text region detection
├── gemini_ocr.py        # Gemini Vision OCR
├── postprocess.py       # Text post-processing
├── utils.py             # Utility functions
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Configuration

All settings are in `config.py`. Key configurations:

```python
# Gemini API
model_name: "gemini-1.5-flash"
temperature: 0.1

# Preprocessing
use_grayscale: True
enhance_contrast: True  # CLAHE

# Text Detection
method: "contour"  # or "east"
min_text_height: 10

# OCR Processing
batch_size: 10
use_batching: True
enable_caching: True

# Post-processing
extract_dates: True
extract_emails: True
```

## Output Format

```json
{
  "raw_text": "Original extracted text...",
  "cleaned_text": "Cleaned and formatted text...",
  "title": "Document Title",
  "date": "2024-01-15",
  "entities": {
    "emails": ["email@example.com"],
    "phones": ["+1234567890"],
    "urls": ["https://example.com"],
    "currency": ["$99.99"],
    "page_count": 2,
    "region_count": 15
  }
}
```

## Performance

- **Processing Speed**: ~1-3 seconds per page (depends on content)
- **API Calls**: Batched for efficiency
- **Caching**: Results cached to avoid redundant API calls
- **Parallel Processing**: Uses multiprocessing for multiple regions

## Troubleshooting

### Common Issues

1. **"Could not load image"**
   - Check file path and format

2. **"Gemini API key not configured"**
   - Set GEMINI_API_KEY environment variable

3. **"PDF page count error"**
   - Install Poppler and add to PATH

4. **"Rate limited"**
   - Reduce batch_size in config

### Debug Mode

```bash
python main.py --file document.pdf --log-level DEBUG
```

## License

MIT License

## Credits

Built with:
- OpenCV - Image processing
- Google Gemini Vision - OCR
- pdf2image - PDF conversion
- Pillow - Image handling