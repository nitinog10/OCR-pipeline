"""
Post-processing Module for OCR Pipeline.

Cleans and structures OCR output using:
- Regex pattern matching
- Spell correction
- Artifact removal
- Line reconstruction
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from collections import defaultdict

from config import PostProcessConfig, get_config

# Try to import spellchecker, but make it optional
try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    logging.warning("pyspellchecker not installed. Spell correction disabled.")

from gemini_ocr import OCRResult

logger = logging.getLogger(__name__)


@dataclass
class StructuredOutput:
    """Structured output from OCR."""
    raw_text: str
    cleaned_text: str
    title: Optional[str] = None
    date: Optional[str] = None
    entities: Dict[str, List[str]] = None

    def __post_init__(self):
        if self.entities is None:
            self.entities = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "title": self.title,
            "date": self.date,
            "entities": self.entities
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class TextPostProcessor:
    """
    Post-processes OCR text output for cleaning and structuring.

    Features:
    - Regex-based pattern extraction
    - Optional spell checking
    - Artifact removal
    - Line and paragraph reconstruction
    """

    def __init__(self, config: Optional[PostProcessConfig] = None):
        """
        Initialize the post-processor.

        Args:
            config: Post-processing configuration.
        """
        self.config = config or get_config().postprocess

        # Initialize spell checker if available and enabled
        self.spell_checker = None
        if self.config.enable_spell_check and SPELLCHECKER_AVAILABLE:
            try:
                self.spell_checker = SpellChecker(language=self.config.spellcheck_language)
                logger.info(f"Spell checker initialized for language: {self.config.spellcheck_language}")
            except Exception as e:
                logger.warning(f"Failed to initialize spell checker: {e}")

        # Compile regex patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile commonly used regex patterns."""
        # Date patterns
        self.date_patterns = [
            # MM/DD/YYYY or DD/MM/YYYY
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            # Month DD, YYYY
            r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}',
            # YYYY-MM-DD (ISO)
            r'\d{4}-\d{2}-\d{2}',
        ]

        # Email pattern
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

        # Phone patterns
        self.phone_patterns = [
            r'\+?1?\s*[-.(]?\d{3}[-.)]\s*\d{3}[-.\s]?\d{4}',  # US
            r'\d{10,}',  # Generic 10+ digits
            r'\+?\d{1,4}[-\s]?\(?\d{1,4}\)?[-\s]?\d{1,4}[-\s]?\d{1,9}',  # International
        ]

        # Currency pattern
        self.currency_pattern = r'[\$£€¥]?\s*\d+(?:,\d{3})*(?:\.\d{2})?'

        # URL pattern
        self.url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'

        # Compile all patterns
        self.compiled_date = [re.compile(p, re.IGNORECASE) for p in self.date_patterns]
        self.compiled_email = re.compile(self.email_pattern)
        self.compiled_phone = [re.compile(p) for p in self.phone_patterns]
        self.compiled_currency = re.compile(self.currency_pattern)
        self.compiled_url = re.compile(self.url_pattern)

    def process(self, ocr_results: List[OCRResult]) -> str:
        """
        Process OCR results and return cleaned text.

        Args:
            ocr_results: List of OCR results from Gemini.

        Returns:
            Cleaned and processed text.
        """
        logger.info(f"Post-processing {len(ocr_results)} OCR results")

        # Combine all text
        combined_text = "\n".join([r.text for r in ocr_results])

        # Clean the text
        cleaned = self.clean_text(combined_text)

        logger.info("Post-processing completed")
        return cleaned

    def process_single(self, text: str) -> str:
        """
        Process a single text string.

        Args:
            text: Input text.

        Returns:
            Cleaned text.
        """
        return self.clean_text(text)

    def clean_text(self, text: str) -> str:
        """
        Apply all cleaning operations to text.

        Args:
            text: Raw text from OCR.

        Returns:
            Cleaned text.
        """
        logger.debug("Cleaning text")

        # Step 1: Remove artifacts
        if self.config.remove_artifacts:
            text = self._remove_artifacts(text)

        # Step 2: Normalize whitespace
        if self.config.normalize_whitespace:
            text = self._normalize_whitespace(text)

        # Step 3: Remove extra spaces
        if self.config.remove_extra_spaces:
            text = self._remove_extra_spaces(text)

        # Step 4: Spell correction (if enabled)
        if self.config.enable_spell_check and self.spell_checker:
            text = self._spell_check(text)

        # Step 5: Reconstruct lines/paragraphs
        if self.config.reconstruct_lines:
            text = self._reconstruct_lines(text)

        logger.debug("Text cleaning completed")
        return text

    def _remove_artifacts(self, text: str) -> str:
        """Remove common OCR artifacts."""
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)

        # Remove unusual unicode characters that aren't text
        # Keep basic punctuation and alphanumeric
        # This is aggressive - might need tuning
        text = re.sub(r'[^\w\s\-.,!?;:\'"()\[\]{}@#$%&*+=/<>\\^_`~]', '', text)

        # Fix common OCR misreads
        replacements = {
            '|': 'I',  # Vertical bar to I
            '0': 'O',  # Only in specific contexts, be careful
            '—': '-',
            '–': '-',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '...': '.',
        }

        # Be careful with these replacements - context matters
        # Only apply in certain contexts

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize various whitespace characters."""
        # Replace multiple spaces with single
        text = re.sub(r'[ \t]+', ' ', text)

        # Replace newlines that should be spaces
        text = re.sub(r'(?<=[a-z])\n(?=[a-z])', ' ', text, flags=re.IGNORECASE)

        # But preserve paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _remove_extra_spaces(self, text: str) -> str:
        """Remove extra spaces around punctuation."""
        # Space before punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)

        # Multiple spaces
        text = re.sub(r' {2,}', ' ', text)

        # Space after opening bracket
        text = re.sub(r'\(\s+', '(', text)

        # Space before closing bracket
        text = re.sub(r'\s+\)', ')', text)

        return text

    def _spell_check(self, text: str) -> str:
        """Apply spell checking and correction."""
        if not self.spell_checker:
            return text

        words = text.split()
        corrected = []

        for word in words:
            # Skip if it's mostly numbers or punctuation
            if re.match(r'^[\d\-\.,!?;:\'"()\[\]{}]+$', word):
                corrected.append(word)
                continue

            # Check spelling
            misspelled = self.spell_checker.unknown([word])

            if word in misspelled:
                # Get suggestions
                suggestions = list(self.spell_checker.candidates(word))
                if suggestions:
                    # Pick the most likely correction
                    corrected.append(suggestions[0])
                else:
                    corrected.append(word)
            else:
                corrected.append(word)

        return ' '.join(corrected)

    def _reconstruct_lines(self, text: str) -> str:
        """Reconstruct lines and paragraphs."""
        # Split into lines
        lines = text.split('\n')

        # Process each line
        processed_lines = []
        for line in lines:
            line = line.strip()
            if line:
                processed_lines.append(line)

        # Join with proper line breaks
        return '\n'.join(processed_lines)

    def extract_structured_info(self, text: str) -> Dict[str, Any]:
        """
        Extract structured information from text.

        Args:
            text: Cleaned text.

        Returns:
            Dictionary with extracted information.
        """
        logger.debug("Extracting structured information")

        result = {
            "title": None,
            "date": None,
            "emails": [],
            "phones": [],
            "urls": [],
            "currency": [],
        }

        # Extract title (first non-empty line that looks like a title)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            # Title is usually the first significant line
            for line in lines[:3]:
                if len(line) > 3 and len(line) < 100:
                    # Exclude lines that are mostly numbers
                    if not re.match(r'^[\d\s\-\./]+$', line):
                        result["title"] = line
                        break

        # Extract dates
        for pattern in self.compiled_date:
            matches = pattern.findall(text)
            result["date"] = matches[0] if matches else None
            if result["date"]:
                break

        # Extract emails
        emails = self.compiled_email.findall(text)
        result["emails"] = list(set(emails))

        # Extract phones
        phones = []
        for pattern in self.compiled_phone:
            found = pattern.findall(text)
            phones.extend(found)
        result["phones"] = list(set(phones))

        # Extract URLs
        urls = self.compiled_url.findall(text)
        result["urls"] = list(set(urls))

        # Extract currency
        currency = self.compiled_currency.findall(text)
        result["currency"] = list(set(currency))

        return result

    def create_structured_output(
        self,
        ocr_results: List[OCRResult]
    ) -> StructuredOutput:
        """
        Create fully structured output from OCR results.

        Args:
            ocr_results: List of OCR results.

        Returns:
            Structured output with all information.
        """
        logger.info("Creating structured output")

        # Get raw text
        raw_text = "\n".join([r.text for r in ocr_results])

        # Clean text
        cleaned_text = self.clean_text(raw_text)

        # Extract structured information
        info = self.extract_structured_info(cleaned_text)

        # Create output
        output = StructuredOutput(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            title=info.get("title"),
            date=info.get("date"),
            entities={
                "emails": info.get("emails", []),
                "phones": info.get("phones", []),
                "urls": info.get("urls", []),
                "currency": info.get("currency", []),
            }
        )

        return output


def postprocess_ocr_results(
    ocr_results: List[OCRResult],
    create_structure: bool = True
) -> Union[str, StructuredOutput]:
    """
    Convenience function to post-process OCR results.

    Args:
        ocr_results: List of OCR results.
        create_structure: Whether to create structured output.

    Returns:
        Cleaned text or structured output.
    """
    processor = TextPostProcessor()

    if create_structure:
        return processor.create_structured_output(ocr_results)
    else:
        return processor.process(ocr_results)