"""Unit tests for the Multilingual Indic OCR Engine."""

import pytest
import unicodedata
import numpy as np
from cognitive.ocr_engine import MultilingualOCREngine


def test_indic_unicode_nfc_normalization():
    """Verify Indic text normalization produces canonical Unicode NFC representation."""
    ocr = MultilingualOCREngine(unicode_normalization="NFC")

    # Hindi Devanagari: Decomposed vs Composed
    # "कि" decomposed is 'क' (U+0915) + 'ि' (U+093F)
    decomposed_hindi = unicodedata.normalize("NFD", "दवा की दुकान 500mg")
    normalized_hindi = ocr.normalize_indic_text(decomposed_hindi)
    assert unicodedata.is_normalized("NFC", normalized_hindi)
    assert "दवा" in normalized_hindi

    # Malayalam: "മരുന്ന്" (Medicine)
    decomposed_mal = unicodedata.normalize("NFD", "മരുന്ന് ഷോപ്പ്")
    normalized_mal = ocr.normalize_indic_text(decomposed_mal)
    assert unicodedata.is_normalized("NFC", normalized_mal)
    assert "മരുന്ന്" in normalized_mal

    # Tamil: "மருந்து" (Medicine)
    decomposed_tam = unicodedata.normalize("NFD", "மருந்து கடை")
    normalized_tam = ocr.normalize_indic_text(decomposed_tam)
    assert unicodedata.is_normalized("NFC", normalized_tam)
    assert "மருந்து" in normalized_tam


def test_ocr_image_preprocessing():
    """Verify grayscale conversion and contrast stretching."""
    ocr = MultilingualOCREngine()

    # Create synthetic gradient image with high-contrast text bar
    test_img = np.full((100, 200, 3), 120, dtype=np.uint8)
    test_img[30:70, 20:180, :] = 240  # Simulated white text block

    binary = ocr.preprocess_image_for_ocr(test_img)
    assert binary.shape == (100, 200)
    assert binary.dtype == np.uint8
    # Should be binarized into 0 and 255
    unique_vals = set(np.unique(binary))
    assert unique_vals.issubset({0, 255})


def test_ocr_text_extraction_synthetic_mode():
    """Verify synthetic text extraction when synthetic_mode is explicitly enabled."""
    ocr = MultilingualOCREngine(synthetic_mode=True)

    test_img = np.zeros((100, 200, 3), dtype=np.uint8)
    test_img[40:60, 40:160, :] = 255  # High-contrast strip

    result = ocr.extract_text(test_img)
    assert "text" in result
    assert result["text"] == "Drishti Pharmacy 500mg"
    assert "confidence" in result
    assert "script" in result
    assert result["confidence"] >= 0.0


def test_ocr_production_no_hardcoded_fallback():
    """Verify production OCR does not emit fake strings on blank or natural frames."""
    ocr = MultilingualOCREngine(synthetic_mode=False)

    test_img = np.zeros((100, 200, 3), dtype=np.uint8)
    test_img[40:60, 40:160, :] = 255

    result = ocr.extract_text(test_img)
    # Should NOT return fake "Drishti Pharmacy 500mg"
    assert result["text"] != "Drishti Pharmacy 500mg"


def test_rapid_ocr_real_text_extraction():
    """Verify neural RapidOCR extracts real printed text from an image."""
    import cv2
    ocr = MultilingualOCREngine(synthetic_mode=False)
    if ocr.rapid_ocr is None:
        pytest.skip("RapidOCR not installed")

    img = np.full((100, 350, 3), 255, dtype=np.uint8)
    cv2.putText(img, "Drishti Wearable", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    res = ocr.extract_text(img)
    assert "Drishti" in res["text"] or "Wearable" in res["text"]
    assert res["confidence"] > 0.4



