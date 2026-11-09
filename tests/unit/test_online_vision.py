"""Unit tests for the Online Vision Cognitive Service."""

import pytest
import numpy as np
from cognitive.online_vision import OnlineVisionService


def test_online_vision_synthetic_mode():
    """Verify deterministic behavior in synthetic mode."""
    service = OnlineVisionService(synthetic_mode=True)
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    narration = service.describe_scene(img, [{"class_name": "chair", "distance_m": 1.2}])
    assert narration["success"] is True
    assert "speech" in narration["structured"]
    assert narration["structured"]["uncertain"] is False

    currency = service.identify_currency(img)
    assert currency["success"] is True
    assert currency["currency"]["denomination"] == 100

    text = service.read_text(img)
    assert text["success"] is True
    assert "Drishti" in text["text"]


def test_online_vision_missing_key_failure():
    """Verify graceful offline failure message when no API key is configured."""
    service = OnlineVisionService(synthetic_mode=False, api_key="")
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    res = service.describe_scene(img, [])
    assert res["success"] is False
    assert "API key not configured" in res["narration"]
    assert "Obstacle guidance remains active" in res["narration"]

    curr_res = service.identify_currency(img)
    assert curr_res["success"] is False
    assert "Internet unavailable" in curr_res["spoken_name"]

    text_res = service.read_text(img)
    assert text_res["success"] is False
    assert "Internet unavailable" in text_res["spoken_name"]


def test_online_vision_jpeg_encoding():
    """Verify JPEG base64 encoding scales large frames down to max_edge."""
    large_img = np.zeros((2000, 1500, 3), dtype=np.uint8)
    b64_str = OnlineVisionService.encode_image_jpeg(large_img, max_edge=800)
    assert isinstance(b64_str, str)
    assert len(b64_str) > 100


def test_online_vision_forbidden_navigation_sanitization(monkeypatch):
    """Verify that any movement or navigation advice returned by cloud vision is sanitized."""
    service = OnlineVisionService(synthetic_mode=False, api_key="sk-test-mock-key")

    # Mock _call_cloud_api to simulate a hallucinating cloud model emitting movement advice
    def mock_call(prompt, b64_img, high_fidelity=False):
        return {
            "success": True,
            "data": {
                "speech": "There is a table on your right. Walk forward and turn left.",
                "visual_context": ["table"],
                "uncertain": False,
            },
            "latency_ms": 250.0,
            "provider": "openai",
        }

    monkeypatch.setattr(service, "_call_cloud_api", mock_call)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = service.describe_scene(img, [])

    assert res["success"] is True
    # The phrases 'walk forward' and 'turn left' must be excised
    assert "walk forward" not in res["narration"].lower()
    assert "turn left" not in res["narration"].lower()
    assert "[navigation advice omitted]" in res["narration"]


def test_online_vision_currency_ambiguity_rejection(monkeypatch):
    """Verify that ambiguous or blurred images do not yield false positive banknotes."""
    service = OnlineVisionService(synthetic_mode=False, api_key="sk-test-mock-key")

    def mock_call(prompt, b64_img, high_fidelity=False):
        return {
            "success": True,
            "data": {
                "denomination_inr": None,
                "confidence": "low",
                "uncertain": True,
                "reason": "Image is blurry with no distinct note",
            },
            "latency_ms": 180.0,
            "provider": "openai",
        }

    monkeypatch.setattr(service, "_call_cloud_api", mock_call)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = service.identify_currency(img)

    assert res["success"] is True
    assert res["currency"]["denomination"] is None
    assert "No banknote clearly detected" in res["currency"]["spoken_name"]


def test_online_vision_multilingual_indic_ocr(monkeypatch):
    """Verify literal transcription across Indic languages."""
    service = OnlineVisionService(synthetic_mode=False, api_key="sk-test-mock-key")

    def mock_call(prompt, b64_img, high_fidelity=False):
        return {
            "success": True,
            "data": {
                "text": "दवा की दुकान 500mg",
                "script": "Devanagari",
                "detected_language": "Hindi",
                "uncertain": False,
            },
            "latency_ms": 320.0,
            "provider": "openai",
        }

    monkeypatch.setattr(service, "_call_cloud_api", mock_call)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = service.read_text(img)

    assert res["success"] is True
    assert "दवा की दुकान" in res["text"]
    assert res["script"] == "Devanagari"
    assert res["detected_language"] == "Hindi"
