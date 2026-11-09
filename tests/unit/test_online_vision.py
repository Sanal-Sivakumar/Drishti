"""Unit tests for the Online Vision Cognitive Service."""

import pytest
import numpy as np
from cognitive.online_vision import OnlineVisionService, VALID_INR_DENOMINATIONS


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


def test_online_vision_currency_strict_denomination_whitelist(monkeypatch):
    """Verify that only valid INR denominations in {10, 20, 50, 100, 200, 500} are accepted."""
    service = OnlineVisionService(synthetic_mode=False, api_key="sk-test-mock-key")
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    # 1. Valid denomination (e.g., 200)
    monkeypatch.setattr(
        service,
        "_call_cloud_api",
        lambda p, b, high_fidelity=False, **kwargs: {
            "success": True,
            "data": {"denomination_inr": 200, "confidence": "high", "uncertain": False},
        },
    )
    res_200 = service.identify_currency(img)
    assert res_200["currency"]["denomination"] == 200
    assert "Two hundred rupees" in res_200["currency"]["spoken_name"]

    # 2. Invalid denomination (e.g. 75, 2000, 0, or non-circulating)
    for bad_denom in [75, 2000, 0, -10, 100.9, "unknown"]:
        monkeypatch.setattr(
            service,
            "_call_cloud_api",
            lambda p, b, high_fidelity=False, d=bad_denom, **kwargs: {
                "success": True,
                "data": {"denomination_inr": d, "confidence": "high", "uncertain": False},
            },
        )
        res_bad = service.identify_currency(img)
        assert res_bad["currency"]["denomination"] is None
        assert "No banknote clearly detected" in res_bad["currency"]["spoken_name"]


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


def test_provider_auto_resolution_gemini_only(monkeypatch):
    """Verify provider 'auto' resolves to Gemini when only GEMINI_API_KEY is present."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyFakeGeminiKey12345")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DRISHTI_VISION_PROVIDER", raising=False)

    service = OnlineVisionService(provider="auto", synthetic_mode=True, load_env=False)
    assert service.provider == "gemini"
    assert service.model == "gemini-3.1-flash-lite"
    assert service.has_key is True


def test_provider_auto_resolution_openai_only(monkeypatch):
    """Verify provider 'auto' resolves to OpenAI when only OPENAI_API_KEY is present."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-FakeOpenAIKey12345")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DRISHTI_VISION_PROVIDER", raising=False)

    service = OnlineVisionService(provider="auto", synthetic_mode=True, load_env=False)
    assert service.provider == "openai"
    assert service.model == "gpt-4o-mini"
    assert service.has_key is True


def test_yaml_model_override_honored(monkeypatch):
    """Verify that models passed from YAML configuration are honored."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyFakeKey123")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    service = OnlineVisionService(
        provider="auto",
        gemini_model="gemini-custom-flash",
        openai_model="gpt-4o-custom",
        synthetic_mode=True,
        load_env=False,
    )
    assert service.provider == "gemini"
    assert service.model == "gemini-custom-flash"


def test_error_sanitization_removes_secrets():
    """Verify that sanitize_error scrubs sensitive API keys and query strings."""
    secret_key = "AIzaSySecretTestingKey999"
    err = f"403 Client Error: Forbidden for url: https://generativelanguage.googleapis.com/v1beta/models?key={secret_key}"

    sanitized = OnlineVisionService.sanitize_error(err, secret=secret_key)
    assert secret_key not in sanitized
    assert "[REDACTED" in sanitized
