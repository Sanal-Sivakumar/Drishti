"""Unit tests for Indian Rupee banknote recognition."""

import pytest
import numpy as np
from perception.currency import IndianCurrencyRecognizer


def test_fast_rgb_to_hsv():
    """Verify vectorized RGB to HSV conversion on primary colors."""
    rec = IndianCurrencyRecognizer()

    # Pure Red: (255, 0, 0) -> Hue = 0, Sat = 1, Val = 1
    red_img = np.zeros((10, 10, 3), dtype=np.uint8)
    red_img[:, :, 0] = 255
    hsv_red = rec.rgb_to_hsv_fast(red_img)
    assert np.isclose(hsv_red[0, 0, 0], 0.0, atol=1.0)
    assert np.isclose(hsv_red[0, 0, 1], 1.0, atol=1e-2)
    assert np.isclose(hsv_red[0, 0, 2], 1.0, atol=1e-2)

    # Pure Green: (0, 255, 0) -> Hue = 120
    green_img = np.zeros((10, 10, 3), dtype=np.uint8)
    green_img[:, :, 1] = 255
    hsv_green = rec.rgb_to_hsv_fast(green_img)
    assert np.isclose(hsv_green[0, 0, 0], 120.0, atol=1.0)

    # Pure Blue: (0, 0, 255) -> Hue = 240
    blue_img = np.zeros((10, 10, 3), dtype=np.uint8)
    blue_img[:, :, 2] = 255
    hsv_blue = rec.rgb_to_hsv_fast(blue_img)
    assert np.isclose(hsv_blue[0, 0, 0], 240.0, atol=1.0)


def test_banknote_classification_profiles():
    """Verify banknote classifier maps color profiles to correct INR denominations."""
    rec = IndianCurrencyRecognizer(confidence_threshold=0.50)

    # 1. Fifty Rupees (Fluorescent Blue, Hue ~ 195)
    # RGB with high cyan/blue: R=60, G=170, B=230 -> Hue ~ 201 deg
    blue_note = np.zeros((66, 135, 3), dtype=np.uint8)
    blue_note[:, :, 0] = 60
    blue_note[:, :, 1] = 170
    blue_note[:, :, 2] = 230
    res_50 = rec.classify_banknote(blue_note, bbox_aspect_ratio=2.04)
    assert res_50["denomination"] == 50
    assert "Fifty" in res_50["spoken_name"]
    assert res_50["confidence"] >= 0.50

    # 2. Ten Rupees (Chocolate Brown, Hue ~ 18 deg)
    # RGB: R=130, G=60, B=30 -> Hue ~ 18 deg
    brown_note = np.zeros((63, 123, 3), dtype=np.uint8)
    brown_note[:, :, 0] = 130
    brown_note[:, :, 1] = 60
    brown_note[:, :, 2] = 30
    res_10 = rec.classify_banknote(brown_note, bbox_aspect_ratio=1.95)
    assert res_10["denomination"] == 10
    assert "Ten" in res_10["spoken_name"]


def test_reject_user_sitting_in_room():
    """Verify that a smooth person/room frame is rejected with denomination None."""
    rec = IndianCurrencyRecognizer(confidence_threshold=0.70)
    # Bluish room frame (user wearing blue shirt in front of camera)
    room_frame = np.full((480, 640, 3), [130, 160, 205], dtype=np.uint8)
    res = rec.classify_banknote(room_frame)
    assert res["denomination"] is None
    assert "No note" in res["spoken_name"]
    assert res["confidence"] == 0.0


