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
    # BGR format: B=230, G=170, R=60 -> Hue ~ 201 deg
    blue_note = np.zeros((66, 135, 3), dtype=np.uint8)
    blue_note[:, :, 0] = 230
    blue_note[:, :, 1] = 170
    blue_note[:, :, 2] = 60
    blue_note[::2, ::2, :] += 20
    res_50 = rec.classify_banknote(blue_note, bbox_aspect_ratio=2.04)
    assert res_50["denomination"] == 50
    assert "Fifty" in res_50["spoken_name"]
    assert res_50["confidence"] >= 0.50

    # 2. Ten Rupees (Chocolate Brown, Hue ~ 18 deg)
    # BGR format: B=30, G=60, R=130 -> Hue ~ 18 deg
    brown_note = np.zeros((63, 123, 3), dtype=np.uint8)
    brown_note[:, :, 0] = 30
    brown_note[:, :, 1] = 60
    brown_note[:, :, 2] = 130
    brown_note[::2, ::2, :] += 20
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


def test_bgr_input_order():
    """Verify BGR camera frames (blue in channel 0) map correctly to ₹50."""
    rec = IndianCurrencyRecognizer(confidence_threshold=0.50)
    # BGR format from camera: B=230 (ch 0), G=170 (ch 1), R=60 (ch 2)
    bgr_note = np.zeros((66, 135, 3), dtype=np.uint8)
    bgr_note[:, :, 0] = 230  # Blue
    bgr_note[:, :, 1] = 170  # Green
    bgr_note[:, :, 2] = 60   # Red
    # Add texture/gradients
    bgr_note[::2, ::2, :] += 25

    # Test bgr_to_hsv_fast
    hsv = rec.bgr_to_hsv_fast(bgr_note, color_order="BGR")
    mean_hue = float(np.mean(hsv[:, :, 0]))
    assert 180.0 <= mean_hue <= 215.0  # Fluorescent blue hue band

    res = rec.classify_banknote(bgr_note, bbox_aspect_ratio=2.04)
    assert res["denomination"] == 50
    assert "Fifty" in res["spoken_name"]


def test_twenty_vs_two_hundred_disambiguation():
    """Verify clear distinction between ₹20 (greenish-yellow) and ₹200 (bright yellow)."""
    rec = IndianCurrencyRecognizer(confidence_threshold=0.50)

    # ₹20 note: hue ~ 41 deg, aspect ratio 2.05
    # BGR format: B=50, G=160, R=210 -> hue = 41.25 deg
    note_20 = np.zeros((63, 129, 3), dtype=np.uint8)
    note_20[:, :, 0] = 50
    note_20[:, :, 1] = 160
    note_20[:, :, 2] = 210
    note_20[::2, ::2, :] += 20
    res_20 = rec.classify_banknote(note_20, bbox_aspect_ratio=2.05)
    assert res_20["denomination"] == 20

    # ₹200 note: hue ~ 55 deg, aspect ratio 2.21
    # BGR format: B=30, G=210, R=230 -> hue = 54.0 deg
    note_200 = np.zeros((66, 146, 3), dtype=np.uint8)
    note_200[:, :, 0] = 30
    note_200[:, :, 1] = 210
    note_200[:, :, 2] = 230
    note_200[::2, ::2, :] = np.clip(note_200[::2, ::2, :].astype(int) + 15, 0, 255).astype(np.uint8)
    res_200 = rec.classify_banknote(note_200, bbox_aspect_ratio=2.21)
    assert res_200["denomination"] == 200


def test_identify_currency_contour_segmentation():
    """Verify identify_currency extracts banknote contour from room frame and classifies."""
    rec = IndianCurrencyRecognizer(confidence_threshold=0.55)

    # 640x480 room frame with user holding ₹50 note
    frame = np.full((480, 640, 3), [120, 140, 160], dtype=np.uint8)
    # Note placed in center: 80x160 pixels (aspect ratio 2.0)
    frame[180:260, 240:400, :] = [230, 170, 60]  # BGR ₹50 note
    frame[180:260:2, 240:400:2, :] += 25

    res = rec.identify_currency(frame, color_order="BGR")
    assert res["denomination"] == 50
    assert "Fifty" in res["spoken_name"]
    assert res["confidence"] >= 0.55
    assert "bbox" in res
    bx, by, bw, bh = res["bbox"]
    assert 230 <= bx <= 250
    assert 170 <= by <= 190


def test_identify_currency_five_hundred_rupees():
    """Verify ₹500 stone-grey/greenish note recognition."""
    rec = IndianCurrencyRecognizer(confidence_threshold=0.50)
    note_500 = np.zeros((66, 145, 3), dtype=np.uint8)
    note_500[:, :] = [100, 160, 130]  # BGR
    note_500[::2, ::2, :] = np.clip(note_500[::2, ::2, :].astype(int) + 20, 0, 255).astype(np.uint8)

    res = rec.classify_banknote(note_500, bbox_aspect_ratio=2.20, color_order="BGR")
    assert res["denomination"] == 500
    assert "Five hundred" in res["spoken_name"]




