"""Unit tests for the Vision-Language Model (VLM) Scene Narrator."""

import pytest
import numpy as np
from cognitive.vlm_narrator import VisionLanguageNarrator


def test_vlm_scene_narration_with_obstacles():
    """Verify VLM generates concise 2-sentence navigational summary referencing obstacles."""
    vlm = VisionLanguageNarrator()
    dummy_img = np.zeros((384, 384, 3), dtype=np.uint8)

    obstacles = [{
        "class_name": "door",
        "distance_m": 2.5,
        "azimuth_deg": 15.0,
    }]

    res = vlm.describe_scene(dummy_img, detected_obstacles=obstacles)
    assert "narration" in res
    text = res["narration"]
    assert "door" in text
    assert "2.5 meters" in text
    assert "right" in text  # Azimuth +15 deg -> to your right

    # Verify brevity: maximum 2 sentences (using regex to ignore decimal numbers like 2.5)
    import re
    sentences = [s.strip() for s in re.split(r'\.\s+', text.rstrip('.')) if s.strip()]
    assert len(sentences) <= 2


def test_vlm_scene_narration_clear_path():
    """Verify VLM generates reassuring navigational summary when no hazards exist."""
    vlm = VisionLanguageNarrator()
    dummy_img = np.zeros((384, 384, 3), dtype=np.uint8)

    res = vlm.describe_scene(dummy_img, detected_obstacles=[])
    text = res["narration"]
    assert "clear" in text.lower()


def test_vlm_empty_image_handling():
    """Verify graceful handling when frame capture fails."""
    vlm = VisionLanguageNarrator()
    res = vlm.describe_scene(None)
    assert "Unable to capture" in res["narration"]
