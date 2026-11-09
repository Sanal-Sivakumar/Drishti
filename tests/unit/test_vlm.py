"""Grounded real-VLM boundary tests using an injected deterministic backend."""

import json
import numpy as np

from cognitive.vlm_narrator import GroundedSmolVLMNarrator


def _image():
    return np.zeros((64, 64, 3), dtype=np.uint8)


def _chair():
    return [{"class_name": "chair", "distance_m": 1.3, "azimuth_deg": 28.0,
             "fusion_source": "tof_laser_fusion", "confirmation_frames": 2}]


def test_grounded_vlm_executes_backend_and_accepts_sensor_consistent_json():
    prompts = []

    def infer(image, prompt):
        prompts.append(prompt)
        return json.dumps({"path_clear": False, "primary_hazard": "chair",
                           "direction": "right", "distance_m": 1.3,
                           "speech": "A chair is 1.3 metres to your right."})

    narrator = GroundedSmolVLMNarrator(inference_fn=infer, temperature=0.9)
    result = narrator.describe_scene(_image(), _chair(), True, "obstructed")
    assert result["success"] is True
    assert result["model_output_accepted"] is True
    assert "TRUSTED_SENSOR_FACTS" in prompts[0]
    assert narrator.temperature == 0.0


def test_grounded_vlm_rejects_invented_distance():
    def infer(image, prompt):
        return json.dumps({"path_clear": False, "primary_hazard": "chair",
                           "direction": "right", "distance_m": 2.8,
                           "speech": "A chair is 2.8 metres to your right."})

    result = GroundedSmolVLMNarrator(inference_fn=infer).describe_scene(
        _image(), _chair(), True, "obstructed"
    )
    assert result["success"] is False
    assert "distance" in result["error"]


def test_grounded_vlm_rejects_clear_path_contradiction():
    def infer(image, prompt):
        return json.dumps({"path_clear": True, "primary_hazard": "stairs",
                           "direction": "ahead", "distance_m": 1.3,
                           "speech": "The path is clear beyond stairs."})

    result = GroundedSmolVLMNarrator(inference_fn=infer).describe_scene(
        _image(), _chair(), True, "obstructed"
    )
    assert result["success"] is False


def test_grounded_vlm_does_not_treat_geometric_prior_as_tof():
    obstacle = [{"class_name": "chair", "distance_m": 1.3, "azimuth_deg": 0.0,
                 "fusion_source": "geometric_perspective_prior", "confirmation_frames": 2}]

    def infer(image, prompt):
        return json.dumps({"path_clear": False, "primary_hazard": "chair",
                           "direction": "ahead", "distance_m": None,
                           "speech": "A chair is directly ahead."})

    result = GroundedSmolVLMNarrator(inference_fn=infer).describe_scene(
        _image(), obstacle, True, "obstructed"
    )
    assert result["success"] is True
    assert result["trusted_facts"][0]["distance_m"] is None


def test_vlm_empty_image_handling():
    result = GroundedSmolVLMNarrator(inference_fn=lambda image, prompt: "{}").describe_scene(None)
    assert result["success"] is False


def test_grounded_vlm_fallback_when_torch_unavailable():
    narrator = GroundedSmolVLMNarrator(inference_fn=None)
    result = narrator.describe_scene(_image(), _chair(), False, "obstructed")
    assert result["success"] is True
    assert "chair" in result["narration"].lower()
    assert "1.3" in result["narration"]
