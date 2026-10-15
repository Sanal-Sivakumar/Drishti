"""Unit tests for Centralized YAML Configuration Loader and Schema Validation."""

import pytest
from pathlib import Path
from config.config_loader import (
    get_pipeline_config,
    get_hardware_config,
    validate_pipeline_config,
    validate_hardware_config,
    ConfigValidationError,
)


def test_load_valid_pipeline_config():
    """Verify default pipeline_config.yaml passes schema validation."""
    cfg = get_pipeline_config(validate=True)
    assert "pipeline" in cfg
    assert "perception" in cfg
    assert "cognitive" in cfg
    assert "safety" in cfg
    assert cfg["pipeline"]["frame_width"] == 640
    assert cfg["pipeline"]["frame_height"] == 480


def test_load_valid_hardware_configs():
    """Verify default hardware_dev.yaml and hardware_pi5.yaml pass validation."""
    dev_cfg = get_hardware_config(profile="dev", validate=True)
    assert dev_cfg["camera"]["backend"] in ("synthetic", "opencv", "v4l2", "auto")

    pi_cfg = get_hardware_config(profile="pi5", validate=True)
    assert pi_cfg["camera"]["backend"] in ("synthetic", "opencv", "v4l2", "libcamera", "auto")


def test_invalid_pipeline_config_raises_validation_error():
    """Verify malformed pipeline configs raise ConfigValidationError with clear description."""
    # Invalid target FPS
    with pytest.raises(ConfigValidationError, match="target_fps"):
        validate_pipeline_config({
            "pipeline": {"target_fps": -5, "frame_width": 640, "frame_height": 480},
            "perception": {"detector": {"confidence_threshold": 0.4}},
            "cognitive": {},
            "safety": {},
        })

    # Out of range confidence threshold
    with pytest.raises(ConfigValidationError, match="confidence_threshold"):
        validate_pipeline_config({
            "pipeline": {"target_fps": 30, "frame_width": 640, "frame_height": 480},
            "perception": {"detector": {"confidence_threshold": 1.5}},
            "cognitive": {},
            "safety": {},
        })


def test_invalid_hardware_config_raises_validation_error():
    """Verify malformed hardware configs raise ConfigValidationError."""
    # Invalid camera backend
    with pytest.raises(ConfigValidationError, match="Invalid camera backend"):
        validate_hardware_config({
            "camera": {"backend": "unsupported_backend_xyz"},
            "distance_sensor": {"backend": "synthetic"},
            "trigger_input": {"backend": "synthetic"},
            "audio_output": {"backend": "headless"},
        })
