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


def test_unverified_calibration_record_forces_calibration_verified_false():
    """Verify that unverified geometry calibration forces calibration_verified to False."""
    cfg = get_pipeline_config(validate=True)
    fusion = cfg["perception"]["tof_fusion"]
    # Even if pipeline_config specifies calibration_verified: true in other contexts,
    # the unverified geometry_calibration.yaml record must force it to False.
    assert fusion["calibration_verified"] is False


def test_invalid_online_vision_config_raises_validation_error():
    """Verify invalid online_vision settings raise ConfigValidationError."""
    base_cfg = {
        "pipeline": {"target_fps": 30, "frame_width": 640, "frame_height": 480},
        "perception": {"detector": {"confidence_threshold": 0.4}},
        "safety": {"telemetry": {"buffer_limit": 500}},
    }

    # Invalid provider
    with pytest.raises(ConfigValidationError, match="Invalid cognitive.online_vision.provider"):
        validate_pipeline_config({
            **base_cfg,
            "cognitive": {"online_vision": {"provider": "unsupported_provider"}},
        })

    # Empty model string
    with pytest.raises(ConfigValidationError, match="openai_model must be a non-empty string"):
        validate_pipeline_config({
            **base_cfg,
            "cognitive": {"online_vision": {"provider": "openai", "openai_model": "   "}},
        })

    # Invalid timeout
    with pytest.raises(ConfigValidationError, match="timeout_s must be a positive number"):
        validate_pipeline_config({
            **base_cfg,
            "cognitive": {"online_vision": {"provider": "openai", "timeout_s": -5.0}},
        })
