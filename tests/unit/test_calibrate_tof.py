"""Unit tests for the ToF zero-offset calibration tool."""

from pathlib import Path
import pytest
import yaml

from scripts.calibrate_tof import compute_calibration_stats, update_hardware_config_offset


def test_compute_calibration_stats_positive_offset():
    # Ground truth 500mm, mean raw reading 482mm -> offset should be +18mm
    samples = [480, 482, 484, 481, 483]
    stats = compute_calibration_stats(samples, ground_truth_mm=500)

    assert stats["ground_truth_mm"] == 500.0
    assert stats["sample_count"] == 5.0
    assert stats["mean_raw_mm"] == pytest.approx(482.0, abs=0.1)
    assert stats["recommended_offset_mm"] == 18.0


def test_compute_calibration_stats_negative_offset():
    # Ground truth 500mm, mean raw reading 515mm -> offset should be -15mm
    samples = [514, 516, 515, 515]
    stats = compute_calibration_stats(samples, ground_truth_mm=500)

    assert stats["mean_raw_mm"] == pytest.approx(515.0, abs=0.1)
    assert stats["recommended_offset_mm"] == -15.0


def test_compute_calibration_stats_empty_raises():
    with pytest.raises(ValueError):
        compute_calibration_stats([], ground_truth_mm=500)


def test_update_hardware_config_offset_tmp(tmp_path: Path):
    cfg_file = tmp_path / "test_hardware.yaml"
    initial_data = {
        "camera": {"backend": "auto"},
        "distance_sensor": {"backend": "esp32_serial", "offset_mm": 0},
    }
    with open(cfg_file, "w") as f:
        yaml.dump(initial_data, f)

    assert update_hardware_config_offset(cfg_file, offset_mm=22) is True

    with open(cfg_file, "r") as f:
        updated = yaml.safe_load(f)

    assert updated["distance_sensor"]["offset_mm"] == 22
