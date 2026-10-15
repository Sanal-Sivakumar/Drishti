"""Configuration Subsystem: Centralized YAML Configuration Loader & Validator.

Loads, validates, and provides structured access to pipeline and hardware
configurations for development workstations and Raspberry Pi 5 edge devices.
"""

from typing import Dict, Any, Optional, List
import os
from pathlib import Path
from dataclasses import dataclass
import yaml
from config.logging_config import get_logger

logger = get_logger("config.loader")

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent


class ConfigValidationError(ValueError):
    """Raised when configuration parameters fail schema or range validation."""
    pass


@dataclass
class PipelineConfigSchema:
    """Validated representation of master pipeline configuration."""
    target_fps: int
    frame_width: int
    frame_height: int
    headless: bool
    confidence_threshold: float
    iou_threshold: float
    target_classes: List[str]
    sample_rate_hz: int
    heartbeat_interval_s: int


def validate_pipeline_config(data: Dict[str, Any]) -> None:
    """Validate pipeline configuration structure and numerical ranges.
    
    Args:
        data: Parsed pipeline configuration dictionary.
        
    Raises:
        ConfigValidationError: If required sections or parameters are invalid.
    """
    if not isinstance(data, dict):
        raise ConfigValidationError("Pipeline configuration must be a mapping.")

    # 1. Pipeline core
    p_sec = data.get("pipeline")
    if not isinstance(p_sec, dict):
        raise ConfigValidationError("Missing required 'pipeline' configuration block.")
    
    fps = p_sec.get("target_fps", 30)
    if not isinstance(fps, (int, float)) or fps <= 0 or fps > 120:
        raise ConfigValidationError(f"Invalid pipeline.target_fps: {fps} (must be in 1..120).")

    width = p_sec.get("frame_width", 640)
    height = p_sec.get("frame_height", 480)
    if width <= 0 or height <= 0:
        raise ConfigValidationError(f"Invalid frame dimensions: {width}x{height}.")

    # 2. Perception subsystem
    perc = data.get("perception")
    if not isinstance(perc, dict):
        raise ConfigValidationError("Missing required 'perception' configuration block.")

    det = perc.get("detector", {})
    conf = det.get("confidence_threshold", 0.35)
    if not (0.0 < conf <= 1.0):
        raise ConfigValidationError(f"detector.confidence_threshold must be in (0, 1], got {conf}.")

    iou = det.get("iou_threshold", 0.45)
    if not (0.0 < iou <= 1.0):
        raise ConfigValidationError(f"detector.iou_threshold must be in (0, 1], got {iou}.")

    # 3. Spatial Audio
    audio = perc.get("spatial_audio", {})
    sr = audio.get("sample_rate_hz", 44100)
    if sr not in (16000, 22050, 44100, 48000):
        raise ConfigValidationError(f"Unsupported spatial_audio.sample_rate_hz: {sr}.")

    # 4. Cognitive subsystem
    cog = data.get("cognitive")
    if not isinstance(cog, dict):
        raise ConfigValidationError("Missing required 'cognitive' configuration block.")

    # 5. Safety subsystem
    safety = data.get("safety")
    if not isinstance(safety, dict):
        raise ConfigValidationError("Missing required 'safety' configuration block.")


def validate_hardware_config(data: Dict[str, Any]) -> None:
    """Validate hardware peripheral mapping and driver profiles.
    
    Args:
        data: Parsed hardware configuration dictionary.
        
    Raises:
        ConfigValidationError: If required peripheral sections are invalid.
    """
    if not isinstance(data, dict):
        raise ConfigValidationError("Hardware configuration must be a mapping.")

    required_peripherals = ["camera", "distance_sensor", "trigger_input", "audio_output"]
    for periph in required_peripherals:
        if periph not in data or not isinstance(data[periph], dict):
            raise ConfigValidationError(f"Missing required hardware configuration section: '{periph}'.")

    cam_backend = data["camera"].get("backend")
    valid_cam = ("synthetic", "opencv", "v4l2", "libcamera", "auto")
    if cam_backend not in valid_cam:
        raise ConfigValidationError(f"Invalid camera backend: '{cam_backend}' (expected one of {valid_cam}).")

    dist_backend = data["distance_sensor"].get("backend")
    valid_dist = ("synthetic", "vl53l1x")
    if dist_backend not in valid_dist:
        raise ConfigValidationError(f"Invalid distance_sensor backend: '{dist_backend}' (expected one of {valid_dist}).")


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file safely.
    
    Args:
        file_path: Path to the target YAML file.
        
    Returns:
        Dict[str, Any]: Parsed configuration mapping.
        
    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def get_pipeline_config(config_path: Optional[str] = None, validate: bool = True) -> Dict[str, Any]:
    """Retrieve and validate master pipeline configuration.
    
    Args:
        config_path: Optional custom path to pipeline YAML configuration.
        validate: If True, executes schema and range validation.
        
    Returns:
        Dict[str, Any]: Complete pipeline parameters.
    """
    path = Path(config_path) if config_path else CONFIG_DIR / "pipeline_config.yaml"
    data = load_yaml(path)
    if validate:
        validate_pipeline_config(data)
    return data


def get_hardware_config(
    profile: str = "auto", config_path: Optional[str] = None, validate: bool = True
) -> Dict[str, Any]:
    """Retrieve and validate hardware configuration for workstation or Raspberry Pi 5.
    
    Args:
        profile: "dev", "pi5", or "auto" (detects host architecture).
        config_path: Optional custom path to hardware YAML configuration.
        validate: If True, executes schema validation.
        
    Returns:
        Dict[str, Any]: Hardware peripheral and pinout configurations.
    """
    if config_path:
        target_path = Path(config_path)
    else:
        if profile == "auto":
            is_pi = False
            try:
                with open("/proc/cpuinfo", "r") as f:
                    content = f.read()
                    is_pi = ("Raspberry Pi" in content) or ("BCM" in content)
            except Exception:
                is_pi = False
            profile = "pi5" if is_pi else "dev"

        if profile in ("pi", "pi5", "rpi"):
            target_path = CONFIG_DIR / "hardware_pi5.yaml"
        else:
            target_path = CONFIG_DIR / "hardware_dev.yaml"

    data = load_yaml(target_path)
    if validate:
        validate_hardware_config(data)
    return data
