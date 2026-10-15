"""Configuration Subsystem: Centralized YAML Configuration Loader.

Loads, validates, and provides structured access to pipeline and hardware
configurations for development workstations and Raspberry Pi 5 edge devices.
"""

from typing import Dict, Any, Optional
import os
from pathlib import Path
import yaml


CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent


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


def get_pipeline_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve master pipeline configuration.
    
    Args:
        config_path: Optional custom path to pipeline YAML configuration.
        
    Returns:
        Dict[str, Any]: Complete pipeline parameters.
    """
    path = Path(config_path) if config_path else CONFIG_DIR / "pipeline_config.yaml"
    return load_yaml(path)


def get_hardware_config(profile: str = "auto", config_path: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve hardware configuration for workstation or Raspberry Pi 5.
    
    Args:
        profile: "dev", "pi5", or "auto" (detects host architecture).
        config_path: Optional custom path to hardware YAML configuration.
        
    Returns:
        Dict[str, Any]: Hardware peripheral and pinout configurations.
    """
    if config_path:
        return load_yaml(Path(config_path))

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

    return load_yaml(target_path)

