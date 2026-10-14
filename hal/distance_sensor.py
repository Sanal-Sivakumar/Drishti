"""Hardware Abstraction Layer: Distance & Time-of-Flight (ToF) Subsystem.

Provides unified interface for STMicroelectronics VL53L1X I2C sensors
and headless synthetic range simulators for testing.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class BaseDistanceSensor(ABC):
    """Abstract base class for distance and depth sensors."""

    @abstractmethod
    def start_ranging(self) -> None:
        """Initialize and begin active sensor ranging."""
        pass

    @abstractmethod
    def stop_ranging(self) -> None:
        """Halt active sensor ranging to conserve power."""
        pass

    @abstractmethod
    def get_distance_meters(self) -> Optional[float]:
        """Query the current measured metric distance in meters.
        
        Returns:
            Optional[float]: Distance in meters (e.g., 1.42), or None if out of range/error.
        """
        pass

    @abstractmethod
    def get_zone_distances(self) -> Dict[str, Optional[float]]:
        """Query multi-zone distance readings across the sensor FoV.
        
        Returns:
            Dict[str, Optional[float]]: Mapping of zone name ('left', 'center', 'right') to metric distance.
        """
        pass


class SyntheticDistanceSensor(BaseDistanceSensor):
    """Deterministic, headless range simulator for testing without I2C hardware."""

    def __init__(
        self,
        base_distance_m: float = 1.50,
        noise_std_m: float = 0.02,
        min_limit_m: float = 0.05,
        max_limit_m: float = 4.00,
    ) -> None:
        self.base_distance = base_distance_m
        self.noise_std = noise_std_m
        self.min_limit = min_limit_m
        self.max_limit = max_limit_m
        self.is_ranging = False
        self.step_count = 0

    def start_ranging(self) -> None:
        self.is_ranging = True

    def stop_ranging(self) -> None:
        self.is_ranging = False

    def get_distance_meters(self) -> Optional[float]:
        if not self.is_ranging:
            return None

        # Simulate natural distance variation (approaching and receding target)
        variation = 0.5 * np.sin(2 * np.pi * 0.1 * self.step_count)
        noise = float(np.random.normal(0, self.noise_std))
        dist = float(self.base_distance + variation + noise)
        self.step_count += 1

        if dist < self.min_limit or dist > self.max_limit:
            return None
        return round(dist, 3)

    def get_zone_distances(self) -> Dict[str, Optional[float]]:
        center_dist = self.get_distance_meters()
        if center_dist is None:
            return {"left": None, "center": None, "right": None}

        # Simulate slight spatial depth variation across zones
        left_dist = round(center_dist + 0.15, 3) if (center_dist + 0.15) <= self.max_limit else None
        right_dist = round(center_dist - 0.10, 3) if (center_dist - 0.10) >= self.min_limit else None
        return {"left": left_dist, "center": center_dist, "right": right_dist}


class VL53L1XSensor(BaseDistanceSensor):
    """Driver for physical VL53L1X Time-of-Flight sensor over I2C."""

    def __init__(self, i2c_bus: int = 1, i2c_address: int = 0x29) -> None:
        self.i2c_bus = i2c_bus
        self.i2c_address = i2c_address
        self.sensor = None
        self.is_ranging = False

        try:
            import VL53L1X
            self.sensor = VL53L1X.VL53L1X(i2c_bus=i2c_bus, i2c_address=i2c_address)
            self.sensor.open()
        except (ImportError, Exception):
            self.sensor = None

    def start_ranging(self) -> None:
        if self.sensor is not None:
            try:
                self.sensor.start_ranging(1)  # 1 = Short mode, 2 = Medium, 3 = Long
                self.is_ranging = True
            except Exception:
                self.is_ranging = False

    def stop_ranging(self) -> None:
        if self.sensor is not None and self.is_ranging:
            try:
                self.sensor.stop_ranging()
            except Exception:
                pass
            self.is_ranging = False

    def get_distance_meters(self) -> Optional[float]:
        if self.sensor is None or not self.is_ranging:
            return None
        try:
            distance_mm = self.sensor.get_distance()
            if distance_mm > 0:
                return round(distance_mm / 1000.0, 3)
            return None
        except Exception:
            return None

    def get_zone_distances(self) -> Dict[str, Optional[float]]:
        # Single-zone fallback when multi-zone SPAD configuration is not configured
        dist = self.get_distance_meters()
        return {"left": dist, "center": dist, "right": dist}

