"""Hardware Abstraction Layer: Distance & Time-of-Flight (ToF) Subsystem.

Provides unified interface for STMicroelectronics VL53L1X I2C sensors
and headless synthetic range simulators for testing.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

from config.logging_config import get_logger
from hal.esp32_link import Esp32Link, LinkStatus

logger = get_logger("hal.distance")


class BaseDistanceSensor(ABC):
    """Abstract base class for distance and depth sensors."""

    @abstractmethod
    def start_ranging(self) -> None:
        """Initialize and begin active sensor ranging."""
        pass

    @property
    def is_synthetic(self) -> bool:
        return False

    @property
    def is_available(self) -> bool:
        return True

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

    @property
    def is_synthetic(self) -> bool:
        return True

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

    def __init__(self, i2c_bus: int = 1, i2c_address: int = 0x29, **kwargs) -> None:
        self.i2c_bus = i2c_bus
        self.i2c_address = kwargs.get("address", i2c_address)
        self.sensor = None
        self.is_ranging = False

        try:
            import VL53L1X
            self.sensor = VL53L1X.VL53L1X(i2c_bus=self.i2c_bus, i2c_address=self.i2c_address)
            self.sensor.open()
        except ImportError:
            logger.warning("VL53L1X driver not installed on host. Run 'pip install VL53L1X' on target.")
            self.sensor = None
        except Exception as e:
            logger.warning("Could not connect to I2C bus %d at 0x%02x: %s", self.i2c_bus, self.i2c_address, e)
            self.sensor = None

    def start_ranging(self) -> None:
        if self.sensor is not None:
            try:
                self.sensor.start_ranging(1)  # 1 = Short mode, 2 = Medium, 3 = Long
                self.is_ranging = True
            except Exception as e:
                logger.error("Error starting VL53L1X ranging: %s", e)
                self.is_ranging = False

    @property
    def is_available(self) -> bool:
        return self.sensor is not None

    def stop_ranging(self) -> None:
        if self.sensor is not None and self.is_ranging:
            try:
                self.sensor.stop_ranging()
            except Exception as e:
                logger.debug("Exception stopping VL53L1X ranging: %s", e)
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


class ESP32DistanceSensor(BaseDistanceSensor):
    """VL53L0X range finder reached through an ESP32 USB-serial bridge.

    The ESP32 owns the I2C timing loop and streams filtered millimetre
    samples. The host only consumes the latest immutable sample, so the
    perception loop can never block on serial I/O.

    A single VL53L0X is a **single-ray** sensor: only the centre of the
    scene has a true laser distance. ``get_zone_distances`` therefore
    reports ``None`` for left/right instead of pretending to have three
    zones.
    """

    def __init__(
        self,
        port: str = "auto",
        baud: int = 921600,
        stale_ms: float = 120.0,
        min_valid_mm: int = 30,
        max_valid_mm: int = 2000,
        offset_mm: int = 0,
        link: Optional[Esp32Link] = None,
    ) -> None:
        self.link = link or Esp32Link(port=port, baud=baud, stale_ms=stale_ms)
        self.stale_ms = float(stale_ms)
        self.min_valid_mm = int(min_valid_mm)
        self.max_valid_mm = int(max_valid_mm)
        self.offset_mm = int(offset_mm)
        self.is_ranging = False

    def start_ranging(self) -> None:
        self.link.start()
        self.is_ranging = True

    def stop_ranging(self) -> None:
        try:
            self.link.stop()
        finally:
            self.is_ranging = False

    def get_distance_meters(self) -> Optional[float]:
        if not self.is_ranging:
            return None

        sample = self.link.latest()
        if sample is None:
            return None
        if self.link.age_ms() > self.stale_ms:
            return None
        if sample.status != 0:
            return None

        millimetres = sample.mm_filtered + self.offset_mm
        if millimetres < self.min_valid_mm or millimetres > self.max_valid_mm:
            return None
        return round(millimetres / 1000.0, 3)

    def get_zone_distances(self) -> Dict[str, Optional[float]]:
        """Single-ray sensor: only the centre zone is physically measured."""
        return {"left": None, "center": self.get_distance_meters(), "right": None}

    def get_link_status(self) -> LinkStatus:
        """Expose link health for diagnostics and the debug viewer."""
        return self.link.status()

    @property
    def is_available(self) -> bool:
        return self.link.latest() is not None and self.link.age_ms() <= self.stale_ms


class UnavailableDistanceSensor(BaseDistanceSensor):
    """Explicit fail-closed range source used when physical ranging is absent."""

    def __init__(self, reason: str = "distance sensor unavailable") -> None:
        self.reason = reason
        self.is_ranging = False

    @property
    def is_available(self) -> bool:
        return False

    def start_ranging(self) -> None:
        self.is_ranging = True

    def stop_ranging(self) -> None:
        self.is_ranging = False

    def get_distance_meters(self) -> Optional[float]:
        return None

    def get_zone_distances(self) -> Dict[str, Optional[float]]:
        return {"left": None, "center": None, "right": None}
