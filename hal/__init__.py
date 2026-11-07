"""Drishti Hardware Abstraction Layer (HAL)."""

from hal.camera import BaseCameraSource, SyntheticCameraSource, OpenCVCameraSource, V4L2CameraSource
from hal.distance_sensor import (
    BaseDistanceSensor,
    SyntheticDistanceSensor,
    VL53L1XSensor,
    ESP32DistanceSensor,
)
from hal.esp32_link import ButtonEvent, Esp32Link, LinkStatus, TofSample
from hal.trigger_input import (
    BaseTriggerInput,
    SyntheticTriggerInput,
    GPIOPushButtonTrigger,
    ESP32ButtonTrigger,
    CompositeTriggerInput,
)
from hal.audio_output import BaseAudioSink, HeadlessAudioRecorder, ALSAAudioSink

__all__ = [
    "BaseCameraSource",
    "SyntheticCameraSource",
    "OpenCVCameraSource",
    "V4L2CameraSource",
    "BaseDistanceSensor",
    "SyntheticDistanceSensor",
    "VL53L1XSensor",
    "ESP32DistanceSensor",
    "Esp32Link",
    "TofSample",
    "ButtonEvent",
    "LinkStatus",
    "BaseTriggerInput",
    "SyntheticTriggerInput",
    "GPIOPushButtonTrigger",
    "ESP32ButtonTrigger",
    "CompositeTriggerInput",
    "BaseAudioSink",
    "HeadlessAudioRecorder",
    "ALSAAudioSink",
]
