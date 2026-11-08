"""Drishti Hardware Abstraction Layer (HAL)."""

from hal.camera import (
    BaseCameraSource,
    OpenCVCameraSource,
    Picamera2CameraSource,
    SyntheticCameraSource,
    V4L2CameraSource,
)
from hal.distance_sensor import (
    BaseDistanceSensor,
    SyntheticDistanceSensor,
    VL53L1XSensor,
    ESP32DistanceSensor,
    UnavailableDistanceSensor,
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
from hal.audio_priority import AudioPriority, PriorityAudioArbiter

__all__ = [
    "BaseCameraSource",
    "SyntheticCameraSource",
    "OpenCVCameraSource",
    "V4L2CameraSource",
    "Picamera2CameraSource",
    "BaseDistanceSensor",
    "SyntheticDistanceSensor",
    "VL53L1XSensor",
    "ESP32DistanceSensor",
    "UnavailableDistanceSensor",
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
    "AudioPriority",
    "PriorityAudioArbiter",
]
