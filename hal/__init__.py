"""Drishti Hardware Abstraction Layer (HAL)."""

from hal.camera import BaseCameraSource, SyntheticCameraSource, OpenCVCameraSource, V4L2CameraSource
from hal.distance_sensor import BaseDistanceSensor, SyntheticDistanceSensor, VL53L1XSensor
from hal.trigger_input import BaseTriggerInput, SyntheticTriggerInput, GPIOPushButtonTrigger
from hal.audio_output import BaseAudioSink, HeadlessAudioRecorder, ALSAAudioSink

__all__ = [
    "BaseCameraSource",
    "SyntheticCameraSource",
    "OpenCVCameraSource",
    "V4L2CameraSource",
    "BaseDistanceSensor",
    "SyntheticDistanceSensor",
    "VL53L1XSensor",
    "BaseTriggerInput",
    "SyntheticTriggerInput",
    "GPIOPushButtonTrigger",
    "BaseAudioSink",
    "HeadlessAudioRecorder",
    "ALSAAudioSink",
]

