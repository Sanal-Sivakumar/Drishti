"""Drishti: Master Pipeline Orchestrator.

Concurrently coordinates:
1. Fast Perception Stream (Continuous, sub-100ms 3D spatial obstacle cues)
2. On-Demand Cognitive Stream (Tactile-triggered Multilingual OCR, Currency, VLM)
3. Priority Preemption Audio Arbiter
4. Edge Privacy Face Anonymization and MQTT Telemetry
"""

from typing import Dict, Any, Optional
import time
import threading
import numpy as np

from hal.camera import BaseCameraSource
from hal.distance_sensor import BaseDistanceSensor
from hal.trigger_input import BaseTriggerInput
from hal.audio_output import BaseAudioSink
from perception.pipeline_fast import FastPerceptionPipeline
from perception.currency import IndianCurrencyRecognizer
from cognitive.ocr_engine import MultilingualOCREngine
from cognitive.vlm_narrator import VisionLanguageNarrator
from cognitive.speech_service import SpeechService
from safety.anonymizer import FaceAnonymizer
from safety.telemetry import TelemetryClient


class DrishtiMasterPipeline:
    """Master orchestrator integrating perception, cognition, safety, and audio."""

    def __init__(
        self,
        camera: BaseCameraSource,
        distance_sensor: BaseDistanceSensor,
        trigger_input: BaseTriggerInput,
        audio_sink: BaseAudioSink,
        telemetry_client: Optional[TelemetryClient] = None,
    ) -> None:
        self.camera = camera
        self.distance_sensor = distance_sensor
        self.trigger_input = trigger_input
        self.audio_sink = audio_sink
        self.telemetry = telemetry_client or TelemetryClient()

        # Core Engines
        self.fast_pipeline = FastPerceptionPipeline(
            camera=self.camera,
            distance_sensor=self.distance_sensor,
            audio_sink=self.audio_sink,
        )
        self.currency_classifier = IndianCurrencyRecognizer()
        self.ocr_engine = MultilingualOCREngine()
        self.vlm_narrator = VisionLanguageNarrator()
        self.speech_service = SpeechService()
        self.anonymizer = FaceAnonymizer()

        # Pipeline lifecycle state
        self.is_running = False
        self.active_thread = None
        self.lock = threading.Lock()
        self.cycle_count = 0
        self.last_heartbeat_time = 0.0

    def step(self) -> Dict[str, Any]:
        """Execute a single synchronized step across perception and trigger inputs.
        
        Returns:
            Dict[str, Any]: Comprehensive cycle telemetry.
        """
        cycle_result = {"timestamp": time.time(), "cycle_index": self.cycle_count}

        # 1. Run Fast Perception Cycle
        fast_res = self.fast_pipeline.step()
        cycle_result["fast_perception"] = fast_res

        # 2. Poll Tactile Triggers (Non-blocking)
        # Button 1: Scene Narration (VLM)
        if self.trigger_input.poll_trigger(button_id=1):
            cycle_result["event"] = "SCENE_NARRATION_TRIGGERED"
            success, frame = self.camera.read_frame()
            if success and frame is not None:
                # Anonymize frame locally
                redacted = self.anonymizer.anonymize_frame(frame)
                # Query VLM with perception context
                obstacles = fast_res.get("obstacles", [])
                vlm_out = self.vlm_narrator.describe_scene(redacted["image"], detected_obstacles=obstacles)
                narration = vlm_out["narration"]
                cycle_result["narration"] = narration
                # Vocalize description
                self.speech_service.speak(narration, play_audio=True)
                # Dispatch caregiver event
                self.telemetry.publish_emergency_alert(
                    alert_type="USER_NARRATION_REQUEST",
                    snapshot_metadata={"faces_redacted": redacted["faces_redacted"]}
                )

        # Button 2: Currency Recognition & Multilingual OCR
        elif self.trigger_input.poll_trigger(button_id=2):
            cycle_result["event"] = "READING_OR_CURRENCY_TRIGGERED"
            success, frame = self.camera.read_frame()
            if success and frame is not None:
                # Check for Indian Rupee banknote first
                currency_res = self.currency_classifier.classify_banknote(frame)
                if currency_res["denomination"] is not None:
                    announcement = currency_res["spoken_name"]
                    cycle_result["currency"] = currency_res
                    self.speech_service.speak(announcement, play_audio=True)
                else:
                    # Fallback to Multilingual Indic OCR reading
                    ocr_res = self.ocr_engine.extract_text(frame)
                    cycle_result["ocr"] = ocr_res
                    readout = ocr_res["text"] if ocr_res["text"] else "No clear text detected."
                    self.speech_service.speak(readout, play_audio=True)

        # 3. Publish Periodic Heartbeat Telemetry (Every 10 seconds)
        now = time.time()
        if now - self.last_heartbeat_time >= 10.0:
            self.telemetry.publish_heartbeat(
                battery_pct=88, cpu_temp_c=51.5, system_state="ACTIVE_NAVIGATION"
            )
            self.last_heartbeat_time = now

        self.cycle_count += 1
        return cycle_result

    def run_loop(self, max_cycles: Optional[int] = None) -> None:
        """Run continuous perception loop until stopped or max cycles reached."""
        self.is_running = True
        count = 0
        while self.is_running:
            self.step()
            count += 1
            if max_cycles is not None and count >= max_cycles:
                break
            time.sleep(0.02)  # Yield CPU to emulate 30-50 FPS pacing

    def shutdown(self) -> None:
        """Gracefully release all hardware and pipeline resources."""
        self.is_running = False
        self.fast_pipeline.shutdown()
        self.telemetry.disconnect()
