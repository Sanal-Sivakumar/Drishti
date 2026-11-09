"""Drishti master pipeline with isolated cognition and fail-closed navigation."""

from __future__ import annotations

import time
import os
from typing import Any, Dict, Optional

import numpy as np

from cognitive.ocr_engine import MultilingualOCREngine
from cognitive.speech_service import SpeechService
from cognitive.vlm_narrator import SpatialSceneNarrator
from cognitive.worker import CognitiveTask, CognitiveWorker
from config.config_loader import get_pipeline_config
from config.logging_config import get_logger
from hal.audio_output import BaseAudioSink
from hal.audio_priority import PriorityAudioArbiter
from hal.camera import BaseCameraSource
from hal.distance_sensor import BaseDistanceSensor
from hal.trigger_input import BaseTriggerInput
from perception.currency import IndianCurrencyRecognizer
from perception.pipeline_fast import FastPerceptionPipeline
from perception.temporal_confirmation import TemporalDetectionConfirmer
from safety.anonymizer import FaceAnonymizer
from safety.telemetry import TelemetryClient

logger = get_logger("pipeline")


class DrishtiMasterPipeline:
    """Coordinate real-time perception without blocking on cognition or speech."""

    def __init__(
        self,
        camera: BaseCameraSource,
        distance_sensor: BaseDistanceSensor,
        trigger_input: BaseTriggerInput,
        audio_sink: BaseAudioSink,
        telemetry_client: Optional[TelemetryClient] = None,
        continuous_currency: bool = False,
        currency_interval: int = 10,
        currency_cooldown_s: float = 4.0,
        pipeline_config: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,
        speech_enabled: bool = True,
    ) -> None:
        self.config = pipeline_config or get_pipeline_config()
        self.mode = mode or ("simulation" if getattr(camera, "is_synthetic", False) else "production")
        self.camera = camera
        self.distance_sensor = distance_sensor
        self.trigger_input = trigger_input
        self.audio_sink = audio_sink
        self.audio_arbiter = PriorityAudioArbiter(audio_sink)
        safety_cfg = self.config.get("safety", {})
        telemetry_cfg = safety_cfg.get("telemetry", {})
        username_env = telemetry_cfg.get("username_env", "DRISHTI_MQTT_USERNAME")
        password_env = telemetry_cfg.get("password_env", "DRISHTI_MQTT_PASSWORD")
        self.telemetry = telemetry_client or TelemetryClient(
            device_id=telemetry_cfg.get("device_id", "drishti-wearable-01"),
            broker_host=telemetry_cfg.get("broker_host", "localhost"),
            broker_port=int(telemetry_cfg.get("broker_port", 8883)),
            use_tls=bool(telemetry_cfg.get("use_tls", True)),
            ca_certs=telemetry_cfg.get("ca_certs"),
            qos=int(telemetry_cfg.get("qos", 1)),
            username=os.environ.get(username_env),
            password=os.environ.get(password_env),
            buffer_limit=int(telemetry_cfg.get("buffer_limit", 500)),
            persistence_path=telemetry_cfg.get("persistence_path"),
            require_auth=bool(telemetry_cfg.get("require_auth", True)),
        )
        self.continuous_currency = continuous_currency
        self.currency_interval = max(1, int(currency_interval))
        self.currency_cooldown_s = float(currency_cooldown_s)
        self.speech_enabled = speech_enabled

        is_synthetic = self.mode == "simulation" or bool(getattr(camera, "is_synthetic", False))
        self.fast_pipeline = FastPerceptionPipeline(
            camera=camera,
            distance_sensor=distance_sensor,
            audio_sink=self.audio_arbiter,
            config=self.config,
            mode=self.mode,
        )

        perception = self.config.get("perception", {})
        cognitive = self.config.get("cognitive", {})
        currency_cfg = perception.get("currency", {})
        self.currency_classifier = IndianCurrencyRecognizer(
            confidence_threshold=float(currency_cfg.get("confidence_threshold", 0.75)),
            supported_denominations=currency_cfg.get("supported_denominations"),
        )
        ocr_cfg = cognitive.get("ocr", {})
        self.ocr_engine = MultilingualOCREngine(
            languages=ocr_cfg.get("languages"),
            min_confidence=float(ocr_cfg.get("min_confidence", 60.0)),
            unicode_normalization=ocr_cfg.get("unicode_normalization", "NFC"),
            synthetic_mode=is_synthetic,
        )
        narrator_cfg = cognitive.get("scene_narrator", cognitive.get("vlm", {}))
        synth_fn = None
        if is_synthetic:
            synth_fn = lambda img, prompt: (
                prompt.split("A minimally valid grounded answer is: ", 1)[1].split(". You may make", 1)[0].strip()
                if "A minimally valid grounded answer is: " in prompt
                else '{"path_clear":false,"primary_hazard":"obstacle","direction":"ahead","distance_m":1.65,"speech":"Obstacle 1.65 metres ahead."}'
            )
        self.scene_narrator = SpatialSceneNarrator(
            model_name=narrator_cfg.get(
                "model_name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
            ),
            model_path=narrator_cfg.get("model_path"),
            max_tokens=int(narrator_cfg.get("max_output_tokens", 64)),
            temperature=float(narrator_cfg.get("temperature", 0.0)),
            timeout_s=float(narrator_cfg.get("timeout_s", 12.0)),
            max_image_edge=int(narrator_cfg.get("max_image_edge", 512)),
            local_files_only=bool(narrator_cfg.get("local_files_only", True)),
            system_prompt=narrator_cfg.get("prompt_template"),
            inference_fn=synth_fn,
        )
        self.vlm_narrator = self.scene_narrator  # deprecated compatibility name

        tts_cfg = cognitive.get("tts", {})
        self.speech_service = SpeechService(
            voice=tts_cfg.get("default_language", "en_US"),
            speaking_rate=float(tts_cfg.get("speaking_rate", 0.95)),
            voice_type=tts_cfg.get("voice_type", "female1"),
            pitch=int(tts_cfg.get("pitch", 0)),
            sample_rate_hz=int(tts_cfg.get("sample_rate_hz", 22050)),
            audio_sink=audio_sink,
            audio_arbiter=self.audio_arbiter,
            piper_model=tts_cfg.get("piper_model"),
            piper_length_scale=tts_cfg.get("piper_length_scale"),
            piper_sentence_silence=float(tts_cfg.get("piper_sentence_silence", 0.25)),
            async_playback=bool(tts_cfg.get("async_playback", True)),
        )
        privacy = safety_cfg.get("privacy", {})
        self.privacy_enabled = bool(privacy.get("enable_face_blur", True))
        self.anonymizer = FaceAnonymizer(
            kernel_size=int(privacy.get("blur_kernel_size", 15)),
            sigma=float(privacy.get("blur_sigma", 7.0)),
        )
        self.cognitive_worker = CognitiveWorker(self._handle_cognitive_task)
        temporal_cfg = perception.get("temporal_confirmation", {})
        self.detection_confirmer = TemporalDetectionConfirmer(
            required_frames=int(temporal_cfg.get("required_frames", 2)),
            max_distance_delta_m=float(temporal_cfg.get("max_distance_delta_m", 0.6)),
        )
        self.confirmed_obstacles: list[Dict[str, Any]] = []

        self.heartbeat_interval_s = float(telemetry_cfg.get("heartbeat_interval_s", 10.0))
        self.is_running = False
        self.cycle_count = 0
        self.last_heartbeat_time = 0.0
        self.last_currency_announcement_time = 0.0
        self.last_currency_denom: Optional[int] = None
        self._currency_pending = False

    def _frame(self) -> Optional[np.ndarray]:
        frame = getattr(self.fast_pipeline, "last_frame", None)
        if frame is None:
            _, frame = self.camera.read_frame()
        return None if frame is None else frame.copy()

    def _handle_cognitive_task(self, task: CognitiveTask) -> Dict[str, Any]:
        frame = task.payload["frame"]
        if task.kind == "narration":
            obstacles = task.payload.get("obstacles", [])
            person_boxes = [o["bbox"] for o in obstacles if o.get("class_name") == "person" and "bbox" in o]
            redacted = (
                self.anonymizer.anonymize_frame(frame, extra_face_boxes=person_boxes)
                if self.privacy_enabled
                else {"image": frame, "faces_redacted": 0}
            )
            output = self.scene_narrator.describe_scene(
                redacted["image"],
                obstacles,
                navigation_authoritative=bool(task.payload.get("navigation_authoritative", False)),
                path_state=str(task.payload.get("path_state", "unknown")),
            )
            if self.speech_enabled and output.get("success"):
                self.speech_service.speak(output["narration"], play_audio=True)
            self.telemetry.publish_event(
                "USER_NARRATION_REQUEST",
                {"faces_redacted": redacted["faces_redacted"], "privacy_filter_enabled": self.privacy_enabled},
            )
            return {**output, "faces_redacted": redacted["faces_redacted"]}

        if task.kind == "continuous_currency":
            currency = self.currency_classifier.identify_currency(frame, color_order="BGR")
            if currency.get("denomination") is not None:
                if self.speech_enabled:
                    self.speech_service.speak(currency["spoken_name"], play_audio=True)
            return {"success": True, "currency": currency}

        # Handle on-demand Button 2 / keyboard 'C' (reading or currency)
        # In simulation / synthetic mode: evaluate immediately on the test frame
        if self.mode == "simulation" or getattr(self.camera, "is_synthetic", False):
            currency = self.currency_classifier.identify_currency(frame, color_order="BGR")
            if currency.get("denomination") is not None:
                if self.speech_enabled:
                    self.speech_service.speak(currency["spoken_name"], play_audio=True)
                return {"success": True, "currency": currency}
            ocr = self.ocr_engine.extract_text(frame)
            if self.speech_enabled:
                self.speech_service.speak(ocr.get("text") or "No clear text detected.", play_audio=True)
            return {"success": True, "ocr": ocr}

        # Production active scanning window: inspect live frames for up to 4.0 seconds
        scan_deadline = time.time() + 4.0
        best_ocr: Optional[Dict[str, Any]] = None
        last_check_time = 0.0

        while time.time() < scan_deadline:
            now = time.time()
            if now - last_check_time >= 0.20:
                last_check_time = now
                live_frame = self._frame()
                if live_frame is None:
                    live_frame = frame

                if live_frame is not None:
                    # 1. Banknote check
                    curr = self.currency_classifier.identify_currency(live_frame, color_order="BGR")
                    if curr.get("denomination") is not None:
                        if self.speech_enabled:
                            self.speech_service.speak(curr["spoken_name"], play_audio=True)
                        return {"success": True, "currency": curr}

                    # 2. OCR check
                    ocr_res = self.ocr_engine.extract_text(live_frame)
                    txt = ocr_res.get("text", "").strip()
                    if txt and len(txt) >= 3:
                        if best_ocr is None or len(txt) > len(best_ocr.get("text", "")):
                            best_ocr = ocr_res
                        if ocr_res.get("confidence", 0.0) >= 0.70 and len(txt) >= 8:
                            break
            time.sleep(0.04)

        if best_ocr is not None and best_ocr.get("text"):
            if self.speech_enabled:
                self.speech_service.speak(best_ocr["text"], play_audio=True)
            return {"success": True, "ocr": best_ocr}

        if self.speech_enabled:
            self.speech_service.speak("No banknote or text detected.", play_audio=True)
        return {"success": False, "ocr": {"text": "", "script": "none", "confidence": 0.0, "lines": []}}

    def _collect_cognitive_results(self) -> list[Dict[str, Any]]:
        results = []
        while True:
            result = self.cognitive_worker.poll()
            if result is None:
                break
            if result.get("kind") == "continuous_currency":
                self._currency_pending = False
                denomination = result.get("currency", {}).get("denomination")
                if denomination is not None:
                    self.last_currency_announcement_time = time.time()
                    self.last_currency_denom = denomination
            results.append(result)
        return results

    def step(self) -> Dict[str, Any]:
        cycle_result: Dict[str, Any] = {
            "timestamp": time.time(), "cycle_index": self.cycle_count,
            "cognitive_results": self._collect_cognitive_results(), "events": [],
        }
        fast_res = self.fast_pipeline.step()
        cycle_result["fast_perception"] = fast_res
        self.confirmed_obstacles = self.detection_confirmer.update(fast_res.get("obstacles", []))

        if self.continuous_currency and not self._currency_pending and self.cycle_count % self.currency_interval == 0:
            frame = self._frame()
            if frame is not None and time.time() - self.last_currency_announcement_time >= self.currency_cooldown_s:
                self._currency_pending = self.cognitive_worker.submit("continuous_currency", {"frame": frame})

        if self.trigger_input.poll_trigger(button_id=1):
            frame = self._frame()
            if self.speech_enabled and self.mode != "simulation":
                self.speech_service.speak("Analyzing scene...", play_audio=True)
            queued = frame is not None and self.cognitive_worker.submit(
                "narration",
                {
                    "frame": frame,
                    "obstacles": list(self.confirmed_obstacles),
                    "navigation_authoritative": fast_res.get("readiness", {}).get("navigation_authoritative", False),
                    "path_state": fast_res.get("path_state", "unknown"),
                },
            )
            cycle_result["events"].append({"type": "SCENE_NARRATION_TRIGGERED", "queued": queued})

        if self.trigger_input.poll_trigger(button_id=2):
            frame = self._frame()
            if self.speech_enabled and self.mode != "simulation":
                self.speech_service.speak("Scanning for banknote or text...", play_audio=True)
            queued = frame is not None and self.cognitive_worker.submit("reading_or_currency", {"frame": frame})
            cycle_result["events"].append({"type": "READING_OR_CURRENCY_TRIGGERED", "queued": queued})

        now = time.time()
        if now - self.last_heartbeat_time >= self.heartbeat_interval_s:
            readiness = fast_res.get("readiness", {})
            state = "ACTIVE_NAVIGATION" if readiness.get("navigation_authoritative") else "NAVIGATION_NOT_READY"
            self.telemetry.publish_heartbeat(system_state=state, readiness=readiness)
            self.last_heartbeat_time = now

        self.cycle_count += 1
        return cycle_result

    def wait_for_cognition(self, timeout: float = 10.0) -> list[Dict[str, Any]]:
        self.cognitive_worker.wait_until_idle(timeout)
        return self._collect_cognitive_results()

    def run_loop(self, max_cycles: Optional[int] = None) -> None:
        self.is_running = True
        target_fps = float(self.config.get("pipeline", {}).get("target_fps", 30))
        count = 0
        while self.is_running:
            self.step()
            count += 1
            if max_cycles is not None and count >= max_cycles:
                break
            time.sleep(1.0 / max(1.0, target_fps))

    def shutdown(self) -> None:
        self.is_running = False
        self.cognitive_worker.stop()
        self.speech_service.stop()
        self.fast_pipeline.shutdown()
        self.telemetry.disconnect()
