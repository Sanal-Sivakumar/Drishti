# Latency, user experience, and optimization plan

Last reviewed: 2026-11-08.

## Known measurements

| Path | Current evidence | Interpretation |
|---|---|---|
| Synthetic perception cycle | Automated test requires mean below 50 ms and at least 20 FPS | Software regression guard only; it is not a physical-device benchmark |
| SmolVLM2 on the development laptop | Real local smoke run: approximately 9.1 seconds | Too slow for live navigation; acceptable only as optional on-demand description after freshness controls |
| Nominal camera/loop setting | 30 FPS / 33.3 ms frame interval | The current main loop sleeps 33.3 ms after processing, so actual period is processing time plus sleep time |
| ESP32/VL53L0X | Firmware timing budget 33 ms; host stale threshold 120 ms | Sensor-level settings, not an end-to-end audible latency result |
| Audio | 60 ms cues, non-blocking bounded output queue | Enqueue-to-ear time has not been measured |

No p50/p95/p99 hardware latency dataset exists yet.

## User-experience risks

1. A narration request can take about nine seconds on the laptop and may be slower on Raspberry Pi.
2. Four queued cognition jobs can create a long backlog and describe old frames.
3. The user receives no dedicated immediate “request accepted/working” cue.
4. A completed narration is validated against request-time facts, not the latest live facts.
5. Spoken narration can occupy the audio route when a new hazard appears.
6. The loop’s unconditional sleep lowers achieved FPS below the configured target.

## Optimize in this order

### P0: preserve safety semantics

1. Mark close-ToF audio as `CRITICAL`, stop/duck speech, flush older PCM, and place the cue at the front of the output path.
2. Timestamp every frame and fact set. At completion, discard or regenerate narration whose evidence is older than the configured freshness limit.
3. Coalesce narration requests: retain at most one pending request and always replace it with the newest frame.
4. Recheck the current live hazard state before speaking any metric distance.
5. Replace “path clear” user wording with “no tracked hazard” until dense, validated free-space perception exists.

### P1: reduce perception latency

1. Change loop scheduling to a deadline: `sleep(max(0, period - processing_time))`.
2. Benchmark the real camera, serial range, YOLO preprocessing, inference, fusion, audio enqueue, and physical output separately.
3. Use an accelerator-appropriate detector: TensorRT on Jetson, an NPU-specific compiled model, or an optimized ONNX Runtime provider.
4. Profile 640×640 versus smaller detector inputs and only accept a speed change after hazard-recall evaluation.
5. Avoid camera buffering; keep the newest frame and drop stale frames.
6. Pin/limit CPU threads only after profiling contention with audio and serial I/O.

### P2: reduce VLM response time

1. Load and warm the VLM during startup or on the first non-critical idle period.
2. Quantize the VLM with a supported runtime and validate structured-output acceptance after quantization.
3. Reduce image edge size and output tokens, then measure scene-description loss.
4. Use an accelerator-backed VLM runtime on the future compute module.
5. Consider a smaller scene classifier/captioner for routine context and reserve the VLM for explicit user questions.
6. Provide a short acknowledgement earcon immediately; never speak “please wait” over hazard cues.

## Required benchmark

Add a hardware benchmark that writes JSONL with monotonic timestamps for:

```text
camera exposure/capture
ToF sample creation
host receipt
preprocess start/end
detector start/end
fusion end
audio enqueue
audio device callback/start
VLM queued/start/end
TTS queued/start/end
```

Report p50, p95, p99, maximum, dropped frames, stale samples, queue depth, thermal state, and battery state. Measure cold start and a 30-minute thermally stable run.

Example engineering targets—not certified limits—should be approved through the risk process before trials:

- critical physical-range event to audible/haptic onset: p95 at or below 150 ms;
- routine detector event to cue onset: p95 at or below 250 ms;
- no accepted narration based on evidence older than 2 seconds;
- no critical warning blocked by speech;
- no unbounded queue growth.

