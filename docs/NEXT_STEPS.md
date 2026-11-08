# Recommended next steps

Work in order. Do not start field trials while any P0 gate is open.

## P0 — safety and truthfulness

- [ ] Give close-ToF cues real `CRITICAL` audio priority; interrupt/duck speech and flush stale PCM.
- [ ] Add frame/fact timestamps, cognition task expiry, and completion-time sensor revalidation.
- [ ] Coalesce narration to one newest pending request; reject stale results.
- [ ] Replace user-facing “path clear” with “no tracked hazard” until dense free-space validation exists.
- [ ] Fix the main-loop deadline so configured FPS includes processing time.
- [ ] Provision and validate `models/yolov8n.onnx` and the Piper voice on the laptop.
- [ ] Record real laptop hardware p50/p95/p99 sensor-to-audio latency.
- [ ] Calibrate camera intrinsics, camera-to-ToF alignment/offset, and the air-conduction headset.

Exit criterion: critical warnings cannot be blocked by speech; stale narration cannot be spoken; hardware readiness is truthful; retained benchmarks meet approved internal targets.

## P1 — perception evidence

- [ ] Define the intended hazard taxonomy with an orientation-and-mobility specialist.
- [ ] Build the consented, route-separated wearable dataset.
- [ ] Add stairs, curbs, drop-offs, poles, glass doors, hanging obstacles and partially visible furniture.
- [ ] Train/export a custom detector and lock its label map/hash.
- [ ] Evaluate critical recall, false-warning rate, direction error, distance error, unsupported narration and latency.
- [ ] Test darkness, glare, glass, motion blur, occlusion, reflective/black surfaces and crowded scenes.
- [ ] Add a downward depth channel before claiming curb/drop-off coverage.

Exit criterion: predefined held-out metrics and confidence intervals pass; failure modes are documented and reflected in user claims.

## P2 — prototype hardware evolution

- [ ] Compare dense stereo/depth modules and 8×8 multizone ToF on the exact use cases.
- [ ] Add synchronized IMU data and compensate for head motion.
- [ ] Add a safety MCU/watchdog and independent haptic critical channel.
- [ ] Benchmark Pi 5/CM5 versus Jetson or another production-supported accelerator.
- [ ] Complete battery, thermal, enclosure, connector and acoustic design experiments.
- [ ] Implement secure boot/update, signed artifacts, rollback and SBOM generation.

Exit criterion: product-prototype architecture survives fault injection and sustained environmental/thermal tests.

## P3 — supervised indoor research

- [ ] Obtain appropriate ethics, consent, privacy and data-retention approvals.
- [ ] Create a controlled route and stopping rules with a sighted supervisor.
- [ ] Keep the participant’s normal mobility aid in use.
- [ ] Start with bench/mannequin tests, then sighted-developer tests, then supervised target-user studies.
- [ ] Record incidents, near misses, discomfort and every readiness transition.

Exit criterion: study protocol objectives are met without unresolved serious hazards. This does not authorize outdoor or unsupervised use.

## P4 — productization

- [ ] Freeze intended use and target markets; obtain regulatory classification advice.
- [ ] Establish quality, risk, software, usability and cybersecurity lifecycle processes.
- [ ] Qualify suppliers and production modules; design the custom carrier/enclosure.
- [ ] Complete verification, validation, regulatory submissions and manufacturing transfer.
- [ ] Establish complaint handling, CAPA, vulnerability response and post-market monitoring.

