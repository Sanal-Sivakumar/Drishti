# Drishti evaluation dataset

This directory defines evidence collection; it does not contain a validated dataset. Dataset capture involving people or identifiable locations requires consent, approved retention/access rules, and separation of identity/consent records from frames.

Use consented, de-identified wearable-camera sequences from the final camera position. Split by route and participant—not by adjacent frame—to prevent leakage. Include daylight, dim indoor light, glare, occlusion, motion blur, and negative scenes.

Each JSONL row represents one evaluated frame:

```json
{"frame_id":"route01-0042","ground_truth":[{"class_name":"curb","critical":true,"azimuth_deg":-18,"distance_m":1.1}],"predictions":[{"class_name":"curb","azimuth_deg":-15,"distance_m":1.2}],"narration_claims":["curb"],"narration_latency_ms":840}
```

Ground-truth distance must come from a measured reference, not monocular estimation. Run `python scripts/evaluate_drishti.py evaluation/results.jsonl`.

## Dataset design

- Split by participant and physical route, never by adjacent frames.
- Retain a locked test set that is not used for training, threshold selection, quantization calibration, or prompt tuning.
- Include empty/negative scenes and hard negatives, not only hazards.
- Cover target camera height/angle, walking/head motion, partial visibility, crowding, glare, darkness, glass, reflective and black surfaces, stairs, curbs, drop-offs, poles, doors and overhead obstacles.
- Record ground truth with an independent measured reference and document its uncertainty.
- Version labels, reviewer disagreement and corrections.
- Store the code commit, model hash, label map, calibration hash and configuration with every result set.

## Reported metrics

At minimum report critical-hazard recall, false-warning rate, direction error, physical-distance error, unsupported narrated objects, narration acceptance/rejection, and end-to-end latency percentiles. Report per-class results and confidence intervals; an average must not hide a safety-critical class failure.

`narration_latency_ms` must be measured from user trigger to accepted output. Add separate timestamps for queue wait, inference, validation, TTS start and audible onset. A narration based on expired evidence should be counted as rejected, not successful.

Do not place participant images or identifiable route/GPS data in this Git repository.
