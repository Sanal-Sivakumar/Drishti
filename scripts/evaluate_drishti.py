#!/usr/bin/env python3
"""Compute safety-oriented Drishti perception and narration metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def angular_error(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def evaluate(rows):
    critical_total = critical_hits = false_warnings = predictions_total = 0
    direction_errors, distance_errors, latencies = [], [], []
    hallucinations = 0
    for row in rows:
        truth = list(row.get("ground_truth", []))
        predictions = list(row.get("predictions", []))
        matched_predictions = set()
        matched_truth_names = set()
        critical_total += sum(bool(item.get("critical")) for item in truth)
        for gt in truth:
            candidates = [
                (index, pred) for index, pred in enumerate(predictions)
                if index not in matched_predictions and pred.get("class_name") == gt.get("class_name")
            ]
            if not candidates:
                continue
            index, pred = min(
                candidates,
                key=lambda pair: angular_error(float(pair[1]["azimuth_deg"]), float(gt["azimuth_deg"])),
            )
            matched_predictions.add(index)
            matched_truth_names.add(str(gt.get("class_name")))
            if gt.get("critical"):
                critical_hits += 1
            direction_errors.append(angular_error(float(pred["azimuth_deg"]), float(gt["azimuth_deg"])))
            if gt.get("distance_m") is not None and pred.get("distance_m") is not None:
                distance_errors.append(abs(float(pred["distance_m"]) - float(gt["distance_m"])))
        predictions_total += len(predictions)
        false_warnings += len(predictions) - len(matched_predictions)
        hallucinations += sum(
            str(claim) not in matched_truth_names for claim in row.get("narration_claims", [])
        )
        if row.get("narration_latency_ms") is not None:
            latencies.append(float(row["narration_latency_ms"]))
    return {
        "critical_hazard_recall": critical_hits / critical_total if critical_total else None,
        "false_warning_rate": false_warnings / predictions_total if predictions_total else 0.0,
        "mean_direction_error_deg": mean(direction_errors) if direction_errors else None,
        "mean_distance_error_m": mean(distance_errors) if distance_errors else None,
        "unsupported_narrated_objects": hallucinations,
        "mean_narration_latency_ms": mean(latencies) if latencies else None,
        "frames": len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_jsonl")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.results_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps(evaluate(rows), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
