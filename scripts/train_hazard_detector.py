#!/usr/bin/env python3
"""Fine-tune and export a wearable-view hazard detector using Ultralytics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="config/hazard_dataset.yaml")
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", default="models/hazard_detector")
    args = parser.parse_args()

    from ultralytics import YOLO

    dataset = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    names = dataset.get("names")
    if not names or not Path(dataset["path"]).exists():
        raise SystemExit("Hazard dataset is absent. Collect and review wearable-camera data first.")

    model = YOLO(args.base_model)
    result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        patience=25,
        close_mosaic=10,
        fliplr=0.5,
        degrees=5.0,
        translate=0.1,
        scale=0.35,
        project="runs/drishti_hazard",
        name="recall_finetune",
    )
    best = Path(result.save_dir) / "weights/best.pt"
    trained = YOLO(str(best))
    # Low confidence validation exposes recall/false-warning trade-offs instead
    # of hiding hazards behind a generic mAP-only threshold.
    trained.val(data=args.data, split="test", conf=0.05, iou=0.6)
    exported = Path(trained.export(format="onnx", imgsz=args.image_size, simplify=True))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "hazard_detector.onnx"
    exported.replace(destination)
    ordered_names = [names[index] if index in names else names[str(index)] for index in range(len(names))]
    (output / "labels.json").write_text(json.dumps(ordered_names, indent=2), encoding="utf-8")
    print(f"Exported {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
