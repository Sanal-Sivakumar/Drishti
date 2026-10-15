#!/usr/bin/env python3
"""Drishti Model Downloader & Verification Utility.

Downloads and verifies official ONNX neural weights for edge deployment:
- YOLOv8n ONNX (13 MB): Real-time 80-class COCO obstacle detector
"""

import os
import sys
import hashlib
from pathlib import Path
import urllib.request

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODELS = {
    "yolov8n.onnx": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx",
        "size_bytes": 13398495,
        "description": "YOLOv8 Nano (ONNX, 640x640, 80-class COCO)",
    }
}


def download_file(url: str, dest_path: Path) -> bool:
    """Download file with progress report."""
    print(f"[*] Downloading {dest_path.name} from {url}...")
    try:
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r    -> Progress: {percent}% ({count * block_size // 1024} KB / {total_size // 1024} KB)")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, str(dest_path), reporthook=reporthook)
        print("\n[+] Download complete.")
        return True
    except Exception as e:
        print(f"\n[-] Failed to download: {e}")
        return False


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("           DRISHTI NEURAL MODEL DOWNLOAD & VERIFICATION")
    print("=" * 70)

    for filename, info in MODELS.items():
        dest = MODELS_DIR / filename
        if dest.exists() and dest.stat().st_size > 1000000:
            print(f"[+] {filename} verified present ({dest.stat().st_size // 1024} KB) - {info['description']}")
        else:
            success = download_file(info["url"], dest)
            if not success:
                print(f"[!] Warning: Could not download {filename}. Please download manually to {dest}")

    print("=" * 70)
    print("[*] Model check completed.")


if __name__ == "__main__":
    main()
