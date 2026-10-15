#!/usr/bin/env python3
"""Drishti Model Downloader & Verification Utility.

Downloads and verifies official ONNX neural weights for edge deployment:
- YOLOv8n ONNX (~12.5 MB): Real-time 80-class COCO obstacle detector
"""

import os
import sys
import socket
import urllib.request
from pathlib import Path

# Prioritize IPv4 resolution to prevent DNS/IPv6 routing hangs on edge/container networks
_orig_getaddrinfo = socket.getaddrinfo


def _getaddrinfo_ipv4_first(host, port, family=0, type=0, proto=0, flags=0):
    res = _orig_getaddrinfo(host, port, family, type, proto, flags)
    v4 = [r for r in res if r[0] == socket.AF_INET]
    v6 = [r for r in res if r[0] == socket.AF_INET6]
    return v4 + v6


socket.getaddrinfo = _getaddrinfo_ipv4_first

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODELS = {
    "yolov8n.onnx": {
        "urls": [
            "https://github.com/yakhyo/people-tracker/releases/download/weights/yolov8n.onnx",
            "https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/yolov8n.onnx",
        ],
        "min_size_bytes": 10_000_000,
        "description": "YOLOv8 Nano (ONNX, 640x640, 80-class COCO)",
    }
}


def download_file(urls: list, dest_path: Path, min_size: int = 1_000_000) -> bool:
    """Download file with progress report and multi-mirror fallback."""
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) DrishtiEdge/1.0")]
    urllib.request.install_opener(opener)

    temp_dest = dest_path.with_suffix(".tmp")
    for url in urls:
        print(f"[*] Attempting download of {dest_path.name} from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as response, open(temp_dest, "wb") as out_file:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 65536
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = int(downloaded * 100 / total_size)
                        sys.stdout.write(f"\r    -> Progress: {pct}% ({downloaded // 1024} KB / {total_size // 1024} KB)")
                    else:
                        sys.stdout.write(f"\r    -> Progress: ({downloaded // 1024} KB)")
                    sys.stdout.flush()

            if temp_dest.exists() and temp_dest.stat().st_size >= min_size:
                temp_dest.replace(dest_path)
                print(f"\n[+] Successfully downloaded {dest_path.name} ({dest_path.stat().st_size // 1024} KB).")
                return True
            else:
                size_found = temp_dest.stat().st_size if temp_dest.exists() else 0
                print(f"\n[-] Downloaded file is too small or incomplete ({size_found} bytes). Trying next mirror...")
                if temp_dest.exists():
                    temp_dest.unlink()
        except Exception as e:
            print(f"\n[-] Mirror download failed: {e}. Trying next mirror...")
            if temp_dest.exists():
                temp_dest.unlink()

    return False


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("           DRISHTI NEURAL MODEL DOWNLOAD & VERIFICATION")
    print("=" * 70)

    failed_models = []
    for filename, info in MODELS.items():
        dest = MODELS_DIR / filename
        min_size = info.get("min_size_bytes", 1_000_000)
        if dest.exists() and dest.stat().st_size >= min_size:
            print(f"[+] {filename} verified present ({dest.stat().st_size // 1024} KB) - {info['description']}")
        else:
            success = download_file(info["urls"], dest, min_size=min_size)
            if not success or not dest.exists() or dest.stat().st_size < min_size:
                print(f"[!] ERROR: Could not obtain valid neural model {filename}.")
                failed_models.append(filename)

    print("=" * 70)
    if failed_models:
        print(f"[!] FAILED: The following required models could not be verified: {', '.join(failed_models)}")
        return 1

    print("[*] All required neural models verified present and operational.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
