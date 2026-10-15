#!/usr/bin/env python3
"""Drishti Model Downloader & Verification Utility.

Downloads and verifies official ONNX neural weights for edge deployment:
- YOLOv8n ONNX (~12.5 MB): Real-time 80-class COCO obstacle detector
"""

import os
import sys
import socket
import hashlib
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
        "sha256_checksums": [
            "9d24faab26bebe6f25de708e235f02ec8733848cb0025ff67e60e4685a37e739",
            "dd48a79dd7fec8ca25fde4eca742ff7bca23b27e2e903eb23bc1d9f83a459bd2",
        ],
        "description": "YOLOv8 Nano (ONNX, 640x640, 80-class COCO)",
    }
}


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a local file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def download_file(urls: list, dest_path: Path, min_size: int = 1_000_000, expected_hashes: list = None) -> bool:
    """Download file with progress report, SHA-256 validation, and multi-mirror fallback."""
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
                # Validate SHA-256 checksum if configured
                if expected_hashes:
                    file_hash = compute_sha256(temp_dest)
                    if file_hash not in expected_hashes:
                        print(f"\n[-] Checksum mismatch for {dest_path.name} (got {file_hash}). Trying next mirror...")
                        temp_dest.unlink()
                        continue
                    print(f"\n[+] Verified SHA-256 integrity: {file_hash[:16]}...")

                temp_dest.replace(dest_path)
                print(f"[+] Successfully installed {dest_path.name} ({dest_path.stat().st_size // 1024} KB).")
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
        expected_hashes = info.get("sha256_checksums", [])

        is_valid = False
        if dest.exists() and dest.stat().st_size >= min_size:
            if expected_hashes:
                file_hash = compute_sha256(dest)
                if file_hash in expected_hashes:
                    print(f"[+] {filename} verified present (SHA-256: {file_hash[:16]}...) - {info['description']}")
                    is_valid = True
                else:
                    print(f"[!] Warning: Existing {filename} hash mismatch ({file_hash[:16]}...). Re-downloading...")
            else:
                print(f"[+] {filename} verified present ({dest.stat().st_size // 1024} KB) - {info['description']}")
                is_valid = True

        if not is_valid:
            success = download_file(info["urls"], dest, min_size=min_size, expected_hashes=expected_hashes)
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
