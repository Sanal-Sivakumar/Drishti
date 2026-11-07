#!/usr/bin/env bash
# ==============================================================================
# Drishti Assistive Wearable: Raspberry Pi 5 Automated Provisioning Script
# Target: Raspberry Pi OS 64-bit (Debian 12 Bookworm)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CURRENT_USER="${SUDO_USER:-$USER}"

echo "=============================================================================="
echo "         DRISHTI ASSISTIVE WEARABLE: RASPBERRY PI 5 PROVISIONING"
echo "=============================================================================="
echo "Project Root: $PROJECT_ROOT"
echo "Target User:  $CURRENT_USER"
echo "------------------------------------------------------------------------------"

# 1. System Package Dependencies
echo "[1/6] Updating APT repositories and installing system packages..."
sudo apt-get update -y
sudo apt-get install -y \
    build-essential git pkg-config \
    libasound2-dev libportaudio2 \
    i2c-tools \
    tesseract-ocr tesseract-ocr-hin tesseract-ocr-tam tesseract-ocr-mal \
    mosquitto mosquitto-clients \
    pipewire pipewire-pulse \
    python3 python3-pip python3-venv

# 2. Hardware User Groups & Permissions
echo "[2/6] Configuring user permissions for Camera, Audio, I2C, and UART..."
sudo usermod -aG video,audio,dialout,i2c "$CURRENT_USER" || true

# 3. Persistent UDEV Rules for ESP32 DevKit V1 (CP2102 and CH340 chips)
echo "[3/6] Installing persistent udev rules for ESP32 sensor link (/dev/drishti-tof)..."
cat << 'EOF' | sudo tee /etc/udev/rules.d/99-drishti-esp32.rules > /dev/null
# Silicon Labs CP2102 USB-to-UART (Standard ESP32 DevKit V1)
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", MODE="0666", GROUP="dialout", SYMLINK+="drishti-tof"
# QinHeng Electronics CH340/CH341 USB-to-UART
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", MODE="0666", GROUP="dialout", SYMLINK+="drishti-tof"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger || true

# 4. Virtual Environment & Python Packages
echo "[4/6] Setting up Python virtual environment..."
cd "$PROJECT_ROOT"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Ensure pip & wheel are modern
.venv/bin/python3 -m pip install --upgrade pip setuptools wheel
.venv/bin/python3 -m pip install -r requirements.txt

# 5. Neural Network Weights & Models
echo "[5/6] Verifying and downloading neural network weights..."
.venv/bin/python3 scripts/download_models.py

# 6. Systemd Background Appliance Service
echo "[6/6] Configuring systemd background auto-start service..."
SERVICE_FILE="/etc/systemd/system/drishti.service"
sudo cp "$PROJECT_ROOT/config/drishti.service" "$SERVICE_FILE"
sudo sed -i "s|/home/pi/Drishti|$PROJECT_ROOT|g" "$SERVICE_FILE"
sudo sed -i "s|User=pi|User=$CURRENT_USER|g" "$SERVICE_FILE"
sudo sed -i "s|Group=pi|Group=$CURRENT_USER|g" "$SERVICE_FILE"
sudo systemctl daemon-reload

echo "=============================================================================="
echo "                 RASPBERRY PI 5 PROVISIONING COMPLETE"
echo "=============================================================================="
echo "Next steps:"
echo "  1. Log out and log back in (or reboot) for user group permissions to take effect:"
echo "     sudo reboot"
echo ""
echo "  2. Test physical hardware health:"
echo "     python3 scripts/check_hardware_health.py"
echo ""
echo "  3. Start the live wearable runner interactively:"
echo "     python3 scripts/run_live_wearable.py --hardware rpi"
echo ""
echo "  4. Enable autonomous boot-up on wearable battery power:"
echo "     sudo systemctl enable --now drishti.service"
echo "=============================================================================="
