# KAVACH ATM-Sentinel

KAVACH ATM-Sentinel is an advanced, real-time edge surveillance module designed to detect physical banking fraud, coercion, and hardware tampering at ATMs.

Unlike traditional surveillance systems that only log events, KAVACH uses a **Dual-Model YOLO Architecture** to instantly translate visual anomalies into structured `Fraud Logic Matrices` (JSON payloads) that can be ingested by a centralized fusion engine.

## Features
- **Dual-Model Inference:** Runs `YOLOv10s` (for general objects) and a custom `YOLO11n-Threat` model (for weapons) simultaneously.
- **NMS-Free & Edge Optimized:** Engineered to run smoothly on edge hardware without computationally expensive Non-Maximum Suppression.
- **Two-Stage Identity Classifier:** Automatically crops faces and runs a lightweight PyTorch MobileNet classifier to detect balaclavas and helmets.

## Threat Detection Matrix
The engine automatically scores the following vectors in real-time:
*   `CRITICAL_THREAT (+90)`: Firearm, Pistol, or Knife detected in the frame.
*   `HARDWARE_TAMPER (+60)`: Laptops, black-box devices, or unauthorized hardware.
*   `SKIMMING_SUSPECTED (+50)`: Backpacks or toolkits placed near the ATM terminal.
*   `IDENTITY_MASKED (+30)`: Balaclavas, ski masks, or helmets detected on a person.
*   `PIN_THEFT_RISK (+25)`: Cell phones near the keypad or multiple people standing too close (Shoulder Surfing).

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/kavach-sentinel.git
cd kavach-sentinel
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

You can run the engine on a local webcam or pass an MP4 file to simulate a CCTV feed.

**Run on Webcam (Live Demo):**
```bash
python sentinel.py --source 0
```

**Run on Video File (CCTV Simulation):**
```bash
python sentinel.py --source "path/to/video.mp4"
```

**Stream Alerts to a Webhook (Fusion Engine):**
```bash
python sentinel.py --source 0 --webhook "http://localhost:5000/api/alerts"
```

## Architecture Notes
The `identity_classifier.pt` and `yolo11n_threat_detection.pt` files are custom trained PyTorch weights. Ensure they remain in the root directory for the script to load them correctly.
