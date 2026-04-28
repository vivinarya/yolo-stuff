"""
KAVACH ATM-Sentinel: Dual-Model Surveillance Module

This module utilizes a dual-model architecture for real-time threat detection:
1. General Model (YOLOv10s): Detects persons, phones, laptops, and everyday objects.
2. Threat Model (YOLO11n): Highly tuned for detecting firearms, pistols, knives, etc.

Usage:
$ python sentinel.py --source 0
"""

import cv2
import threading
import time
import math
import requests
import argparse
from datetime import datetime, timezone
from typing import Dict, Any
import os

from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

class VideoProcessor:
    def __init__(self, source):
        self.source = source
        try:
            self.source = int(source)
        except ValueError:
            pass

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source: {self.source}")
            
        self.ret, self.frame = self.cap.read()
        self.running = True
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.frame = frame
                else:
                    self.running = False 
            time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

class ATMSentinel:
    def __init__(self, general_model_path="yolov10s.pt", threat_model_path="yolo11n_threat_detection.pt", webhook_url=None):
        print("[*] Loading General Model (YOLOv10s)...")
        self.general_model = YOLO(general_model_path) 
        
        print(f"[*] Loading Threat Model ({threat_model_path})...")
        self.threat_model = YOLO(threat_model_path)
        
        self.webhook_url = webhook_url
        self.SURFING_DISTANCE_THRESHOLD = 180.0 

        # --- General Model (COCO) Classes ---
        self.GEN_PERSON = 0
        self.GEN_BACKPACK = 24
        self.GEN_LAPTOP = 63
        self.GEN_CELL_PHONE = 67

        # --- Threat Model Classes ---
        # {0: 'ammo', 1: 'firearm', 2: 'grenade', 3: 'knife', 4: 'pistol', 5: 'rocket'}
        self.THREAT_CLASSES = [0, 1, 2, 3, 4, 5]

        # --- Identity Mask Classifier ---
        self.mask_classifier = None
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        mask_model_path = "identity_classifier.pt"
        if os.path.exists(mask_model_path):
            print("[*] Loading Identity Mask Classifier...")
            self.mask_classifier = models.mobilenet_v3_small()
            num_ftrs = self.mask_classifier.classifier[3].in_features
            self.mask_classifier.classifier[3] = nn.Linear(num_ftrs, 2)
            self.mask_classifier.load_state_dict(torch.load(mask_model_path, map_location="cpu", weights_only=True))
            self.mask_classifier.eval()
        else:
            print("[!] Identity Mask Classifier not found. Skipping face mask detection.")

    def _calculate_distance(self, box1, box2):
        x1_center = (box1[0] + box1[2]) / 2
        y1_center = (box1[1] + box1[3]) / 2
        
        x2_center = (box2[0] + box2[2]) / 2
        y2_center = (box2[1] + box2[3]) / 2
        
        return math.hypot(x2_center - x1_center, y2_center - y1_center)

    def _check_face_mask(self, frame, box):
        if not self.mask_classifier:
            return False
        x1, y1, x2, y2 = map(int, box)
        head_height = max(10, int((y2 - y1) * 0.20))
        y2_head = min(frame.shape[0], y1 + head_height)
        y1, x1, x2 = max(0, y1), max(0, x1), min(frame.shape[1], x2)
        head_crop = frame[y1:y2_head, x1:x2]
        if head_crop.size == 0:
            return False
        head_img = Image.fromarray(cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB))
        input_tensor = self.mask_transform(head_img).unsqueeze(0)
        with torch.no_grad():
            outputs = self.mask_classifier(input_tensor)
            _, predicted = torch.max(outputs, 1)
        # 0 = clear_face, 1 = masked_face
        return predicted.item() == 1

    def analyze_frame(self, frame, gen_results, threat_results) -> Dict[str, Any]:
        """Fraud Logic Layer combining both models"""
        events = []
        persons = []
        phones = []
        laptops = []
        backpacks = []
        weapons = []
        
        highest_threat_level = "GREEN" 

        # 1. Parse General Objects
        for box in gen_results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            if cls_id == self.GEN_PERSON: 
                persons.append({"box": xyxy, "conf": conf})
                # Check for Mask/Balaclava
                if self._check_face_mask(frame, xyxy):
                    highest_threat_level = "YELLOW" if highest_threat_level == "GREEN" else highest_threat_level
                    events.append({
                        "event": "IDENTITY_MASKED", "score": 30, "confidence": conf,
                        "details": "Deliberate identity concealment (Mask/Helmet) detected.",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
            elif cls_id == self.GEN_CELL_PHONE: phones.append({"box": xyxy, "conf": conf})
            elif cls_id == self.GEN_LAPTOP: laptops.append({"box": xyxy, "conf": conf})
            elif cls_id == self.GEN_BACKPACK: backpacks.append({"box": xyxy, "conf": conf})

        # 2. Parse Threat Objects (Weapons)
        for box in threat_results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            if cls_id in self.THREAT_CLASSES:
                weapons.append({"box": xyxy, "conf": conf, "name": self.threat_model.names[cls_id]})

        # --- EVENT SCORING MATRIX ---

        # 1. CRITICAL_THREAT (+90)
        if weapons:
            highest_threat_level = "RED"
            top_weapon = max(weapons, key=lambda w: w["conf"])
            events.append({
                "event": "CRITICAL_THREAT", "score": 90, "confidence": top_weapon["conf"], 
                "details": f"Physical robbery or coercion ({top_weapon['name'].upper()} DETECTED).", 
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        # 2. HARDWARE_TAMPER (+60)
        if laptops:
            highest_threat_level = "RED"
            events.append({
                "event": "HARDWARE_TAMPER", "score": 60, "confidence": max([l["conf"] for l in laptops]), 
                "details": "Black Box attack / unauthorized hardware.", 
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        # 3. SKIMMING_SUSPECTED (+50)
        if backpacks and len(persons) >= 1:
            highest_threat_level = "RED" if highest_threat_level != "RED" else "RED"
            events.append({
                "event": "SKIMMING_SUSPECTED", "score": 50, "confidence": max([b["conf"] for b in backpacks]), 
                "details": "Potential skimmer installation or tamper kit.", 
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        # 4. PIN_THEFT_RISK (+25) (Cell Phones)
        if phones:
            highest_threat_level = "YELLOW" if highest_threat_level == "GREEN" else highest_threat_level
            events.append({
                "event": "PIN_THEFT_RISK", "score": 25, "confidence": max([p["conf"] for p in phones]), 
                "details": "Cell phone near keypad.", 
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        # 5. PIN_THEFT_RISK (+25) (Shoulder Surfing)
        if len(persons) >= 2:
            for i in range(len(persons)):
                for j in range(i + 1, len(persons)):
                    dist = self._calculate_distance(persons[i]["box"], persons[j]["box"])
                    if dist < self.SURFING_DISTANCE_THRESHOLD:
                        highest_threat_level = "YELLOW" if highest_threat_level == "GREEN" else highest_threat_level
                        events.append({
                            "event": "PIN_THEFT_RISK", "score": 25, "confidence": min(persons[i]["conf"], persons[j]["conf"]), 
                            "details": "Shoulder Surfing detected.", 
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                        break

        return {"status": highest_threat_level, "events": events}

    def emit_event(self, event_data: Dict[str, Any]):
        print(f"[ALERT] {event_data['event']} (+{event_data['score']}) - {event_data['details']}")
        if self.webhook_url:
            try: requests.post(self.webhook_url, json=event_data, timeout=1)
            except: pass

    def draw_overlay(self, frame, gen_results, threat_results, analysis):
        status_color = {"GREEN": (0, 255, 0), "YELLOW": (0, 255, 255), "RED": (0, 0, 255)}
        color = status_color.get(analysis["status"], (0, 255, 0))
        
        # Draw General Objects (in Light Blue)
        for box in gen_results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{self.general_model.names[cls_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
            cv2.putText(frame, label, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

        # Draw Threat Objects (in Bold RED, overrides general if overlapping)
        for box in threat_results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"THREAT: {self.threat_model.names[cls_id].upper()} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, label, (x1, max(10, y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw HUD Overlay
        cv2.rectangle(frame, (10, 10), (380, 80), (0, 0, 0), -1)
        cv2.putText(frame, "KAVACH ATM-Sentinel (DUAL-MODEL)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Risk Level: {analysis['status']}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame

    def run(self, source):
        print(f"[*] Starting Dual-Model Surveillance on source: {source}")
        processor = VideoProcessor(source)
        
        try:
            while processor.running:
                ret, frame = processor.read()
                if not ret or frame is None:
                    continue

                gen_results = self.general_model.predict(
                    source=frame, 
                    classes=[self.GEN_PERSON, self.GEN_BACKPACK, self.GEN_LAPTOP, self.GEN_CELL_PHONE],
                    verbose=False, 
                    conf=0.45
                )
                
                # 2. Run Threat Detection (Detects ONLY weapons)
                threat_results = self.threat_model.predict(source=frame, verbose=False, conf=0.35)

                # 3. Analyze combined logic
                analysis = self.analyze_frame(frame, gen_results, threat_results)

                # 4. Emit
                for event in analysis["events"]:
                    self.emit_event(event)

                # 5. Render
                display_frame = self.draw_overlay(frame.copy(), gen_results, threat_results, analysis)
                cv2.imshow("KAVACH Security Monitor", display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[*] Shutdown signal received.")
                    break

        except KeyboardInterrupt:
            print("[*] Keyboard interrupt detected.")
        finally:
            processor.release()
            cv2.destroyAllWindows()
            print("[*] Surveillance terminated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KAVACH Dual-Model ATM-Sentinel")
    parser.add_argument("--source", type=str, default="0", help="Video source")
    parser.add_argument("--webhook", type=str, default=None, help="Fusion engine webhook URL")
    args = parser.parse_args()

    sentinel = ATMSentinel(webhook_url=args.webhook)
    sentinel.run(args.source)
