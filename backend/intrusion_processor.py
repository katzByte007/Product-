"""Intrusion detection — zone alerts + orange vest/helmet PPE check."""
import logging
import threading
import time

import cv2
import numpy as np

from backend import beta_ai
from backend.alerts import save_alert_event
from backend.analytics_runtime import point_in_polygon
from config import OWLV2_INFER_MAX_WIDTH

logger = logging.getLogger(__name__)

PPE_VEST_LABELS = ("orange safety vest",)
PPE_HELMET_LABELS = ("orange safety helmet",)


def _bbox_touches_zone(bbox, zone):
    if len(zone) < 3:
        return False
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    if point_in_polygon((cx, cy), zone):
        return True
    for px, py in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
        if point_in_polygon((px, py), zone):
            return True
    return False


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-6)


def _center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _crop(frame, box, pad=16):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


def _is_person_label(label):
    t = (label or "").lower()
    return "person" in t or t in ("people", "human", "worker", "intruder")


def _is_vest_label(label):
    t = (label or "").lower()
    return "vest" in t or "jacket" in t


def _is_helmet_label(label):
    t = (label or "").lower()
    return "helmet" in t or "hard hat" in t or "hardhat" in t


def _center_in_box(cx, cy, box):
    x1, y1, x2, y2 = box
    return x1 <= cx <= x2 and y1 <= cy <= y2


class IntrusionDetectionProcessor:
    def __init__(self, camera_id, camera_reader, zone_points, prompt_text, canvas_size=(800, 450), conf=0.25):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.prompt_text = (prompt_text or "").strip()
        self.conf = max(0.05, float(conf))
        self._zones_converted = False
        self.zone_video = []
        self.canvas_w, self.canvas_h = canvas_size
        self.zone_canvas = [self._xy(p) for p in (zone_points or [])]

        self.running = False
        self.lock = threading.Lock()
        self._thread = None
        self.annotated_frame = None
        self._overlay_dets = []
        self._alert_active = False
        self.stats = {"detections": 0, "status": "Initializing", "alert": False}
        self._last_detections = []

    def _xy(self, p):
        if isinstance(p, dict):
            return (float(p.get("x", 0)), float(p.get("y", 0)))
        return (float(p[0]), float(p[1]))

    def _labels(self):
        user = [s.strip() for s in self.prompt_text.split(",") if s.strip()]
        extra = ["person"] + list(PPE_VEST_LABELS) + list(PPE_HELMET_LABELS)
        out, seen = [], set()
        for lbl in user + extra:
            key = lbl.lower()
            if key not in seen:
                seen.add(key)
                out.append(lbl)
        return out

    def _convert_zone(self, fw, fh):
        self.zone_video = [
            (int(p[0] * fw / self.canvas_w), int(p[1] * fh / self.canvas_h))
            for p in self.zone_canvas
        ]
        self._zones_converted = True

    def _prime_overlay(self):
        frame = self.camera_reader.get_frame() if self.camera_reader else None
        if frame is None:
            return
        h, w = frame.shape[:2]
        if not self._zones_converted:
            self._convert_zone(w, h)
        vis = frame.copy()
        self._draw_zones(vis)
        cv2.putText(vis, "Intrusion | detecting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 140, 0), 2)
        with self.lock:
            self.annotated_frame = vis

    def start(self):
        labels = self._labels()
        if not labels:
            self.stats["status"] = "No prompt labels"
            return
        if not beta_ai.owlv2_available():
            self.stats["status"] = "OWLv2 not available"
            self._prime_overlay()
            return
        self.running = True
        self._prime_overlay()
        self._thread = threading.Thread(target=self._prepare_scheduler, daemon=True, name=f"Intrusion-{self.camera_id}")
        self._thread.start()

    def stop(self):
        self.running = False
        if getattr(self, "_scheduler", None) is not None:
            self._scheduler.unregister(getattr(self, "_scheduler_key", self.camera_id))
        if self._thread:
            self._thread.join(timeout=5.0)

    def _prepare_scheduler(self):
        from PIL import Image

        proc, model = beta_ai._get_owlv2()
        if proc is None or model is None:
            self.stats["status"] = "OWLv2 unavailable"
            return
        from backend.owlv2_scheduler import get_owlv2_scheduler
        self._scheduler = get_owlv2_scheduler()
        self._scheduler_key = f"intrusion:{self.camera_id}"
        self._scheduler.register(
            self._scheduler_key,
            self.camera_reader,
            lambda frame: self._run_loop(frame, proc, model),
            interval=max(beta_ai._infer_min_interval(next(model.parameters()).device), 0.1),
        )
        self.stats["status"] = "Active"

    def _run_loop(self, scheduled_frame=None, proc=None, model=None):
        from PIL import Image

        if proc is None or model is None:
            proc, model = beta_ai._get_owlv2()
        if proc is None or model is None:
            self.stats["status"] = "OWLv2 unavailable"
            return

        labels = self._labels()
        text_labels = [labels]
        last_t = 0.0
        post_thresh = max(0.01, self.conf)
        device = next(model.parameters()).device
        min_interval = beta_ai._infer_min_interval(device)

        while self.running:
            try:
                now = time.time()
                if min_interval > 0 and now - last_t < min_interval:
                    time.sleep(0.01)
                    continue
                last_t = now

                frame = scheduled_frame if scheduled_frame is not None else self.camera_reader.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                h, w = frame.shape[:2]
                if not self._zones_converted:
                    self._convert_zone(w, h)

                infer = frame
                max_w = OWLV2_INFER_MAX_WIDTH
                if w > max_w:
                    scale = max_w / float(w)
                    infer = cv2.resize(frame, (max_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
                ih, iw = infer.shape[:2]
                rgb = cv2.cvtColor(infer, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                inputs = proc(text=text_labels, images=pil_image, return_tensors="pt")
                device = next(model.parameters()).device
                t = beta_ai.torch
                if t is not None:
                    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
                    with beta_ai._owlv2_infer_lock:
                        with t.inference_mode():
                            outputs = model(**inputs)
                else:
                    with beta_ai._owlv2_infer_lock:
                        outputs = model(**inputs)

                raw = beta_ai._owlv2_decode_boxes(proc, outputs, inputs, ih, iw, post_thresh, text_labels)
                sx, sy = w / float(iw), h / float(ih)
                mapped = []
                for d in raw:
                    x1, y1, x2, y2 = d["box"]
                    bbox = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
                    mapped.append({"box": bbox, "label": d["label"], "score": d["score"]})

                persons = [d for d in mapped if _is_person_label(d["label"])]
                vests = [d for d in mapped if _is_vest_label(d["label"])]
                helmets = [d for d in mapped if _is_helmet_label(d["label"])]

                vis = frame.copy()
                self._draw_zones(vis)

                last_dets = []
                zone_hits = []
                ppe_violations = []

                for det in mapped:
                    x1, y1, x2, y2 = det["box"]
                    lbl = f"{det['label']} {det['score']:.2f}"
                    color = (80, 200, 255)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(vis, lbl, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                for person in persons:
                    box = person["box"]
                    in_zone = _bbox_touches_zone(box, self.zone_video)
                    has_vest = False
                    for v in vests:
                        cx, cy = _center(v["box"])
                        if _iou(box, v["box"]) > 0.05 or _center_in_box(cx, cy, box):
                            has_vest = True
                            break
                    # helmet: overlap or center in upper 45% of person box
                    px1, py1, px2, py2 = box
                    head_y = py1 + 0.45 * max(1, py2 - py1)
                    has_helmet = False
                    for hel in helmets:
                        cx, cy = _center(hel["box"])
                        if _iou(box, hel["box"]) > 0.02 and cy <= head_y:
                            has_helmet = True
                            break
                        if px1 <= cx <= px2 and py1 <= cy <= head_y:
                            has_helmet = True
                            break

                    missing = []
                    if not has_vest:
                        missing.append("orange vest")
                    if not has_helmet:
                        missing.append("orange helmet")
                    ppe_ok = not missing

                    x1, y1, x2, y2 = box
                    if in_zone and not ppe_ok:
                        color = (0, 0, 255)
                        tag = "INTRUSION + NO PPE"
                        zone_hits.append(person)
                    elif in_zone:
                        color = (0, 0, 255)
                        tag = "INTRUSION"
                        zone_hits.append(person)
                    elif not ppe_ok:
                        color = (0, 140, 255)
                        tag = "NO PPE " + ",".join(missing)
                    else:
                        color = (0, 200, 80)
                        tag = "PPE OK"
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(vis, tag, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    last_dets.append((box, tag, color))

                    if in_zone:
                        save_alert_event(
                            self.camera_id,
                            "intrusion",
                            _crop(frame, box),
                            [(box, tag)],
                            severity="high",
                            meta={"message": "Person in intrusion zone"},
                        )
                    if not ppe_ok:
                        ppe_violations.append(person)
                        save_alert_event(
                            self.camera_id,
                            "no_ppe",
                            _crop(frame, box),
                            [(box, tag)],
                            severity="high",
                            meta={"message": f"No PPE: missing {', '.join(missing)}"},
                        )

                alert_now = len(zone_hits) > 0 or len(ppe_violations) > 0
                if zone_hits:
                    cv2.putText(vis, "INTRUSION ALERT", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                elif ppe_violations:
                    cv2.putText(vis, "NO PPE ALERT", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 140, 255), 3)

                with self.lock:
                    self.annotated_frame = vis
                    self._overlay_dets = mapped
                    self._alert_active = alert_now
                    self._last_detections = last_dets

                self.stats = {"detections": len(mapped), "status": "Active", "alert": alert_now}
                if scheduled_frame is not None:
                    return

            except Exception as e:
                logger.error("Intrusion %s: %s", self.camera_id, e)
                if scheduled_frame is not None:
                    return
                time.sleep(0.2)

    def _draw_zones(self, frame):
        if frame is None:
            return
        if not self._zones_converted:
            h, w = frame.shape[:2]
            self._convert_zone(w, h)
        if len(self.zone_video) >= 3:
            pts = np.array(self.zone_video, dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (255, 140, 0))
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            cv2.polylines(frame, [pts], True, (255, 140, 0), 2)
            cx = int(sum(p[0] for p in self.zone_video) / len(self.zone_video))
            cy = int(sum(p[1] for p in self.zone_video) / len(self.zone_video))
            cv2.putText(frame, "INTRUSION", (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2)

    def get_annotated_frame(self):
        with self.lock:
            if self.annotated_frame is not None:
                return self.annotated_frame.copy()
        live = self.camera_reader.get_frame() if self.camera_reader else None
        if live is None:
            return None
        if not self._zones_converted:
            h, w = live.shape[:2]
            self._convert_zone(w, h)
        vis = live.copy()
        self._draw_zones(vis)
        return vis

    def get_last_detections(self):
        with self.lock:
            return list(self._last_detections)
