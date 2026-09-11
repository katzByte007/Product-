"""Workforce monitoring — worker dwell time in machine zone with away alerts."""
import logging
import threading
import time

import cv2
import numpy as np

from backend.alerts import save_alert_event
from backend.analytics_runtime import HeadCountTracker, get_inference_engine, nms_detections, point_in_polygon

logger = logging.getLogger(__name__)

AWAY_ALERT_SEC = 3.0
OCCLUSION_GRACE_FRAMES = 45


class WorkforceMonitoringProcessor:
    """Track one worker in a machine zone; alert if away >3s; resume dwell time on return."""

    def __init__(self, camera_id, camera_reader, machine_zone, canvas_size=(800, 450), engine=None, conf=0.5, away_alert_sec=3.0):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.engine = engine
        self.conf = conf
        self.away_alert_sec = max(1.0, float(away_alert_sec))
        self.running = False
        self.annotated_frame = None
        self.lock = threading.Lock()

        self.machine_zone_canvas = machine_zone
        self.canvas_w, self.canvas_h = canvas_size
        self.machine_zone_video = []
        self._zones_converted = False

        self.tracker = HeadCountTracker(max_age=150, min_hits=1, iou_threshold=0.35)
        self.worker_track_id = 1
        self.active_worker_id = None
        self.accumulated_sec = 0.0
        self.last_tick = None
        self.in_zone = False
        self.away_since = None
        self.alert_active = False
        self._last_known_in_zone = False
        self._loop_generation = getattr(camera_reader, "loop_generation", 0)

        self.stats = {
            "worker_id": None,
            "dwell_sec": 0,
            "status": "Initializing",
            "alert": False,
            "state": "waiting",
        }
        self._last_detections = []

    def _xy(self, p):
        if isinstance(p, dict):
            return float(p.get("x", 0)), float(p.get("y", 0))
        return float(p[0]), float(p[1])

    def _convert_zones(self, frame_w, frame_h):
        self.machine_zone_video = [
            (int(self._xy(p)[0] * frame_w / self.canvas_w), int(self._xy(p)[1] * frame_h / self.canvas_h))
            for p in self.machine_zone_canvas
        ]
        self._zones_converted = True

    def start(self):
        self.running = True
        self._prime_overlay()
        self.engine.register(self.camera_id, self.camera_reader, self._on_result, self.conf)

    def _prime_overlay(self):
        frame = self.camera_reader.get_frame() if self.camera_reader else None
        if frame is None:
            return
        h, w = frame.shape[:2]
        if not self._zones_converted:
            self._convert_zones(w, h)
        vis = frame.copy()
        self._draw_zones(vis)
        self._draw_status_bar(vis, ["Workforce  |  Worker 1  |  Time at machine: 0s", "Detecting…"], alert=False)
        with self.lock:
            self.annotated_frame = vis

    def stop(self):
        self.running = False
        self.engine.unregister(self.camera_id)

    def _reset_for_loop(self):
        self.tracker = HeadCountTracker(max_age=150, min_hits=1, iou_threshold=0.35)
        self.active_worker_id = None
        self.in_zone = False
        self.away_since = None
        self.alert_active = False
        self._last_known_in_zone = False
        self.last_tick = None

    def _in_machine(self, person):
        zone = self.machine_zone_video
        if person is None or len(zone) < 3:
            return False
        x1, y1, x2, y2 = [int(v) for v in person["bbox"]]
        cx, cy = person.get("center") or ((x1 + x2) // 2, (y1 + y2) // 2)
        samples = [
            (cx, cy),
            ((x1 + x2) // 2, max(y1, y2 - 8)),
            ((x1 + x2) // 2, (y1 + y2) // 2),
            (x1, y1),
            (x2, y2),
            (x1, y2),
            (x2, y1),
        ]
        if any(point_in_polygon(p, zone) for p in samples):
            return True
        zx = [p[0] for p in zone]
        zy = [p[1] for p in zone]
        ix1, iy1 = max(x1, min(zx)), max(y1, min(zy))
        ix2, iy2 = min(x2, max(zx)), min(y2, max(zy))
        return ix2 > ix1 and iy2 > iy1

    def _find_worker_track(self, tracked, worker_id):
        for person in tracked:
            if person["id"] == worker_id:
                return person
        for t in self.tracker.trackers:
            if t.id == worker_id and t.time_since_update <= OCCLUSION_GRACE_FRAMES:
                bbox = t.get_state()
                return {
                    "id": worker_id,
                    "bbox": bbox,
                    "center": ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
                    "occluded": True,
                }
        return None

    def _on_result(self, frame, detections, _masks):
        if not self.running:
            return
        h, w = frame.shape[:2]
        if not self._zones_converted:
            self._convert_zones(w, h)

        try:
            loop_gen = getattr(self.camera_reader, "loop_generation", 0)
            if loop_gen != self._loop_generation:
                self._loop_generation = loop_gen
                self._reset_for_loop()

            scores = [(d[2] - d[0]) * (d[3] - d[1]) for d in detections]
            detections = nms_detections(detections, scores, iou_thresh=0.35)
            tracked, current_ids = self.tracker.update(detections)
            now = time.time()

            if self.active_worker_id is None:
                in_zone_people = [p for p in tracked if self._in_machine(p)]
                pick = next((p for p in in_zone_people if p["id"] == self.worker_track_id), None)
                if pick is None and in_zone_people:
                    pick = in_zone_people[0]
                if pick is None:
                    pick = next((p for p in tracked if p["id"] == self.worker_track_id), None)
                if pick is not None:
                    self.active_worker_id = pick["id"]
                    in_machine = self._in_machine(pick)
                    self.in_zone = in_machine
                    self._last_known_in_zone = in_machine
                    self.last_tick = now if in_machine else None
                    if not in_machine:
                        self.away_since = now

            worker = None
            if self.active_worker_id is not None:
                worker = self._find_worker_track(tracked, self.active_worker_id)

            if worker is not None:
                in_machine = self._in_machine(worker)
                occluded = worker.get("occluded", False)
                at_machine = in_machine or (occluded and self._last_known_in_zone)

                if at_machine:
                    if self.last_tick is not None:
                        dt = now - self.last_tick
                        if 0.0 < dt < 8.0:
                            self.accumulated_sec += dt
                    self.last_tick = now
                    self.away_since = None
                    self.alert_active = False
                    self.in_zone = True
                    if in_machine:
                        self._last_known_in_zone = True
                else:
                    if self.in_zone or self._last_known_in_zone:
                        if self.away_since is None:
                            self.away_since = now
                        self.in_zone = False
                        self.last_tick = None
                        self._last_known_in_zone = False
                    if self.away_since and (now - self.away_since) >= self.away_alert_sec and not self.alert_active:
                        self.alert_active = True
            elif self.active_worker_id is not None and self.active_worker_id not in current_ids:
                if self.in_zone and self.away_since is None:
                    self.away_since = now
                    self.in_zone = False
                    self.last_tick = None
                if self.away_since and (now - self.away_since) >= self.away_alert_sec and not self.alert_active:
                    self.alert_active = True

            vis = frame.copy()
            self._draw_zones(vis, alert=self.alert_active)

            last_dets = []
            for person in tracked:
                x1, y1, x2, y2 = person["bbox"]
                tid = person["id"]
                is_worker = tid == self.worker_track_id or tid == self.active_worker_id
                color = (0, 255, 255) if is_worker else (180, 180, 180)
                thickness = 3 if is_worker else 2
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                label = "Worker 1" if is_worker else f"ID:{tid}"
                cv2.putText(vis, label, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                last_dets.append((tuple(person["bbox"]), label, color))

            dwell = int(self.accumulated_sec)
            state_txt = "WORKING"
            if self.alert_active:
                state_txt = "ALERT — WORKER 1 AWAY"
            elif self.active_worker_id is None:
                state_txt = "WAITING FOR WORKER 1"
            elif self.away_since:
                state_txt = f"AWAY {now - self.away_since:.1f}s / alert at {self.away_alert_sec:.0f}s"

            self._draw_status_bar(
                vis,
                [
                    f"Workforce  |  Worker 1  |  Time at machine: {dwell}s",
                    f"{state_txt}  |  Away alert: {self.away_alert_sec:.0f}s",
                ],
                alert=self.alert_active,
            )
            if self.alert_active:
                cv2.putText(vis, "ALERT", (w // 2 - 80, 88), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)

            self.stats = {
                "worker_id": self.active_worker_id,
                "dwell_sec": dwell,
                "status": "Active",
                "alert": self.alert_active,
                "state": state_txt.lower(),
            }

            with self.lock:
                self.annotated_frame = vis
                self._last_detections = last_dets

            if self.alert_active:
                crop = vis
                if worker is not None:
                    x1, y1, x2, y2 = worker["bbox"]
                    pad = 12
                    crop = vis[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
                    if crop is None or crop.size == 0:
                        crop = vis
                save_alert_event(
                    self.camera_id,
                    "workforce",
                    crop if crop is not None and crop.size else vis,
                    last_dets,
                    severity="high",
                    meta={
                        "worker_id": self.active_worker_id,
                        "dwell_sec": dwell,
                        "message": f"Worker {self.active_worker_id} away from machine >{self.away_alert_sec:.0f}s",
                    },
                )

        except Exception as e:
            logger.error("Workforce %s: %s", self.camera_id, e)

    def _draw_zones(self, frame, alert=False):
        if frame is None:
            return
        if not self._zones_converted:
            h, w = frame.shape[:2]
            self._convert_zones(w, h)
        if len(self.machine_zone_video) >= 3:
            pts = np.array(self.machine_zone_video, dtype=np.int32)
            zone_color = (0, 0, 255) if alert else (0, 200, 120)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], zone_color)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            thickness = 4 if alert else 2
            cv2.polylines(frame, [pts], True, zone_color, thickness)
            cx = int(sum(p[0] for p in self.machine_zone_video) / len(self.machine_zone_video))
            cy = int(sum(p[1] for p in self.machine_zone_video) / len(self.machine_zone_video))
            cv2.putText(frame, "MACHINE", (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "MACHINE", (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone_color, 1)

    def _draw_status_bar(self, vis, lines, alert=False):
        if vis is None or not lines:
            return
        h, w = vis.shape[:2]
        bar_w = min(w - 12, 820)
        bar_h = 16 + 26 * len(lines)
        overlay = vis.copy()
        cv2.rectangle(overlay, (6, 6), (6 + bar_w, 6 + bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.78, vis, 0.22, 0, vis)
        border = (0, 0, 255) if alert else (0, 220, 255)
        cv2.rectangle(vis, (6, 6), (6 + bar_w, 6 + bar_h), border, 2)
        color = (0, 0, 255) if alert else (255, 255, 255)
        y = 30
        for line in lines:
            cv2.putText(vis, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
            y += 26

    def get_annotated_frame(self):
        with self.lock:
            if self.annotated_frame is not None:
                return self.annotated_frame.copy()
        live = self.camera_reader.get_frame() if self.camera_reader else None
        if live is None:
            return None
        if not self._zones_converted:
            h, w = live.shape[:2]
            self._convert_zones(w, h)
        vis = live.copy()
        self._draw_zones(vis)
        return vis

    def get_last_detections(self):
        with self.lock:
            return list(self._last_detections)
