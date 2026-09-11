"""ANPR detection: vehicle type, license plate OCR, and speed estimation via LibreYOLO."""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime

import cv2
import numpy as np

from backend.db import get_db
from config import (
    ANPR_DEFAULT_CONF,
    ANPR_DEFAULT_SPEED_THRESHOLD_KMH,
    ANPR_IMGSZ,
    ANPR_IOU,
    ANPR_MAX_DET,
    ANPR_METERS_PER_PIXEL,
    ANPR_PLATE_CONF,
    ANPR_PLATE_DETECT_INTERVAL,
    ANPR_PLATE_MODEL_PATH,
    ANPR_SPEED_SAMPLE_FRAMES,
    ANPR_VEHICLE_CLASS_NAMES,
    ANPR_VEHICLE_CONF,
    ANPR_VEHICLE_MODEL_PATH,
)

logger = logging.getLogger(__name__)

try:
    from libreyolo import ByteTracker, LibreYOLO, TrackConfig

    _LIBREYOLO_AVAILABLE = True
except ImportError:
    ByteTracker = LibreYOLO = TrackConfig = None
    _LIBREYOLO_AVAILABLE = False

try:
    from paddleocr import PaddleOCR

    _PADDLEOCR_AVAILABLE = True
except ImportError:
    PaddleOCR = None
    _PADDLEOCR_AVAILABLE = False

OCR_REFRESH_FRAMES = 45
MAX_OCR_PLATES_PER_FRAME = 3
SPEED_LINE_Y_RATIO = 0.55

_models_lock = threading.Lock()
_plate_model = None
_vehicle_model = None
_vehicle_names = None
_vehicle_classes = None
_ocr_model = None


def anpr_dependencies_ready():
    return _LIBREYOLO_AVAILABLE and _PADDLEOCR_AVAILABLE


def _class_label(names: dict | list, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, names.get(str(cls_id), cls_id)))
    if isinstance(names, list) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def _vehicle_class_filter(names: dict | list) -> list[int] | None:
    class_ids: list[int] = []
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for cls_id, name in items:
        if str(name).lower() in ANPR_VEHICLE_CLASS_NAMES:
            class_ids.append(int(cls_id))
    return class_ids or None


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(track_id))
    return tuple(int(v) for v in rng.integers(60, 255, size=3))


def _draw_label(frame_bgr, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    y0 = max(0, y - th - 10)
    cv2.rectangle(frame_bgr, (x, y0), (x + tw + 8, y), color, -1)
    cv2.putText(
        frame_bgr, text, (x + 4, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
    )


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return intersection / float(area_a + area_b - intersection)


class PlateOcrCache:
    def __init__(self):
        self.entries: list[dict] = []

    def get(self, box, frame_idx):
        best_entry, best_iou = None, 0.0
        for entry in self.entries:
            score = _iou(box, entry['box'])
            if score > best_iou:
                best_iou, best_entry = score, entry
        if best_entry is None or best_iou < 0.35:
            return '', True
        best_entry['box'] = box
        best_entry['last_seen'] = frame_idx
        text = best_entry.get('text', '')
        should_refresh = (
            not text
            or frame_idx - int(best_entry.get('ocr_frame', 0)) >= OCR_REFRESH_FRAMES
        )
        return text, should_refresh

    def update(self, box, text, frame_idx):
        best_entry, best_iou = None, 0.0
        for entry in self.entries:
            score = _iou(box, entry['box'])
            if score > best_iou:
                best_iou, best_entry = score, entry
        if best_entry is not None and best_iou >= 0.35:
            best_entry.update({'box': box, 'last_seen': frame_idx, 'ocr_frame': frame_idx})
            if text:
                best_entry['text'] = text
        else:
            self.entries.append({
                'box': box, 'text': text, 'last_seen': frame_idx, 'ocr_frame': frame_idx,
            })
        self.entries = [
            e for e in self.entries
            if frame_idx - int(e.get('last_seen', frame_idx)) <= OCR_REFRESH_FRAMES
        ]


def _ocr_plate_text(plate_crop, ocr_model) -> str:
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    ocr_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    result = ocr_model.predict(ocr_input)
    texts = []
    if result:
        ocr_res = result[0]
        for txt, score in zip(ocr_res.get('rec_texts', []), ocr_res.get('rec_scores', [])):
            if score > 0.4:
                texts.append(txt)
    plate_text = ' '.join(texts)
    return ''.join(ch for ch in plate_text if ch.isalnum()).upper()


def _detect_plates(frame_bgr, plate_model, ocr_model, ocr_cache, frame_idx):
    result = plate_model(
        frame_bgr, conf=ANPR_PLATE_CONF, iou=ANPR_IOU, imgsz=ANPR_IMGSZ,
        max_det=ANPR_MAX_DET, save=False, color_format='bgr',
    )
    boxes = result.boxes
    plates = []
    if boxes is None or len(boxes) == 0:
        return plates
    h, w = frame_bgr.shape[:2]
    xyxy = boxes.xyxy.cpu().numpy()
    conf_arr = boxes.conf.cpu().numpy()
    ocr_count = 0
    for box, conf in zip(xyxy, conf_arr):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        plate_box = (x1, y1, x2, y2)
        text, should_ocr = ocr_cache.get(plate_box, frame_idx)
        crop = frame_bgr[y1:y2, x1:x2]
        if should_ocr and crop.size > 0 and ocr_count < MAX_OCR_PLATES_PER_FRAME:
            try:
                text = _ocr_plate_text(crop, ocr_model)
                ocr_cache.update(plate_box, text, frame_idx)
                ocr_count += 1
            except Exception as e:
                logger.debug('ANPR OCR error: %s', e)
        plates.append({
            'box': plate_box,
            'text': text or 'PLATE',
            'conf': float(conf),
            'center': ((x1 + x2) / 2, (y1 + y2) / 2),
        })
    return plates


def _associate_plate(vehicle_box, plates):
    x1, y1, x2, y2 = vehicle_box
    best_plate, best_score = None, -1.0
    for plate in plates:
        cx, cy = plate['center']
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            continue
        px1, py1, px2, py2 = plate['box']
        area = max(1, (px2 - px1) * (py2 - py1))
        score = float(plate['conf']) + area / 100000.0
        if score > best_score:
            best_score, best_plate = score, plate
    if best_plate is None:
        return ''
    text = best_plate['text']
    return '' if text == 'PLATE' else text


class VehicleSpeedEstimator:
    def __init__(self, names, fps, frame_shape, meters_per_pixel, speed_threshold_kmh):
        self.names = names
        self.fps = fps if fps > 0 else 30.0
        self.meters_per_pixel = meters_per_pixel
        self.speed_threshold_kmh = speed_threshold_kmh
        self.track_history = defaultdict(list)
        self.last_points = {}
        self.last_frames = {}
        self.speed_history = defaultdict(lambda: deque(maxlen=10))
        self.speeds = {}
        self.plates = {}
        self.vehicle_types = {}
        self.line_y = int(frame_shape[0] * SPEED_LINE_Y_RATIO)

    def _speed_kmh(self, track_id, point, frame_idx):
        if track_id not in self.last_points:
            return None
        prev_point = self.last_points[track_id]
        prev_frame = self.last_frames[track_id]
        frame_delta = max(1, frame_idx - prev_frame)
        if frame_delta < ANPR_SPEED_SAMPLE_FRAMES:
            return None
        seconds = frame_delta / self.fps
        pixel_distance = float(np.hypot(point[0] - prev_point[0], point[1] - prev_point[1]))
        speed = (pixel_distance * self.meters_per_pixel / seconds) * 3.6
        self.speed_history[track_id].append(speed)
        return float(np.mean(self.speed_history[track_id]))

    def process(self, frame_bgr, tracked_result, plates, frame_idx):
        h, w = frame_bgr.shape[:2]
        cv2.line(frame_bgr, (80, self.line_y), (w - 80, self.line_y), (255, 0, 255), 2)
        cv2.putText(
            frame_bgr, 'Speed measurement line', (80, max(25, self.line_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA,
        )
        detections = []
        boxes = tracked_result.boxes
        track_ids = tracked_result.track_id
        if boxes is None or len(boxes) == 0 or track_ids is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        ids = track_ids.cpu().numpy().astype(int)

        for box, conf, cls_id, track_id in zip(xyxy, confs, classes, ids):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            bottom_center = ((x1 + x2) / 2, y2)
            speed = self._speed_kmh(track_id, bottom_center, frame_idx)
            if speed is not None:
                self.speeds[track_id] = speed
                self.last_points[track_id] = bottom_center
                self.last_frames[track_id] = frame_idx
            if track_id not in self.last_points:
                self.last_points[track_id] = bottom_center
                self.last_frames[track_id] = frame_idx

            plate_text = _associate_plate((x1, y1, x2, y2), plates)
            if plate_text:
                self.plates[track_id] = plate_text

            class_name = _class_label(self.names, cls_id)
            self.vehicle_types[track_id] = class_name

            history = self.track_history[track_id]
            history.append(center)
            if len(history) > 30:
                history.pop(0)

            color = _color_for_id(track_id)
            speed_text = f'{self.speeds[track_id]:.1f} km/h' if track_id in self.speeds else '-- km/h'
            plate_label = self.plates.get(track_id, 'NO PLATE')
            label = f'{class_name} ID:{track_id} {plate_label} {speed_text}'
            is_overspeed = track_id in self.speeds and self.speeds[track_id] > self.speed_threshold_kmh

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            _draw_label(frame_bgr, label, x1, y1, color)
            if is_overspeed:
                cv2.putText(
                    frame_bgr, 'Overspeeding!', (x1, min(h - 10, y2 + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
                )
            if len(history) > 1:
                points = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame_bgr, [points], False, color, 2, cv2.LINE_AA)

            detections.append({
                'track_id': int(track_id),
                'vehicle_type': class_name,
                'plate_number': plate_label if plate_label not in ('NO PLATE', 'PLATE') else '',
                'speed_kmh': float(self.speeds[track_id]) if track_id in self.speeds else None,
                'is_overspeeding': is_overspeed,
                'confidence': float(conf),
                'bbox': (x1, y1, x2, y2),
                'label': label,
                'color': color,
            })
        return detections


def _load_shared_models():
    global _plate_model, _vehicle_model, _vehicle_names, _vehicle_classes, _ocr_model
    with _models_lock:
        if _plate_model is not None:
            return
        if not anpr_dependencies_ready():
            raise RuntimeError(
                'ANPR requires libreyolo and paddleocr. '
                'pip install libreyolo paddleocr'
            )
        logger.info('ANPR: Loading plate model from %s', ANPR_PLATE_MODEL_PATH)
        _plate_model = LibreYOLO(ANPR_PLATE_MODEL_PATH)
        fuse_fn = getattr(_plate_model, 'fuse', None)
        if callable(fuse_fn):
            fuse_fn()
        logger.info('ANPR: Loading vehicle model from %s', ANPR_VEHICLE_MODEL_PATH)
        _vehicle_model = LibreYOLO(ANPR_VEHICLE_MODEL_PATH)
        fuse_fn = getattr(_vehicle_model, 'fuse', None)
        if callable(fuse_fn):
            fuse_fn()
        _vehicle_names = _vehicle_model.names
        _vehicle_classes = _vehicle_class_filter(_vehicle_names)
        logger.info('ANPR: Loading PaddleOCR')
        _ocr_model = PaddleOCR(
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            enable_mkldnn=False,
            lang='en',
        )
        logger.info('ANPR: Models loaded successfully')


class ANPRDetectionProcessor:
    """Live ANPR processor using LibreYOLO vehicle + plate models and PaddleOCR."""

    def __init__(
        self,
        camera_id,
        camera_reader,
        conf=None,
        speed_threshold_kmh=None,
        meters_per_pixel=None,
    ):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.conf = conf if conf is not None else ANPR_DEFAULT_CONF
        self.speed_threshold_kmh = (
            speed_threshold_kmh if speed_threshold_kmh is not None else ANPR_DEFAULT_SPEED_THRESHOLD_KMH
        )
        self.meters_per_pixel = meters_per_pixel if meters_per_pixel is not None else ANPR_METERS_PER_PIXEL
        self.running = False
        self.annotated_frame = None
        self.lock = threading.Lock()
        self._thread = None
        self._ready = False
        self._init_error = None
        self.stats = {
            'status': 'Initializing',
            'vehicles_detected': 0,
            'plates_read': 0,
            'overspeeding': 0,
        }
        self._last_detections = []
        self._frame_idx = 0
        self._tracker = None
        self._speed_estimator = None
        self._ocr_cache = PlateOcrCache()
        self._cached_plates = []
        self._saved_tracks = {}

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def get_last_detections(self):
        with self.lock:
            return list(self._last_detections)

    def _run(self):
        try:
            _load_shared_models()
            fps = 25.0
            self._tracker = ByteTracker(
                config=TrackConfig(
                    track_high_thresh=max(self.conf, ANPR_VEHICLE_CONF),
                    track_low_thresh=0.1,
                    new_track_thresh=max(self.conf, ANPR_VEHICLE_CONF),
                    match_thresh=0.8,
                    track_buffer=30,
                    frame_rate=max(1, int(fps)),
                    minimum_consecutive_frames=1,
                )
            )
            self._ready = True
            self.stats['status'] = 'Active'
            logger.info('ANPR %s: Processor started', self.camera_id)
        except Exception as e:
            self._init_error = str(e)
            self.stats['status'] = f'Init error: {e}'
            logger.error('ANPR %s: Init error - %s', self.camera_id, e)
            return

        while self.running:
            frame = self.camera_reader.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue
            self._frame_idx += 1
            try:
                out_frame, dets = self._process_frame(frame)
                plates_read = sum(1 for d in dets if d.get('plate_number'))
                overspeeding = sum(1 for d in dets if d.get('is_overspeeding'))
                self.stats = {
                    'status': 'Active',
                    'vehicles_detected': len(dets),
                    'plates_read': plates_read,
                    'overspeeding': overspeeding,
                }
                overlay = []
                for d in dets:
                    x1, y1, x2, y2 = d['bbox']
                    overlay.append((d['bbox'], d['label'], d['color']))
                with self.lock:
                    self.annotated_frame = out_frame
                    self._last_detections = overlay
                self._maybe_save_events(dets, out_frame)
            except Exception as e:
                logger.error('ANPR %s: Frame error - %s', self.camera_id, e)
                time.sleep(0.05)

    def _process_frame(self, frame):
        vehicle_result = _vehicle_model(
            frame,
            conf=self.conf,
            iou=ANPR_IOU,
            imgsz=ANPR_IMGSZ,
            classes=_vehicle_classes,
            max_det=ANPR_MAX_DET,
            save=False,
            color_format='bgr',
        )
        out_frame = frame.copy()
        tracked_result = self._tracker.update(vehicle_result)

        if self._frame_idx == 1 or self._frame_idx % ANPR_PLATE_DETECT_INTERVAL == 0:
            self._cached_plates = _detect_plates(
                frame, _plate_model, _ocr_model, self._ocr_cache, self._frame_idx,
            )
        for plate in self._cached_plates:
            x1, y1, x2, y2 = plate['box']
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            _draw_label(out_frame, f"{plate['text']} | {plate['conf']:.2f}", x1, y1, (0, 160, 0))

        if self._speed_estimator is None:
            h, w = frame.shape[:2]
            self._speed_estimator = VehicleSpeedEstimator(
                _vehicle_names, 25.0, (h, w),
                self.meters_per_pixel, self.speed_threshold_kmh,
            )
        else:
            self._speed_estimator.speed_threshold_kmh = self.speed_threshold_kmh

        dets = self._speed_estimator.process(
            out_frame, tracked_result, self._cached_plates, self._frame_idx,
        )
        return out_frame, dets

    def _maybe_save_events(self, dets, frame):
        now = time.time()
        for d in dets:
            tid = d['track_id']
            plate = d.get('plate_number') or ''
            has_plate = len(plate) >= 3
            is_overspeed = d.get('is_overspeeding')
            if not has_plate and not is_overspeed:
                continue
            prev = self._saved_tracks.get(tid, {})
            if (
                prev.get('plate') == plate
                and prev.get('overspeed') == is_overspeed
                and now - prev.get('ts', 0) < 30
            ):
                continue
            self._save_event(d)
            self._saved_tracks[tid] = {'plate': plate, 'overspeed': is_overspeed, 'ts': now}
            if is_overspeed:
                self._save_overspeed_alert(d, frame)

    def _save_event(self, det):
        try:
            conn = get_db()
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conn.execute(
                '''INSERT INTO anpr_events
                   (camera_id, timestamp, vehicle_type, plate_number, speed_kmh,
                    is_overspeeding, track_id, confidence)
                   VALUES (?,?,?,?,?,?,?,?)''',
                (
                    self.camera_id, ts,
                    det.get('vehicle_type', ''),
                    det.get('plate_number') or '',
                    det.get('speed_kmh'),
                    1 if det.get('is_overspeeding') else 0,
                    det.get('track_id'),
                    det.get('confidence'),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error('ANPR DB save error (%s): %s', self.camera_id, e)

    def _save_overspeed_alert(self, det, frame):
        try:
            from services.alerts.service import save_alert_event

            x1, y1, x2, y2 = det['bbox']
            label = f"Overspeed {det.get('speed_kmh', 0):.0f} km/h"
            if det.get('plate_number'):
                label += f" | {det['plate_number']}"
            dets = [(det['bbox'], label, (0, 0, 255))]
            save_alert_event(
                self.camera_id, 'anpr', frame, dets, severity='high',
                meta={
                    'vehicle_type': det.get('vehicle_type'),
                    'plate_number': det.get('plate_number'),
                    'speed_kmh': det.get('speed_kmh'),
                    'track_id': det.get('track_id'),
                },
            )
        except Exception as e:
            logger.debug('ANPR alert save error: %s', e)
