"""AI analytics runtime: tracking, inference engine, processors, VLM, FR (Phase 1b)."""
import asyncio
import colorsys
import csv
import gc
import hashlib
import inspect
import io
import json
import logging
import math
import os
import pickle
import re
import socket
import subprocess
import threading
import time
import uuid
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.parse import urlsplit

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from flask import jsonify
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from backend.db import get_db
from backend.runtime import (
    beta_procs,
    video_readers,
    entryexit_procs,
    fire_smoke_procs,
    flapgate_procs,
    fr_procs,
    headcount_procs,
    ppe_procs,
)
from config import (
    BASE_DIR,
    CALLBACK_POOL_WORKERS,
    DATA_DIR,
    ENTRYEXIT_MODEL_PATH,
    FIRE_SMOKE_MODEL_PATH,
    FLAPGATE_MODEL_PATH,
    FLORENCE2_LOCAL_PATH,
    FR_ENABLE_GENDER_AGE,
    FR_ORT_THREADS,
    FR_TARGET_FPS,
    HEADCOUNT_MODEL_PATH,
    INFERENCE_IMGSZ,
    INFERENCE_MAX_INFLIGHT_FACTOR,
    MICRO_BATCH,
    OWLV2_LOCAL_PATH,
    OWLV2_MODEL_ID,
    PPE_MODEL_PATH,
    QWEN25VL_LOCAL_PATH,
    TARGET_FPS,
    USE_FP16,
    USE_SAHI_PPE,
    USE_TENSORRT_IF_AVAILABLE,
    VISION_FLORENCE_RULE_ALERT_COOLDOWN_SEC,
    VISION_QWEN_ALLOW_GPU_WITH_BETA,
    VISION_QWEN_DEVICE,
    VISION_QWEN_GPU_MIN_FREE_GB,
    VISION_QWEN_IMAGE_MAX_SIDE,
    VISION_QWEN_MAX_PIXELS,
    VISION_QWEN_MIN_PIXELS,
    VISION_QWEN_UNLOAD_AFTER_USE,
    VISION_QWEN_USE_4BIT,
    VISION_VLM_DEVICE,
    VISION_VLM_EVICT_OTHERS,
    resolve_pt_model,
)
from backend.alerts import save_alert_event as _save_alert_event

logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    psutil = None
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_ok = True
except Exception:
    _nvml_ok = False
try:
    import torch
    _torch_cuda_available = torch.cuda.is_available()
except Exception:
    torch = None
    _torch_cuda_available = False

warnings.filterwarnings('ignore')

# Optional deps (remain in app.py until ai_analytics service extraction)
try:
    from sahi.predict import get_sliced_prediction
    from sahi import AutoDetectionModel
    _SAHI_AVAILABLE = True
except Exception:
    get_sliced_prediction = None

# Optional: OWLv2 VLM for Beta (open-vocabulary) detection
_OWLV2_AVAILABLE = False
try:
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from PIL import Image
    _OWLV2_AVAILABLE = True
except Exception:
    Owlv2Processor = None
    Owlv2ForObjectDetection = None
    Image = None
    AutoDetectionModel = None
    _SAHI_AVAILABLE = False

# Optional: Florence-2 for Beta (phrase grounding with user-provided label list)
_FLORENCE2_AVAILABLE = False
_FlorenceProcessorClass = None
_FlorenceModelClass = None
try:
    from transformers import AutoProcessor as _FlorenceProcessorClass
    try:
        from transformers import Florence2ForConditionalGeneration as _FlorenceModelClass
    except ImportError:
        from transformers import AutoModelForCausalLM as _FlorenceModelClass
    _FLORENCE2_AVAILABLE = True
except Exception:
    _FlorenceProcessorClass = None
    _FlorenceModelClass = None

# Optional: Qwen vision-language model for incident investigation
_QWEN25VL_AVAILABLE = False
_QwenProcessorClass = None
_QwenModelClass = None
try:
    from transformers import AutoProcessor as _QwenProcessorClass
    try:
        from transformers import AutoModelForImageTextToText as _QwenModelClass
    except Exception:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as _QwenModelClass
        except Exception:
            from transformers import AutoModelForVision2Seq as _QwenModelClass
    _QWEN25VL_AVAILABLE = True
except Exception:
    _QwenProcessorClass = None
    _QwenModelClass = None

try:
    import edge_tts as _edge_tts_mod
    _EDGE_TTS_AVAILABLE = True
except ImportError:
    _edge_tts_mod = None
    _EDGE_TTS_AVAILABLE = False

try:
    from pypdf import PdfReader as _PdfReaderClass
    _PYPDF_AVAILABLE = True
except ImportError:
    _PdfReaderClass = None
    _PYPDF_AVAILABLE = False

if Image is None:
    try:
        from PIL import Image as _PILImage
        Image = _PILImage
    except ImportError:
        Image = None

_QWEN25VL_MODEL_TYPES = frozenset({'qwen2_5_vl', 'qwen2_vl'})


def _is_qwen25vl_model_dir(model_dir):
    """True only for Qwen2.x-VL checkpoints (excludes Qwen3.5 text/VLM variants)."""
    cfg_path = os.path.join(model_dir, 'config.json')
    if not os.path.isfile(cfg_path):
        return False
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception:
        return False
    model_type = (cfg.get('model_type') or '').strip().lower()
    if model_type in _QWEN25VL_MODEL_TYPES:
        return True
    arch = cfg.get('architectures') or []
    arch_s = ' '.join(str(a) for a in arch).lower()
    return 'qwen2_5_vl' in arch_s or 'qwen2_vl' in arch_s


def _qwen25vl_local_dir():
    candidates = []
    if QWEN25VL_LOCAL_PATH:
        p = QWEN25VL_LOCAL_PATH
        if os.path.isfile(p):
            p = os.path.dirname(p)
        candidates.append(p)
    candidates.extend([
        os.path.join(BASE_DIR, '.hf_qwen25vl_cache', 'Qwen2.5-VL-3B-Instruct'),
        os.path.join(BASE_DIR, 'hf_qwen25vl_cache', 'Qwen2.5-VL-3B-Instruct'),
        os.path.join(BASE_DIR, '.hf_qwen25vl_cache', 'Qwen2.5-VL'),
        os.path.join(BASE_DIR, 'hf_qwen25vl_cache', 'Qwen2.5-VL'),
        os.path.join(BASE_DIR, '.hf_qwen25vl_cache'),
        os.path.join(BASE_DIR, 'hf_qwen25vl_cache'),
        os.path.join(
            BASE_DIR, 'hf_qwen2vl_3b_4bit', 'hf_qwen2vl_3b_4bit',
            'Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit',
        ),
    ])
    for d in candidates:
        if d and _is_qwen25vl_model_dir(d):
            return d
    return None


def _qwen25vl_ready():
    return bool(_QWEN25VL_AVAILABLE and _QwenProcessorClass and _QwenModelClass and _qwen25vl_local_dir() and Image is not None)


_vlm_global_lock = threading.Lock()
_qwen25vl_processor = None
_qwen25vl_model = None
_qwen25vl_lock = threading.Lock()
_qwen25vl_infer_lock = threading.Lock()
_qwen25vl_device = None  # 'cuda' | 'cpu' used for last load


def _torch_free_cuda_memory():
    try:
        gc.collect()
        if torch is not None and _torch_cuda_available:
            torch.cuda.empty_cache()
    except Exception:
        pass


def _beta_vlm_camera_count():
    """How many beta (OWLv2/Florence) streams are active — used to avoid GPU Qwen colliding with beta."""
    bp = globals().get('beta_procs')
    if not isinstance(bp, dict):
        return 0
    return len(bp)


def _vlm_beta_device():
    d = (VISION_VLM_DEVICE or "auto").strip().lower()
    if d in ("auto", "", "cuda"):
        d = "cuda"
    if d == "cuda" and not _torch_cuda_available:
        return "cpu"
    return d


def _qwen_resolve_device():
    """Resolve Qwen device (default cuda). Can fall back to CPU if CUDA unavailable."""
    raw_cfg = os.environ.get('VISION_QWEN_DEVICE', VISION_QWEN_DEVICE).strip().lower()
    if raw_cfg in ('cuda', 'auto', ''):
        raw = 'cuda'
    elif raw_cfg == 'cpu':
        raw = 'cpu'
    else:
        raw = 'cuda' if _torch_cuda_available else 'cpu'
    if raw == 'cuda' and not _torch_cuda_available:
        return 'cpu'
    if raw == 'cuda' and torch is not None and _torch_cuda_available:
        try:
            free_b, _total_b = torch.cuda.mem_get_info()
            free_gb = float(free_b) / (1024 ** 3)
            if free_gb < VISION_QWEN_GPU_MIN_FREE_GB:
                logger.warning(
                    f"Qwen vision: free VRAM {free_gb:.2f}GB < {VISION_QWEN_GPU_MIN_FREE_GB:.2f}GB; using CPU for safety."
                )
                return 'cpu'
        except Exception:
            pass
    if raw == 'cuda' and _beta_vlm_camera_count() > 0 and not VISION_QWEN_ALLOW_GPU_WITH_BETA:
        logger.warning(
            'Qwen2.5-VL: GPU requested but Beta VLM is active; using CPU. '
            'Set VISION_QWEN_ALLOW_GPU_WITH_BETA=1 to allow GPU (may OOM with YOLO).'
        )
        return 'cpu'
    return raw


def _unload_qwen25vl():
    global _qwen25vl_processor, _qwen25vl_model, _qwen25vl_device
    with _qwen25vl_infer_lock:
        with _qwen25vl_lock:
            try:
                if _qwen25vl_model is not None:
                    del _qwen25vl_model
            except Exception:
                pass
            _qwen25vl_model = None
            _qwen25vl_processor = None
            _qwen25vl_device = None
    _torch_free_cuda_memory()


def _unload_owlv2():
    global _owlv2_processor, _owlv2_model
    with _owlv2_infer_lock:
        with _owlv2_lock:
            try:
                if _owlv2_model is not None:
                    del _owlv2_model
            except Exception:
                pass
            _owlv2_model = None
            _owlv2_processor = None
    _torch_free_cuda_memory()


def _unload_florence2():
    global _florence2_processor, _florence2_model
    with _florence2_infer_lock:
        with _florence2_lock:
            try:
                if _florence2_model is not None:
                    del _florence2_model
            except Exception:
                pass
            _florence2_model = None
            _florence2_processor = None
    _torch_free_cuda_memory()


def _evict_vlms_except(keep):
    """Unload other VLMs to reduce VRAM before loading `keep` (owlv2 | florence2 | qwen)."""
    if not VISION_VLM_EVICT_OTHERS:
        return
    with _vlm_global_lock:
        if keep != 'owlv2':
            _unload_owlv2()
        if keep != 'florence2':
            _unload_florence2()
        if keep != 'qwen':
            _unload_qwen25vl()


def _get_qwen25vl():
    global _qwen25vl_processor, _qwen25vl_model, _qwen25vl_device
    if not _qwen25vl_ready():
        return None, None
    dev = _qwen_resolve_device()
    with _qwen25vl_lock:
        need_reload = _qwen25vl_processor is None or _qwen25vl_device != dev
    if need_reload and _qwen25vl_processor is not None:
        # Must not hold _qwen25vl_lock while unloading (unload takes the same locks).
        _unload_qwen25vl()
    # Evict other VLMs before taking Qwen locks (avoids lock-order inversion with _vlm_global_lock).
    if dev == 'cuda' and VISION_VLM_EVICT_OTHERS:
        _evict_vlms_except('qwen')
    with _qwen25vl_lock:
        if _qwen25vl_processor is None:
            local_dir = _qwen25vl_local_dir()
            if not local_dir:
                return None, None
            try:
                _qwen25vl_processor = _QwenProcessorClass.from_pretrained(
                    local_dir,
                    local_files_only=True,
                    trust_remote_code=True,
                    min_pixels=VISION_QWEN_MIN_PIXELS,
                    max_pixels=VISION_QWEN_MAX_PIXELS,
                )
                load_kw = {'local_files_only': True, 'trust_remote_code': True}
                if torch is not None:
                    if dev == 'cuda' and VISION_QWEN_USE_4BIT:
                        load_kw['load_in_4bit'] = True
                        load_kw['device_map'] = 'auto'
                    else:
                        load_kw['dtype'] = torch.float16 if dev == 'cuda' else torch.float32
                load_kw['low_cpu_mem_usage'] = True
                _qwen25vl_model = _QwenModelClass.from_pretrained(local_dir, **load_kw)
                if torch is not None and not (dev == 'cuda' and VISION_QWEN_USE_4BIT):
                    _qwen25vl_model = _qwen25vl_model.to(dev)
                try:
                    _qwen25vl_model.eval()
                except Exception:
                    pass
                _qwen25vl_device = dev
                logger.info(f"Qwen vision model loaded ({dev}) from {local_dir}")
            except Exception as e:
                logger.error(f"Qwen2.5-VL load failed: {e}")
                _qwen25vl_processor = None
                _qwen25vl_model = None
                _qwen25vl_device = None
        return _qwen25vl_processor, _qwen25vl_model


def maybe_unload_qwen_after_incident():
    """Free VRAM/RAM after incident investigation when VISION_QWEN_UNLOAD_AFTER_USE=1."""
    if VISION_QWEN_UNLOAD_AFTER_USE:
        _unload_qwen25vl()


def _maybe_unload_beta_vlms_if_idle():
    """If no beta processors are active, release OWLv2 and Florence models."""
    try:
        if _beta_vlm_camera_count() == 0:
            _unload_owlv2()
            _unload_florence2()
    except Exception:
        pass


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def nms_detections(detections, scores, iou_thresh=0.35):
    """Greedy NMS to suppress duplicate bounding boxes before tracking.
    Keeps the highest-confidence box when two boxes overlap above iou_thresh.
    detections: list of [x1,y1,x2,y2], scores: list of float confidences.
    Returns filtered list of [x1,y1,x2,y2]."""
    if not detections:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(detections[i])
        for j in order:
            if j != i and j not in suppressed:
                if compute_iou(detections[i], detections[j]) >= iou_thresh:
                    suppressed.add(j)
    return keep


class KalmanTracker:
    _id_counter = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        dt = 1.0
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        cx = x1 + w / 2.
        cy = y1 + h / 2.
        s = w * h
        r = w / h
        self.kf.x[:4] = np.array([cx, cy, s, r]).reshape((4, 1))

        self.time_since_update = 0
        self.id = KalmanTracker._id_counter
        KalmanTracker._id_counter += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.last_valid_bbox = bbox
        self.position_history = deque(maxlen=30)
        self.position_history.append((cx, cy))

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        cx = x1 + w / 2.
        cy = y1 + h / 2.
        s = w * h
        r = w / h
        self.kf.update(np.array([cx, cy, s, r]).reshape((4, 1)))
        self.last_valid_bbox = bbox
        self.position_history.append((cx, cy))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.kf.x

    def get_state(self):
        try:
            x, y, s, r = self.kf.x[:4].flatten()
            if any(math.isnan(v) for v in [x, y, s, r]):
                return self.last_valid_bbox
            w = np.sqrt(abs(s) * r)
            h = abs(s) / w
            return [max(0, int(x - w / 2)), max(0, int(y - h / 2)), int(x + w / 2), int(y + h / 2)]
        except:
            return self.last_valid_bbox


class MultiObjectTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.next_id = 1
        self.free_ids = set()
        self.active_ids = set()

    def _get_next_id(self):
        if self.free_ids:
            nid = min(self.free_ids)
            self.free_ids.remove(nid)
            return nid
        nid = self.next_id
        self.next_id += 1
        return nid

    def _release_id(self, tid):
        if tid < self.next_id:
            self.free_ids.add(tid)
        self.active_ids.discard(tid)

    def update(self, detections):
        self.frame_count += 1
        for t in self.trackers:
            t.predict()

        matches, unmatched_dets, unmatched_trks = self._associate(detections)

        for d_idx, t_idx in matches:
            self.trackers[t_idx].update(detections[d_idx])

        for d_idx in unmatched_dets:
            new_id = self._get_next_id()
            tracker = KalmanTracker(detections[d_idx])
            tracker.id = new_id
            self.trackers.append(tracker)
            self.active_ids.add(new_id)

        dead = [i for i, t in enumerate(self.trackers) if t.time_since_update > self.max_age]
        for i in sorted(dead, reverse=True):
            self._release_id(self.trackers[i].id)
            self.trackers.pop(i)

        self.active_ids.clear()
        active = []
        for t in self.trackers:
            if t.time_since_update < 2:
                self.active_ids.add(t.id)
                bbox = t.get_state()
                active.append({
                    'id': t.id,
                    'bbox': bbox,
                    'center': ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
                    'foot': ((bbox[0] + bbox[2]) // 2, bbox[3])
                })
        return active, self.active_ids

    def _associate(self, detections):
        if not self.trackers:
            return [], list(range(len(detections))), []
        if not detections:
            return [], [], list(range(len(self.trackers)))

        iou_matrix = np.zeros((len(detections), len(self.trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(self.trackers):
                iou_matrix[d, t] = compute_iou(det, trk.get_state())

        try:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matches, um_d, um_t = [], list(range(len(detections))), list(range(len(self.trackers)))
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matches.append((r, c))
                    if r in um_d:
                        um_d.remove(r)
                    if c in um_t:
                        um_t.remove(c)
            return matches, um_d, um_t
        except:
            return [], list(range(len(detections))), list(range(len(self.trackers)))


# ==============================================================
# HEAD COUNT TRACKER — high-accuracy, persistent-ID tracker
# ==============================================================

class HeadCountTracker:
    """Specialized tracker for head-count accuracy in confined spaces.

    Key improvements over the generic MultiObjectTracker:
    - IDs are NEVER recycled: once assigned, an ID is permanent for the
      lifetime of the processor, so the same person always keeps the same ID.
    - Two-stage association: first match to recently updated tracks (strict IoU);
      then match remaining detections to "lost" tracks (recovery) by predicted
      position so bending/walking re-associates to the same ID instead of a new one.
    - When the track has high predicted velocity (person moving), stage-1 IoU
      threshold is relaxed so the same person keeps the same ID while moving.
    - Longer max_age (90 frames) keeps tracks alive through brief occlusions.
    """

    # Recovery: max centre distance (in multiples of mean box diagonal) to re-match
    # a detection to a track that was lost for a few frames (bending/walking).
    RECOVERY_DIST_MULT = 4.0
    # Velocity (pixels/frame) above which we relax IoU for stage-1 matching.
    # Keep conservative: too-loose matching causes ID switches in crowds.
    VELOCITY_RELAX_THRESH = 8.0
    IOU_RELAXED = 0.25   # used when track is moving

    def __init__(self, max_age=120, min_hits=2, iou_threshold=0.4):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self._next_id = 1          # monotonically increasing — never reused

    def _new_id(self):
        nid = self._next_id
        self._next_id += 1
        return nid

    def update(self, detections):
        """detections: list of [x1,y1,x2,y2] (already NMS-filtered).
        Returns (active_list, active_id_set)."""
        self.frame_count += 1

        for t in self.trackers:
            t.predict()

        matches, unmatched_dets, _ = self._associate(detections)

        for d_idx, t_idx in matches:
            self.trackers[t_idx].update(detections[d_idx])

        for d_idx in unmatched_dets:
            tracker = KalmanTracker(detections[d_idx])
            tracker.id = self._new_id()
            self.trackers.append(tracker)

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        active = []
        active_ids = set()
        recent_thresh = 2
        for t in self.trackers:
            if t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                if t.time_since_update < recent_thresh:
                    bbox = t.get_state()
                    active.append({
                        'id': t.id,
                        'bbox': bbox,
                        'center': ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
                        'foot': ((bbox[0] + bbox[2]) // 2, bbox[3])
                    })
                    active_ids.add(t.id)

        return active, active_ids

    def _associate(self, detections):
        if not self.trackers:
            return [], list(range(len(detections))), []
        if not detections:
            return [], [], list(range(len(self.trackers)))

        n_det = len(detections)
        n_trk = len(self.trackers)
        confirmed_idx = [i for i in range(n_trk) if self.trackers[i].time_since_update < 2]
        recovery_idx = [i for i in range(n_trk) if 2 <= self.trackers[i].time_since_update <= self.max_age]

        # ----- Stage 1: match detections to confirmed (recently updated) tracks -----
        # Use IoU + centre distance; relax IoU when track has high velocity (moving person).
        cost1 = np.full((n_det, n_trk), -1e9)
        iou_required = {}
        for t in range(n_trk):
            trk = self.trackers[t]
            vx = float(trk.kf.x[4])
            vy = float(trk.kf.x[5])
            vel = math.sqrt(vx * vx + vy * vy)
            iou_required[t] = self.IOU_RELAXED if vel > self.VELOCITY_RELAX_THRESH else self.iou_threshold

        for d, det in enumerate(detections):
            dcx = (det[0] + det[2]) / 2.0
            dcy = (det[1] + det[3]) / 2.0
            dw = max(det[2] - det[0], 1)
            dh = max(det[3] - det[1], 1)
            for t in confirmed_idx:
                trk = self.trackers[t]
                tb = trk.get_state()
                iou_val = compute_iou(det, tb)
                tcx = (tb[0] + tb[2]) / 2.0
                tcy = (tb[1] + tb[3]) / 2.0
                norm = (math.sqrt(dw * dw + dh * dh) + math.sqrt(max(tb[2] - tb[0], 1) ** 2 + max(tb[3] - tb[1], 1) ** 2)) / 2.0 + 1e-6
                dist_sim = max(0.0, 1.0 - math.sqrt((dcx - tcx) ** 2 + (dcy - tcy) ** 2) / norm)
                cost1[d, t] = 0.7 * iou_val + 0.3 * dist_sim

        matched_d, matched_t = set(), set()
        matches = []
        if confirmed_idx:
            try:
                row_ind, col_ind = linear_sum_assignment(-cost1)
                for r, c in zip(row_ind, col_ind):
                    if c not in confirmed_idx:
                        continue
                    det = detections[r]
                    tb = self.trackers[c].get_state()
                    iou_val = compute_iou(det, tb)

                    # Anti-ID-switch gating: avoid an existing ID jumping to a different person.
                    da = max((det[2] - det[0]) * (det[3] - det[1]), 1)
                    ta = max((tb[2] - tb[0]) * (tb[3] - tb[1]), 1)
                    area_ratio = max(da, ta) / max(1, min(da, ta))
                    if area_ratio > 2.5:
                        continue

                    dcx = (det[0] + det[2]) / 2.0
                    dcy = (det[1] + det[3]) / 2.0
                    tcx = (tb[0] + tb[2]) / 2.0
                    tcy = (tb[1] + tb[3]) / 2.0
                    diag = math.sqrt(max(det[2] - det[0], 1) ** 2 + max(det[3] - det[1], 1) ** 2) + 1e-6
                    if math.sqrt((dcx - tcx) ** 2 + (dcy - tcy) ** 2) > 1.2 * diag:
                        continue

                    if iou_val >= iou_required[c] and cost1[r, c] >= max(0.35, iou_required[c] * 0.8):
                        matches.append((r, c))
                        matched_d.add(r)
                        matched_t.add(c)
            except Exception:
                pass

        unmatched_dets = [i for i in range(n_det) if i not in matched_d]
        unmatched_trks_recovery = [i for i in recovery_idx if i not in matched_t]

        # ----- Stage 2 (recovery): match unmatched detections to "lost" tracks by position -----
        # When person bends/walks we can lose them for a few frames; re-associate by predicted centre.
        if unmatched_dets and unmatched_trks_recovery:
            cost2 = np.zeros((len(unmatched_dets), len(unmatched_trks_recovery)))
            recovery_radius = []
            for i, t in enumerate(unmatched_trks_recovery):
                trk = self.trackers[t]
                tb = trk.get_state()
                tw = max(tb[2] - tb[0], 1)
                th = max(tb[3] - tb[1], 1)
                recovery_radius.append((i, t, math.sqrt(tw * tw + th * th)))
            for j, d in enumerate(unmatched_dets):
                det = detections[d]
                dcx = (det[0] + det[2]) / 2.0
                dcy = (det[1] + det[3]) / 2.0
                dw = max(det[2] - det[0], 1)
                dh = max(det[3] - det[1], 1)
                det_diag = math.sqrt(dw * dw + dh * dh)
                for i, (_, t, trk_diag) in enumerate(recovery_radius):
                    trk = self.trackers[t]
                    tb = trk.get_state()
                    tcx = (tb[0] + tb[2]) / 2.0
                    tcy = (tb[1] + tb[3]) / 2.0
                    dist = math.sqrt((dcx - tcx) ** 2 + (dcy - tcy) ** 2)
                    mean_diag = (det_diag + trk_diag) / 2.0 + 1e-6
                    if dist <= self.RECOVERY_DIST_MULT * mean_diag:
                        cost2[j, i] = 1.0 / (1.0 + dist / mean_diag)
                    else:
                        cost2[j, i] = 0.0
            try:
                row_ind2, col_ind2 = linear_sum_assignment(-cost2)
                for r2, c2 in zip(row_ind2, col_ind2):
                    if cost2[r2, c2] <= 0:
                        continue
                    d_idx = unmatched_dets[r2]
                    _, t_idx, _ = recovery_radius[c2]
                    if t_idx in matched_t:
                        continue
                    matches.append((d_idx, t_idx))
                    matched_d.add(d_idx)
                    matched_t.add(t_idx)
            except Exception:
                pass

        unmatched_dets_final = [i for i in range(n_det) if i not in matched_d]
        unmatched_trks_final = [i for i in range(n_trk) if i not in matched_t]
        return matches, unmatched_dets_final, unmatched_trks_final


# ==============================================================
# GEOMETRY UTILITIES
# ==============================================================

def point_in_polygon(point, polygon):
    """Ray-casting algorithm for point-in-polygon test."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ==============================================================
# PRODUCTION INFERENCE INFRASTRUCTURE
# ==============================================================

def _export_pt_to_tensorrt_engine(pt_path, engine_path):
    """Export .pt to TensorRT .engine using app's INFERENCE_IMGSZ and USE_FP16. No accuracy change (FP16).
    Ultralytics writes .engine alongside .pt by default (same dir, same base name)."""
    export_cwd = os.path.join(DATA_DIR, '.runtime_tmp', 'trt_export')
    os.makedirs(export_cwd, exist_ok=True)
    prev_cwd = os.getcwd()
    try:
        os.chdir(export_cwd)
        logger.info(f"ModelRegistry: Exporting {pt_path} to TensorRT (imgsz={INFERENCE_IMGSZ}, fp16={USE_FP16}). This may take 2-5 min per model ...")
        m = YOLO(pt_path)
        # Ultralytics: half= is deprecated → quantize=16 (FP16) / 32 (FP32)
        kwargs = {"format": "engine", "device": 0, "quantize": (16 if USE_FP16 else 32), "imgsz": INFERENCE_IMGSZ, "verbose": False}
        try:
            m.export(**{**kwargs, "dynamo": False})
        except TypeError as te:
            if "dynamo" in str(te):
                m.export(**kwargs)
            else:
                raise
        if os.path.isfile(engine_path):
            return True
        base = os.path.splitext(os.path.basename(pt_path))[0]
        for d in [os.path.dirname(os.path.abspath(pt_path)), export_cwd, prev_cwd]:
            alt = os.path.join(d, base + ".engine")
            if os.path.isfile(alt):
                if os.path.abspath(alt) != os.path.abspath(engine_path):
                    import shutil
                    shutil.copy2(alt, engine_path)
                return True
        return False
    except Exception as e:
        err_msg = str(e)
        if "dynamo" in err_msg:
            logger.warning(
                "ModelRegistry: TensorRT export failed (dynamo API mismatch). Using PyTorch .pt. "
                "To avoid this, set VISION_USE_TENSORRT=0 or pin ultralytics to a compatible version."
            )
        else:
            logger.warning(f"ModelRegistry: TensorRT export failed ({e}). Using PyTorch .pt (no accuracy change).")
        return False
    finally:
        try:
            os.chdir(prev_cwd)
        except OSError:
            pass


class ModelRegistry:
    """Loads each YOLO model exactly once; returns the shared instance + its lock.
    With USE_TENSORRT_IF_AVAILABLE: uses .engine when present, or auto-exports .pt to .engine
    (same imgsz/FP16 as inference). No compromise on accuracy or detection."""
    _models = {}
    _locks = {}
    _init_lock = threading.Lock()

    @staticmethod
    def _is_trt_runtime_mismatch(err):
        """Detect common TensorRT engine/runtime incompatibility signatures."""
        s = str(err).lower()
        markers = (
            "tensorrt model exported with a different version",
            "serialization assertion safeversionread",
            "serialized engine version",
            "create_execution_context",
            "deserialize_cuda_engine",
            "version tag does not match",
        )
        return any(m in s for m in markers)

    @staticmethod
    def _probe_model_runtime(model):
        """Run one tiny dry-run to surface lazy TensorRT backend failures early."""
        side = max(160, int(INFERENCE_IMGSZ))
        probe = np.zeros((side, side, 3), dtype=np.uint8)
        model([probe], conf=0.01, verbose=False, imgsz=side, quantize=(16 if USE_FP16 else 32), max_det=1)

    @classmethod
    def get(cls, model_path):
        if model_path not in cls._models:
            with cls._init_lock:
                if model_path not in cls._models:
                    load_path = model_path
                    engine_path = None
                    if USE_TENSORRT_IF_AVAILABLE and model_path.lower().endswith('.pt') and os.path.isfile(model_path):
                        engine_path = model_path[:-3] + '.engine'
                        if os.path.isfile(engine_path):
                            load_path = engine_path
                            logger.info(f"ModelRegistry: Using TensorRT engine {engine_path}")
                        else:
                            if _export_pt_to_tensorrt_engine(model_path, engine_path):
                                load_path = engine_path
                                logger.info(f"ModelRegistry: TensorRT engine ready {engine_path}")
                    if load_path == model_path:
                        logger.info(f"ModelRegistry: Loading {model_path} ...")

                    loaded_model = YOLO(load_path)
                    if load_path.lower().endswith('.engine'):
                        try:
                            cls._probe_model_runtime(loaded_model)
                        except Exception as e:
                            if cls._is_trt_runtime_mismatch(e):
                                logger.warning(
                                    "ModelRegistry: TensorRT engine runtime mismatch for %s (%s). "
                                    "Falling back to .pt and regenerating engine if possible.",
                                    load_path, e
                                )
                                if engine_path and os.path.isfile(engine_path):
                                    try:
                                        os.remove(engine_path)
                                        logger.warning("ModelRegistry: Removed incompatible engine %s", engine_path)
                                    except Exception as rm_e:
                                        logger.warning("ModelRegistry: Could not remove %s (%s)", engine_path, rm_e)

                                # Best-effort rebuild with current runtime, then retry once.
                                reloaded = False
                                if engine_path and _export_pt_to_tensorrt_engine(model_path, engine_path) and os.path.isfile(engine_path):
                                    try:
                                        loaded_model = YOLO(engine_path)
                                        cls._probe_model_runtime(loaded_model)
                                        load_path = engine_path
                                        reloaded = True
                                        logger.info("ModelRegistry: Rebuilt TensorRT engine %s", engine_path)
                                    except Exception as rebuild_e:
                                        logger.warning("ModelRegistry: Rebuilt engine failed (%s). Using .pt.", rebuild_e)

                                if not reloaded:
                                    load_path = model_path
                                    loaded_model = YOLO(model_path)
                            else:
                                raise

                    cls._models[model_path] = loaded_model
                    cls._locks[model_path] = threading.Lock()
                    logger.info(f"ModelRegistry: {load_path} ready")
        return cls._models[model_path], cls._locks[model_path]


class InferenceEngine:
    """Centralized per-detection-type engine.

    One engine per detection type (headcount / entryexit / flapgate).
    Collects the latest frame from every registered camera, runs
    micro-batched YOLO inference through a shared model, and dispatches
    results to per-camera callbacks on a bounded thread pool so the
    inference loop never stalls.
    """

    def __init__(self, model_path, name, target_fps=8, micro_batch=8,
                 imgsz=736, half=False, pool_workers=12, classes=None, nms_iou=0.5):
        self.name = name
        self.model, self.model_lock = ModelRegistry.get(model_path)
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.micro_batch = micro_batch
        self.imgsz = imgsz
        self.half = half
        self.pool_workers = pool_workers
        self.classes = classes  # None = all classes; [0] = persons only
        self.nms_iou = nms_iou  # YOLO NMS IoU threshold; lower = fewer duplicate boxes

        max_inflight = max(4, int(pool_workers * INFERENCE_MAX_INFLIGHT_FACTOR))
        self._inflight_sem = threading.Semaphore(max_inflight)
        self._dropped_frames = 0

        self._streams = {}
        self._streams_lock = threading.Lock()

        self.running = False
        self._thread = None

        self.heartbeat = time.time()
        self.total_processed = 0
        self.errors = 0

    def register(self, cam_id, reader, callback, conf=0.5):
        with self._streams_lock:
            self._streams[cam_id] = {
                'reader': reader,
                'callback': callback,
                'conf': conf,
                'last_t': 0,
                'busy': False,
            }

    def unregister(self, cam_id):
        with self._streams_lock:
            self._streams.pop(cam_id, None)

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"IE-{self.name}")
        self._thread.start()
        logger.info(f"InferenceEngine[{self.name}] started")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def stream_count(self):
        return len(self._streams)

    def _dispatch(self, info, frame, dets, masks):
        try:
            info['callback'](frame, dets, masks)
        except Exception as e:
            logger.error(f"IE[{self.name}] callback error: {e}")
            self.errors += 1
        finally:
            info['busy'] = False
            self._inflight_sem.release()

    def _submit_callback(self, info, frame, dets, masks):
        if not self._inflight_sem.acquire(blocking=False):
            self._dropped_frames += 1
            if self._dropped_frames == 1 or self._dropped_frames % 500 == 0:
                logger.warning(
                    "IE[%s] callback backlog full — dropped %d frame(s); inference continues",
                    self.name, self._dropped_frames,
                )
            return False
        info['busy'] = True
        pool_submit = getattr(self, '_pool_submit', None)
        if pool_submit is None:
            self._inflight_sem.release()
            info['busy'] = False
            return False
        pool_submit(self._dispatch, info, frame, dets, masks)
        return True

    def _run(self):
        try:
            while self.running:
                self.heartbeat = time.time()
                now = time.time()

                with self._streams_lock:
                    snap = [(cid, info) for cid, info in self._streams.items()
                            if now - info['last_t'] >= self.frame_interval]

                pending = []
                for cid, info in snap:
                    frame = info['reader'].get_frame()
                    if frame is not None:
                        pending.append((cid, frame, info))

                if not pending:
                    time.sleep(0.01)
                    continue

                for i in range(0, len(pending), self.micro_batch):
                    mb = pending[i:i + self.micro_batch]
                    frames = [x[1] for x in mb]
                    confs = [x[2]['conf'] for x in mb]
                    batch_conf = min(confs) if confs else 0.5

                    results = None
                    try:
                        with self.model_lock:
                            results = self.model(
                                frames, conf=batch_conf, classes=self.classes,
                                iou=self.nms_iou, verbose=False, max_det=50,
                                imgsz=self.imgsz, quantize=(16 if self.half else 32))
                    except Exception as e:
                        err_str = str(e)
                        if "not equal to max model size" in err_str or "batch size" in err_str.lower():
                            results = []
                            with self.model_lock:
                                for (cid, frame, info) in mb:
                                    r = self.model(
                                        [frame], conf=info['conf'], classes=self.classes,
                                        iou=self.nms_iou, verbose=False, max_det=50,
                                        imgsz=self.imgsz, quantize=(16 if self.half else 32))
                                    results.append(r[0])
                        else:
                            self.errors += 1
                            logger.error(f"IE[{self.name}] inference: {e}")
                            time.sleep(0.05)
                            continue

                    if results is None:
                        continue

                    try:
                        for j, (cid, frame, info) in enumerate(mb):
                            res = results[j]
                            dets = []
                            masks_list = []
                            has_masks = res.masks is not None
                            h, w = frame.shape[:2]

                            if res.boxes is not None:
                                if has_masks:
                                    mdata = res.masks.data.cpu().numpy()
                                for k, box in enumerate(res.boxes):
                                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                                    if box.conf[0].item() < info['conf']:
                                        continue
                                    dets.append([int(x1), int(y1),
                                                 int(x2), int(y2)])
                                    if has_masks:
                                        mr = cv2.resize(mdata[k], (w, h))
                                        masks_list.append(
                                            (mr > 0.5).astype(np.uint8) * 255)

                            third_arg = masks_list if has_masks else res
                            try:
                                info['callback'](frame, dets, third_arg)
                                self.total_processed += 1
                            except Exception as e:
                                logger.error(f"IE[{self.name}] callback error: {e}")
                                self.errors += 1
                            info['last_t'] = now
                    except Exception as e:
                        self.errors += 1
                        logger.error(f"IE[{self.name}] post-inference: {e}")
                        time.sleep(0.05)
                    finally:
                        if results is not None:
                            del results
                        results = None
        finally:
            pass


class CUDACacheSteward:
    """Runs torch.cuda.empty_cache() periodically to curb VRAM creep from PyTorch's allocator cache."""
    _interval = 90
    _running = False
    _thread = None

    @classmethod
    def start(cls):
        if not _torch_cuda_available or cls._running:
            return
        def _loop():
            cls._running = True
            while cls._running:
                time.sleep(cls._interval)
                try:
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        cls._thread = threading.Thread(target=_loop, daemon=True, name="CUDACacheSteward")
        cls._thread.start()
        logger.info("CUDACacheSteward started (empty_cache every %ss)", cls._interval)


_engines = {}
_engine_init_lock = threading.Lock()


def get_inference_engine(name, model_path, target_fps=None, classes=None):
    """Lazy-create one InferenceEngine per detection type. Uses resource-tuning constants if target_fps not given."""
    if target_fps is None:
        target_fps = TARGET_FPS
    if name not in _engines:
        with _engine_init_lock:
            if name not in _engines:
                # Headcount uses a tighter YOLO NMS threshold to reduce duplicate boxes
                # on closely spaced heads before the tracker even sees them.
                nms_iou = 0.35 if name == 'headcount' else 0.5
                # PPE (pharmappenew.pt): smaller batch for stability when used with other streams.
                micro_batch = 8 if name == 'ppe' else MICRO_BATCH
                eng = InferenceEngine(
                    model_path, name,
                    target_fps=target_fps,
                    micro_batch=micro_batch,
                    imgsz=INFERENCE_IMGSZ,
                    half=USE_FP16,
                    pool_workers=CALLBACK_POOL_WORKERS,
                    classes=classes,
                    nms_iou=nms_iou,
                )
                eng.start()
                _engines[name] = eng
                logger.info(f"Created InferenceEngine '{name}' (imgsz={INFERENCE_IMGSZ}, fps={target_fps}, fp16={USE_FP16}, classes={classes}, nms_iou={nms_iou})")
    return _engines[name]


class Watchdog:
    """Monitors InferenceEngine heartbeats, auto-restarts stalled engines."""
    STALE_THRESHOLD = 30

    def __init__(self):
        self._engines = []
        self.running = False
        self._thread = None

    def register(self, engine):
        self._engines.append(engine)

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="Watchdog")
        self._thread.start()
        logger.info("Watchdog started")

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            for eng in self._engines:
                if not eng.running:
                    continue
                age = time.time() - eng.heartbeat
                if age > self.STALE_THRESHOLD and eng.stream_count > 0:
                    logger.warning(
                        f"Watchdog: engine '{eng.name}' heartbeat stale "
                        f"({age:.0f}s), restarting ...")
                    try:
                        eng.stop()
                    except Exception:
                        pass
                    eng.running = False
                    eng.start()
            time.sleep(5)


watchdog = Watchdog()
startup_time = time.time()


# ==============================================================
# CAMERA READER — single RTSP connection per camera
# ==============================================================

class HeadCountProcessor:
    def __init__(self, camera_id, camera_reader, engine, conf=0.1):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.engine = engine
        self.conf = conf
        self.running = False
        self.annotated_frame = None
        self.lock = threading.Lock()

        self.tracker = HeadCountTracker(max_age=100, min_hits=2, iou_threshold=0.45)

        self.total_entries = 0
        self.seen_ids = set()
        self.current_ids = set()
        self.max_id_seen = 0
        self.stats = {'current_count': 0, 'total_entries': 0, 'max_id': 0, 'status': 'Initializing'}
        self._last_detections = []  # [(bbox, label, color), ...] for combined feed overlay
        self._loop_generation = getattr(camera_reader, 'loop_generation', 0)

        self.last_db_save = time.time()
        self.db_interval = 300

    def start(self):
        self.running = True
        self.engine.register(self.camera_id, self.camera_reader,
                             self._on_result, self.conf)

    def stop(self):
        self.running = False
        self.engine.unregister(self.camera_id)

    def _reset_tracking_for_loop(self):
        """File demos rewind to frame 0 — reset tracker so Total/IDs don't inflate forever."""
        self.tracker = HeadCountTracker(max_age=100, min_hits=2, iou_threshold=0.45)
        self.total_entries = 0
        self.seen_ids = set()
        self.current_ids = set()
        self.max_id_seen = 0

    def _on_result(self, frame, detections, masks):
        if not self.running:
            return
        try:
            loop_gen = getattr(self.camera_reader, 'loop_generation', 0)
            if loop_gen != self._loop_generation:
                self._loop_generation = loop_gen
                self._reset_tracking_for_loop()

            # Suppress duplicate/overlapping boxes before tracking.
            # Use box area as a confidence proxy (larger = more complete detection).
            scores = [(d[2] - d[0]) * (d[3] - d[1]) for d in detections]
            detections = nms_detections(detections, scores, iou_thresh=0.35)

            tracked, current_ids = self.tracker.update(detections)

            new_ids = current_ids - self.seen_ids
            if new_ids:
                self.total_entries += len(new_ids)
                self.seen_ids.update(new_ids)
                for nid in new_ids:
                    if nid > self.max_id_seen:
                        self.max_id_seen = nid

            self.current_ids = current_ids
            self.stats = {
                'current_count': len(current_ids),
                'total_entries': self.total_entries,
                'max_id': self.max_id_seen,
                'status': 'Active'
            }

            vis = frame.copy()
            last_dets = []
            for h in tracked:
                x1, y1, x2, y2 = h['bbox']
                tid = h['id']
                color = ((tid * 50) % 256, (tid * 80 + 100) % 256, (tid * 120 + 50) % 256)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"ID:{tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                last_dets.append((tuple(h['bbox']), f"ID:{tid}", color))

            cv2.putText(vis, f"Heads: {len(tracked)}  |  Total: {self.total_entries}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            with self.lock:
                self.annotated_frame = vis
                self._last_detections = last_dets

            now = time.time()
            if now - self.last_db_save >= self.db_interval:
                self._save_to_db()
                self.last_db_save = now

        except Exception as e:
            logger.error(f"HeadCount {self.camera_id}: Error - {e}")

    def _save_to_db(self):
        try:
            conn = get_db()
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conn.execute(
                'INSERT INTO headcounts (camera_id, timestamp, current_count, total_entries) VALUES (?,?,?,?)',
                (self.camera_id, ts, self.stats['current_count'], self.stats['total_entries'])
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"HeadCount DB save error: {e}")

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def get_last_detections(self):
        """Return list of (bbox, label, color) for combined feed overlay."""
        with self.lock:
            return list(self._last_detections)


# ==============================================================
# ENTRY/EXIT PROCESSOR
# ==============================================================

class EntryExitProcessor:
    def __init__(self, camera_id, camera_reader, entry_zone, exit_zone,
                 canvas_size=(800, 450), engine=None, conf=0.5):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.engine = engine
        self.conf = conf
        self.running = False
        self.annotated_frame = None
        self.lock = threading.Lock()

        self.entry_zone_canvas = entry_zone
        self.exit_zone_canvas = exit_zone
        self.canvas_w, self.canvas_h = canvas_size
        self.entry_zone_video = []
        self.exit_zone_video = []
        self._zones_converted = False
        self.entry_zone_mask = None
        self.exit_zone_mask = None

        # Use HeadCountTracker for persistent IDs and recovery. min_hits=1 so multiple people entering/exiting in one frame are all counted.
        self.tracker = HeadCountTracker(max_age=120, min_hits=1, iou_threshold=0.4)

        self.last_zone = {}
        # Separate far-apart doors: count on rising-edge at each door independently.
        self._present_in_entry = set()
        self._present_in_exit = set()
        self._inside_ids = set()
        self._seen_ids = set()
        self.ids_pending_entry = set()
        self.ids_pending_exit = set()
        self.total_entries = 0
        self.total_exits = 0
        self.current_inside = 0
        self.stats = {'entries': 0, 'exits': 0, 'current_inside': 0, 'status': 'Initializing'}

        self.last_db_save = time.time()
        self.db_interval = 300
        self._last_detections = []
        self._loop_generation = getattr(camera_reader, 'loop_generation', 0)

    def _convert_zones(self, frame_w, frame_h):
        self.entry_zone_video = [
            (int(p[0] * frame_w / self.canvas_w), int(p[1] * frame_h / self.canvas_h))
            for p in self.entry_zone_canvas
        ]
        self.exit_zone_video = [
            (int(p[0] * frame_w / self.canvas_w), int(p[1] * frame_h / self.canvas_h))
            for p in self.exit_zone_canvas
        ]
        self.entry_zone_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        if len(self.entry_zone_video) >= 3:
            cv2.fillPoly(self.entry_zone_mask, [np.array(self.entry_zone_video, dtype=np.int32)], 255)
        self.exit_zone_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        if len(self.exit_zone_video) >= 3:
            cv2.fillPoly(self.exit_zone_mask, [np.array(self.exit_zone_video, dtype=np.int32)], 255)
        self._zones_converted = True

    def start(self):
        self.running = True
        self.engine.register(self.camera_id, self.camera_reader,
                             self._on_result, self.conf)

    def stop(self):
        self.running = False
        self.engine.unregister(self.camera_id)

    def _on_result(self, frame, detections, det_masks):
        if not self.running:
            return

        h, w = frame.shape[:2]
        if not self._zones_converted:
            self._convert_zones(w, h)

        try:
            loop_gen = getattr(self.camera_reader, 'loop_generation', 0)
            if loop_gen != self._loop_generation:
                self._loop_generation = loop_gen
                self.tracker = HeadCountTracker(max_age=120, min_hits=1, iou_threshold=0.4)
                self.last_zone = {}
                self._present_in_entry = set()
                self._present_in_exit = set()
                self._inside_ids = set()
                self._seen_ids = set()
                self.ids_pending_entry = set()
                self.ids_pending_exit = set()

            tracked, current_ids = self.tracker.update(detections)

            # Independent doors: ENTRY counts when a person first steps into the entry
            # polygon; EXIT counts when someone already inside first steps into the exit
            # polygon. No handshake between doors (they can be far apart).
            # People first seen in the open yard (neither door) are already inside.
            for person in tracked:
                tid = person['id']
                centroid = person['center']
                in_entry = point_in_polygon(centroid, self.entry_zone_video) if len(self.entry_zone_video) >= 3 else False
                in_exit = point_in_polygon(centroid, self.exit_zone_video) if len(self.exit_zone_video) >= 3 else False

                if tid not in self._seen_ids:
                    self._seen_ids.add(tid)
                    if not in_entry and not in_exit:
                        self._inside_ids.add(tid)

                if in_entry and tid not in self._present_in_entry:
                    if tid not in self._inside_ids:
                        self.total_entries += 1
                        self._inside_ids.add(tid)
                    self.last_zone[tid] = 'entry'
                if in_entry:
                    self._present_in_entry.add(tid)
                else:
                    self._present_in_entry.discard(tid)

                if in_exit and tid not in self._present_in_exit:
                    if tid in self._inside_ids:
                        self.total_exits += 1
                        self._inside_ids.discard(tid)
                    self.last_zone[tid] = 'exit'
                if in_exit:
                    self._present_in_exit.add(tid)
                else:
                    self._present_in_exit.discard(tid)

            self.current_inside = max(0, len(self._inside_ids))

            self.stats = {
                'entries': self.total_entries,
                'exits': self.total_exits,
                'current_inside': self.current_inside,
                'status': 'Active'
            }

            vis = frame.copy()
            self._draw_zones(vis)

            last_dets = []
            for person in tracked:
                x1, y1, x2, y2 = person['bbox']
                tid = person['id']
                centroid = person['center']
                last_z = self.last_zone.get(tid)
                inside = tid in self._inside_ids or last_z == 'entry'
                st_display = 'inside' if inside else 'outside'
                color = (0, 255, 0) if inside else (0, 165, 255)

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"ID:{tid} ({st_display})", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(vis, centroid, 6, (0, 0, 255), -1)
                cv2.circle(vis, centroid, 8, (255, 255, 255), 1)
                last_dets.append((tuple(person['bbox']), f"ID:{tid} ({st_display})", color))

            info_bg = np.zeros((80, 350, 3), dtype=np.uint8)
            vis[5:85, 5:355] = cv2.addWeighted(vis[5:85, 5:355], 0.4, info_bg, 0.6, 0)
            cv2.putText(vis, f"Entries: {self.total_entries}  Exits: {self.total_exits}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, f"Inside: {self.current_inside}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            with self.lock:
                self.annotated_frame = vis
                self._last_detections = last_dets

            now = time.time()
            if now - self.last_db_save >= self.db_interval:
                self._save_to_db()
                self.last_db_save = now

        except Exception as e:
            logger.error(f"EntryExit {self.camera_id}: Error - {e}")

    def _draw_zones(self, frame):
        if len(self.entry_zone_video) >= 3:
            pts = np.array(self.entry_zone_video, dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (255, 150, 0))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.polylines(frame, [pts], True, (255, 150, 0), 2)
            cx = sum(p[0] for p in self.entry_zone_video) // len(self.entry_zone_video)
            cy = sum(p[1] for p in self.entry_zone_video) // len(self.entry_zone_video)
            cv2.putText(frame, "ENTRY", (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)

        if len(self.exit_zone_video) >= 3:
            pts = np.array(self.exit_zone_video, dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
            cx = sum(p[0] for p in self.exit_zone_video) // len(self.exit_zone_video)
            cy = sum(p[1] for p in self.exit_zone_video) // len(self.exit_zone_video)
            cv2.putText(frame, "EXIT", (cx - 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def _save_to_db(self):
        try:
            conn = get_db()
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conn.execute(
                'INSERT INTO entryexit_counts (camera_id, timestamp, entries, exits, current_inside) VALUES (?,?,?,?,?)',
                (self.camera_id, ts, self.total_entries, self.total_exits, self.current_inside)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"EntryExit DB save error: {e}")

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def get_last_detections(self):
        with self.lock:
            return list(self._last_detections)


# ==============================================================
# FLAP GATE TRESPASSING PROCESSOR
# ==============================================================

def _generate_zone_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        colors.append((int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255)))
    return colors

FLAPGATE_ZONE_COLORS = _generate_zone_colors(3)


class FlapGateProcessor:
    def __init__(self, camera_id, camera_reader, gate_zones,
                 canvas_size=(800, 450), engine=None, conf=0.65):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.engine = engine
        self.conf = conf
        self.running = False
        self.annotated_frame = None
        self.lock = threading.Lock()

        self.gate_zones_canvas = gate_zones
        self.canvas_w, self.canvas_h = canvas_size
        self.gate_zones_video = {}
        self._zones_converted = False

        self.person_tracker = {}
        self.next_person_id = 1
        self.zone_occupants = {1: set(), 2: set(), 3: set()}
        self.zone_gate_history = {1: [], 2: [], 3: []}
        self.gate_statuses = {1: 'red', 2: 'red', 3: 'red'}
        self.gate_force_green = {1: False, 2: False, 3: False}
        self.gate_timer_start = {1: 0, 2: 0, 3: 0}
        self.gate_timer_start_ts = {1: 0.0, 2: 0.0, 3: 0.0}
        self.gate_allowed_person = {1: None, 2: None, 3: None}
        self.gate_green_frames = {1: 0, 2: 0, 3: 0}
        self.frame_count = 0
        self.fps_estimate = 25
        self.min_green_seconds = 2.68

        self.total_trespassing = 0
        self.trespassing_per_gate = {1: 0, 2: 0, 3: 0}
        self.stats = {
            'gate_statuses': {'1': 'red', '2': 'red', '3': 'red'},
            'occupants': {'1': 0, '2': 0, '3': 0},
            'trespassing_total': 0,
            'trespassing_per_gate': {'1': 0, '2': 0, '3': 0},
            'persons_in_frame': 0,
            'status': 'Initializing'
        }

        self.last_db_save = time.time()
        self.db_interval = 300
        self._last_detections = []

    def _convert_zones(self, frame_w, frame_h):
        for zone_id, zone_points in self.gate_zones_canvas.items():
            if zone_points and len(zone_points) >= 3:
                self.gate_zones_video[zone_id] = [
                    (int(p[0] * frame_w / self.canvas_w), int(p[1] * frame_h / self.canvas_h))
                    for p in zone_points
                ]
        self._zones_converted = True

    def start(self):
        self.running = True
        self.engine.register(self.camera_id, self.camera_reader,
                             self._on_result, self.conf)

    def stop(self):
        self.running = False
        self.engine.unregister(self.camera_id)

    def _detect_gate_color(self, frame, zone_points):
        if len(zone_points) < 3:
            return 'red'
        height, width = frame.shape[:2]
        zone_mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.array(zone_points, dtype=np.int32)
        cv2.fillPoly(zone_mask, [pts], 255)
        zone_area = cv2.bitwise_and(frame, frame, mask=zone_mask)
        hsv = cv2.cvtColor(zone_area, cv2.COLOR_BGR2HSV)

        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )
        green_pixels = cv2.countNonZero(green_mask)
        red_pixels = cv2.countNonZero(red_mask)

        if green_pixels > 50 and green_pixels > red_pixels:
            return 'green'
        return 'red'

    def _draw_zones(self, frame):
        """Draw configured gate zones on a live frame (used by unified camera feed)."""
        h, w = frame.shape[:2]
        if not self._zones_converted:
            self._convert_zones(w, h)
        for zone_id, zpts in self.gate_zones_video.items():
            if len(zpts) < 3:
                continue
            pts_arr = np.array(zpts, dtype=np.int32)
            zc = FLAPGATE_ZONE_COLORS[zone_id - 1]
            overlay = frame.copy()
            alpha = 0.15 if self.gate_statuses[zone_id] == 'red' else 0.25
            cv2.fillPoly(overlay, [pts_arr], zc)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            border_c = (0, 0, 255) if self.gate_statuses[zone_id] == 'red' else (0, 255, 0)
            cv2.polylines(frame, [pts_arr], True, border_c, 2)
            for pt in zpts:
                cv2.circle(frame, pt, 4, zc, -1)
            cx = sum(p[0] for p in zpts) // len(zpts)
            cy = sum(p[1] for p in zpts) // len(zpts)
            st = f"Gate {zone_id}: {self.gate_statuses[zone_id].upper()}"
            if self.gate_statuses[zone_id] == 'green' and self.gate_force_green[zone_id]:
                elapsed = time.time() - self.gate_timer_start_ts[zone_id]
                remaining = max(0, self.min_green_seconds - elapsed)
                st += f" ({remaining:.1f}s)"
            cv2.putText(frame, st, (cx - 60, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_c, 2)

    def _on_result(self, frame, detections, det_masks):
        if not self.running:
            return

        h, w = frame.shape[:2]
        if not self._zones_converted:
            self._convert_zones(w, h)

        try:
            display = frame.copy()

            for zone_id, zpts in self.gate_zones_video.items():
                if len(zpts) < 3:
                    continue
                raw_status = self._detect_gate_color(frame, zpts)
                self.zone_gate_history[zone_id].append(raw_status)
                if len(self.zone_gate_history[zone_id]) > 10:
                    self.zone_gate_history[zone_id].pop(0)

                if len(self.zone_gate_history[zone_id]) >= 5:
                    recent = self.zone_gate_history[zone_id][-5:]
                    green_count = recent.count('green')

                    if self.gate_force_green[zone_id]:
                        elapsed = time.time() - self.gate_timer_start_ts[zone_id]
                        if elapsed < self.min_green_seconds:
                            self.gate_statuses[zone_id] = 'green'
                        else:
                            self.gate_force_green[zone_id] = False
                            self.gate_allowed_person[zone_id] = None
                            self.gate_statuses[zone_id] = 'green' if green_count >= 3 else 'red'
                    else:
                        self.gate_statuses[zone_id] = 'green' if green_count >= 3 else 'red'
                        if self.gate_statuses[zone_id] == 'green' and self.gate_green_frames[zone_id] == 0:
                            self.gate_timer_start[zone_id] = self.frame_count
                            self.gate_timer_start_ts[zone_id] = time.time()
                            self.gate_force_green[zone_id] = True
                            self.gate_allowed_person[zone_id] = None
                else:
                    self.gate_statuses[zone_id] = raw_status

                if self.gate_statuses[zone_id] == 'green':
                    self.gate_green_frames[zone_id] += 1
                else:
                    self.gate_green_frames[zone_id] = 0
                    self.gate_force_green[zone_id] = False
                    self.gate_allowed_person[zone_id] = None

            self._draw_zones(display)

            current_frame_persons = []
            trespassing_this_frame = set()
            last_dets = []

            # Centroid-based: iterate over detections only (no mask); match persons by bbox IOU
            for i, bbox in enumerate(detections):
                centroid_x = int((bbox[0] + bbox[2]) / 2)
                centroid_y = int((bbox[1] + bbox[3]) / 2)
                centroid = (centroid_x, centroid_y)

                person_id = None
                max_iou = 0.3
                for pid, last_data in self.person_tracker.items():
                    iou = compute_iou(bbox, last_data['bbox'])
                    if iou > max_iou:
                        max_iou = iou
                        person_id = pid

                if person_id is None:
                    person_id = self.next_person_id
                    self.next_person_id += 1

                self.person_tracker[person_id] = {
                    'frame': self.frame_count,
                    'centroid': centroid,
                    'bbox': bbox
                }
                current_frame_persons.append(person_id)

                person_in_zones = []
                for zone_id, zpts in self.gate_zones_video.items():
                    if len(zpts) >= 3:
                        pts_arr = np.array(zpts, dtype=np.int32)
                        if cv2.pointPolygonTest(pts_arr, (float(centroid[0]), float(centroid[1])), False) >= 0:
                            person_in_zones.append(zone_id)
                            self.zone_occupants[zone_id].add(person_id)

                for zone_id in person_in_zones:
                    if self.gate_statuses[zone_id] == 'green':
                        if len(self.zone_occupants[zone_id]) > 1:
                            if self.gate_allowed_person[zone_id] is None:
                                self.gate_allowed_person[zone_id] = person_id
                                cx_p = int((bbox[0] + bbox[2]) / 2)
                                cy_p = int(bbox[1]) - 15
                                cv2.putText(display, f"ALLOWED! Gate {zone_id}",
                                            (max(0, cx_p - 80), max(20, cy_p)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            elif person_id != self.gate_allowed_person[zone_id]:
                                self.total_trespassing += 1
                                self.trespassing_per_gate[zone_id] += 1
                                trespassing_this_frame.add(person_id)
                                cx_p = int((bbox[0] + bbox[2]) / 2)
                                cy_p = int(bbox[1]) - 15
                                cv2.putText(display, f"TRESPASSING! Gate {zone_id}",
                                            (max(0, cx_p - 100), max(20, cy_p)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                cv2.rectangle(display,
                                              (int(bbox[0]), int(bbox[1])),
                                              (int(bbox[2]), int(bbox[3])),
                                              (0, 0, 255), 3)
                    else:
                        self.gate_allowed_person[zone_id] = None

                box_color = (0, 0, 255) if person_id in trespassing_this_frame else (0, 255, 0)
                cv2.rectangle(display, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), box_color, 2)
                cv2.putText(display, f"ID:{person_id}", (centroid[0] - 15, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(display, centroid, 6, (255, 0, 0), -1)
                cv2.circle(display, centroid, 8, (255, 255, 255), 1)
                last_dets.append((tuple(map(int, bbox)), f"ID:{person_id}", box_color))

            for zone_id in self.zone_occupants:
                self.zone_occupants[zone_id] = {
                    pid for pid in self.zone_occupants[zone_id] if pid in current_frame_persons
                }

            info_bg = np.zeros((110, 400, 3), dtype=np.uint8)
            y_start, x_start = 5, 5
            display[y_start:y_start+110, x_start:x_start+400] = cv2.addWeighted(
                display[y_start:y_start+110, x_start:x_start+400], 0.4, info_bg, 0.6, 0
            )
            cv2.putText(display, f"Persons: {len(current_frame_persons)}  Trespassing: {self.total_trespassing}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            y_pos = 50
            for zid in [1, 2, 3]:
                cnt = len(self.zone_occupants[zid])
                st = self.gate_statuses[zid]
                clr = (0, 0, 255) if st == 'red' else (0, 255, 0)
                txt = f"Gate {zid}: {cnt} person(s) [{st.upper()}]"
                if st == 'green' and self.gate_force_green[zid]:
                    elapsed = time.time() - self.gate_timer_start_ts[zid]
                    remaining = max(0, self.min_green_seconds - elapsed)
                    txt += f" ({remaining:.1f}s)"
                cv2.putText(display, txt, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 2)
                y_pos += 22

            self.stats = {
                'gate_statuses': {str(k): v for k, v in self.gate_statuses.items()},
                'occupants': {str(k): len(v) for k, v in self.zone_occupants.items()},
                'trespassing_total': self.total_trespassing,
                'trespassing_per_gate': {str(k): v for k, v in self.trespassing_per_gate.items()},
                'persons_in_frame': len(current_frame_persons),
                'status': 'Active'
            }

            with self.lock:
                self.annotated_frame = display
                self._last_detections = last_dets

            if len(last_dets) > 0 and (len(trespassing_this_frame) > 0 or self.total_trespassing > 0):
                _save_alert_event(
                    self.camera_id, 'flapgate', display, last_dets, severity='high',
                    meta={'trespassing_frame': len(trespassing_this_frame), 'trespassing_total': self.total_trespassing}
                )

            self.frame_count += 1
            if self.frame_count % 10 == 0:
                to_remove = [pid for pid, d in self.person_tracker.items()
                             if self.frame_count - d['frame'] > 30]
                for pid in to_remove:
                    del self.person_tracker[pid]
                    for zid in self.zone_occupants:
                        self.zone_occupants[zid].discard(pid)

            now = time.time()
            if now - self.last_db_save >= self.db_interval:
                self._save_to_db()
                self.last_db_save = now

        except Exception as e:
            logger.error(f"FlapGate {self.camera_id}: Error - {e}")

    def _save_to_db(self):
        try:
            conn = get_db()
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conn.execute(
                'INSERT INTO flapgate_events (camera_id, timestamp, trespassing_count, persons_in_frame, gate1_status, gate2_status, gate3_status) VALUES (?,?,?,?,?,?,?)',
                (self.camera_id, ts, self.total_trespassing, self.stats['persons_in_frame'],
                 self.gate_statuses.get(1, 'red'), self.gate_statuses.get(2, 'red'), self.gate_statuses.get(3, 'red'))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"FlapGate DB save error: {e}")

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def get_last_detections(self):
        with self.lock:
            return list(self._last_detections)


# ==============================================================
# PPE SAHI HELPERS (optional sliced inference for small/far people)
# ==============================================================

def _SAHI_empty_result():
    """Result-like object with no boxes for _on_result when SAHI returns nothing."""
    torch = __import__('torch')
    class _Boxes:
        xyxy = torch.tensor([]).reshape(0, 4)
        cls = torch.tensor([], dtype=torch.long)
        conf = torch.tensor([])
        def __len__(self):
            return 0
    class _R:
        boxes = _Boxes()
        names = {}
    return _R()

def _SAHI_result_from_lists(xyxy_list, cls_list, conf_list, names):
    """Build Result-like object from SAHI prediction lists for _on_result."""
    torch = __import__('torch')
    if not xyxy_list:
        return _SAHI_empty_result()
    class _Boxes:
        xyxy = torch.tensor(xyxy_list, dtype=torch.float32)
        cls = torch.tensor(cls_list, dtype=torch.long)
        conf = torch.tensor(conf_list, dtype=torch.float32)
        def __len__(self):
            return self.xyxy.shape[0]
    names_dict = dict(names) if names else {}
    class _R:
        boxes = _Boxes()
        names = names_dict
    return _R()


# ==============================================================
# PPE DETECTION PROCESSOR (pharmappenew.pt via InferenceEngine or SAHI)
# ==============================================================

def _draw_premium_ppe_box(frame, x1, y1, x2, y2, label, color_bgr):
    """Draw a premium, readable bounding box and label for PPE (enterprise-grade look)."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # Subtle outer border for depth and readability on any background
    border_thick = 2
    shadow = (30, 30, 30)
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), shadow, 1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, border_thick)
    # Label: pill-style background for clarity
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    thickness_text = 2
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness_text)
    pad_x, pad_y = 10, 5
    lx1 = x1
    ly1 = max(y1 - th - 2 * pad_y, 4)
    lx2 = min(x1 + tw + 2 * pad_x, frame.shape[1] - 2)
    ly2 = y1
    # Filled label background (same color as box, slightly transparent look via border)
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color_bgr, -1)
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (255, 255, 255), 1)
    # White text for high contrast and readability
    text_color = (255, 255, 255)
    cv2.putText(frame, label, (x1 + pad_x, y1 - pad_y - 2), font, font_scale, text_color, thickness_text, cv2.LINE_AA)

class PPEDetectionProcessor:
    """Detects PPE compliance (helmets, vests, etc.) using pharmappenew.pt.
    Annotates frames with bounding boxes, class labels and a compliance summary.
    Tracks compliant vs. violation counts and persists to ppe_events table."""

    # Class IDs that indicate a PPE *violation* (no helmet / no hard hat etc.) — drawn in red bbox.
    VIOLATION_KEYWORDS = ('no-helmet', 'no-hardhat', 'no_hard_hat', 'no-gloves',
                          'no-mask', 'no-glasses', 'no-boots', 'no-earmuff',
                          'no_helmet', 'no_hardhat', 'no_gloves',
                          'no_mask', 'no_glasses', 'no_boots', 'no_earmuff',
                          'no hard hat', 'no hardhat',
                          'violation', 'non-compliant', 'noncompliant')
    # Require higher confidence for violations to reduce false positives (e.g. false "No vest").
    VIOLATION_CONF_MIN = 0.1
    # Min bbox area (pixels²) to count as violation; smaller = far/tiny = skip to avoid false no-vest.
    MIN_VIOLATION_AREA = 2500

    def __init__(self, camera_id, camera_reader, engine, conf=0.4):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.engine = engine
        self.conf = conf
        self.running = False
        self.annotated_frame = None
        self.lock = threading.Lock()

        self.compliant_count = 0
        self.violation_count = 0
        self.total_persons = 0
        self.stats = {
            'compliant_count': 0,
            'violation_count': 0,
            'total_persons': 0,
            'status': 'Initializing'
        }

        self.last_db_save = time.time()
        self.db_interval = 300
        self.use_sahi = USE_SAHI_PPE and _SAHI_AVAILABLE
        self._sahi_thread = None
        self._last_detections = []

    def start(self):
        self.running = True
        if self.use_sahi and get_sliced_prediction and AutoDetectionModel:
            self._sahi_thread = threading.Thread(target=self._sahi_loop, daemon=True)
            self._sahi_thread.start()
        else:
            self.engine.register(self.camera_id, self.camera_reader,
                                self._on_result, self.conf)

    def stop(self):
        self.running = False
        if self.use_sahi:
            if self._sahi_thread is not None:
                self._sahi_thread.join(timeout=5.0)
        else:
            self.engine.unregister(self.camera_id)

    def _sahi_loop(self):
        """Run PPE inference via SAHI (sliced) for small/far people when VISION_PPE_USE_SAHI=1."""
        try:
            import torch
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            detection_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path=PPE_MODEL_PATH,
                confidence_threshold=self.conf,
                device=device,
            )
        except Exception as e:
            logger.error(f"PPE SAHI init {self.camera_id}: {e}")
            return
        last_t = 0
        frame_interval = 1.0 / 9.0
        while self.running:
            try:
                now = time.time()
                if now - last_t < frame_interval:
                    time.sleep(0.02)
                    continue
                frame = self.camera_reader.get_frame()
                if frame is None:
                    time.sleep(0.04)
                    continue
                last_t = now
                result = get_sliced_prediction(
                    frame,
                    detection_model,
                    slice_height=736,
                    slice_width=736,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                )
                if not result or not getattr(result, 'object_prediction_list', None):
                    self._on_result(frame, [], _SAHI_empty_result())
                    continue
                preds = result.object_prediction_list
                xyxy_list = []
                cls_list = []
                conf_list = []
                names = {}
                for p in preds:
                    b = p.bbox
                    xyxy_list.append([b.minx, b.miny, b.maxx, b.maxy])
                    cid = getattr(p.category, 'id', 0)
                    cls_list.append(cid)
                    names[cid] = getattr(p.category, 'name', str(cid))
                    conf_list.append(float(p.score.value))
                import torch as _torch
                raw = _SAHI_result_from_lists(xyxy_list, cls_list, conf_list, names)
                dets = [[int(a), int(b), int(c), int(d)] for a, b, c, d in xyxy_list]
                self._on_result(frame, dets, raw)
            except Exception as e:
                logger.error(f"PPE SAHI {self.camera_id}: {e}")
                time.sleep(0.1)

    def _is_violation(self, class_name):
        name_lower = class_name.lower()
        return any(kw in name_lower for kw in self.VIOLATION_KEYWORDS)

    def _is_vest_blue(self, frame, xyxy):
        """Check if the vest region in the bbox is blue/shade of blue (BGR frame). Returns True for Vest, False for No vest."""
        try:
            h_img, w_img = frame.shape[:2]
            x1 = max(0, min(int(xyxy[0]), w_img - 1))
            y1 = max(0, min(int(xyxy[1]), h_img - 1))
            x2 = max(0, min(int(xyxy[2]), w_img))
            y2 = max(0, min(int(xyxy[3]), h_img))
            if x2 <= x1 + 5 or y2 <= y1 + 5:
                return False
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return False
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            # Blue in OpenCV HSV: H ~100-130 (0-180), S and V reasonable
            h, s, v = cv2.split(hsv)
            mask_bright = (v > 30) & (s > 25)
            if not np.any(mask_bright):
                return False
            h_mean = np.mean(h[mask_bright])
            s_mean = np.mean(s[mask_bright])
            # Blue hue range (shades of blue): ~95 to 135
            if 95 <= h_mean <= 135 and s_mean > 25:
                return True
            return False
        except Exception:
            return False

    def _on_result(self, frame, detections, raw_result):
        if not self.running:
            return
        try:
            vis = frame.copy()

            compliant = 0
            violations = 0
            boxes_with_cls = []

            # raw_result is the YOLO Results object (passed by InferenceEngine when no masks).
            if raw_result is not None and hasattr(raw_result, 'boxes') and raw_result.boxes is not None:
                names = raw_result.names if hasattr(raw_result, 'names') else {}
                boxes_data = raw_result.boxes
                for i in range(len(boxes_data)):
                    try:
                        xyxy = boxes_data.xyxy[i].cpu().numpy().astype(int)
                        cls_id = int(boxes_data.cls[i].cpu().numpy())
                        conf_val = float(boxes_data.conf[i].cpu().numpy())
                        cls_name = names.get(cls_id, str(cls_id)) if names else str(cls_id)
                        cn = cls_name.lower()
                        # Omit NO_vest entirely: do not detect or show "No vest".
                        if 'vest' in cn and ('no' in cn or 'no_vest' in cn or 'no-vest' in cn):
                            continue
                        # Vest: only show bbox for blue vests; skip yellow/orange (no bbox).
                        if 'vest' in cn and not self._is_vest_blue(vis, xyxy):
                            continue
                        is_viol_class = self._is_violation(cls_name)
                        if is_viol_class:
                            if conf_val < max(self.conf, self.VIOLATION_CONF_MIN):
                                continue
                            # Skip tiny/far boxes for violations to avoid false positives on small people.
                            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                            if area < self.MIN_VIOLATION_AREA:
                                continue
                        else:
                            if conf_val < self.conf:
                                continue
                        if 'glove' in cn or 'boot' in cn:
                            continue
                        boxes_with_cls.append((xyxy, cls_id, cls_name, conf_val))
                    except Exception:
                        continue
            else:
                # Fallback when raw_result is a masks list (shouldn't happen for PPE but handled safely)
                for bbox in detections:
                    boxes_with_cls.append((bbox, 0, 'detection', 0.0))

            last_dets = []
            for (xyxy, cls_id, cls_name, conf_val) in boxes_with_cls:
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                display_name = cls_name
                # NO hard Hat / no hardhat etc. are violations (red bbox). Vest "No vest" is not shown (filtered above).
                is_viol = self._is_violation(cls_name)
                # Vest (compliant): only show Vest when blue; other vest classes already filtered
                if 'vest' in cls_name.lower():
                    if self._is_vest_blue(vis, xyxy):
                        display_name = 'Vest'
                        is_viol = False
                if is_viol:
                    violations += 1
                    color = (0, 0, 255)   # BGR red for violations
                    label = f"{display_name} {conf_val:.2f}"
                else:
                    compliant += 1
                    color = (0, 180, 90)   # BGR emerald green for compliant
                    label = f"{display_name} {conf_val:.2f}"

                _draw_premium_ppe_box(vis, x1, y1, x2, y2, label, color)
                last_dets.append(((x1, y1, x2, y2), label, color))

            total = compliant + violations
            self.compliant_count = compliant
            self.violation_count = violations
            self.total_persons = total

            self.stats = {
                'compliant_count': compliant,
                'violation_count': violations,
                'total_persons': total,
                'status': 'Active'
            }

            # Premium info strip: dark semi-transparent panel with clear typography
            info_h, info_w = 72, 400
            info_bg = np.zeros((info_h, info_w, 3), dtype=np.uint8)
            info_bg[:] = (28, 28, 32)
            vis[6:6+info_h, 8:8+info_w] = cv2.addWeighted(vis[6:6+info_h, 8:8+info_w], 0.35, info_bg, 0.65, 0)
            cv2.rectangle(vis, (8, 6), (8+info_w, 6+info_h), (70, 70, 78), 1)
            cv2.putText(vis, f"PPE  |  Compliant: {compliant}   Violations: {violations}",
                        (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
            viol_color = (0, 0, 255) if violations > 0 else (0, 200, 100)
            cv2.putText(vis, f"Total: {total}",
                        (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, viol_color, 2, cv2.LINE_AA)

            with self.lock:
                self.annotated_frame = vis
                self._last_detections = last_dets

            if violations > 0:
                _save_alert_event(
                    self.camera_id, 'ppe', vis, last_dets, severity='high',
                    meta={'violations': violations, 'compliant': compliant, 'total': total}
                )

            now = time.time()
            if now - self.last_db_save >= self.db_interval:
                self._save_to_db()
                self.last_db_save = now

        except Exception as e:
            logger.error(f"PPE {self.camera_id}: Error - {e}")

    def _save_to_db(self):
        try:
            conn = get_db()
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conn.execute(
                'INSERT INTO ppe_events (camera_id, timestamp, compliant_count, violation_count, total_persons) VALUES (?,?,?,?,?)',
                (self.camera_id, ts, self.stats['compliant_count'],
                 self.stats['violation_count'], self.stats['total_persons'])
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"PPE DB save error: {e}")

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def get_last_detections(self):
        with self.lock:
            return list(self._last_detections)


class FireSmokeDetectionProcessor:
    """Detect fire/smoke events using dedicated model weights."""
    FIRE_KEYWORDS = ('fire', 'flame')
    SMOKE_KEYWORDS = ('smoke',)

    def __init__(self, camera_id, camera_reader, engine, conf=0.35):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.engine = engine
        self.conf = conf
        self.running = False
        self.annotated_frame = None
        self.lock = threading.Lock()
        self.stats = {'fire_count': 0, 'smoke_count': 0, 'total': 0, 'status': 'Initializing'}
        self.last_db_save = time.time()
        self.db_interval = 120
        self._last_detections = []

    def start(self):
        self.running = True
        self.engine.register(self.camera_id, self.camera_reader, self._on_result, self.conf)

    def stop(self):
        self.running = False
        self.engine.unregister(self.camera_id)

    def _kind(self, cls_name):
        n = (cls_name or '').lower()
        if any(k in n for k in self.FIRE_KEYWORDS):
            return 'fire'
        if any(k in n for k in self.SMOKE_KEYWORDS):
            return 'smoke'
        return ''

    def _on_result(self, frame, detections, raw_result):
        if not self.running:
            return
        try:
            vis = frame.copy()
            names = raw_result.names if (raw_result is not None and hasattr(raw_result, 'names')) else {}
            boxes_data = raw_result.boxes if (raw_result is not None and hasattr(raw_result, 'boxes')) else None
            fire_count, smoke_count = 0, 0
            last_dets = []
            if boxes_data is not None:
                for i in range(len(boxes_data)):
                    try:
                        xyxy = boxes_data.xyxy[i].cpu().numpy().astype(int)
                        cls_id = int(boxes_data.cls[i].cpu().numpy())
                        conf_val = float(boxes_data.conf[i].cpu().numpy())
                        if conf_val < self.conf:
                            continue
                        cls_name = names.get(cls_id, str(cls_id)) if names else str(cls_id)
                        kind = self._kind(cls_name)
                        if not kind:
                            continue
                        x1, y1, x2, y2 = map(int, xyxy)
                        if kind == 'fire':
                            fire_count += 1
                            color = (0, 0, 255)
                            label = f"Fire {conf_val:.2f}"
                        else:
                            smoke_count += 1
                            color = (0, 165, 255)
                            label = f"Smoke {conf_val:.2f}"
                        _draw_premium_ppe_box(vis, x1, y1, x2, y2, label, color)
                        last_dets.append(((x1, y1, x2, y2), label, color))
                    except Exception:
                        continue
            total = fire_count + smoke_count
            self.stats = {'fire_count': fire_count, 'smoke_count': smoke_count, 'total': total, 'status': 'Active'}
            with self.lock:
                self.annotated_frame = vis
                self._last_detections = last_dets

            if total > 0:
                _save_alert_event(
                    self.camera_id, 'fire_smoke', vis, last_dets, severity='high',
                    meta={'fire_count': fire_count, 'smoke_count': smoke_count, 'total': total}
                )
            if time.time() - self.last_db_save >= self.db_interval:
                self._save_to_db()
                self.last_db_save = time.time()
        except Exception as e:
            logger.error(f"FireSmoke {self.camera_id}: Error - {e}")

    def _save_to_db(self):
        try:
            conn = get_db()
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conn.execute(
                'INSERT INTO fire_smoke_events (camera_id, timestamp, fire_count, smoke_count, total_detections) VALUES (?,?,?,?,?)',
                (self.camera_id, ts, int(self.stats['fire_count']), int(self.stats['smoke_count']), int(self.stats['total']))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"FireSmoke DB save error: {e}")

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def get_last_detections(self):
        with self.lock:
            return list(self._last_detections)


# ==============================================================
# BETA MODEL DETECTION (OWLv2 VLM open-vocabulary)
# ==============================================================

_owlv2_processor = None
_owlv2_model = None
_owlv2_lock = threading.Lock()
_owlv2_infer_lock = threading.Lock()


def _get_owlv2():
    """Lazy-load OWLv2 processor and model once (shared across all beta processors)."""
    global _owlv2_processor, _owlv2_model
    with _owlv2_lock:
        if _owlv2_processor is not None:
            return _owlv2_processor, _owlv2_model
    if not _OWLV2_AVAILABLE:
        return None, None
    # Evict Florence + Qwen outside OWLv2 locks to avoid deadlocks with VLM global lock.
    _evict_vlms_except('owlv2')
    dev = _vlm_beta_device()
    with _owlv2_lock:
        if _owlv2_processor is None:
            try:
                # Prefer local model path (offline, faster cold start). Accept either a snapshot dir or a file inside it.
                local_path = OWLV2_LOCAL_PATH
                if local_path:
                    if os.path.isfile(local_path):
                        local_path = os.path.dirname(local_path)
                    _owlv2_processor = Owlv2Processor.from_pretrained(local_path, local_files_only=True)
                    _owlv2_model = Owlv2ForObjectDetection.from_pretrained(local_path, local_files_only=True)
                    logger.info(f"OWLv2 loaded from local path: {local_path}")
                else:
                    _owlv2_processor = Owlv2Processor.from_pretrained(OWLV2_MODEL_ID)
                    _owlv2_model = Owlv2ForObjectDetection.from_pretrained(OWLV2_MODEL_ID)
                    logger.info(f"OWLv2 loaded: {OWLV2_MODEL_ID}")
                if torch is not None:
                    _owlv2_model = _owlv2_model.to(dev)
                    if dev == 'cuda':
                        try:
                            _owlv2_model = _owlv2_model.half()
                        except Exception:
                            pass
                try:
                    _owlv2_model.eval()
                except Exception:
                    pass
                logger.info(f"OWLv2 using device: {dev} (set VISION_VLM_DEVICE=cuda|cpu)")
            except Exception as e:
                logger.error(f"OWLv2 load failed: {e}")
        return _owlv2_processor, _owlv2_model


_florence2_processor = None
_florence2_model = None
_florence2_lock = threading.Lock()
_florence2_infer_lock = threading.Lock()


def _florence2_local_dir():
    """Resolve local Florence-2 snapshot directory (must contain config.json)."""
    candidates = []
    if FLORENCE2_LOCAL_PATH:
        p = FLORENCE2_LOCAL_PATH
        if os.path.isfile(p):
            p = os.path.dirname(p)
        candidates.append(p)
    candidates.extend([
        os.path.join(BASE_DIR, 'hf_florence2_cache', 'Florence-2-large'),
        os.path.join(BASE_DIR, '.hf_florence2_cache', 'Florence-2-large'),
        os.path.join(BASE_DIR, 'florence2_model2', 'Florence-2-large'),
        os.path.join(BASE_DIR, 'hf_florence2_cache', 'Florence-2-base'),
        os.path.join(BASE_DIR, '.hf_florence2_cache', 'Florence-2-base'),
    ])
    for d in candidates:
        if d and os.path.isfile(os.path.join(d, 'config.json')):
            return d
    return None


def _florence2_ready():
    return bool(_FLORENCE2_AVAILABLE and _florence2_local_dir() and Image is not None)


def _get_florence2():
    """Lazy-load Florence-2 processor and model once (shared across beta processors)."""
    global _florence2_processor, _florence2_model
    if not _FLORENCE2_AVAILABLE or _FlorenceProcessorClass is None or _FlorenceModelClass is None:
        return None, None
    with _florence2_lock:
        if _florence2_processor is not None:
            return _florence2_processor, _florence2_model
    local_dir = _florence2_local_dir()
    if not local_dir:
        return None, None
    _evict_vlms_except('florence2')
    dev = _vlm_beta_device()
    with _florence2_lock:
        if _florence2_processor is None:
            try:
                _florence2_processor = _FlorenceProcessorClass.from_pretrained(
                    local_dir, local_files_only=True, trust_remote_code=True
                )
                load_kw = {'local_files_only': True, 'trust_remote_code': True}
                if torch is not None:
                    load_kw['torch_dtype'] = torch.float16 if dev == 'cuda' else torch.float32
                _florence2_model = _FlorenceModelClass.from_pretrained(local_dir, **load_kw)
                if torch is not None:
                    _florence2_model = _florence2_model.to(dev)
                try:
                    _florence2_model.eval()
                except Exception:
                    pass
                logger.info(f"Florence-2 loaded ({dev}) from local path: {local_dir}")
            except Exception as e:
                logger.error(f"Florence-2 load failed: {e}")
                _florence2_processor = None
                _florence2_model = None
        return _florence2_processor, _florence2_model


def _parse_prompt_labels(prompt_text):
    """Parse comma-separated prompt into list of label strings (strip, non-empty)."""
    if not prompt_text or not isinstance(prompt_text, str):
        return []
    return [s.strip() for s in prompt_text.split(',') if s.strip()]


class BetaDetectionProcessor:
    """Open-vocabulary detection using OWLv2. Labels come from user prompts; each camera can have multiple prompts."""
    BETA_FRAME_INTERVAL = 1.0 / 3.0  # ~3 fps inference to reduce load

    def __init__(self, camera_id, camera_reader, prompt_texts, conf=0.2):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.prompt_texts = list(prompt_texts) if prompt_texts else []
        self.conf = conf
        self.running = False
        self.annotated_frame = None
        self.lock = threading.Lock()
        self._last_detections = []
        self._thread = None
        self.stats = {'detections': 0, 'status': 'Initializing'}

    def start(self):
        if not _OWLV2_AVAILABLE:
            self.stats['status'] = 'OWLv2 not available'
            return
        labels = self._all_labels()
        if not labels:
            self.stats['status'] = 'No labels'
            return
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _all_labels(self):
        out = []
        seen = set()
        for pt in self.prompt_texts:
            for lbl in _parse_prompt_labels(pt):
                if lbl and lbl.lower() not in seen:
                    seen.add(lbl.lower())
                    out.append(lbl)
        return out

    def stop(self):
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run_loop(self):
        proc, model = _get_owlv2()
        if proc is None or model is None:
            self.stats['status'] = 'Model unavailable'
            self.running = False
            return
        labels = self._all_labels()
        if not labels:
            self.stats['status'] = 'No labels'
            self.running = False
            return
        text_labels = [labels]
        last_t = 0
        while self.running:
            try:
                now = time.time()
                if now - last_t < self.BETA_FRAME_INTERVAL:
                    time.sleep(0.03)
                    continue
                frame = self.camera_reader.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                last_t = now
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                inputs = proc(text=text_labels, images=pil_image, return_tensors="pt")
                if torch is not None and hasattr(model, 'device'):
                    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                # Avoid concurrent forwards on a single shared model (keeps VRAM stable on multi-camera setups).
                with _owlv2_infer_lock:
                    if torch is not None:
                        with torch.inference_mode():
                            if hasattr(model, "device") and str(model.device) != "cpu":
                                with torch.autocast(device_type="cuda", dtype=torch.float16):
                                    outputs = model(**inputs)
                            else:
                                outputs = model(**inputs)
                    else:
                        outputs = model(**inputs)
                target_sizes = torch.tensor([(pil_image.height, pil_image.width)])
                if torch is not None and hasattr(model, 'device') and str(model.device) != 'cpu':
                    target_sizes = target_sizes.to(model.device)
                results = proc.post_process_grounded_object_detection(
                    outputs=outputs, target_sizes=target_sizes, threshold=self.conf, text_labels=text_labels
                )
                vis = frame.copy()
                last_dets = []
                if results and len(results) > 0:
                    r = results[0]
                    boxes = r.get("boxes")
                    scores = r.get("scores")
                    text_labels_out = r.get("text_labels", [])
                    if boxes is not None and scores is not None:
                        for box, score, label in zip(boxes, scores, text_labels_out):
                            if score.item() < self.conf:
                                continue
                            x1, y1, x2, y2 = box.tolist()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            color = (0, 200, 120)
                            lbl = f"{label} {score.item():.2f}"
                            _draw_premium_ppe_box(vis, x1, y1, x2, y2, lbl, color)
                            last_dets.append(((x1, y1, x2, y2), lbl, color))
                info_h, info_w = 72, 380
                info_bg = np.zeros((info_h, info_w, 3), dtype=np.uint8)
                info_bg[:] = (28, 28, 32)
                vis[6:6+info_h, 8:8+info_w] = cv2.addWeighted(vis[6:6+info_h, 8:8+info_w], 0.35, info_bg, 0.65, 0)
                cv2.rectangle(vis, (8, 6), (8+info_w, 6+info_h), (70, 70, 78), 1)
                cv2.putText(vis, f"Beta (OWLv2)  |  Detections: {len(last_dets)}",
                            (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(vis, f"Labels: {', '.join(labels[:6])}{'...' if len(labels) > 6 else ''}",
                            (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 2, cv2.LINE_AA)
                with self.lock:
                    self.annotated_frame = vis
                    self._last_detections = last_dets
                self.stats = {'detections': len(last_dets), 'status': 'Active'}
                if len(last_dets) > 0:
                    _save_alert_event(
                        self.camera_id, 'beta_owlv2', vis, last_dets, severity='medium',
                        meta={'model': 'owlv2', 'detections': len(last_dets)}
                    )
            except Exception as e:
                logger.error(f"Beta {self.camera_id}: {e}")
                self.stats['status'] = str(e)[:40]
                time.sleep(0.2)

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def get_last_detections(self):
        with self.lock:
            return list(self._last_detections)


def _florence_parse_grounding_boxes(parsed, task_prompt):
    """Extract bboxes + labels from Florence-2 post_process_generation output."""
    if not parsed:
        return [], []
    if task_prompt in parsed and isinstance(parsed[task_prompt], dict):
        d = parsed[task_prompt]
        return list(d.get('bboxes') or []), list(d.get('labels') or [])
    for _k, v in parsed.items():
        if isinstance(v, dict) and 'bboxes' in v:
            return list(v.get('bboxes') or []), list(v.get('labels') or [])
    return [], []


class BetaFlorenceDetectionProcessor:
    """Instruction-driven anomaly monitoring via Florence-2 VQA-style prompting."""
    BETA_FRAME_INTERVAL = 1.0 / 3.0
    _FLO_TASK = "<VQA>"

    def __init__(self, camera_id, camera_reader, prompt_texts, conf=0.2):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.prompt_texts = list(prompt_texts) if prompt_texts else []
        self.conf = conf
        self.running = False
        self.annotated_frame = None
        self.lock = threading.Lock()
        self._last_detections = []
        self._thread = None
        self.stats = {'detections': 0, 'alerts': 0, 'status': 'Initializing'}
        self._rule_alert_last_ts = {}

    def start(self):
        if not _florence2_ready():
            self.stats['status'] = 'Florence-2 not available'
            return
        rules = self._all_rules()
        if not rules:
            self.stats['status'] = 'No instructions'
            return
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _all_rules(self):
        out = []
        seen = set()
        for pt in self.prompt_texts:
            instr = (pt or '').strip()
            if instr and instr.lower() not in seen:
                seen.add(instr.lower())
                out.append(instr)
        return out

    def _build_monitor_prompt(self, instruction):
        return (
            f"{self._FLO_TASK} You are an intelligent video anomaly monitor. "
            f"Instruction: {instruction}\n"
            f"Check only what is visible in this frame. "
            f"If an instructed anomaly/safety violation is present, respond exactly in one line: "
            f"ALERT:<short_label>. Otherwise respond exactly: CLEAR"
        )

    @staticmethod
    def _parse_monitor_response(text):
        t = (text or '').strip()
        tl = t.lower()
        if tl.startswith('alert:'):
            return True, t.split(':', 1)[1].strip() or 'anomaly'
        if 'alert:' in tl:
            idx = tl.find('alert:')
            tail = t[idx + len('alert:'):].strip()
            tail = tail.splitlines()[0].strip() if tail else ''
            return True, tail or 'anomaly'
        if tl.startswith('alert'):
            tail = t[len('alert'):].strip(' :.-')
            return True, (tail.splitlines()[0].strip() if tail else 'anomaly')
        if '\n' in tl:
            first = tl.splitlines()[0].strip()
            if first.startswith('alert:'):
                lbl = first.split(':', 1)[1].strip()
                return True, lbl or 'anomaly'
        return False, ''

    def stop(self):
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run_loop(self):
        proc, model = _get_florence2()
        if proc is None or model is None:
            self.stats['status'] = 'Florence-2 not loaded'
            self.running = False
            return
        rules = self._all_rules()
        if not rules:
            self.stats['status'] = 'No instructions'
            self.running = False
            return
        last_t = 0
        total_alerts = 0
        while self.running:
            try:
                now = time.time()
                if now - last_t < self.BETA_FRAME_INTERVAL:
                    time.sleep(0.03)
                    continue
                frame = self.camera_reader.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                last_t = now
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                vis = frame.copy()
                last_dets = []
                triggered_label = ''
                triggered_rule = ''
                for rule in rules:
                    prompt = self._build_monitor_prompt(rule)
                    inputs = proc(text=prompt, images=pil_image, return_tensors="pt")
                    if torch is not None and hasattr(model, 'parameters'):
                        try:
                            device = next(model.parameters()).device
                        except Exception:
                            device = None
                        if device is not None:
                            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                    with _florence2_infer_lock:
                        if torch is not None:
                            with torch.inference_mode():
                                dev = next(model.parameters()).device
                                if str(dev) != "cpu":
                                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                                        generated_ids = model.generate(
                                            **inputs,
                                            max_new_tokens=int(os.environ.get('VISION_FLORENCE_MONITOR_MAX_NEW_TOKENS', '48')),
                                            num_beams=int(os.environ.get('VISION_FLORENCE_MONITOR_NUM_BEAMS', '3')),
                                            do_sample=False
                                        )
                                else:
                                    generated_ids = model.generate(
                                        **inputs,
                                        max_new_tokens=int(os.environ.get('VISION_FLORENCE_MONITOR_MAX_NEW_TOKENS', '48')),
                                        num_beams=int(os.environ.get('VISION_FLORENCE_MONITOR_NUM_BEAMS', '3')),
                                        do_sample=False
                                    )
                        else:
                            generated_ids = model.generate(
                                **inputs,
                                max_new_tokens=int(os.environ.get('VISION_FLORENCE_MONITOR_MAX_NEW_TOKENS', '48')),
                                num_beams=int(os.environ.get('VISION_FLORENCE_MONITOR_NUM_BEAMS', '3')),
                                do_sample=False
                            )
                    generated_text = proc.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    is_alert, label = self._parse_monitor_response(generated_text)
                    if is_alert:
                        triggered_label = label
                        triggered_rule = rule
                        break

                emit_alert = False
                if triggered_label:
                    h, w = vis.shape[:2]
                    x1, y1, x2, y2 = 8, 8, max(12, w - 8), max(12, h - 8)
                    _draw_premium_ppe_box(vis, x1, y1, x2, y2, f"Anomaly: {triggered_label}", (0, 0, 255))
                    last_dets.append(((x1, y1, x2, y2), f"anomaly:{triggered_label}", (0, 0, 255)))
                    # Tiny per-instruction cooldown: prevent identical consecutive-frame alert spam.
                    key = ((triggered_rule or '').strip().lower(), (triggered_label or '').strip().lower())
                    now_ts = time.time()
                    last_ts = self._rule_alert_last_ts.get(key, 0.0)
                    emit_alert = (now_ts - last_ts) >= max(0.0, VISION_FLORENCE_RULE_ALERT_COOLDOWN_SEC)
                    if emit_alert:
                        self._rule_alert_last_ts[key] = now_ts
                        total_alerts += 1
                info_h, info_w = 72, 420
                info_bg = np.zeros((info_h, info_w, 3), dtype=np.uint8)
                info_bg[:] = (28, 28, 32)
                vis[6:6+info_h, 8:8+info_w] = cv2.addWeighted(vis[6:6+info_h, 8:8+info_w], 0.35, info_bg, 0.65, 0)
                cv2.rectangle(vis, (8, 6), (8+info_w, 6+info_h), (70, 70, 78), 1)
                cv2.putText(vis, f"Beta (Florence-2 Monitor)  |  Alerts: {total_alerts}",
                            (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
                rule_txt = triggered_rule if triggered_rule else "No instructed anomaly in frame"
                cv2.putText(vis, f"{rule_txt[:60]}{'...' if len(rule_txt) > 60 else ''}",
                            (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 2, cv2.LINE_AA)
                with self.lock:
                    self.annotated_frame = vis
                    self._last_detections = last_dets
                self.stats = {'detections': len(last_dets), 'alerts': total_alerts, 'status': 'Active'}
                if len(last_dets) > 0 and emit_alert:
                    _save_alert_event(
                        self.camera_id, 'beta_florence2_anomaly', vis, last_dets, severity='high',
                        meta={'model': 'florence2_monitor', 'instruction': triggered_rule, 'alert_label': triggered_label}
                    )
            except Exception as e:
                logger.error(f"Beta (Florence-2) {self.camera_id}: {e}")
                self.stats['status'] = str(e)[:40]
                time.sleep(0.2)

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def get_last_detections(self):
        with self.lock:
            return list(self._last_detections)


class BetaQwenDetectionProcessor:
    """Instruction-driven anomaly monitoring via Qwen vision-language model."""
    BETA_FRAME_INTERVAL = 1.0 / 2.0

    def __init__(self, camera_id, camera_reader, prompt_texts, conf=0.2):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.prompt_texts = list(prompt_texts) if prompt_texts else []
        self.conf = conf
        self.running = False
        self.annotated_frame = None
        self.lock = threading.Lock()
        self._last_detections = []
        self._thread = None
        self.stats = {'detections': 0, 'alerts': 0, 'status': 'Initializing'}
        self._anomaly_state = {}  # (rule, label) -> {'active': bool, 'last_seen': ts, 'last_emit': ts}

    def start(self):
        if not _qwen25vl_ready():
            self.stats['status'] = 'Qwen not available'
            return
        rules = self._all_rules()
        if not rules:
            self.stats['status'] = 'No instructions'
            return
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _all_rules(self):
        out = []
        seen = set()
        for pt in self.prompt_texts:
            instr = (pt or '').strip()
            if instr and instr.lower() not in seen:
                seen.add(instr.lower())
                out.append(instr)
        return out

    @staticmethod
    def _parse_monitor_response(text):
        t = (text or '').strip()
        tl = t.lower()
        if tl.startswith('alert:'):
            return True, t.split(':', 1)[1].strip() or 'anomaly'
        if 'alert:' in tl:
            idx = tl.find('alert:')
            tail = t[idx + len('alert:'):].strip()
            tail = tail.splitlines()[0].strip() if tail else ''
            return True, tail or 'anomaly'
        if tl.startswith('alert'):
            tail = t[len('alert'):].strip(' :.-')
            return True, (tail.splitlines()[0].strip() if tail else 'anomaly')
        return False, ''

    @staticmethod
    def _rule_categories(rule_text):
        s = (rule_text or '').lower()
        cats = []
        if any(k in s for k in ('ppe', 'hard hat', 'hardhat', 'helmet', 'lab coat', 'safety vest', 'no ppe')):
            cats.append('no_ppe')
        if any(k in s for k in ('dropped box', 'dropped boxes', 'box on floor', 'boxes on the floor', 'fallen box')):
            cats.append('dropped_box')
        if any(k in s for k in ('ladder',)):
            cats.append('ladder')
        if any(k in s for k in ('obstruction', 'obstructions', 'blocked aisle', 'aisle')):
            cats.append('aisle_obstruction')
        if not cats:
            cats.append('generic_anomaly')
        return cats

    @staticmethod
    def _category_prompt(rule_text, category):
        base = (
            "You are an intelligent video anomaly monitor. "
            f"Instruction context: {rule_text}\n"
            "Check only what is visible in this frame. "
        )
        if category == 'no_ppe':
            return base + (
                "Task: detect people without required PPE (white hard hat and safety blue lab coat). "
                "If present respond exactly: ALERT:No PPE usage. Otherwise respond exactly: CLEAR"
            )
        if category == 'dropped_box':
            return base + (
                "Task: detect dropped boxes on the floor (not on shelves, not held by people). "
                "If present respond exactly: ALERT:Dropped boxes on floor. Otherwise respond exactly: CLEAR"
            )
        if category == 'ladder':
            return base + (
                "Task: detect unsafe ladder usage. "
                "If present respond exactly: ALERT:Unsafe ladder usage. Otherwise respond exactly: CLEAR"
            )
        if category == 'aisle_obstruction':
            return base + (
                "Task: detect aisle obstruction/blocking by objects that hinder movement. "
                "If present respond exactly: ALERT:Aisle obstruction. Otherwise respond exactly: CLEAR"
            )
        return base + (
            "Task: detect any anomaly relevant to instruction. "
            "If present respond exactly: ALERT:Anomaly detected. Otherwise respond exactly: CLEAR"
        )

    def _qwen_gen(self, proc, model, text_in, image_obj):
        ins = None
        if image_obj is not None and hasattr(proc, 'apply_chat_template'):
            try:
                msgs = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_obj},
                        {"type": "text", "text": text_in},
                    ],
                }]
                ins = proc.apply_chat_template(
                    msgs,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except Exception:
                ins = None
        if ins is None:
            ins = proc(text=text_in, images=image_obj, return_tensors='pt')
        if torch is not None and hasattr(model, 'parameters'):
            dvc = next(model.parameters()).device
            ins = {k: v.to(dvc) if hasattr(v, 'to') else v for k, v in ins.items()}
        with _qwen25vl_infer_lock:
            if torch is not None:
                with torch.inference_mode():
                    oids = model.generate(
                        **ins,
                        max_new_tokens=int(os.environ.get('VISION_QWEN_BETA_MONITOR_MAX_NEW_TOKENS', '32')),
                        num_beams=int(os.environ.get('VISION_QWEN_BETA_MONITOR_NUM_BEAMS', '1')),
                        do_sample=False,
                        repetition_penalty=1.05,
                    )
            else:
                oids = model.generate(
                    **ins,
                    max_new_tokens=int(os.environ.get('VISION_QWEN_BETA_MONITOR_MAX_NEW_TOKENS', '32')),
                    num_beams=int(os.environ.get('VISION_QWEN_BETA_MONITOR_NUM_BEAMS', '1')),
                    do_sample=False,
                    repetition_penalty=1.05,
                )
        try:
            in_len = int(ins["input_ids"].shape[-1]) if "input_ids" in ins else 0
            gen_only = oids[:, in_len:] if in_len > 0 else oids
            out = proc.batch_decode(gen_only, skip_special_tokens=True)[0].strip()
        except Exception:
            out = proc.batch_decode(oids, skip_special_tokens=True)[0].strip()
        return _clean_qwen_text(out)

    def stop(self):
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run_loop(self):
        proc, model = _get_qwen25vl()
        if proc is None or model is None:
            self.stats['status'] = 'Qwen not loaded'
            self.running = False
            return
        rules = self._all_rules()
        if not rules:
            self.stats['status'] = 'No instructions'
            self.running = False
            return
        last_t = 0
        total_alerts = 0
        clear_grace = float(os.environ.get('VISION_QWEN_BETA_CLEAR_GRACE_SEC', '20'))
        reminder_sec = float(os.environ.get('VISION_QWEN_BETA_REMINDER_SEC', '900'))
        while self.running:
            try:
                now = time.time()
                if now - last_t < self.BETA_FRAME_INTERVAL:
                    time.sleep(0.03)
                    continue
                frame = self.camera_reader.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                last_t = now
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                if max(pil_image.size) > 1024:
                    pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
                vis = frame.copy()
                last_dets = []
                triggered_labels = []
                triggered_rule = ''
                for rule in rules:
                    rule_hits = []
                    for cat in self._rule_categories(rule):
                        out = self._qwen_gen(proc, model, self._category_prompt(rule, cat), pil_image)
                        is_alert, lbl = self._parse_monitor_response(out)
                        if is_alert:
                            rule_hits.append(lbl or cat.replace('_', ' '))
                    if rule_hits:
                        triggered_rule = rule
                        triggered_labels = rule_hits
                        break

                emit_labels = []
                if triggered_labels:
                    h, w = vis.shape[:2]
                    x1, y1, x2, y2 = 8, 8, max(12, w - 8), max(12, h - 8)
                    label_text = ", ".join(triggered_labels[:3])
                    _draw_premium_ppe_box(vis, x1, y1, x2, y2, f"Anomaly: {label_text}", (0, 0, 255))
                    last_dets.append(((x1, y1, x2, y2), f"anomaly:{label_text}", (0, 0, 255)))
                    now_ts = time.time()
                    seen_keys = set()
                    for lbl in triggered_labels:
                        key = ((triggered_rule or '').strip().lower(), (lbl or '').strip().lower())
                        seen_keys.add(key)
                        st = self._anomaly_state.get(key, {'active': False, 'last_seen': 0.0, 'last_emit': 0.0})
                        should_emit = (not st['active']) or ((now_ts - st['last_emit']) >= max(0.0, reminder_sec))
                        st['active'] = True
                        st['last_seen'] = now_ts
                        if should_emit:
                            st['last_emit'] = now_ts
                            emit_labels.append(lbl)
                            total_alerts += 1
                        self._anomaly_state[key] = st
                    # mark stale active anomalies as cleared after grace period
                    for k, st in list(self._anomaly_state.items()):
                        if k not in seen_keys and st.get('active') and (now_ts - st.get('last_seen', 0.0)) >= max(0.0, clear_grace):
                            st['active'] = False
                            self._anomaly_state[k] = st
                else:
                    now_ts = time.time()
                    for k, st in list(self._anomaly_state.items()):
                        if st.get('active') and (now_ts - st.get('last_seen', 0.0)) >= max(0.0, clear_grace):
                            st['active'] = False
                            self._anomaly_state[k] = st
                info_h, info_w = 72, 430
                info_bg = np.zeros((info_h, info_w, 3), dtype=np.uint8)
                info_bg[:] = (28, 28, 32)
                vis[6:6+info_h, 8:8+info_w] = cv2.addWeighted(vis[6:6+info_h, 8:8+info_w], 0.35, info_bg, 0.65, 0)
                cv2.rectangle(vis, (8, 6), (8+info_w, 6+info_h), (70, 70, 78), 1)
                cv2.putText(vis, f"Beta (Qwen Monitor)  |  Alerts: {total_alerts}",
                            (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
                rule_txt = triggered_rule if triggered_rule else "No instructed anomaly in frame"
                cv2.putText(vis, f"{rule_txt[:60]}{'...' if len(rule_txt) > 60 else ''}",
                            (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 2, cv2.LINE_AA)
                with self.lock:
                    self.annotated_frame = vis
                    self._last_detections = last_dets
                self.stats = {'detections': len(last_dets), 'alerts': total_alerts, 'status': 'Active'}
                if len(last_dets) > 0 and emit_labels:
                    _save_alert_event(
                        self.camera_id, 'beta_qwen_anomaly', vis, last_dets, severity='high',
                        meta={
                            'model': 'qwen_monitor',
                            'instruction': triggered_rule,
                            'alert_labels': triggered_labels,
                            'new_alert_labels': emit_labels,
                            'alert_label': ", ".join(triggered_labels),
                        }
                    )
            except Exception as e:
                logger.error(f"Beta (Qwen) {self.camera_id}: {e}")
                self.stats['status'] = str(e)[:40]
                time.sleep(0.2)
        _maybe_unload_beta_vlms_if_idle()

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def get_last_detections(self):
        with self.lock:
            return list(self._last_detections)


def _make_beta_processor(camera_id, camera_reader, prompt_texts, conf, backend):
    """Choose OWLv2 or Florence-2 beta processor without altering the OWLv2 class."""
    b = (backend or 'owlv2').strip().lower()
    if b == 'florence2':
        return BetaFlorenceDetectionProcessor(camera_id, camera_reader, prompt_texts, conf=conf)
    if b == 'qwen':
        return BetaQwenDetectionProcessor(camera_id, camera_reader, prompt_texts, conf=conf)
    return BetaDetectionProcessor(camera_id, camera_reader, prompt_texts, conf=conf)

# ==============================================================
# FACE RECOGNITION (FR) MODULE
# Integrates logic from trialinsightfaceattend_1.py and
# ultimatefacedataset_streamlit_2.py exactly as-is, adapted for Flask.
# ==============================================================

try:
    import pickle
    import hashlib
    import insightface
    from insightface.model_zoo import get_model as _insightface_get_model
    from insightface.model_zoo.model_zoo import ModelRouter as _FR_ModelRouter
    import onnxruntime as _ort
    try:
        _ort.set_default_logger_severity(3)
    except Exception:
        pass
    _FR_INSIGHTFACE_AVAILABLE = True
except ImportError:
    _insightface_get_model = None
    _FR_ModelRouter = None
    _FR_INSIGHTFACE_AVAILABLE = False
    logger.warning('FR: insightface not installed. pip install insightface onnxruntime')

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort as _FR_DeepSort
    _FR_DEEPSORT_AVAILABLE = True
except ImportError:
    _FR_DeepSort = None
    _FR_DEEPSORT_AVAILABLE = False

_FR_YOLO_AVAILABLE = True  # YOLO already imported at module level

# ── FR paths ──
# DATA_DIR is already resolved from VISION_DATA_DIR env var (set to /app/data in Docker,
# or BASE_DIR/data locally). All FR paths derive from it so they always point inside the
# bind-mounted volume — exactly where the employee images live on disk.
FR_AUTOFACEDATA_BASE_DIR = os.path.join(DATA_DIR, 'autofacedata')
FR_AUTOFACEDATA_CSV     = os.path.join(DATA_DIR, 'autofacedata.csv')
# AntelopeV2 model files: override with VISION_FR_ANTELOPEV2_PATH env var if needed,
# otherwise fall back to data/antelopev2/ (inside the same bind-mounted volume).
FR_ANTELOPEV2_PATH = os.environ.get('VISION_FR_ANTELOPEV2_PATH', '').strip() or os.path.join(DATA_DIR, 'antelopev2')
FR_FACE_MODEL_PATH = resolve_pt_model('bestempfacereal.pt')
FR_DETECTION_MODEL = os.path.join(FR_ANTELOPEV2_PATH, 'scrfd_10g_bnkps.onnx')
FR_RECOGNITION_MODEL = os.path.join(FR_ANTELOPEV2_PATH, 'glintr100.onnx')
FR_GENDER_AGE_MODEL = os.path.join(FR_ANTELOPEV2_PATH, 'genderage.onnx')

logger.info(
    f'FR paths resolved — DATA_DIR={DATA_DIR!r}  '
    f'base_dir={FR_AUTOFACEDATA_BASE_DIR!r}  '
    f'csv={FR_AUTOFACEDATA_CSV!r}  '
    f'antelopev2={FR_ANTELOPEV2_PATH!r}'
)

# ── FR recognition constants (from trialinsightfaceattend_1.py) ──
FR_RECOGNITION_THRESHOLD = 0.30
FR_CACHE_FILENAME = 'insightface_embeddings_antelopev2.pkl'
FR_MIN_SIDE_FOR_EMBEDDING = 300
# scrfd_10g_bnkps.onnx (AntelopeV2) is exported for 640×640 — 480 causes ONNX shape warnings.
_fr_det_env = os.environ.get('VISION_FR_INSIGHTFACE_DET_SIZE', '640,640').strip()
try:
    _fr_det_parts = [int(x.strip()) for x in _fr_det_env.split(',')]
    FR_INSIGHTFACE_DET_SIZE = tuple(_fr_det_parts[:2]) if len(_fr_det_parts) >= 2 else (640, 640)
except Exception:
    FR_INSIGHTFACE_DET_SIZE = (640, 640)
FR_DETECTION_RESIZE_WIDTH = int(
    os.environ.get('VISION_FR_DETECTION_RESIZE_WIDTH', str(FR_INSIGHTFACE_DET_SIZE[0]))
)
FR_INSIGHTFACE_DET_THRESH = 0.25
FR_DISPLAY_CONFIDENCE_THRESHOLD = 0.55
FR_FACE_PALETTE = [
    (72, 187, 99), (255, 165, 0), (203, 192, 255),
    (147, 255, 255), (255, 144, 238), (128, 255, 203), (185, 218, 255),
]
FR_UNKNOWN_COLOR = (80, 80, 255)
FR_BOX_THICKNESS = 3
FR_CORNER_RADIUS = 10
FR_LABEL_PADDING = 10
FR_FONT_SCALE_LABEL = 0.8
FR_FONT_SCALE_SUB = 0.65
FR_FONT_THICKNESS = 2
FR_TILED_DETECTION = os.environ.get(
    'VISION_FR_TILED_DETECTION', '1'
).strip().lower() in ('1', 'true', 'yes')
FR_TILE_OVERLAP = min(0.35, max(0.05, float(os.environ.get('VISION_FR_TILE_OVERLAP', '0.16'))))
FR_RESULT_TTL_SEC = max(0.5, float(os.environ.get('VISION_FR_RESULT_TTL_SEC', '6')))


def _fr_get_model_bounded(model_path):
    """Create an InsightFace model without an all-core ONNX thread pool."""
    options = _ort.SessionOptions()
    options.intra_op_num_threads = FR_ORT_THREADS
    options.inter_op_num_threads = 1
    options.execution_mode = _ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    available = set(_ort.get_available_providers())
    providers = [
        provider
        for provider in ('CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider')
        if provider in available
    ]
    if not providers:
        providers = ['CPUExecutionProvider']
    model = _FR_ModelRouter(model_path).get_model(
        sess_options=options,
        providers=providers,
    )
    logger.info(
        'FR model session: file=%s providers=%s ort_threads=%d',
        os.path.basename(model_path),
        providers,
        FR_ORT_THREADS,
    )
    return model

# ── FR data collection constants (from ultimatefacedataset_streamlit_2.py) ──
FR_NUM_IMAGES_PER_FACE = 50
FR_FACE_CLASS_ID = 0
FR_CONF_THRESHOLD = 0.05
FR_NMS_IOU = 0.40
FR_FACE_BOX_EXPAND_RATIO = 1.2
FR_DETECT_EVERY_N_FRAMES = 2
FR_SHARPNESS_THRESHOLD = 100.0


# ── Drawing utilities from trialinsightfaceattend_1.py ──

def _fr_resize_for_detection(frame, max_width=None):
    if max_width is None:
        max_width = FR_DETECTION_RESIZE_WIDTH
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0, 1.0
    scale = max_width / w
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return small, w / new_w, h / new_h


def _fr_scale_bbox(bbox, sx, sy):
    x1, y1, x2, y2 = bbox
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


def _fr_color_for_identity(name):
    if name is None:
        return FR_UNKNOWN_COLOR
    idx = hash(name) % len(FR_FACE_PALETTE)
    return FR_FACE_PALETTE[idx]


def _fr_draw_rounded_rect(img, x1, y1, x2, y2, color, thickness, radius=None):
    if radius is None:
        radius = FR_CORNER_RADIUS
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    w, h = x2 - x1, y2 - y1
    radius = min(radius, w // 2, h // 2, 12)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)


def _fr_draw_bbox_with_label(img, x1, y1, x2, y2, label, sublabel, color):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FR_FONT_SCALE_LABEL, FR_FONT_THICKNESS)
    tw2, th2 = 0, 0
    if sublabel:
        (tw2, th2), _ = cv2.getTextSize(sublabel, cv2.FONT_HERSHEY_SIMPLEX, FR_FONT_SCALE_SUB, FR_FONT_THICKNESS)
    label_w = max(tw, tw2) + FR_LABEL_PADDING * 2
    label_h = th + (th2 + FR_LABEL_PADDING if sublabel else 0) + FR_LABEL_PADDING
    ly1 = max(0, y1 - label_h - 4)
    ly2 = y1 - 4
    lx1 = max(0, min(x1, img.shape[1] - label_w))
    lx2 = min(img.shape[1], lx1 + label_w)
    cv2.rectangle(img, (lx1, ly1), (lx2, ly2), color, -1)
    cv2.rectangle(img, (lx1, ly1), (lx2, ly2), color, FR_BOX_THICKNESS)
    cv2.putText(img, label, (lx1 + FR_LABEL_PADDING, ly1 + th + FR_LABEL_PADDING),
                cv2.FONT_HERSHEY_SIMPLEX, FR_FONT_SCALE_LABEL, (0, 0, 0), FR_FONT_THICKNESS + 2, cv2.LINE_AA)
    cv2.putText(img, label, (lx1 + FR_LABEL_PADDING, ly1 + th + FR_LABEL_PADDING),
                cv2.FONT_HERSHEY_SIMPLEX, FR_FONT_SCALE_LABEL, (255, 255, 255), FR_FONT_THICKNESS, cv2.LINE_AA)
    if sublabel:
        cv2.putText(img, sublabel, (lx1 + FR_LABEL_PADDING, ly1 + th + FR_LABEL_PADDING + th2 + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, FR_FONT_SCALE_SUB, (0, 0, 0), FR_FONT_THICKNESS + 2, cv2.LINE_AA)
        cv2.putText(img, sublabel, (lx1 + FR_LABEL_PADDING, ly1 + th + FR_LABEL_PADDING + th2 + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, FR_FONT_SCALE_SUB, (255, 255, 255), FR_FONT_THICKNESS, cv2.LINE_AA)
    _fr_draw_rounded_rect(img, x1, y1, x2, y2, color, FR_BOX_THICKNESS)


def _fr_cache_path(csv_path):
    key = f'v8_antelopev2_final|{csv_path}|{FR_AUTOFACEDATA_BASE_DIR}|antelopev2'
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return os.path.join(FR_AUTOFACEDATA_BASE_DIR, f'{FR_CACHE_FILENAME}.{h}')


# ── CSV helpers from ultimatefacedataset_streamlit_2.py ──

_fr_csv_lock = threading.RLock()


def _fr_ensure_dirs(person_name=None):
    os.makedirs(FR_AUTOFACEDATA_BASE_DIR, exist_ok=True)
    if person_name:
        person_dir = os.path.join(FR_AUTOFACEDATA_BASE_DIR, person_name)
        os.makedirs(person_dir, exist_ok=True)
        return person_dir
    return FR_AUTOFACEDATA_BASE_DIR


def _fr_get_or_create_csv():
    fieldnames = ['person_name', 'file_path']
    if os.path.exists(FR_AUTOFACEDATA_CSV):
        rows = []
        try:
            with open(FR_AUTOFACEDATA_CSV, 'r', encoding='utf-8', newline='') as f:
                reader_csv = csv.DictReader(f, restkey='_extra')
                fn = list(reader_csv.fieldnames or fieldnames)
                fn = [c for c in fn if c and c != '_extra']
                if 'person_name' not in fn:
                    fn = fieldnames
                for row in reader_csv:
                    extra = row.pop('_extra', None) or []
                    name = str(row.get('person_name', '') or '').strip()
                    path_str = str(row.get('file_path', '') or '').strip()
                    extras = [str(p).strip().strip('"') for p in extra if str(p).strip()]
                    if extras:
                        parts = [p.strip().strip('"') for p in path_str.split(',') if p.strip()] if path_str else []
                        path_str = ','.join(parts + extras)
                    if name:
                        rows.append({'person_name': name, 'file_path': path_str})
            return rows, fieldnames
        except Exception as e:
            # Log but do NOT recreate/wipe the file — return empty rows so callers
            # can still append new data without losing whatever is on disk.
            logger.warning(f'FR: Could not read CSV at {FR_AUTOFACEDATA_CSV}: {e}')
            return [], fieldnames
    # CSV doesn't exist yet — create it with the header only.
    _fr_ensure_dirs()
    try:
        with open(FR_AUTOFACEDATA_CSV, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
        logger.info(f'FR: Created new CSV at {FR_AUTOFACEDATA_CSV}')
    except Exception as e:
        logger.error(f'FR: Could not create CSV at {FR_AUTOFACEDATA_CSV}: {e}')
    return [], fieldnames


def _fr_write_csv_atomic(rows, fieldnames):
    """Write rows atomically. Caller must hold _fr_csv_lock."""
    tmp_path = FR_AUTOFACEDATA_CSV + f'.{uuid.uuid4().hex}.tmp'
    try:
        _fr_ensure_dirs()
        with open(tmp_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction='ignore',
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()
            writer.writerows(rows)
        # Windows may briefly lock the target CSV (antivirus, Excel, concurrent readers).
        last_err = None
        for attempt in range(8):
            try:
                os.replace(tmp_path, FR_AUTOFACEDATA_CSV)
                last_err = None
                break
            except OSError as e:
                last_err = e
                winerr = getattr(e, 'winerror', None)
                if attempt < 7 and (winerr in (32, 2) or e.errno in (13, 16)):
                    time.sleep(0.05 * (attempt + 1))
                    continue
                raise
        if last_err is not None:
            raise last_err
        logger.info(f'FR: CSV saved — {len(rows)} row(s) → {FR_AUTOFACEDATA_CSV}')
        return True
    except Exception as e:
        logger.error(f'FR: CSV save failed ({FR_AUTOFACEDATA_CSV}): {e}')
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return False


def _fr_save_csv(rows, fieldnames):
    """Write rows to the CSV atomically (write to .tmp then rename) to avoid corruption."""
    with _fr_csv_lock:
        return _fr_write_csv_atomic(rows, fieldnames)


def _fr_update_csv_for_person(person_name, paths_list):
    with _fr_csv_lock:
        rows, fieldnames = _fr_get_or_create_csv()
        path_str = ','.join(paths_list)
        found = False
        for row in rows:
            if str(row.get('person_name', '')).strip() == str(person_name).strip():
                row['file_path'] = path_str
                found = True
                break
        if not found:
            rows.append({'person_name': person_name, 'file_path': path_str})
        return _fr_write_csv_atomic(rows, fieldnames)


# ── Data collection helpers from ultimatefacedataset_streamlit_2.py ──

def _fr_expand_bbox(bbox, frame_shape, ratio=None):
    if ratio is None:
        ratio = FR_FACE_BOX_EXPAND_RATIO
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame_shape[:2]
    cw, ch = x2 - x1, y2 - y1
    if cw <= 0 or ch <= 0:
        return (x1, y1, x2, y2)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw, nh = cw * ratio, ch * ratio
    nx1, ny1 = int(cx - nw / 2), int(cy - nh / 2)
    nx2, ny2 = int(cx + nw / 2), int(cy + nh / 2)
    return (max(0, nx1), max(0, ny1), min(w, nx2), min(h, ny2))


def _fr_extract_face_crop(frame, bbox, expanded=True):
    if expanded:
        bbox = _fr_expand_bbox(bbox, frame.shape)
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def _fr_is_frame_usable(frame):
    if frame is None or frame.size == 0:
        return False
    try:
        mean = np.mean(frame)
        std = np.std(frame)
        if std < 5.0 or mean < 3 or mean > 252:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return lap_var >= 15
    except Exception:
        return False


def _fr_is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < FR_SHARPNESS_THRESHOLD


def _fr_is_crop_usable(crop):
    if crop is None or crop.size == 0:
        return False
    try:
        h, w = crop.shape[:2]
        if w < 20 or h < 20:
            return False
        if _fr_is_blurry(crop):
            return False
        mean = np.mean(crop)
        std = np.std(crop)
        if std < 8 or mean < 8 or mean > 247:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        return cv2.Laplacian(gray, cv2.CV_64F).var() >= 12
    except Exception:
        return False


class _FR_SimpleTrack:
    __slots__ = ('track_id', '_bbox')

    def __init__(self, tid, bbox):
        self.track_id = tid
        self._bbox = bbox

    def to_ltrb(self):
        return self._bbox

    def is_confirmed(self):
        return True


def _fr_reader_thread(rtsp_url, state):
    """Dedicated RTSP reader for FR data collection."""
    from services.stream_ingestion.camera_reader import CameraReader

    ffmpeg_opts_prev = os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS')
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = CameraReader._RTSP_OPTIONS
    try:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
        if not cap.isOpened():
            state['error'] = 'Could not open RTSP stream. Check URL and network.'
            return
        while not state.get('stop', False):
            ret, frame = cap.read()
            if ret and frame is not None:
                old_frame = state.get('latest_frame')
                state['latest_frame'] = frame
                if old_frame is not None:
                    del old_frame
                state['latest_ok'] = True
            else:
                state['latest_ok'] = False
            time.sleep(0)
        cap.release()
    except Exception as e:
        state['error'] = str(e)
    finally:
        if ffmpeg_opts_prev is not None:
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = ffmpeg_opts_prev
        else:
            os.environ.pop('OPENCV_FFMPEG_CAPTURE_OPTIONS', None)


def _fr_processor_thread(state, person_name):
    """Face detection + image saving thread – exact logic from ultimatefacedataset_streamlit_2.py."""
    from collections import defaultdict as _fr_dd
    _fr_ensure_dirs(person_name)
    if not _FR_YOLO_AVAILABLE:
        state['error'] = 'ultralytics not installed. pip install ultralytics'
        return
    if not _FR_DEEPSORT_AVAILABLE:
        state['error'] = 'deep_sort_realtime not installed. pip install deep-sort-realtime'
        return
    if not os.path.exists(FR_FACE_MODEL_PATH):
        state['error'] = f'Face model not found: {FR_FACE_MODEL_PATH}'
        return

    model = YOLO(FR_FACE_MODEL_PATH)
    tracker = _FR_DeepSort(
        max_age=30, n_init=3, nms_max_overlap=0.7,
        max_cosine_distance=0.3, nn_budget=100,
        embedder='mobilenet', half=True, bgr=True, embedder_gpu=True,
    )
    next_persistent_id = 1
    track_to_persistent = {}
    persistent_counts = _fr_dd(int)
    persistent_paths = _fr_dd(list)
    last_bbox_by_track = {}
    frame_index = 0
    last_good_display = None
    last_save_time = time.time()
    save_interval = 1.0 / 3.0

    try:
        while not state.get('stop', False):
            frame = state.get('latest_frame')
            if frame is None:
                time.sleep(0.01)
                continue
            frame = frame.copy()
            frame_index += 1

            if not _fr_is_frame_usable(frame) or _fr_is_blurry(frame):
                if last_good_display is not None:
                    state['frame'] = last_good_display
                time.sleep(0.001)
                continue

            display_frame = frame.copy()
            run_detection = (frame_index % FR_DETECT_EVERY_N_FRAMES == 1)

            if run_detection:
                results = model(frame, conf=FR_CONF_THRESHOLD, iou=FR_NMS_IOU,
                                classes=[FR_FACE_CLASS_ID], verbose=False)
                detections = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        if int(box.cls[0]) != FR_FACE_CLASS_ID:
                            continue
                        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                        bconf = float(box.conf[0])
                        detections.append(([int(bx1), int(by1), int(bx2 - bx1), int(by2 - by1)], bconf, 'face'))
                tracked = tracker.update_tracks(detections, frame=frame)
            else:
                tracked = []
                for tid, bbox in list(last_bbox_by_track.items()):
                    if tid in track_to_persistent:
                        tracked.append(_FR_SimpleTrack(tid, bbox))

            for track in tracked:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                raw_bbox = (x1, y1, x2, y2)
                bbox = _fr_expand_bbox(raw_bbox, frame.shape)
                x1, y1, x2, y2 = bbox
                last_bbox_by_track[track_id] = bbox

                if track_id not in track_to_persistent:
                    track_to_persistent[track_id] = next_persistent_id
                    next_persistent_id += 1
                persistent_id = track_to_persistent[track_id]

                color = (0, 255, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f'ID {persistent_id}',
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                current_time = time.time()
                if (persistent_counts[persistent_id] < FR_NUM_IMAGES_PER_FACE and
                        current_time - last_save_time >= save_interval):
                    crop = _fr_extract_face_crop(frame, bbox, expanded=True)
                    if crop is not None and _fr_is_crop_usable(crop):
                        crop = cv2.resize(crop, (480, 480), interpolation=cv2.INTER_LANCZOS4)
                        person_dir = os.path.join(FR_AUTOFACEDATA_BASE_DIR, person_name)
                        fname = f'face_{person_name}_{persistent_counts[persistent_id]:03d}.jpg'
                        out_path = os.path.join(person_dir, fname)
                        cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        abs_path = os.path.join(FR_AUTOFACEDATA_BASE_DIR, person_name, fname)
                        persistent_paths[persistent_id].append(abs_path)
                        persistent_counts[persistent_id] += 1
                        last_save_time = current_time
                        if persistent_counts[persistent_id] == FR_NUM_IMAGES_PER_FACE:
                            _fr_update_csv_for_person(person_name, persistent_paths[persistent_id])
                            state['completed_ids'] = state.get('completed_ids', set()) | {persistent_id}
                            state['csv_updated'] = True

                cv2.putText(display_frame, f'{persistent_counts[persistent_id]}/{FR_NUM_IMAGES_PER_FACE}',
                            (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            if run_detection:
                current_ids = {t.track_id for t in tracked if t.is_confirmed()}
                for tid in list(last_bbox_by_track.keys()):
                    if tid not in current_ids:
                        last_bbox_by_track.pop(tid, None)

            out_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            last_good_display = out_rgb
            state['frame'] = out_rgb
            state['counts'] = dict(persistent_counts)
            state['completed'] = list(state.get('completed_ids', set()))
    except Exception as e:
        state['error'] = str(e)
        logger.error(f'FR data collection processor error: {e}')


class FaceRecognitionProcessor:
    """
    Live face recognition processor for a camera stream.
    Core logic from trialinsightfaceattend_1.py (AttendanceSystem), adapted for Flask.
    """

    def __init__(self, camera_id, camera_reader, threshold=None, draw_threshold=None, det_threshold=None):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.threshold = threshold if threshold is not None else FR_RECOGNITION_THRESHOLD
        self.draw_threshold = (
            draw_threshold if draw_threshold is not None else FR_DISPLAY_CONFIDENCE_THRESHOLD
        )
        self.det_threshold = det_threshold if det_threshold is not None else FR_INSIGHTFACE_DET_THRESH
        self.running = False
        self.annotated_frame = None
        self.lock = threading.Lock()
        self._thread = None
        self.face_app = None
        self.known_embeddings = []
        self.known_names = []
        self.known_embeddings_matrix = None
        self._ready = False
        self._init_error = None
        self.stats = {'status': 'Initializing', 'faces_detected': 0, 'recognized': 0}
        self.last_detections = []
        self._diag_tick = 0
        self._scan_region_index = 0
        self._result_cache = []

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def get_last_detections(self):
        with self.lock:
            return list(self.last_detections)

    def _run(self):
        try:
            self._init_insightface()
            self._load_embeddings()
            self._ready = True
            self.stats['status'] = 'Active'
        except Exception as e:
            self._init_error = str(e)
            self.stats['status'] = f'Init error: {e}'
            logger.error(f'FR {self.camera_id}: Init error - {e}')
            return

        inference_interval = 1.0 / FR_TARGET_FPS
        next_inference_at = 0.0
        while self.running:
            now = time.monotonic()
            if now < next_inference_at:
                time.sleep(min(0.05, next_inference_at - now))
                continue
            frame = self.camera_reader.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue
            next_inference_at = time.monotonic() + inference_interval
            try:
                annotated, results = self._process_frame(frame)
                recognized = sum(1 for r in results if r.get('name') is not None)
                self.stats = {
                    'status': 'Active',
                    'faces_detected': len(results),
                    'recognized': recognized,
                }
                with self.lock:
                    self.annotated_frame = annotated
                    # Keep a generic overlay list so /api/feed_snapshot composite mode can draw FR too.
                    dets = []
                    for r in results:
                        bbox = r.get('bbox')
                        if not bbox or len(bbox) != 4:
                            continue
                        name = r.get('name')
                        conf = float(r.get('confidence', 0.0))
                        color = _fr_color_for_identity(name)
                        label = (name if name else 'Unknown') + f' {conf:.0%}'
                        dets.append((bbox, label, color))
                    self.last_detections = dets
                self._diag_tick += 1
                if self._diag_tick % 90 == 0:
                    logger.info(
                        f'FR {self.camera_id}: frame_diag faces_detected={len(results)} '
                        f'recognized={recognized} known_embs={len(self.known_embeddings)} '
                        f'match_threshold={self.threshold} draw_threshold={self.draw_threshold} '
                        f'det_threshold={self.det_threshold}'
                    )
            except Exception as e:
                logger.error(f'FR {self.camera_id}: Frame processing error - {e}')
                time.sleep(0.05)

    def _init_insightface(self):
        """Initialize AntelopeV2 directly. From trialinsightfaceattend_1.py _init_antelopev2_direct()."""
        if not _FR_INSIGHTFACE_AVAILABLE:
            raise RuntimeError('insightface not installed. pip install insightface onnxruntime')
        if not os.path.exists(FR_DETECTION_MODEL):
            raise RuntimeError(f'Detection model not found: {FR_DETECTION_MODEL}')
        if not os.path.exists(FR_RECOGNITION_MODEL):
            raise RuntimeError(f'Recognition model not found: {FR_RECOGNITION_MODEL}')

        os.environ['INSIGHTFACE_MODELS_DIR'] = FR_ANTELOPEV2_PATH
        det_model = _fr_get_model_bounded(FR_DETECTION_MODEL)
        det_model.prepare(ctx_id=0, input_size=FR_INSIGHTFACE_DET_SIZE)
        if hasattr(det_model, 'threshold'):
            det_model.threshold = self.det_threshold
        logger.info(
            f'FR {self.camera_id}: SCRFD det_size={FR_INSIGHTFACE_DET_SIZE} '
            f'resize_width={FR_DETECTION_RESIZE_WIDTH} tiled={FR_TILED_DETECTION} '
            f'match={self.threshold} draw={self.draw_threshold} det={self.det_threshold}'
        )

        rec_model = _fr_get_model_bounded(FR_RECOGNITION_MODEL)
        rec_model.prepare(ctx_id=0)

        genderage_model = None
        if FR_ENABLE_GENDER_AGE and os.path.exists(FR_GENDER_AGE_MODEL):
            genderage_model = _fr_get_model_bounded(FR_GENDER_AGE_MODEL)
            genderage_model.prepare(ctx_id=0)

        det_thresh = self.det_threshold

        class _SimpleInsightFace:
            def __init__(self_, det, rec, ga, thresh):
                self_.det_model = det
                self_.rec_model = rec
                self_.genderage_model = ga
                self_.det_thresh = thresh

            def get(self_, img):
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                try:
                    bboxes, kpss = self_.det_model.detect(img)
                except Exception:
                    bboxes, kpss = self_.det_model.detect(img, threshold=self_.det_thresh)
                if bboxes is None or bboxes.shape[0] == 0:
                    return []
                if bboxes.shape[1] >= 5:
                    mask = bboxes[:, 4] >= self_.det_thresh
                    bboxes = bboxes[mask]
                    if kpss is not None:
                        kpss = kpss[mask]
                if bboxes.shape[0] == 0:
                    return []
                from insightface.app.common import Face
                faces = []
                for i in range(bboxes.shape[0]):
                    bbox = bboxes[i][:4]
                    kps = kpss[i] if kpss is not None else None
                    face = Face(bbox=bbox, kps=kps)
                    face.det_score = float(bboxes[i][4]) if bboxes.shape[1] >= 5 else 1.0
                    self_.rec_model.get(img, face)
                    faces.append(face)
                return faces

        self.face_app = _SimpleInsightFace(det_model, rec_model, genderage_model, det_thresh)
        logger.info(f'FR {self.camera_id}: AntelopeV2 initialized successfully')

    def _load_embeddings(self):
        """Load or build embeddings from CSV. From trialinsightfaceattend_1.py _ensure_embeddings()."""
        if not os.path.exists(FR_AUTOFACEDATA_CSV):
            logger.warning(f'FR: CSV not found at {FR_AUTOFACEDATA_CSV}. No faces recognized until data is collected.')
            return
        import pandas as pd
        rows, _ = _fr_get_or_create_csv()
        if not rows:
            logger.warning('FR: CSV has no person rows after parse. No faces recognized until data is collected.')
            return
        self._df = pd.DataFrame(rows)
        if 'file_path' not in self._df.columns:
            self._df['file_path'] = ''

        logger.info(f'FR _load_embeddings: CSV has {len(self._df)} row(s), '
                    f'columns={list(self._df.columns)}')

        if 'person_name' not in self._df.columns or 'file_path' not in self._df.columns:
            logger.error(f'FR: CSV missing expected columns. '
                         f'Got {list(self._df.columns)}, need person_name + file_path')
            return

        cache_file = _fr_cache_path(FR_AUTOFACEDATA_CSV)
        current_employees = set(str(r['person_name']) for _, r in self._df.iterrows())
        cached_embs, cached_names = [], []
        cache_exists = False

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                if data.get('model') == 'antelopev2' and data.get('names'):
                    cached_embs = data['embeddings']
                    cached_names = data['names']
                    cache_exists = True
            except Exception:
                pass

        cached_set = set(cached_names)
        new_employees = current_employees - cached_set

        if not new_employees and cache_exists:
            self.known_embeddings = cached_embs
            self.known_names = cached_names
            if self.known_embeddings:
                self.known_embeddings_matrix = np.array(self.known_embeddings, dtype=np.float32)
            logger.info(f'FR: Loaded {len(self.known_embeddings)} cached embeddings for {len(set(self.known_names))} people')
            return

        new_embs, new_names = [], []
        logger.info(f'FR: Building embeddings for {len(new_employees)} new employee(s): {sorted(new_employees)}')
        for _, row in self._df.iterrows():
            name = str(row['person_name'])
            if name not in new_employees:
                continue
            file_path = str(row.get('file_path', ''))
            paths = [p.strip() for p in file_path.split(',') if p.strip()]
            person_ok = 0
            person_missing = 0
            person_no_face = 0
            for img_path in paths:
                # Absolute paths (written by current code) are used directly.
                # Relative paths (legacy entries) are resolved against the base dir.
                if os.path.isabs(img_path):
                    full = img_path
                else:
                    full = os.path.join(FR_AUTOFACEDATA_BASE_DIR, name, os.path.basename(img_path))
                    if not os.path.exists(full):
                        full = os.path.join(FR_AUTOFACEDATA_BASE_DIR, img_path)
                    if not os.path.exists(full):
                        full = img_path
                if not os.path.exists(full):
                    person_missing += 1
                    continue
                img = cv2.imread(full)
                if img is None:
                    person_missing += 1
                    continue
                h, w = img.shape[:2]
                if max(h, w) < FR_MIN_SIDE_FOR_EMBEDDING:
                    scale = FR_MIN_SIDE_FOR_EMBEDDING / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
                emb = self._get_embedding(img)
                if emb is not None:
                    new_embs.append(emb)
                    new_names.append(name)
                    person_ok += 1
                else:
                    person_no_face += 1
            logger.info(f'FR embeddings: {name!r} — paths={len(paths)}  '
                        f'ok={person_ok}  missing={person_missing}  no_face={person_no_face}')

        if cache_exists:
            self.known_embeddings = cached_embs + new_embs
            self.known_names = cached_names + new_names
        else:
            self.known_embeddings = new_embs
            self.known_names = new_names

        if self.known_embeddings:
            self.known_embeddings_matrix = np.array(self.known_embeddings, dtype=np.float32)

        try:
            os.makedirs(FR_AUTOFACEDATA_BASE_DIR, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'model': 'antelopev2',
                    'embeddings': self.known_embeddings,
                    'names': self.known_names,
                    'rollnos': self.known_names,
                }, f)
        except Exception as e:
            logger.warning(f'FR: Cache save error: {e}')

        logger.info(f'FR: Built {len(self.known_embeddings)} embeddings for {len(set(self.known_names))} people')

    def reload_embeddings(self):
        """Rebuild embeddings after adding or removing people."""
        import glob as _fr_glob
        self.known_embeddings = []
        self.known_names = []
        self.known_embeddings_matrix = None
        try:
            for f in _fr_glob.glob(os.path.join(FR_AUTOFACEDATA_BASE_DIR, f'{FR_CACHE_FILENAME}.*')):
                os.remove(f)
        except Exception:
            pass
        try:
            self._load_embeddings()
        except Exception as e:
            logger.error(f'FR {self.camera_id}: Reload embeddings error: {e}')

    def _get_embedding(self, img):
        if self.face_app is None:
            return None
        try:
            # InsightFace's OpenCV API expects BGR input.
            faces = self.face_app.get(img)
            if not faces:
                return None
            best = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            emb = best.normed_embedding
            if emb is None or len(emb) == 0:
                return None
            emb = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb
        except Exception:
            return None

    def _match_embedding_fast(self, query_emb):
        """Fast cosine similarity match. From trialinsightfaceattend_1.py."""
        if query_emb is None or self.known_embeddings_matrix is None or len(self.known_embeddings) == 0:
            return None, 0.0
        sims = self.known_embeddings_matrix @ query_emb
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score > self.threshold:
            return self.known_names[best_idx], best_score
        return None, best_score

    @staticmethod
    def _bbox_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        intersection = iw * ih
        if intersection <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return intersection / max(1.0, area_a + area_b - intersection)

    def _next_scan_region(self, frame):
        """Alternate overlapping half-width views without adding inference calls."""
        h, w = frame.shape[:2]
        full = (0, 0, w, h)
        if not FR_TILED_DETECTION or w <= FR_DETECTION_RESIZE_WIDTH:
            return full

        tile_scale = 0.5 + (FR_TILE_OVERLAP / 2.0)
        tile_w = min(w, max(FR_DETECTION_RESIZE_WIDTH, int(round(w * tile_scale))))
        x2 = max(0, w - tile_w)
        regions = [
            (0, 0, tile_w, h),
            (x2, 0, w, h),
        ]
        region = regions[self._scan_region_index % len(regions)]
        self._scan_region_index = (self._scan_region_index + 1) % len(regions)
        return region

    def _merge_result_cache(self, current_results, now):
        cache = [
            item for item in self._result_cache
            if now - item['_last_seen'] <= FR_RESULT_TTL_SEC
        ]
        for result in current_results:
            best_index = -1
            best_iou = 0.0
            for index, existing in enumerate(cache):
                iou = self._bbox_iou(result['bbox'], existing['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_index = index
            item = dict(result)
            item['_last_seen'] = now
            if best_index >= 0 and best_iou >= 0.25:
                cache[best_index] = item
            else:
                cache.append(item)
        self._result_cache = cache
        return [
            {key: value for key, value in item.items() if key != '_last_seen'}
            for item in cache
        ]

    def _process_frame(self, frame):
        """Recognize one full frame or tile while retaining recent overlays."""
        if frame is None or self.face_app is None:
            return frame, []
        display = frame.copy()
        try:
            rx1, ry1, rx2, ry2 = self._next_scan_region(frame)
            region = frame[ry1:ry2, rx1:rx2]
            small, sx, sy = _fr_resize_for_detection(
                region,
                max_width=FR_DETECTION_RESIZE_WIDTH,
            )
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            faces = self.face_app.get(rgb_small)
            current_results = []
            for face in faces:
                bbox_small = face.bbox.astype(int).tolist()
                lx1, ly1, lx2, ly2 = _fr_scale_bbox(bbox_small, sx, sy)
                bbox_full = (
                    max(0.0, min(float(frame.shape[1]), lx1 + rx1)),
                    max(0.0, min(float(frame.shape[0]), ly1 + ry1)),
                    max(0.0, min(float(frame.shape[1]), lx2 + rx1)),
                    max(0.0, min(float(frame.shape[0]), ly2 + ry1)),
                )
                embedding = face.normed_embedding
                if embedding is not None:
                    emb = np.array(embedding, dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    name, confidence = self._match_embedding_fast(emb)
                else:
                    name, confidence = None, 0.0
                current_results.append({
                    'bbox': bbox_full,
                    'name': name,
                    'confidence': float(confidence),
                })

            results = self._merge_result_cache(current_results, time.monotonic())
            for result in results:
                confidence = result['confidence']
                if confidence >= self.draw_threshold:
                    x1, y1, x2, y2 = result['bbox']
                    name = result['name']
                    color = _fr_color_for_identity(name)
                    label = name if name else 'Unknown'
                    sublabel = f'{confidence:.0%}'
                    _fr_draw_bbox_with_label(display, x1, y1, x2, y2, label, sublabel, color)
            return display, results
        except Exception:
            return display, []


# Star-import from app.py needs private names (_engines, _OWLV2_AVAILABLE, etc.).
__all__ = [name for name in globals() if not name.startswith('__')]

