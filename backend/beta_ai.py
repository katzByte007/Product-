"""Beta model detection — OWLv2 open vocabulary + Qwen2.5-VL instruction monitoring."""
import json
import logging
import os
import re
import threading
import time

import cv2
import numpy as np

from config import (
    OWLV2_INFER_FPS,
    OWLV2_INFER_MAX_WIDTH,
    OWLV2_LOCAL_PATH,
    OWLV2_MAX_DETECTIONS,
    OWLV2_MIN_BOX_AREA,
    OWLV2_MODEL_ID,
    OWLV2_NMS_IOU,
    QWEN25VL_LOCAL_PATH,
    TORCH_THREADS,
    VLM_DEVICE,
    _has_model_weights,
)

logger = logging.getLogger(__name__)

torch = None
Image = None
Owlv2ForObjectDetection = None
Owlv2Processor = None
AutoModelForVision2Seq = None
AutoProcessor = None
_OWLV2_AVAILABLE = None
_QWEN_IMPORT_OK = None


def _ensure_torch():
    global torch
    if torch is None:
        try:
            import torch as _torch
            torch = _torch
            torch.set_num_threads(TORCH_THREADS)
            torch.set_num_interop_threads(max(1, TORCH_THREADS // 2))
        except Exception:
            torch = False
    return torch if torch is not False else None


def _ensure_pil():
    global Image
    if Image is None:
        try:
            from PIL import Image as _Image
            Image = _Image
        except Exception:
            Image = False
    return Image if Image is not False else None


_OWLV2_IMPORT_ERROR = ""
_QWEN_IMPORT_ERROR = ""


def _ensure_owlv2_imports():
    global _OWLV2_AVAILABLE, Owlv2ForObjectDetection, Owlv2Processor, _OWLV2_IMPORT_ERROR
    if _OWLV2_AVAILABLE is True:
        return True
    _ensure_torch()
    _ensure_pil()
    err = ""
    try:
        from transformers import Owlv2ForObjectDetection as _M, Owlv2Processor as _P

        Owlv2ForObjectDetection = _M
        Owlv2Processor = _P
        _OWLV2_AVAILABLE = True
        _OWLV2_IMPORT_ERROR = ""
        return True
    except Exception as e:
        err = str(e)
    try:
        from transformers.models.owlv2.modeling_owlv2 import Owlv2ForObjectDetection as _M
        from transformers.models.owlv2.processing_owlv2 import Owlv2Processor as _P

        Owlv2ForObjectDetection = _M
        Owlv2Processor = _P
        _OWLV2_AVAILABLE = True
        _OWLV2_IMPORT_ERROR = ""
        return True
    except Exception as e:
        err = err or str(e)
    _OWLV2_IMPORT_ERROR = err
    logger.warning("OWLv2 import failed: %s", err)
    return False


def _ensure_qwen_imports():
    global _QWEN_IMPORT_OK, AutoModelForVision2Seq, AutoProcessor, _QWEN_IMPORT_ERROR
    if _QWEN_IMPORT_OK is True:
        return True
    _ensure_torch()
    _ensure_pil()
    err = ""
    try:
        from transformers import AutoProcessor as _P

        AutoProcessor = _P
        try:
            from transformers import AutoModelForVision2Seq as _M
        except Exception:
            from transformers import Qwen2_5_VLForConditionalGeneration as _M
        AutoModelForVision2Seq = _M
        _QWEN_IMPORT_OK = True
        _QWEN_IMPORT_ERROR = ""
        return True
    except Exception as e:
        err = str(e)
        logger.warning("Qwen import failed: %s", err)
        _QWEN_IMPORT_ERROR = err
        return False

_owlv2_processor = None
_owlv2_model = None
_owlv2_lock = threading.Lock()
_owlv2_infer_lock = threading.Lock()

_qwen_processor = None
_qwen_model = None
_qwen_lock = threading.Lock()
_qwen_infer_lock = threading.Lock()
_active_backend = None


def _device():
    _ensure_torch()
    pref = (VLM_DEVICE or "auto").strip().lower()
    if torch is not None and torch.cuda.is_available() and pref in ("cuda", "auto", ""):
        return "cuda"
    if pref == "mps" and torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _device_is_gpu(device) -> bool:
    return "cuda" in str(device) or str(device).startswith("mps")


def _infer_min_interval(device) -> float:
    """GPU: run as fast as the card allows unless VISION_OWLV2_INFER_FPS is set."""
    if OWLV2_INFER_FPS > 0:
        return 1.0 / max(0.5, OWLV2_INFER_FPS)
    if _device_is_gpu(device):
        return 0.0
    return 1.0 / 2.0


def _owlv2_local_dir():
    if OWLV2_LOCAL_PATH and _has_model_weights(OWLV2_LOCAL_PATH):
        return OWLV2_LOCAL_PATH
    return None


def _qwen_config_dir():
    """Directory with Qwen config.json (weights may still be missing)."""
    p = QWEN25VL_LOCAL_PATH
    if not p:
        return None
    if os.path.isfile(p):
        p = os.path.dirname(p)
    if os.path.isfile(os.path.join(p, "config.json")):
        return p
    return None


def _qwen_needs_4bit(path: str) -> bool:
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        return bool(cfg.get("quantization_config", {}).get("_load_in_4bit"))
    except Exception:
        return False


def _bitsandbytes_ok() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False


def owlv2_available():
    return _ensure_owlv2_imports() and _ensure_pil() is not None


def owlv2_ready():
    """Dependencies OK and local weights exist or HuggingFace hub fallback is possible."""
    if not owlv2_available():
        return False
    return _owlv2_local_dir() is not None or bool(OWLV2_MODEL_ID)


def qwen_available():
    """True when Qwen code + local weights exist. bitsandbytes is optional (FP16 fallback)."""
    if not _ensure_qwen_imports() or _ensure_pil() is None:
        return False
    p = _qwen_config_dir()
    if not p:
        return False
    if not _has_model_weights(p):
        return False
    return True


def beta_model_status():
    """Detailed status for UI / API."""
    owlv2_deps = owlv2_available()
    qwen_deps = _ensure_qwen_imports() and _ensure_pil() is not None
    owlv2_local = _owlv2_local_dir()
    qwen_dir = _qwen_config_dir()
    qwen_weights = bool(qwen_dir and _has_model_weights(qwen_dir))
    qwen_4bit = bool(qwen_dir and _qwen_needs_4bit(qwen_dir))
    bnb = _bitsandbytes_ok()
    status = {
        "owlv2_available": owlv2_deps,
        "owlv2_ready": owlv2_ready(),
        "owlv2_local_weights": owlv2_local is not None,
        "owlv2_will_download": owlv2_deps and owlv2_local is None,
        "owlv2_local_path": owlv2_local or OWLV2_LOCAL_PATH or "",
        "vlm_device_pref": VLM_DEVICE,
        "vlm_device": _device() if owlv2_deps else VLM_DEVICE,
        "qwen_available": qwen_available(),
        "qwen_config_present": qwen_dir is not None,
        "qwen_weights_present": qwen_weights,
        "qwen_needs_bitsandbytes": qwen_4bit,
        "qwen_bitsandbytes_ok": bnb,
        "qwen_local_path": qwen_dir or QWEN25VL_LOCAL_PATH or "",
    }
    if not owlv2_deps:
        extra = f" ({_OWLV2_IMPORT_ERROR})" if _OWLV2_IMPORT_ERROR else ""
        status["owlv2_message"] = f"Open vocabulary is not importable{extra}"
    elif owlv2_local:
        status["owlv2_message"] = "OWLv2 ready (local weights)"
    else:
        status["owlv2_message"] = "OWLv2 will download from HuggingFace on first use (local folder has config only, no weights)"
    if not qwen_deps:
        extra = f" ({_QWEN_IMPORT_ERROR})" if _QWEN_IMPORT_ERROR else ""
        status["qwen_message"] = f"Instruction monitoring is not importable{extra}"
    elif not qwen_dir:
        status["qwen_message"] = "Qwen model folder not found under qwen25vl_model2/"
    elif not qwen_weights:
        status["qwen_message"] = "Qwen config found but model weights (.safetensors) are missing — download the full model"
    elif qwen_4bit and not bnb:
        status["qwen_message"] = "Qwen ready (will load in FP16; bitsandbytes not required)"
    else:
        status["qwen_message"] = "Qwen ready"
    return status


def _unload_qwen():
    global _qwen_processor, _qwen_model
    with _qwen_lock:
        _qwen_model = None
        _qwen_processor = None
    t = _ensure_torch()
    if t is not None:
        try:
            t.cuda.empty_cache()
        except Exception:
            pass


def _unload_owlv2():
    global _owlv2_processor, _owlv2_model
    with _owlv2_lock:
        _owlv2_model = None
        _owlv2_processor = None
    t = _ensure_torch()
    if t is not None:
        try:
            t.cuda.empty_cache()
        except Exception:
            pass


def _evict_except(keep: str):
    global _active_backend
    if keep != "owlv2":
        _unload_owlv2()
    if keep != "qwen":
        _unload_qwen()
    _active_backend = keep


def _get_owlv2():
    global _owlv2_processor, _owlv2_model
    if not owlv2_available():
        return None, None
    with _owlv2_lock:
        if _owlv2_processor is not None and _owlv2_model is not None:
            want = _device()
            try:
                have = str(next(_owlv2_model.parameters()).device)
            except Exception:
                have = "cpu"
            if (want == "cuda" and "cuda" not in have) or (want == "cpu" and "cuda" in have):
                logger.info("OWLv2 device mismatch (have %s, want %s) — reloading", have, want)
            else:
                return _owlv2_processor, _owlv2_model
    _evict_except("owlv2")
    _unload_owlv2()
    dev = _device()
    with _owlv2_lock:
        if _owlv2_processor is None:
            try:
                local = _owlv2_local_dir()
                if local:
                    logger.info("Loading OWLv2 from local weights: %s", local)
                    _owlv2_processor = Owlv2Processor.from_pretrained(local, local_files_only=True)
                    _owlv2_model = Owlv2ForObjectDetection.from_pretrained(local, local_files_only=True)
                else:
                    if OWLV2_LOCAL_PATH and not _has_model_weights(OWLV2_LOCAL_PATH):
                        logger.warning(
                            "OWLv2 folder has config but no weight files — downloading %s from HuggingFace",
                            OWLV2_MODEL_ID,
                        )
                    logger.info("Loading OWLv2 from HuggingFace: %s", OWLV2_MODEL_ID)
                    _owlv2_processor = Owlv2Processor.from_pretrained(OWLV2_MODEL_ID)
                    _owlv2_model = Owlv2ForObjectDetection.from_pretrained(OWLV2_MODEL_ID)
                if torch is not None:
                    _owlv2_model = _owlv2_model.to(dev)
                    if dev == "cuda":
                        try:
                            _owlv2_model = _owlv2_model.half()
                        except Exception:
                            pass
                    try:
                        gpu_name = torch.cuda.get_device_name(0) if dev == "cuda" else dev
                    except Exception:
                        gpu_name = dev
                    logger.info("OWLv2 loaded on %s (%s)", dev, gpu_name)
                _owlv2_model.eval()
            except Exception as e:
                logger.error("OWLv2 load failed: %s", e, exc_info=True)
                return None, None
        return _owlv2_processor, _owlv2_model


def _qwen_local_dir():
    p = _qwen_config_dir()
    if p and _has_model_weights(p):
        return p
    return None


def _get_qwen():
    global _qwen_processor, _qwen_model
    if not qwen_available():
        return None, None
    with _qwen_lock:
        if _qwen_processor is not None:
            return _qwen_processor, _qwen_model
    _evict_except("qwen")
    dev = _device()
    local = _qwen_local_dir()
    with _qwen_lock:
        if _qwen_processor is None:
            try:
                proc_kw = {"local_files_only": True, "trust_remote_code": True}
                model_kw = dict(proc_kw)
                t = _ensure_torch()
                use_4bit = bool(local and _qwen_needs_4bit(local) and _bitsandbytes_ok() and t is not None)
                if use_4bit:
                    from transformers import BitsAndBytesConfig

                    model_kw["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=t.bfloat16 if dev == "cuda" else t.float32,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    model_kw["device_map"] = "auto" if dev == "cuda" else None
                elif t is not None:
                    model_kw["torch_dtype"] = torch.float16 if dev == "cuda" else torch.float32
                _qwen_processor = AutoProcessor.from_pretrained(local, **proc_kw)
                _qwen_model = AutoModelForVision2Seq.from_pretrained(local, **model_kw)
                if torch is not None and not model_kw.get("device_map"):
                    _qwen_model = _qwen_model.to(dev)
                _qwen_model.eval()
                logger.info("Qwen2.5-VL loaded on %s from %s (4bit=%s)", dev, local, use_4bit)
            except Exception as e:
                logger.error("Qwen load failed: %s", e)
                return None, None
        return _qwen_processor, _qwen_model


def _parse_labels(prompt_text):
    if not prompt_text:
        return []
    return [s.strip() for s in prompt_text.split(",") if s.strip()]


def _attach_text_labels(detection_output, text_labels):
    """Map class index outputs to label strings (older transformers)."""
    if not text_labels:
        for image_output in detection_output:
            image_output["text_labels"] = None
        return detection_output
    if len(text_labels) != len(detection_output):
        raise ValueError("text_labels batch size must match detection output batch size")
    for image_output, image_text_labels in zip(detection_output, text_labels):
        labels = image_output.get("labels")
        if labels is None:
            image_output["text_labels"] = None
            continue
        idxs = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        image_output["text_labels"] = [
            image_text_labels[i] if 0 <= i < len(image_text_labels) else str(i) for i in idxs
        ]
    return detection_output


def _box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _filter_detections_norm(dets, conf, frame_w, frame_h, nms_iou=OWLV2_NMS_IOU, max_det=OWLV2_MAX_DETECTIONS, min_area=OWLV2_MIN_BOX_AREA):
    """NMS + size filter. dets use pixel boxes on the source frame."""
    parsed = []
    for det in dets:
        sc = float(det["score"])
        if sc < conf:
            continue
        x1, y1, x2, y2 = det["box"]
        if x2 <= x1 or y2 <= y1:
            continue
        if _box_area((x1, y1, x2, y2)) < min_area:
            continue
        # Store normalized so overlay stays correct if live frame size differs
        parsed.append(
            {
                "box_norm": (x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h),
                "box": (x1, y1, x2, y2),
                "score": sc,
                "label": str(det["label"]),
            }
        )

    parsed.sort(key=lambda d: d["score"], reverse=True)
    kept = []
    for det in parsed:
        if len(kept) >= max_det:
            break
        if any(_box_iou(det["box"], k["box"]) >= nms_iou for k in kept):
            continue
        kept.append(det)
    return kept


def _center_to_corners_xyxy(boxes):
    """cx,cy,w,h (normalized or absolute) → x1,y1,x2,y2."""
    cx, cy, bw, bh = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - 0.5 * bw
    y1 = cy - 0.5 * bh
    x2 = cx + 0.5 * bw
    y2 = cy + 0.5 * bh
    return torch.stack((x1, y1, x2, y2), dim=-1)


def _owlv2_processor_input_size(proc):
    """Fixed square size OWLv2 image processor pads/resizes to."""
    img_proc = getattr(proc, "image_processor", None) or proc
    size = getattr(img_proc, "size", None)
    if isinstance(size, dict):
        return int(size.get("height") or size.get("width") or size.get("longest_edge") or 960)
    if isinstance(size, (int, float)):
        return int(size)
    return 960


def _owlv2_decode_boxes(proc, outputs, inputs, orig_h, orig_w, threshold, text_labels):
    """
    Map OWLv2 pred_boxes onto the original frame — accounting for resize+pad-to-square.
    Naive target_sizes scaling ignores padding and shifts boxes (usually upward).
    """
    if torch is None or not hasattr(outputs, "pred_boxes"):
        # Last-resort: library post-process (may be misaligned on older transformers)
        target_sizes = torch.tensor([(orig_h, orig_w)])
        return _post_process_owlv2_legacy(proc, outputs, target_sizes, threshold, text_labels)

    pred_boxes = outputs.pred_boxes  # (B, N, 4) cxcywh in [0,1] relative to padded square
    logits = outputs.logits  # (B, N, num_queries/labels)
    scores = torch.sigmoid(logits)
    best_scores, best_labels = scores.max(dim=-1)

    # Padded model input size (usually square, e.g. 960)
    pv = inputs.get("pixel_values") if isinstance(inputs, dict) else None
    if pv is not None and hasattr(pv, "shape") and len(pv.shape) == 4:
        pad_h, pad_w = int(pv.shape[2]), int(pv.shape[3])
    else:
        side = _owlv2_processor_input_size(proc)
        pad_h = pad_w = side

    # OWLv2: scale so max(h,w) → pad size, then pad bottom/right to square
    scale = min(pad_w / float(orig_w), pad_h / float(orig_h))
    # content size inside padded canvas
    content_w = orig_w * scale
    content_h = orig_h * scale

    boxes_xyxy = _center_to_corners_xyxy(pred_boxes[0].float().cpu())
    # From normalized padded coords → pixel on padded canvas → unpad/unscale → orig frame
    boxes_xyxy = boxes_xyxy * torch.tensor([pad_w, pad_h, pad_w, pad_h], dtype=torch.float32)
    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]] / scale
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]] / scale

    labels_list = text_labels[0] if text_labels else []
    sc = best_scores[0].float().cpu()
    lb = best_labels[0].cpu()

    dets = []
    for i in range(boxes_xyxy.shape[0]):
        score = float(sc[i].item())
        if score < threshold:
            continue
        x1, y1, x2, y2 = [float(v) for v in boxes_xyxy[i].tolist()]
        x1 = max(0.0, min(float(orig_w - 1), x1))
        x2 = max(0.0, min(float(orig_w), x2))
        y1 = max(0.0, min(float(orig_h - 1), y1))
        y2 = max(0.0, min(float(orig_h), y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        # Drop boxes that mostly sit in the padded void (after unscale they can be OOB)
        if x1 >= orig_w or y1 >= orig_h:
            continue
        li = int(lb[i].item())
        label = labels_list[li] if 0 <= li < len(labels_list) else str(li)
        dets.append({"box": (int(x1), int(y1), int(x2), int(y2)), "score": score, "label": label})

    # Silence unused (kept for clarity if pad mode changes)
    _ = (content_w, content_h)
    return dets


def _post_process_owlv2_legacy(proc, outputs, target_sizes, threshold, text_labels):
    """Compatible with old and new transformers Owlv2Processor APIs (fallback only)."""
    if hasattr(proc, "post_process_grounded_object_detection"):
        return proc.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold,
            text_labels=text_labels,
        )

    if hasattr(proc, "post_process_object_detection"):
        results = proc.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold,
        )
        return _attach_text_labels(results, text_labels)

    img_proc = getattr(proc, "image_processor", None)
    if img_proc is not None and hasattr(img_proc, "post_process_object_detection"):
        results = img_proc.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold,
        )
        return _attach_text_labels(results, text_labels)

    raise AttributeError(
        "Owlv2Processor has no supported post-processing method. "
        "Upgrade transformers: pip install -U 'transformers>=4.46'"
    )


def _draw_box(frame, x1, y1, x2, y2, label, color=(0, 200, 120)):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_hud(vis, title, subtitle, color=(70, 70, 78)):
    info_h, info_w = 72, 420
    overlay = vis[6 : 6 + info_h, 8 : 8 + info_w].copy()
    bg = np.zeros_like(overlay)
    bg[:] = (28, 28, 32)
    vis[6 : 6 + info_h, 8 : 8 + info_w] = cv2.addWeighted(overlay, 0.35, bg, 0.65, 0)
    cv2.rectangle(vis, (8, 6), (8 + info_w, 6 + info_h), color, 1)
    cv2.putText(vis, title, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, subtitle[:62] + ("..." if len(subtitle) > 62 else ""), (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 2, cv2.LINE_AA)


def _clean_qwen_text(text):
    t = (text or "").strip()
    t = re.sub(r"<\|[^|]+\|>", "", t)
    return t.strip()


def _parse_monitor_response(text, lenient=False):
    t = (text or "").strip()
    tl = t.lower().lstrip()
    if tl.startswith("clear") or tl in ("ok", "none", "no"):
        return False, ""
    if tl.startswith("alert:") or tl.startswith("seen:"):
        return True, t.split(":", 1)[1].strip() or "person"
    if "alert:" in tl:
        idx = tl.find("alert:")
        tail = t[idx + len("alert:") :].strip().splitlines()[0].strip()
        return True, tail or "person"
    if lenient:
        keys = ("person", "man", "woman", "worker", "enter", "exit", "door", "hallway", "corridor", "room")
        if any(k in tl for k in keys) and not tl.startswith("no "):
            return True, t.splitlines()[0].strip()[:220]
    return False, ""


class BetaOwlv2Processor:
    INFER_MAX_WIDTH = OWLV2_INFER_MAX_WIDTH

    def __init__(self, camera_id, camera_reader, prompt_texts, conf=0.2):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.prompt_texts = list(prompt_texts or [])
        self.conf = max(0.05, float(conf))
        self.running = False
        self.lock = threading.Lock()
        self._thread = None
        self._overlay_dets = []
        self._overlay_labels = []
        self._frame_size = (0, 0)
        self._annotated_frame = None
        self.stats = {"detections": 0, "status": "Initializing", "device": VLM_DEVICE}

    def _all_labels(self):
        out, seen = [], set()
        for pt in self.prompt_texts:
            for lbl in _parse_labels(pt):
                key = lbl.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(lbl)
        return out

    def start(self):
        if not owlv2_available():
            self.stats["status"] = "OWLv2 not available"
            return
        if not self._all_labels():
            self.stats["status"] = "No labels"
            return
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run_loop(self):
        proc, model = _get_owlv2()
        if proc is None or model is None:
            self.stats["status"] = "Model unavailable"
            self.running = False
            return
        labels = self._all_labels()
        text_labels = [labels]
        last_t = 0.0
        post_thresh = max(0.01, self.conf)
        device = next(model.parameters()).device if torch is not None else "cpu"
        min_interval = _infer_min_interval(device)
        logger.info(
            "Beta OWLv2 %s: device=%s interval=%.3fs (boxes locked to inferred frame)",
            self.camera_id,
            device,
            min_interval,
        )
        while self.running:
            try:
                now = time.time()
                if min_interval > 0:
                    wait = min_interval - (now - last_t)
                    if wait > 0:
                        time.sleep(min(wait, 0.01))
                        continue
                # Always take the newest frame so boxes match what is on screen.
                frame = self.camera_reader.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                last_t = time.time()
                h, w = frame.shape[:2]
                infer_frame = frame
                if w > self.INFER_MAX_WIDTH:
                    scale = self.INFER_MAX_WIDTH / float(w)
                    infer_frame = cv2.resize(
                        frame,
                        (self.INFER_MAX_WIDTH, max(1, int(round(h * scale)))),
                        interpolation=cv2.INTER_AREA,
                    )
                ih, iw = infer_frame.shape[:2]
                rgb = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                inputs = proc(text=text_labels, images=pil_image, return_tensors="pt")
                if torch is not None:
                    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
                with _owlv2_infer_lock:
                    if torch is not None:
                        with torch.inference_mode():
                            if _device_is_gpu(device):
                                with torch.autocast(device_type="cuda", dtype=torch.float16):
                                    outputs = model(**inputs)
                            else:
                                outputs = model(**inputs)
                    else:
                        outputs = model(**inputs)

                raw_dets = _owlv2_decode_boxes(
                    proc, outputs, inputs, ih, iw, post_thresh, text_labels
                )
                sx = w / float(iw)
                sy = h / float(ih)
                mapped = []
                for d in raw_dets:
                    x1, y1, x2, y2 = d["box"]
                    mapped.append(
                        {
                            "box": (
                                int(round(x1 * sx)),
                                int(round(y1 * sy)),
                                int(round(x2 * sx)),
                                int(round(y2 * sy)),
                            ),
                            "score": d["score"],
                            "label": d["label"],
                        }
                    )
                min_area = max(120, int(OWLV2_MIN_BOX_AREA * (w * h) / (1920 * 1080) * 0.35))
                kept = _filter_detections_norm(
                    mapped, self.conf, w, h, min_area=min_area
                )
                vis = frame.copy()
                for det in kept:
                    x1, y1, x2, y2 = det["box"]
                    lbl = f"{det['label']} {det['score']:.2f}"
                    _draw_box(vis, x1, y1, x2, y2, lbl)
                if not self._decorate_frame(vis, kept, labels):
                    dev_short = "GPU" if _device_is_gpu(device) else "CPU"
                    _draw_hud(
                        vis,
                        f"Beta (OWLv2/{dev_short})  |  Detections: {len(kept)}",
                        f"Labels: {', '.join(labels[:6])}{'...' if len(labels) > 6 else ''}",
                    )
                self._on_detections(frame, vis, kept, labels)
                with self.lock:
                    self._overlay_dets = kept
                    self._overlay_labels = labels
                    self._frame_size = (w, h)
                    self._annotated_frame = vis
                self.stats = {
                    "detections": len(kept),
                    "status": "Active",
                    "device": str(device),
                }
            except Exception as e:
                logger.error("Beta OWLv2 %s: %s", self.camera_id, e)
                self.stats["status"] = str(e)[:40]
                time.sleep(0.05)

    def _decorate_frame(self, vis, kept, labels):
        return False

    def _on_detections(self, frame, vis, kept, labels):
        return

    def render_display_frame(self, live_frame):
        """Prefer the frame that was actually inferred so boxes stay locked to objects."""
        with self.lock:
            if self._annotated_frame is not None:
                return self._annotated_frame.copy()
        if live_frame is None:
            return None
        vis = live_frame.copy()
        with self.lock:
            dets = list(self._overlay_dets)
            labels = list(self._overlay_labels)
        for det in dets:
            if "box_norm" in det:
                lh, lw = vis.shape[:2]
                nx1, ny1, nx2, ny2 = det["box_norm"]
                x1, y1, x2, y2 = int(nx1 * lw), int(ny1 * lh), int(nx2 * lw), int(ny2 * lh)
            else:
                x1, y1, x2, y2 = det["box"]
            lbl = f"{det['label']} {det['score']:.2f}"
            _draw_box(vis, x1, y1, x2, y2, lbl)
        device_tag = self.stats.get("device", VLM_DEVICE)
        dev_short = "GPU" if _device_is_gpu(device_tag) else "CPU"
        _draw_hud(
            vis,
            f"Beta (OWLv2/{dev_short})  |  Detections: {len(dets)}",
            f"Labels: {', '.join(labels[:6])}{'...' if len(labels) > 6 else ''}",
        )
        return vis

    def get_annotated_frame(self):
        with self.lock:
            if self._annotated_frame is not None:
                return self._annotated_frame.copy()
        live = self.camera_reader.get_frame()
        if live is None:
            return None
        return self.render_display_frame(live)

    def get_last_detections(self):
        with self.lock:
            dets = list(self._overlay_dets)
            fw, fh = self._frame_size
        out = []
        for det in dets:
            if det.get("box"):
                x1, y1, x2, y2 = det["box"]
            elif det.get("box_norm") and fw and fh:
                nx1, ny1, nx2, ny2 = det["box_norm"]
                x1, y1, x2, y2 = int(nx1 * fw), int(ny1 * fh), int(nx2 * fw), int(ny2 * fh)
            else:
                continue
            out.append(((x1, y1, x2, y2), f"{det['label']} {det['score']:.2f}", (0, 200, 120)))
        return out


class BetaQwenProcessor:
    BETA_FRAME_INTERVAL = 1.0 / 1.0
    INFER_MAX_SIZE = 640

    def __init__(
        self,
        camera_id,
        camera_reader,
        prompt_texts,
        conf=0.2,
        *,
        alert_detection_type=None,
        prompt_mode="food_plant",
    ):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.prompt_texts = list(prompt_texts or [])
        self.conf = conf
        self.alert_detection_type = alert_detection_type
        self.prompt_mode = prompt_mode
        self.running = False
        self.lock = threading.Lock()
        self._thread = None
        self._alert_label = ""
        self._alert_rule = ""
        self._total_alerts = 0
        self._annotated_frame = None
        self.stats = {"detections": 0, "alerts": 0, "status": "Initializing", "device": VLM_DEVICE}

    def _all_rules(self):
        out, seen = [], set()
        for pt in self.prompt_texts:
            s = (pt or "").strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out

    def start(self):
        if not qwen_available():
            self.stats["status"] = "Qwen not available"
            return
        if not self._all_rules():
            self.stats["status"] = "No instructions"
            return
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _emit_alert(self, frame, label, rule):
        from backend.alerts import save_alert_event

        save_alert_event(
            self.camera_id,
            self.alert_detection_type,
            frame,
            [(0, label)],
            severity="high",
            meta={"rule": rule, "message": label},
        )

    def _build_prompt(self, instruction):
        if self.prompt_mode == "instruction":
            return (
                f"{instruction.strip()}\n"
                "Check only what is visible in this frame. "
                "If an anomaly or safety violation matching the instruction is present, "
                "respond exactly in one line: ALERT:<short one-line description>. "
                "Otherwise respond exactly: CLEAR"
            )
        return (
            "You are an intelligent food processing plant video monitor. "
            f"Instruction: {instruction}\n"
            "Check only what is visible in this frame. "
            "If an instructed anomaly/safety violation is present, respond exactly in one line: "
            "ALERT:<short_label>. Otherwise respond exactly: CLEAR"
        )

    def _qwen_gen(self, proc, model, text_in, pil_image):
        ins = None
        if hasattr(proc, "apply_chat_template"):
            try:
                msgs = [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": text_in}]}]
                ins = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
            except Exception:
                ins = None
        if ins is None:
            ins = proc(text=text_in, images=pil_image, return_tensors="pt")
        if torch is not None:
            dvc = next(model.parameters()).device
            ins = {k: v.to(dvc) if hasattr(v, "to") else v for k, v in ins.items()}
        max_new = int(getattr(self, "max_new_tokens", 32))
        with _qwen_infer_lock:
            if torch is not None:
                with torch.inference_mode():
                    oids = model.generate(**ins, max_new_tokens=max_new, num_beams=1, do_sample=False)
            else:
                oids = model.generate(**ins, max_new_tokens=max_new, num_beams=1, do_sample=False)
        try:
            in_len = int(ins["input_ids"].shape[-1])
            gen_only = oids[:, in_len:]
            out = proc.batch_decode(gen_only, skip_special_tokens=True)[0].strip()
        except Exception:
            out = proc.batch_decode(oids, skip_special_tokens=True)[0].strip()
        return _clean_qwen_text(out)

    def _run_loop(self):
        proc, model = _get_qwen()
        if proc is None or model is None:
            self.stats["status"] = "Qwen not loaded"
            self.running = False
            return
        rules = self._all_rules()
        last_t = 0.0
        total_alerts = 0
        try:
            model_device = next(model.parameters()).device
        except Exception:
            model_device = VLM_DEVICE
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
                if max(pil_image.size) > self.INFER_MAX_SIZE:
                    pil_image.thumbnail((self.INFER_MAX_SIZE, self.INFER_MAX_SIZE), Image.Resampling.LANCZOS)
                triggered_label = ""
                triggered_rule = ""
                for rule in rules:
                    prompt = self._build_prompt(rule)
                    response = self._qwen_gen(proc, model, prompt, pil_image)
                    is_alert, label = _parse_monitor_response(
                        response, lenient=getattr(self, "parse_lenient", False)
                    )
                    if is_alert:
                        triggered_label = label
                        triggered_rule = rule
                        total_alerts += 1
                        if self.alert_detection_type:
                            self._emit_alert(frame, label, rule)
                        break
                with self.lock:
                    self._alert_label = triggered_label
                    self._alert_rule = triggered_rule if triggered_rule else (
                        "Watching for people" if getattr(self, "parse_lenient", False) else "Monitoring — no anomaly in frame"
                    )
                    self._total_alerts = total_alerts
                    vis = frame.copy()
                    if triggered_label:
                        h, w = vis.shape[:2]
                        _draw_box(vis, 8, 8, w - 8, h - 8, f"Anomaly: {triggered_label}", (0, 0, 255))
                    dev_short = "GPU" if _device_is_gpu(model_device) else "CPU"
                    _draw_hud(
                        vis,
                        f"Beta (Qwen/{dev_short})  |  Alerts: {total_alerts}",
                        triggered_rule if triggered_rule else "Monitoring — no anomaly in frame",
                        (180, 60, 60),
                    )
                    self._annotated_frame = vis
                self.stats = {
                    "detections": 1 if triggered_label else 0,
                    "alerts": total_alerts,
                    "status": "Active",
                    "device": str(model_device),
                }
            except Exception as e:
                logger.error("Beta Qwen %s: %s", self.camera_id, e)
                self.stats["status"] = str(e)[:40]
                time.sleep(0.3)

    def render_display_frame(self, live_frame):
        with self.lock:
            if self._annotated_frame is not None:
                return self._annotated_frame.copy()
        if live_frame is None:
            return None
        vis = live_frame.copy()
        with self.lock:
            label = self._alert_label
            rule = self._alert_rule
            alerts = self._total_alerts
        if label:
            h, w = vis.shape[:2]
            _draw_box(vis, 8, 8, w - 8, h - 8, f"Anomaly: {label}", (0, 0, 255))
        dev_short = "GPU" if _device_is_gpu(self.stats.get("device", "")) else "CPU"
        _draw_hud(vis, f"Beta (Qwen/{dev_short})  |  Alerts: {alerts}", rule, (180, 60, 60))
        return vis

    def get_annotated_frame(self):
        with self.lock:
            if self._annotated_frame is not None:
                return self._annotated_frame.copy()
        live = self.camera_reader.get_frame()
        if live is None:
            return None
        return self.render_display_frame(live)


def make_beta_processor(camera_id, camera_reader, prompt_texts, conf, backend):
    if backend == "qwen":
        return BetaQwenProcessor(camera_id, camera_reader, prompt_texts, conf)
    return BetaOwlv2Processor(camera_id, camera_reader, prompt_texts, conf)


def unload_beta_models_if_idle(active_count):
    if active_count == 0:
        _unload_owlv2()
        _unload_qwen()
