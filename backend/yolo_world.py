"""Shared Ultralytics YOLO-World runtime for prompt-based Autotrack."""
import logging
import threading

from config import VLM_DEVICE, YOLO_WORLD_MODEL

logger = logging.getLogger(__name__)


def normalize_labels(labels):
    normalized = []
    seen = set()
    for label in labels or []:
        value = str(label).strip()
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            normalized.append(value)
    return tuple(normalized)


class YoloWorldRunner:
    """Own one YOLO-World model and serialize prompt updates and forwards."""

    def __init__(self, model_path=YOLO_WORLD_MODEL, device=VLM_DEVICE):
        self.model_path = model_path
        self.device = device
        self._lock = threading.Lock()
        self._model = None
        self._labels = ()

    @staticmethod
    def available():
        try:
            from ultralytics import YOLOWorld
            return YOLOWorld is not None
        except Exception:
            return False

    def _load(self):
        if self._model is None:
            from ultralytics import YOLOWorld
            self._model = YOLOWorld(self.model_path)
            logger.info("YOLO-World loaded from %s", self.model_path)
        return self._model

    def prepare(self, labels):
        prompt_labels = normalize_labels(labels)
        if not prompt_labels:
            raise ValueError("YOLO-World requires at least one prompt label")
        with self._lock:
            model = self._load()
            if self._labels != prompt_labels:
                model.set_classes(list(prompt_labels))
                self._labels = prompt_labels

    def detect(self, frame, labels, confidence=0.2, imgsz=640):
        prompt_labels = normalize_labels(labels)
        if frame is None or not prompt_labels:
            return []

        with self._lock:
            model = self._load()
            if self._labels != prompt_labels:
                model.set_classes(list(prompt_labels))
                self._labels = prompt_labels
            options = {
                "conf": float(confidence),
                "imgsz": int(imgsz),
                "verbose": False,
                "stream": False,
            }
            if self.device in ("cpu", "cuda", "mps"):
                options["device"] = self.device
            results = model.predict(source=frame, **options)

        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        names = getattr(result, "names", {})
        detections = []
        for box in boxes:
            coords = box.xyxy[0]
            if hasattr(coords, "cpu"):
                coords = coords.cpu()
            if hasattr(coords, "tolist"):
                coords = coords.tolist()
            class_id = int(box.cls[0])
            score = float(box.conf[0])
            label = names.get(class_id, prompt_labels[class_id] if class_id < len(prompt_labels) else "object")
            detections.append({
                "box": tuple(int(round(float(value))) for value in coords[:4]),
                "score": score,
                "label": str(label),
            })
        return detections


_runner = None
_runner_lock = threading.Lock()


def get_yolo_world_runner():
    global _runner
    with _runner_lock:
        if _runner is None:
            _runner = YoloWorldRunner()
        return _runner