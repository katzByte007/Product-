"""Application settings — portable paths for Linux/Windows demo deployments."""
from __future__ import annotations

import glob
import os
import socket

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VIDEOS_DIR = os.path.normpath(os.environ.get("VISION_VIDEOS_DIR", os.path.join(DATA_DIR, "videos")))
DB_PATH = os.path.join(DATA_DIR, "vision_ai.db")

# Server bind / public URL (Linux AI server / Jupyter)
# Use FOOD_VISION_PORT for this app. VISION_PORT=8080 is ignored when set to 8080.
HOST = os.environ.get("VISION_HOST", "0.0.0.0")
DEFAULT_PORT = 8765


def resolve_port() -> int:
    if os.environ.get("FOOD_VISION_PORT", "").strip():
        return int(os.environ["FOOD_VISION_PORT"])
    env = os.environ.get("VISION_PORT", "").strip()
    if env and env != "8080":
        return int(env)
    return DEFAULT_PORT


PORT = resolve_port()
PUBLIC_HOST = os.environ.get("VISION_PUBLIC_HOST", "").strip()

# Display streaming (CPU-friendly for 6 simultaneous feeds)
DISPLAY_FPS = float(os.environ.get("VISION_DISPLAY_FPS", "20"))
DISPLAY_JPEG_QUALITY = int(os.environ.get("VISION_DISPLAY_JPEG_QUALITY", "82"))
DISPLAY_MAX_WIDTH = int(os.environ.get("VISION_DISPLAY_MAX_WIDTH", "960"))

# Video file reader
VIDEO_BUFFER_SIZE = 1
VIDEO_READ_SLEEP = 0.001

# Beta AI models (weights are not in git — run: python download_owlv2.py)
# Override local folder with VISION_OWLV2_LOCAL_PATH if the model lives elsewhere.
OWLV2_MODEL_ID = os.environ.get("VISION_OWLV2_MODEL", "google/owlv2-base-patch16-ensemble")
QWEN25VL_LOCAL_PATH = os.environ.get("VISION_QWEN25VL_LOCAL_PATH", "").strip()
OWLV2_LOCAL_PATH = os.environ.get("VISION_OWLV2_LOCAL_PATH", "").strip()


def _default_vlm_device() -> str:
    # Do NOT import torch here — that alone can hang Windows startup for 20–40s.
    # "auto" resolves to CUDA at first model load if a GPU is present (DGX / AI server).
    explicit = os.environ.get("VISION_VLM_DEVICE", "").strip().lower()
    if explicit in ("cuda", "cpu", "mps", "auto"):
        return explicit
    return "auto"


VLM_DEVICE = _default_vlm_device()

# OWLv2 tuning — 0 FPS = run as fast as the device allows (GPU) / ~2 FPS on CPU
OWLV2_INFER_FPS = float(os.environ.get("VISION_OWLV2_INFER_FPS", "0"))
OWLV2_WORKERS = max(1, int(os.environ.get("VISION_OWLV2_WORKERS", "1")))
OWLV2_SCHEDULER_FPS = max(1.0, float(os.environ.get("VISION_OWLV2_SCHEDULER_FPS", "10")))
OWLV2_INFER_MAX_WIDTH = int(os.environ.get("VISION_OWLV2_INFER_MAX_WIDTH", "640"))
OWLV2_NMS_IOU = float(os.environ.get("VISION_OWLV2_NMS_IOU", "0.42"))
OWLV2_MAX_DETECTIONS = int(os.environ.get("VISION_OWLV2_MAX_DETECTIONS", "8"))
OWLV2_MIN_BOX_AREA = int(os.environ.get("VISION_OWLV2_MIN_BOX_AREA", "900"))

# CPU thread limits (avoid oversubscription on demo machines)
TORCH_THREADS = int(os.environ.get("VISION_TORCH_THREADS", "2"))

SECRET_KEY = os.environ.get("VISION_SECRET_KEY", "food-industry-demo-secret")

# ── YOLO / detection model paths (Utthunga-compatible) ───────────────────────
MODELS_DIR = os.path.join(BASE_DIR, "models")
ALERTS_DIR = os.path.join(DATA_DIR, "alerts")
ALERT_COOLDOWN_SEC = float(os.environ.get("VISION_ALERT_COOLDOWN_SEC", "18"))


def resolve_pt_model(filename: str) -> str:
    in_models = os.path.join(MODELS_DIR, filename)
    in_root = os.path.join(BASE_DIR, filename)
    if os.path.isfile(in_models):
        return in_models
    if os.path.isfile(in_root):
        return in_root
    return in_models


def resolve_model_with_fallback(filename: str, fallback: str = "yolov8m.pt") -> str:
    path = resolve_pt_model(filename)
    if os.path.isfile(path):
        return path
    fb = resolve_pt_model(fallback)
    if os.path.isfile(fb):
        return fb
    # Ultralytics auto-downloads standard hub names when local .pt is missing
    return fallback


HEADCOUNT_MODEL_PATH = resolve_model_with_fallback("best10xemphead.pt")
ENTRYEXIT_MODEL_PATH = resolve_model_with_fallback("yolov8m.pt")
FLAPGATE_MODEL_PATH = resolve_model_with_fallback("yolov8m.pt")
WORKFORCE_MODEL_PATH = resolve_model_with_fallback("yolov8m.pt")
PPE_MODEL_PATH = resolve_model_with_fallback("pharmappenew.pt")
FIRE_SMOKE_MODEL_PATH = resolve_model_with_fallback("firensmoke.pt")
ANPR_PLATE_MODEL_PATH = resolve_model_with_fallback("ANPRlib.pt")
ANPR_VEHICLE_MODEL_PATH = resolve_model_with_fallback("LibreYOLO9s.pt")

INFERENCE_IMGSZ = int(os.environ.get("VISION_INFERENCE_IMGSZ", "640"))
TARGET_FPS = float(os.environ.get("VISION_TARGET_FPS", "20"))
MICRO_BATCH = int(os.environ.get("VISION_MICRO_BATCH", "2"))
CALLBACK_POOL_WORKERS = int(os.environ.get("VISION_CALLBACK_POOL_WORKERS", "4"))
USE_FP16 = os.environ.get("VISION_USE_FP16", "1").strip().lower() in ("1", "true", "yes")
USE_TENSORRT_IF_AVAILABLE = os.environ.get("VISION_USE_TENSORRT", "0").strip().lower() in ("1", "true", "yes")
USE_SAHI_PPE = os.environ.get("VISION_PPE_USE_SAHI", "0").strip().lower() in ("1", "true", "yes")
INFERENCE_MAX_INFLIGHT_FACTOR = float(os.environ.get("VISION_INFERENCE_MAX_INFLIGHT_FACTOR", "1.5"))

ANPR_DEFAULT_CONF = float(os.environ.get("VISION_ANPR_CONF", "0.25"))
ANPR_DEFAULT_SPEED_THRESHOLD_KMH = float(os.environ.get("VISION_ANPR_SPEED_THRESHOLD", "50"))
ANPR_METERS_PER_PIXEL = float(os.environ.get("VISION_ANPR_METERS_PER_PIXEL", "0.15"))
ANPR_PLATE_CONF = float(os.environ.get("VISION_ANPR_PLATE_CONF", "0.1"))
ANPR_VEHICLE_CONF = float(os.environ.get("VISION_ANPR_VEHICLE_CONF", "0.25"))
ANPR_IOU = float(os.environ.get("VISION_ANPR_IOU", "0.45"))
ANPR_IMGSZ = int(os.environ.get("VISION_ANPR_IMGSZ", "640"))
ANPR_MAX_DET = int(os.environ.get("VISION_ANPR_MAX_DET", "300"))
ANPR_PLATE_DETECT_INTERVAL = int(os.environ.get("VISION_ANPR_PLATE_INTERVAL", "2"))
ANPR_SPEED_SAMPLE_FRAMES = int(os.environ.get("VISION_ANPR_SPEED_SAMPLE_FRAMES", "5"))
ANPR_VEHICLE_CLASS_NAMES = frozenset(
    {"bicycle", "bike", "bus", "car", "motorbike", "motorcycle", "truck", "van"}
)

FLORENCE2_LOCAL_PATH = os.environ.get("VISION_FLORENCE2_LOCAL_PATH", "").strip()
VISION_VLM_DEVICE = VLM_DEVICE
VISION_QWEN_DEVICE = os.environ.get("VISION_QWEN_DEVICE", VLM_DEVICE).strip().lower()
VISION_QWEN_UNLOAD_AFTER_USE = os.environ.get("VISION_QWEN_UNLOAD_AFTER_USE", "0").strip().lower() in ("1", "true", "yes")
VISION_VLM_EVICT_OTHERS = os.environ.get("VISION_VLM_EVICT_OTHERS", "1").strip().lower() in ("1", "true", "yes")
VISION_QWEN_ALLOW_GPU_WITH_BETA = os.environ.get("VISION_QWEN_ALLOW_GPU_WITH_BETA", "1").strip().lower() in ("1", "true", "yes")
VISION_QWEN_MIN_PIXELS = int(os.environ.get("VISION_QWEN_MIN_PIXELS", str(256 * 28 * 28)))
VISION_QWEN_MAX_PIXELS = int(os.environ.get("VISION_QWEN_MAX_PIXELS", str(1280 * 28 * 28)))
VISION_QWEN_IMAGE_MAX_SIDE = int(os.environ.get("VISION_QWEN_IMAGE_MAX_SIDE", "1024"))
VISION_QWEN_USE_4BIT = os.environ.get("VISION_QWEN_USE_4BIT", "0").strip().lower() in ("1", "true", "yes")
VISION_QWEN_GPU_MIN_FREE_GB = float(os.environ.get("VISION_QWEN_GPU_MIN_FREE_GB", "20"))
VISION_FLORENCE_RULE_ALERT_COOLDOWN_SEC = float(os.environ.get("VISION_FLORENCE_RULE_ALERT_COOLDOWN_SEC", "6"))

FR_TARGET_FPS = max(0.5, float(os.environ.get("VISION_FR_TARGET_FPS", "4")))
FR_ORT_THREADS = max(1, int(os.environ.get("VISION_FR_ORT_THREADS", "4")))
FR_ENABLE_GENDER_AGE = os.environ.get("VISION_FR_ENABLE_GENDER_AGE", "0").strip().lower() in ("1", "true", "yes")


def ensure_data_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ALERTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)


ensure_data_dirs()


def _has_model_config(path: str) -> bool:
    return bool(path) and os.path.isfile(os.path.join(path, "config.json"))


def _has_model_weights(path: str) -> bool:
    """True when at least one weight shard exists (not just config/tokenizer)."""
    if not path or not os.path.isdir(path):
        return False
    try:
        names = os.listdir(path)
    except OSError:
        return False
    for name in names:
        lower = name.lower()
        if lower.endswith((".safetensors", ".bin")) and not name.startswith("."):
            return True
        if lower == "model.safetensors.index.json":
            return True
    for hit in glob.glob(os.path.join(path, "**", "*.safetensors"), recursive=True):
        if os.path.isfile(hit):
            return True
    for hit in glob.glob(os.path.join(path, "**", "*.bin"), recursive=True):
        if os.path.isfile(hit) and "tokenizer" not in os.path.basename(hit).lower():
            return True
    return False


def _first_config_dir(*roots: str) -> str:
    """Find first directory under roots that contains config.json and model weights."""
    for root in roots:
        if not root:
            continue
        root = os.path.normpath(root)
        if _has_model_config(root) and _has_model_weights(root):
            return root
        snapshots = os.path.join(root, "snapshots")
        if os.path.isdir(snapshots):
            for name in sorted(os.listdir(snapshots), reverse=True):
                candidate = os.path.join(snapshots, name)
                if _has_model_config(candidate) and _has_model_weights(candidate):
                    return candidate
        for hit in glob.glob(os.path.join(root, "**", "config.json"), recursive=True):
            candidate = os.path.dirname(hit)
            if _has_model_weights(candidate):
                return candidate
    return ""


def discover_owlv2_local_path() -> str:
    env_path = os.environ.get("VISION_OWLV2_LOCAL_PATH", "").strip()
    if env_path:
        p = env_path
        if os.path.isfile(p):
            p = os.path.dirname(p)
        if _has_model_config(p) and _has_model_weights(p):
            return p
    hf_home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub"
    )
    return _first_config_dir(
        os.path.join(BASE_DIR, "models", "owlv2"),
        os.path.join(BASE_DIR, "models--google--owlv2-base-patch16-ensemble"),
        os.path.join(BASE_DIR, "owlv2"),
        os.path.join(hf_home, "models--google--owlv2-base-patch16-ensemble"),
        os.path.join(BASE_DIR, "models", "models--google--owlv2-base-patch16-ensemble"),
    )


def discover_qwen_local_path() -> str:
    env_path = os.environ.get("VISION_QWEN25VL_LOCAL_PATH", "").strip()
    if env_path:
        p = env_path
        if os.path.isfile(p):
            p = os.path.dirname(p)
        if _has_model_config(p):
            return p
    # Config dir may exist without weights — used for status messages
    for root in (
        os.path.join(BASE_DIR, "qwen25vl_model2", "Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit"),
        os.path.join(BASE_DIR, "qwen25vl_model2"),
        os.path.join(BASE_DIR, "qwen25vl_model"),
        os.path.join(BASE_DIR, "models", "qwen25vl"),
    ):
        root = os.path.normpath(root)
        if _has_model_config(root):
            return root
        snapshots = os.path.join(root, "snapshots")
        if os.path.isdir(snapshots):
            for name in sorted(os.listdir(snapshots), reverse=True):
                candidate = os.path.join(snapshots, name)
                if _has_model_config(candidate):
                    return candidate
    return ""


# Resolved at import — relative to project root, works on any OS
OWLV2_LOCAL_PATH = discover_owlv2_local_path()
QWEN25VL_LOCAL_PATH = discover_qwen_local_path()


def server_access_urls(port: int | None = None) -> list[str]:
    """URLs to print at startup (public host, hostname, LAN IP, localhost)."""
    port = port or PORT
    urls: list[str] = []

    def add(host: str):
        if not host:
            return
        url = f"http://{host}:{port}"
        if url not in urls:
            urls.append(url)

    add(PUBLIC_HOST)

    try:
        add(socket.gethostname())
    except OSError:
        pass

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            add(s.getsockname()[0])
    except OSError:
        pass

    add("127.0.0.1")
    return urls


def pick_listen_port(preferred: int | None = None) -> int:
    """Return first free port starting from preferred (default 8765)."""
    start = preferred if preferred is not None else PORT
    candidates = [start] + [p for p in (8765, 8766, 8767, 8770, 8780, 8790) if p != start]
    for p in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((HOST if HOST != "" else "0.0.0.0", p))
                return p
            except OSError:
                continue
    raise SystemExit(f"No free port found (tried {candidates})")


def log_paths(logger) -> None:
    logger.info("Project root: %s", BASE_DIR)
    logger.info("Videos dir:   %s", VIDEOS_DIR)
    logger.info("Database:     %s", DB_PATH)
    logger.info("OWLv2 local:  %s", OWLV2_LOCAL_PATH or "(not found — will use HuggingFace hub)")
    logger.info("Qwen local:   %s", QWEN25VL_LOCAL_PATH or "(not found)")
    logger.info("YOLO models:  %s", MODELS_DIR)
    logger.info("VLM device:   %s (resolved at first model load if auto)", VLM_DEVICE)
