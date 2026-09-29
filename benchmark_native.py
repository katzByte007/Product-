"""Native YOLO + OWLv2 capacity benchmark.

This benchmark uses the real local model weights and cloned video readers. It
keeps one latest frame per logical camera/task, measures queue and model timing,
and writes one row per mode/camera-count combination. It is intentionally
separate from the application runtime so a benchmark failure cannot alter the
running VMS process.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import statistics
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import psutil

SOURCE_FEEDS = [
    "data/videos/cam4_industry_ppe.mp4",
    "data/videos/cam5_pharma_ppe.mp4",
    "data/videos/cam6_lab_ppe.mp4",
]
COUNTS = (3, 6, 12, 25, 50, 75, 100)
MODES = (
    "yolo-single",
    "yolo",
    "owlv2-multiprompt",
    "owlv2",
    "mixed",
    "baseline",
    "optimized",
)

TASKS_BY_MODE = {
    "yolo-single": ("yolo-headcount",),
    "yolo": ("yolo-headcount", "yolo-entryexit"),
    "owlv2-multiprompt": ("owlv2-multiprompt",),
    "owlv2": ("owlv2-person", "owlv2-forklift", "owlv2-person-hardhat"),
    "mixed": (
        "yolo-headcount",
        "yolo-entryexit",
        "owlv2-person",
        "owlv2-forklift",
        "owlv2-person-hardhat",
    ),
    "baseline": ("owlv2-person", "owlv2-forklift", "owlv2-person-hardhat"),
    "optimized": ("owlv2-person", "owlv2-forklift", "owlv2-person-hardhat"),
}


def percentile(values, fraction):
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    index = min(len(values) - 1, int(round((len(values) - 1) * fraction)))
    return float(values[index])


def feeds_for(count):
    return [SOURCE_FEEDS[index % len(SOURCE_FEEDS)] for index in range(count)]


def resource_snapshot(process, torch_module=None, device="cpu"):
    snapshot = {
        "cpu_percent": process.cpu_percent(None),
        "rss_gb": process.memory_info().rss / (1024 ** 3),
        "gpu_memory_gb": None,
        "gpu_utilization_pct": None,
        "gpu_temperature_c": None,
    }
    if torch_module is not None and str(device).startswith("cuda"):
        try:
            snapshot["gpu_memory_gb"] = torch_module.cuda.memory_allocated() / (1024 ** 3)
        except Exception:
            pass
    return snapshot


class NativeScheduler:
    """Fair latest-frame scheduler with timing metadata for each admitted job."""

    def __init__(self, worker_count):
        self.worker_count = max(1, int(worker_count))
        self.condition = threading.Condition()
        self.jobs = {}
        self.ready = []
        self.running = True
        self.submitted = 0
        self.dropped = 0
        self.completed = 0
        self.errors = 0
        self.submitted_by_camera = defaultdict(int)
        self._threads = [threading.Thread(target=self._worker, daemon=True) for _ in range(self.worker_count)]
        for thread in self._threads:
            thread.start()

    def register(self, key, reader, callback, interval):
        with self.condition:
            self.jobs[key] = {"reader": reader, "callback": callback, "interval": interval, "last_submit": 0.0, "frame": None, "queued": False}

    def submit_due(self):
        now = time.monotonic()
        with self.condition:
            jobs = list(self.jobs.items())
        for key, job in jobs:
            if now - job["last_submit"] < job["interval"]:
                continue
            frame = job["reader"].get_frame()
            if frame is None:
                continue
            submitted_at = time.perf_counter()
            with self.condition:
                current = self.jobs.get(key)
                if current is None:
                    continue
                current["last_submit"] = now
                camera_id = key.split(":", 2)[1]
                self.submitted_by_camera[camera_id] += 1
                if current["frame"] is not None:
                    self.dropped += 1
                current["frame"] = (frame, submitted_at, time.time())
                self.submitted += 1
                if not current["queued"]:
                    current["queued"] = True
                    self.ready.append(key)
                self.condition.notify()

    def _worker(self):
        while True:
            with self.condition:
                while self.running and not self.ready:
                    self.condition.wait(0.1)
                if not self.running:
                    return
                key = self.ready.pop(0)
                job = self.jobs.get(key)
                if job is None or job["frame"] is None:
                    continue
                frame_data = job["frame"]
                job["frame"] = None
                job["queued"] = False
            try:
                job["callback"](key, *frame_data)
                self.completed += 1
            except Exception:
                self.errors += 1

    def stop(self):
        with self.condition:
            self.running = False
            self.jobs.clear()
            self.ready.clear()
            self.condition.notify_all()
        for thread in self._threads:
            thread.join(timeout=3.0)


class Metrics:
    def __init__(self, mode, count):
        self.mode = mode
        self.count = count
        self.lock = threading.Lock()
        self.samples = defaultdict(list)
        self.per_camera = defaultdict(lambda: {"requested": 0, "completed": 0, "frame_ages": []})
        for camera in range(count):
            self.per_camera[str(camera)]
        self.detections = 0
        self.alerts = 0
        self.skipped = 0
        self.actual_forwards = 0
        self.model_errors = 0
        self.error_messages = []

    def add(self, camera, values, detections=0, alert=False):
        with self.lock:
            for key, value in values.items():
                self.samples[key].append(float(value))
            item = self.per_camera[str(camera)]
            item["completed"] += 1
            item["frame_ages"].append(float(values.get("frame_age_ms", 0.0)))
            self.detections += int(detections)
            self.alerts += int(bool(alert))

    def request(self, camera):
        with self.lock:
            self.per_camera[str(camera)]["requested"] += 1

    def skip(self):
        with self.lock:
            self.skipped += 1


def load_readers(count):
    from backend.video_reader import VideoFileReader
    readers = []
    for index, path in enumerate(feeds_for(count)):
        reader = VideoFileReader(f"native-bench-{index}", path)
        reader.start()
        readers.append(reader)
    return readers


def stop_readers(readers):
    for reader in readers:
        reader.running = False
    join_threads = []
    for reader in readers:
        if reader._thread and reader._thread.is_alive():
            join_thread = threading.Thread(target=reader._thread.join, kwargs={"timeout": 5.0}, daemon=True)
            join_thread.start()
            join_threads.append(join_thread)
    for join_thread in join_threads:
        join_thread.join(timeout=5.5)


class YoloRunner:
    def __init__(self, device, imgsz, tasks):
        from ultralytics import YOLO
        from config import ENTRYEXIT_MODEL_PATH, HEADCOUNT_MODEL_PATH
        self.device = device
        self.imgsz = imgsz
        model_paths = {
            "yolo-headcount": HEADCOUNT_MODEL_PATH,
            "yolo-entryexit": ENTRYEXIT_MODEL_PATH,
        }
        self.models = {
            task: YOLO(model_paths[task])
            for task in tasks
            if task in model_paths
        }
        self.locks = {key: threading.Lock() for key in self.models}

    def infer(self, task, frame):
        started = time.perf_counter()
        with self.locks[task]:
            results = self.models[task](frame, imgsz=self.imgsz, device=self.device, verbose=False, stream=False)
        model_ms = (time.perf_counter() - started) * 1000.0
        detections = 0
        if results:
            try:
                detections = len(results[0].boxes)
            except Exception:
                detections = 0
        return model_ms, detections


class Owlv2Runner:
    def __init__(self, device, max_width, labels):
        os.environ["VISION_VLM_DEVICE"] = device
        from backend import beta_ai
        self.beta = beta_ai
        self.device = device
        self.max_width = max_width
        self.labels = beta_ai._get_cached_text_labels(labels)
        self.text_labels = [self.labels]
        self.processor, self.model = beta_ai._get_owlv2()
        if self.processor is None or self.model is None:
            raise RuntimeError("OWLv2 could not be loaded; check models/owlv2 and Transformers")
        self.actual_device = str(next(self.model.parameters()).device)
        self.previous = {}
        self.last_trigger = defaultdict(float)
        self.lock = threading.Lock()

    def infer(self, camera, frame, optimized):
        beta = self.beta
        started = time.perf_counter()
        previous = self.previous.get(camera)
        self.previous[camera] = frame.copy()
        roi = None
        if optimized and previous is not None:
            now = time.monotonic()
            motion = beta._motion_score(previous, frame)
            heartbeat = now - self.last_trigger[camera] >= 1.5
            if motion <= 0.012 and not heartbeat:
                return 0.0, 0.0, 0.0, 0, True
            self.last_trigger[camera] = now
            roi = beta._motion_roi(previous, frame, threshold=0.012, pad=32)
        infer_frame = frame
        offset_x = offset_y = 0
        if roi is not None:
            x1, y1, x2, y2 = roi
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                infer_frame = crop
                offset_x, offset_y = x1, y1
        prep_start = time.perf_counter()
        if infer_frame.shape[1] > self.max_width:
            scale = self.max_width / float(infer_frame.shape[1])
            infer_frame = cv2.resize(infer_frame, (self.max_width, max(1, int(round(infer_frame.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
        ih, iw = infer_frame.shape[:2]
        image = self.beta.Image.fromarray(cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(text=self.text_labels, images=image, return_tensors="pt")
        inputs = {key: value.to(self.actual_device) if hasattr(value, "to") else value for key, value in inputs.items()}
        prep_ms = (time.perf_counter() - prep_start) * 1000.0
        forward_start = time.perf_counter()
        with self.lock:
            with self.beta.torch.inference_mode():
                if self.actual_device == "cuda" and beta.OWLV2_USE_AUTOCast and beta.OWLV2_USE_FP16:
                    with self.beta.torch.autocast(device_type="cuda", dtype=self.beta.torch.float16):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)
        model_ms = (time.perf_counter() - forward_start) * 1000.0
        post_start = time.perf_counter()
        raw = beta._owlv2_decode_boxes(self.processor, outputs, inputs, ih, iw, 0.2, self.text_labels)
        post_ms = (time.perf_counter() - post_start) * 1000.0
        _ = (offset_x, offset_y)
        return prep_ms, model_ms, post_ms, len(raw), False


def run_mode(mode, count, duration, workers, interval, device, imgsz, owl_width):
    from backend import beta_ai
    process = psutil.Process()
    readers = load_readers(count)
    metrics = Metrics(mode, count)
    tasks = TASKS_BY_MODE[mode]
    yolo = YoloRunner(device, imgsz, tasks) if any(task.startswith("yolo") for task in tasks) else None
    owl = Owlv2Runner(device, owl_width, ("person", "forklift", "hardhat")) if any(task.startswith("owlv2") for task in tasks) else None
    scheduler = NativeScheduler(workers)
    for camera, reader in enumerate(readers):
        for task in tasks:
            key = f"camera:{camera}:task:{task}"
            scheduler.register(key, reader, lambda key, frame, submitted_at, frame_ts, task=task, camera=camera: callback(key, frame, submitted_at, frame_ts, task, camera, mode, yolo, owl, metrics, process, device), interval=interval)
    time.sleep(1.0)
    process.cpu_percent(None)
    start = time.perf_counter()
    while time.perf_counter() - start < duration:
        scheduler.submit_due()
        time.sleep(0.005)
    time.sleep(max(1.0, interval * 2.0))
    elapsed = time.perf_counter() - start
    scheduler.stop()
    stop_readers(readers)
    stats = scheduler.__dict__
    completed = metrics.samples["end_to_end_ms"]
    row = {
        "mode": mode, "camera_count": count, "device": getattr(owl, "actual_device", device), "workers": workers,
        "logical_tasks_per_camera": len(tasks),
        "prompts_per_owlv2_forward": 3 if owl is not None else 0,
        "task_interval_sec": interval, "run_duration_sec": duration,
        "requested_forwards": metrics.actual_forwards + metrics.skipped,
        "actual_model_forwards": metrics.actual_forwards,
        "completed_forwards": len(completed), "useful_detections": metrics.detections,
        "alerts": metrics.alerts, "skipped_by_gate": metrics.skipped,
        "submitted": scheduler.submitted, "dropped": scheduler.dropped, "errors": scheduler.errors + metrics.model_errors,
        "drop_rate": scheduler.dropped / max(1, scheduler.submitted),
        "completion_ratio": len(completed) / max(1, scheduler.submitted),
        "actual_forwards_per_sec": metrics.actual_forwards / max(elapsed, 1e-6),
        "requested_forwards_per_sec": (metrics.actual_forwards + metrics.skipped) / max(elapsed, 1e-6),
        "completed_forwards_per_sec": len(completed) / max(elapsed, 1e-6),
        "useful_throughput_per_sec": metrics.detections / max(elapsed, 1e-6),
        "latency_p50_ms": percentile(completed, 0.50), "latency_p95_ms": percentile(completed, 0.95), "latency_p99_ms": percentile(completed, 0.99),
        "frame_age_p50_ms": percentile(metrics.samples["frame_age_ms"], 0.50), "frame_age_p95_ms": percentile(metrics.samples["frame_age_ms"], 0.95), "frame_age_p99_ms": percentile(metrics.samples["frame_age_ms"], 0.99),
        "queue_wait_p50_ms": percentile(metrics.samples["queue_wait_ms"], 0.50), "queue_wait_p95_ms": percentile(metrics.samples["queue_wait_ms"], 0.95), "queue_wait_p99_ms": percentile(metrics.samples["queue_wait_ms"], 0.99),
        "preprocess_p50_ms": percentile(metrics.samples["preprocess_ms"], 0.50), "preprocess_p95_ms": percentile(metrics.samples["preprocess_ms"], 0.95),
        "model_p50_ms": percentile(metrics.samples["model_ms"], 0.50), "model_p95_ms": percentile(metrics.samples["model_ms"], 0.95), "model_p99_ms": percentile(metrics.samples["model_ms"], 0.99),
        "postprocess_p50_ms": percentile(metrics.samples["postprocess_ms"], 0.50), "postprocess_p95_ms": percentile(metrics.samples["postprocess_ms"], 0.95),
        "cpu_percent": process.cpu_percent(None), "rss_gb": process.memory_info().rss / (1024 ** 3),
        "gpu_memory_gb": None, "gpu_utilization_pct": None, "gpu_temperature_c": None,
        "per_camera_completion": json.dumps({str(k): v["completed"] / max(1, scheduler.submitted_by_camera.get(str(k), 0)) for k, v in metrics.per_camera.items()}),
        "per_camera_frame_age_p95_ms": json.dumps({str(k): percentile(v["frame_ages"], 0.95) for k, v in metrics.per_camera.items()}),
        "camera_starvation_count": sum(1 for camera in range(count) if metrics.per_camera[str(camera)]["completed"] == 0),
        "alert_latency_p95_ms": percentile(metrics.samples["end_to_end_ms"], 0.95),
        "source_feeds": json.dumps(feeds_for(count)),
        "error_messages": json.dumps(metrics.error_messages[:10]),
    }
    return row


def callback(key, frame, submitted_at, frame_ts, task, camera, mode, yolo, owl, metrics, process, device):
    metrics.request(camera)
    queue_wait_ms = (time.perf_counter() - submitted_at) * 1000.0
    frame_age_ms = (time.time() - frame_ts) * 1000.0
    try:
        if task.startswith("yolo"):
            prep_start = time.perf_counter()
            prep_ms = (time.perf_counter() - prep_start) * 1000.0
            model_ms, detections = yolo.infer(task, frame)
            post_ms = 0.0
            metrics.actual_forwards += 1
        else:
            result = owl.infer(camera, frame, mode == "optimized")
            if result[4]:
                metrics.skip()
                return
            prep_ms, model_ms, post_ms, detections, _skipped = result
            metrics.actual_forwards += 1
        end_to_end_ms = queue_wait_ms + prep_ms + model_ms + post_ms
        metrics.add(camera, {"queue_wait_ms": queue_wait_ms, "frame_age_ms": frame_age_ms, "preprocess_ms": prep_ms, "model_ms": model_ms, "postprocess_ms": post_ms, "end_to_end_ms": end_to_end_ms}, detections=detections, alert=detections > 0)
    except Exception as exc:
        metrics.model_errors += 1
        with metrics.lock:
            metrics.error_messages.append(f"{task}: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}")


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Native YOLO + OWLv2 benchmark")
    parser.add_argument("--camera-counts", default=",".join(map(str, COUNTS)))
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--task-interval", type=float, default=0.5)
    parser.add_argument("--device", default="mps", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--owl-width", type=int, default=480)
    parser.add_argument("--output", default="data/benchmarks/native_model_benchmark.csv")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    os.environ["VISION_VLM_DEVICE"] = args.device
    counts = [int(value) for value in args.camera_counts.split(",") if value.strip()]
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    rows = []
    output_path = Path(args.output)
    if args.append and output_path.exists():
        with output_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        completed = {(row["mode"], int(row["camera_count"])) for row in rows}
    else:
        completed = set()
    for mode in modes:
        for count in counts:
            if (mode, count) in completed:
                print(json.dumps({"skipping_existing": mode, "cameras": count}), flush=True)
                continue
            print(json.dumps({"starting": mode, "cameras": count}), flush=True)
            rows.append(run_mode(mode, count, args.duration, args.workers, args.task_interval, args.device, args.imgsz, args.owl_width))
            write_csv(rows, output_path)
            print(json.dumps(rows[-1], sort_keys=True), flush=True)
    print(json.dumps({"rows": len(rows), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
