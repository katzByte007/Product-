"""Comprehensive GPU/app capacity benchmark for cloned video feeds.

This is a deterministic workload benchmark intended to separate:
- ingest capacity
- fixed-model capacity
- OWLv2 capacity
- end-to-end scheduler behavior
- latency and stale-work pressure

It replays a small set of local MP4 files as many virtual cameras and records a
CSV of measured throughput, latency, queue health, and resource use.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import socket
import statistics
import threading
import time
from pathlib import Path

import psutil

from backend.owlv2_scheduler import Owlv2Scheduler
from backend.video_reader import VideoFileReader

SOURCE_FEEDS = [
    "data/videos/cam4_industry_ppe.mp4",
    "data/videos/cam5_pharma_ppe.mp4",
    "data/videos/cam6_lab_ppe.mp4",
]

TASK_LIBRARY = {
    "ingest": [],
    "fixed": ["yolo-headcount", "yolo-entryexit"],
    "owlv2": ["owlv2-person", "owlv2-forklift", "owlv2-person-hardhat"],
    "full": ["yolo-headcount", "yolo-entryexit", "owlv2-person", "owlv2-forklift", "owlv2-person-hardhat"],
}

MODEL_WORK_MS = {
    "yolo-headcount": 6.0,
    "yolo-entryexit": 8.0,
    "owlv2-person": 40.0,
    "owlv2-forklift": 45.0,
    "owlv2-person-hardhat": 52.0,
}


def iter_camera_feeds(camera_count: int):
    for idx in range(camera_count):
        yield SOURCE_FEEDS[idx % len(SOURCE_FEEDS)]


def make_readers(camera_count: int):
    readers = []
    for idx, feed in enumerate(iter_camera_feeds(camera_count)):
        reader = VideoFileReader(f"bench-{idx}", feed)
        reader.start()
        readers.append(reader)
    return readers


def stop_readers(readers):
    for reader in readers:
        try:
            reader.stop()
        except Exception:
            pass


def _record_callback(latencies, callback_lock, work_ms, reader, task_id):
    started = time.perf_counter()
    if work_ms > 0:
        time.sleep(work_ms / 1000.0)
    latency_ms = (time.perf_counter() - started) * 1000.0
    with callback_lock:
        latencies.append({
            "task": task_id,
            "latency_ms": latency_ms,
            "frame_age_ms": (time.time() - reader.frame_ts) * 1000.0 if reader.frame_ts else 0.0,
        })


def run_mode(mode: str, camera_count: int, duration_sec: float, workers: int, task_interval: float):
    readers = make_readers(camera_count)
    process = psutil.Process()
    metrics = {
        "mode": mode,
        "camera_count": camera_count,
        "total_tasks": len(TASK_LIBRARY[mode]),
        "source_feeds": list(iter_camera_feeds(camera_count)),
        "run_duration_sec": duration_sec,
        "workers": workers,
        "task_interval_sec": task_interval,
    }

    if mode == "ingest":
        time.sleep(1.0)
        start = time.perf_counter()
        process_cpu_start = process.cpu_times()
        read_count = 0
        read_age_samples = []
        healthy_reader_samples = 0
        unhealthy_reader_samples = 0
        per_reader_health = {
            str(reader.camera_id): {"healthy_samples": 0, "unhealthy_samples": 0}
            for reader in readers
        }
        deadline = start + duration_sec
        while time.perf_counter() < deadline:
            for reader in readers:
                health = reader.health
                reader_metrics = per_reader_health[str(reader.camera_id)]
                if health["healthy"]:
                    healthy_reader_samples += 1
                    reader_metrics["healthy_samples"] += 1
                else:
                    unhealthy_reader_samples += 1
                    reader_metrics["unhealthy_samples"] += 1
                frame = reader.get_frame()
                if frame is not None:
                    read_count += 1
                    if health["frame_age_sec"] is not None:
                        read_age_samples.append(health["frame_age_sec"] * 1000.0)
            time.sleep(0.02)
        elapsed = time.perf_counter() - start
        cpu_end = process.cpu_times()
        cpu_seconds = (cpu_end.user + cpu_end.system) - (process_cpu_start.user + process_cpu_start.system)
        readers_healthy_at_end = sum(1 for reader in readers if reader.health["healthy"])
        metrics.update({
            "read_frames": read_count,
            "read_fps": read_count / max(elapsed, 1e-6),
            "frame_age_p50_ms": statistics.median(read_age_samples) if read_age_samples else 0.0,
            "frame_age_p95_ms": statistics.quantiles(read_age_samples, n=100)[94] if len(read_age_samples) >= 10 else (max(read_age_samples) if read_age_samples else 0.0),
            "frame_age_p99_ms": statistics.quantiles(read_age_samples, n=100)[98] if len(read_age_samples) >= 100 else (max(read_age_samples) if read_age_samples else 0.0),
            "readers_configured": len(readers),
            "readers_healthy_at_end": readers_healthy_at_end,
            "healthy_reader_samples": healthy_reader_samples,
            "unhealthy_reader_samples": unhealthy_reader_samples,
            "reader_health_ratio": healthy_reader_samples / max(1, healthy_reader_samples + unhealthy_reader_samples),
            "reader_health_by_camera": per_reader_health,
            "rss_gb": process.memory_info().rss / (1024 ** 3),
            "cpu_seconds": cpu_seconds,
        })
        stop_readers(readers)
        return metrics

    scheduler = Owlv2Scheduler(workers)
    callback_lock = threading.Lock()
    callback_latencies = []
    per_task_completed = {}
    job_keys = []
    tasks = TASK_LIBRARY[mode]

    for camera_index, reader in enumerate(readers):
        for task_id in tasks:
            key = f"camera:{camera_index}:task:{task_id}"
            job_keys.append(key)
            scheduler.register(
                key,
                reader,
                lambda _frame, task_id=task_id, reader=reader: (
                    _record_callback(callback_latencies, callback_lock, MODEL_WORK_MS.get(task_id, 12.0), reader, task_id),
                    per_task_completed.setdefault(task_id, 0) or per_task_completed.__setitem__(task_id, per_task_completed.get(task_id, 0) + 1),
                ),
                interval=task_interval,
            )

    time.sleep(1.0)
    process_cpu_start = process.cpu_times()
    start = time.perf_counter()
    deadline = start + duration_sec
    while time.perf_counter() < deadline:
        time.sleep(0.05)
    elapsed = time.perf_counter() - start
    cpu_end = process.cpu_times()
    cpu_seconds = (cpu_end.user + cpu_end.system) - (process_cpu_start.user + process_cpu_start.system)
    stats = scheduler.stats

    for key in job_keys:
        scheduler.unregister(key)
    scheduler.stop()
    stop_readers(readers)

    latencies = [item["latency_ms"] for item in callback_latencies]
    frame_ages = [item["frame_age_ms"] for item in callback_latencies]
    latencies = latencies or [0.0]
    frame_ages = frame_ages or [0.0]

    metrics.update({
        "submitted": stats.get("submitted", 0),
        "completed": stats.get("completed", 0),
        "dropped": stats.get("dropped", 0),
        "errors": stats.get("errors", 0),
        "completion_ratio": stats.get("completion_ratio", 0.0),
        "callback_count": len(callback_latencies),
        "callback_fps": len(callback_latencies) / max(elapsed, 1e-6),
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": statistics.quantiles(latencies, n=100)[94] if len(latencies) >= 10 else max(latencies),
        "latency_p99_ms": statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 20 else max(latencies),
        "frame_age_p50_ms": statistics.median(frame_ages),
        "frame_age_p95_ms": statistics.quantiles(frame_ages, n=100)[94] if len(frame_ages) >= 10 else max(frame_ages),
        "frame_age_p99_ms": statistics.quantiles(frame_ages, n=100)[98] if len(frame_ages) >= 20 else max(frame_ages),
        "rss_gb": process.memory_info().rss / (1024 ** 3),
        "cpu_seconds": cpu_seconds,
        "model_task_map": {k: MODEL_WORK_MS.get(k, 12.0) for k in tasks},
    })
    return metrics


def build_rows(camera_counts, modes, duration_sec, workers, task_interval):
    rows = []
    for mode in modes:
        for camera_count in camera_counts:
            row = run_mode(mode, camera_count, duration_sec, workers, task_interval)
            row["machine"] = platform.platform()
            row["hostname"] = socket.gethostname()
            row["python_version"] = platform.python_version()
            row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            rows.append(row)
    return rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ts", "hostname", "machine", "python_version",
        "mode", "camera_count", "total_tasks", "workers", "task_interval_sec",
        "run_duration_sec", "submitted", "completed", "dropped", "errors", "completion_ratio",
        "callback_count", "callback_fps",
        "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
        "frame_age_p50_ms", "frame_age_p95_ms", "frame_age_p99_ms",
        "read_frames", "read_fps", "readers_configured", "readers_healthy_at_end",
        "healthy_reader_samples", "unhealthy_reader_samples", "reader_health_ratio",
        "reader_health_by_camera", "rss_gb", "cpu_seconds", "source_feeds", "model_task_map",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: json.dumps(v) if isinstance(v, (list, dict, tuple)) else v
                for k, v in row.items() if k in fieldnames
            })


def main():
    parser = argparse.ArgumentParser(description="Comprehensive cloned-feed benchmark")
    parser.add_argument("--camera-counts", default="3,6,12,25,50,75,100")
    parser.add_argument("--modes", default="ingest,fixed,owlv2,full")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--task-interval", type=float, default=0.2)
    parser.add_argument("--output", default="data/benchmarks/comprehensive_cloned_feed_benchmark.csv")
    args = parser.parse_args()

    counts = [int(x.strip()) for x in args.camera_counts.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    rows = build_rows(counts, modes, args.duration, args.workers, args.task_interval)
    write_csv(rows, Path(args.output))
    print(json.dumps({
        "rows": len(rows),
        "counts": counts,
        "modes": modes,
        "output": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
