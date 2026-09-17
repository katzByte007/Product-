"""Replay one local feed as 100 cameras and compare scheduler behavior.

The default run exercises real OpenCV readers and three logical tasks per
camera. It measures scheduling and ingest, not neural-network accuracy.
"""
import argparse
import statistics
import threading
import time

import psutil

from backend.owlv2_scheduler import Owlv2Scheduler
from backend.video_reader import VideoFileReader


TASKS = ("yolo", "owlv2:person+helmet", "owlv2:forklift")


def make_readers(video_path, cameras):
    readers = [VideoFileReader(f"bench-{index}", video_path) for index in range(cameras)]
    for reader in readers:
        reader.start()
    return readers


def stop_readers(readers):
    for reader in readers:
        reader.stop()


def run_scheduler(video_path, cameras, duration, workers, task_interval, callback_ms):
    readers = make_readers(video_path, cameras)
    scheduler = Owlv2Scheduler(workers)
    callback_times = []
    callback_lock = threading.Lock()
    for camera, reader in enumerate(readers):
        for task in TASKS:
            key = f"camera:{camera}:{task}"
            scheduler.register(
                key,
                reader,
                lambda _frame: record_callback(callback_times, callback_lock, callback_ms),
                interval=task_interval,
            )
    process = psutil.Process()
    time.sleep(1.0)
    start = time.perf_counter()
    cpu_start = process.cpu_times()
    time.sleep(duration)
    elapsed = time.perf_counter() - start
    cpu_end = process.cpu_times()
    active = sum(reader.status == "active" for reader in readers)
    rss_gb = process.memory_info().rss / (1024 ** 3)
    cpu_s = (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system)
    for camera in range(cameras):
        for task in TASKS:
            scheduler.unregister(f"camera:{camera}:{task}")
    scheduler.stop()
    stats = scheduler.stats
    stop_readers(readers)
    return {
        "mode": "shared_scheduler",
        "feed": video_path,
        "cameras": cameras,
        "model_tasks_per_camera": len(TASKS),
        "active_readers": active,
        "duration_s": round(elapsed, 2),
        "callback_count": len(callback_times),
        "callback_fps": round(len(callback_times) / elapsed, 2),
        "callback_p50_ms": round(statistics.median(callback_times) * 1000, 2) if callback_times else 0,
        "rss_gb": round(rss_gb, 3),
        "process_cpu_seconds": round(cpu_s, 2),
        "scheduler": stats,
    }


def run_legacy(video_path, cameras, duration, task_interval, callback_ms):
    """Control run approximating the old one-polling-thread-per-task design."""
    readers = make_readers(video_path, cameras)
    callback_count = 0
    callback_times = []
    count_lock = threading.Lock()
    stop_event = threading.Event()

    def task_loop(reader):
        nonlocal callback_count
        while not stop_event.is_set():
            frame = reader.get_frame()
            if frame is not None:
                started = time.perf_counter()
                if callback_ms:
                    time.sleep(callback_ms / 1000.0)
                with count_lock:
                    callback_count += 1
                    callback_times.append(time.perf_counter() - started)
            stop_event.wait(task_interval)

    threads = [
        threading.Thread(target=task_loop, args=(reader,), daemon=True)
        for reader in readers for _task in TASKS
    ]
    for thread in threads:
        thread.start()
    process = psutil.Process()
    time.sleep(1.0)
    start = time.perf_counter()
    cpu_start = process.cpu_times()
    time.sleep(duration)
    elapsed = time.perf_counter() - start
    cpu_end = process.cpu_times()
    stop_event.set()
    for thread in threads:
        thread.join(timeout=2.0)
    active = sum(reader.status == "active" for reader in readers)
    cpu_s = (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system)
    result = {
        "mode": "legacy_polling_threads",
        "feed": video_path,
        "cameras": cameras,
        "model_tasks_per_camera": len(TASKS),
        "polling_threads": len(threads),
        "active_readers": active,
        "duration_s": round(elapsed, 2),
        "callback_count": callback_count,
        "callback_fps": round(callback_count / elapsed, 2),
        "callback_p50_ms": round(statistics.median(callback_times) * 1000, 2) if callback_times else 0,
        "rss_gb": round(process.memory_info().rss / (1024 ** 3), 3),
        "process_cpu_seconds": round(cpu_s, 2),
    }
    stop_readers(readers)
    return result


def record_callback(callback_times, callback_lock, callback_ms):
    started = time.perf_counter()
    if callback_ms:
        time.sleep(callback_ms / 1000.0)
    with callback_lock:
        callback_times.append(time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed", default="data/videos/cam4_industry_ppe.mp4")
    parser.add_argument("--cameras", type=int, default=100)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--callback-ms", type=float, default=4)
    parser.add_argument("--compare-legacy", action="store_true")
    args = parser.parse_args()
    print(run_scheduler(args.feed, args.cameras, args.duration, args.workers, args.interval, args.callback_ms))
    if args.compare_legacy:
        print(run_legacy(args.feed, args.cameras, args.duration, args.interval, args.callback_ms))


if __name__ == "__main__":
    main()
