"""Run a dependency-light 100-camera / three-task scheduler benchmark.

This measures admission, batching compatibility, queue drops, and callback
throughput. It is not a substitute for a real GPU inference benchmark.
"""
import argparse
import time

from backend.inference_protocol import InferenceRequest, group_compatible_requests
from backend.owlv2_scheduler import Owlv2Scheduler


class SyntheticReader:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.sequence = 0

    def get_frame(self):
        self.sequence += 1
        return (self.camera_id, self.sequence)


def run(duration, workers, interval):
    scheduler = Owlv2Scheduler(workers)
    seen = []
    readers = []
    for camera in range(100):
        reader = SyntheticReader(camera)
        readers.append(reader)
        for task, labels in (
            ("yolo", ()),
            ("owlv2", ("person", "helmet")),
            ("owlv2", ("forklift",)),
        ):
            key = f"camera:{camera}:{task}:{','.join(labels)}"
            scheduler.register(
                key,
                reader,
                lambda frame, task=task: seen.append(task),
                interval=interval,
            )
    started = time.perf_counter()
    time.sleep(duration)
    elapsed = time.perf_counter() - started
    stats = scheduler.stats
    for camera in range(100):
        for task, labels in (
            ("yolo", ()),
            ("owlv2", ("person", "helmet")),
            ("owlv2", ("forklift",)),
        ):
            scheduler.unregister(f"camera:{camera}:{task}:{','.join(labels)}")
    scheduler.stop()
    return {
        "cameras": 100,
        "model_tasks_per_camera": 3,
        "elapsed_s": round(elapsed, 2),
        "callbacks": len(seen),
        "callbacks_per_second": round(len(seen) / elapsed, 2),
        "scheduler": stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--interval", type=float, default=0.1)
    args = parser.parse_args()
    print(run(args.duration, args.workers, args.interval))


if __name__ == "__main__":
    main()
