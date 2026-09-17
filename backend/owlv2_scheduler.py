"""Bounded latest-frame scheduler for shared OWLv2 inference."""
import collections
import logging
import threading
import time

from config import OWLV2_WORKERS

logger = logging.getLogger(__name__)


class Owlv2Scheduler:
    """Run camera jobs with bounded workers and no stale frame backlog."""

    def __init__(self, worker_count=1):
        self.worker_count = max(1, int(worker_count))
        self._condition = threading.Condition()
        self._jobs = {}
        self._ready = collections.deque()
        self._cursor = 0
        self._workers = []
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="OWLv2Dispatcher"
        )
        self._running = True
        self.submitted = 0
        self.completed = 0
        self.dropped = 0
        self.errors = 0
        for index in range(self.worker_count):
            worker = threading.Thread(
                target=self._worker,
                daemon=True,
                name=f"OWLv2Worker-{index + 1}",
            )
            self._workers.append(worker)
            worker.start()
        self._dispatcher.start()

    def register(self, key, reader, callback, interval=0.0):
        with self._condition:
            self._jobs[key] = {
                "reader": reader,
                "callback": callback,
                "interval": max(0.0, float(interval)),
                "last_submit": 0.0,
                "frame": None,
                "queued": False,
                "last_dispatch": 0.0,
            }

    def unregister(self, key):
        with self._condition:
            self._jobs.pop(key, None)
            self._condition.notify_all()

    def submit(self, key, frame):
        with self._condition:
            job = self._jobs.get(key)
            if job is None:
                return False
            if job["frame"] is not None:
                self.dropped += 1
            job["frame"] = frame
            self.submitted += 1
            if not job["queued"]:
                job["queued"] = True
                self._ready.append(key)
            self._condition.notify()
            return True

    def _worker(self):
        while True:
            with self._condition:
                while self._running and not self._ready:
                    self._condition.wait()
                if not self._running:
                    return
                key = self._ready.popleft()
                job = self._jobs.get(key)
                if job is None:
                    continue
                frame = job["frame"]
                job["frame"] = None
                job["queued"] = False
                job["last_dispatch"] = time.monotonic()
            try:
                job["callback"](frame)
                self.completed += 1
            except Exception:
                self.errors += 1
                logger.exception("OWLv2 job failed for camera %s", key)

    def _dispatch_loop(self):
        while True:
            with self._condition:
                if not self._running:
                    return
                jobs = list(self._jobs.items())
                if jobs:
                    start = self._cursor % len(jobs)
                    jobs = jobs[start:] + jobs[:start]
                    self._cursor = (start + 1) % len(jobs)
            now = time.monotonic()
            for key, job in jobs:
                if now - job["last_submit"] < job["interval"]:
                    continue
                frame = job["reader"].get_frame()
                if frame is not None:
                    with self._condition:
                        current = self._jobs.get(key)
                        if current is not None:
                            current["last_submit"] = now
                    self.submit(key, frame)
            time.sleep(0.005)

    @property
    def stats(self):
        with self._condition:
            return {
                "workers": self.worker_count,
                "active_cameras": len(self._jobs),
                "queued_cameras": len(self._ready),
                "submitted": self.submitted,
                "completed": self.completed,
                "dropped": self.dropped,
                "errors": self.errors,
                "completion_ratio": round(
                    self.completed / max(1, self.submitted), 4
                ),
            }

    def stop(self):
        with self._condition:
            self._running = False
            self._jobs.clear()
            self._ready.clear()
            self._condition.notify_all()
        for worker in self._workers:
            worker.join(timeout=2.0)
        self._dispatcher.join(timeout=2.0)


_default_scheduler = None
_scheduler_lock = threading.Lock()


def get_owlv2_scheduler(worker_count=None):
    global _default_scheduler
    with _scheduler_lock:
        if _default_scheduler is None:
            _default_scheduler = Owlv2Scheduler(
                OWLV2_WORKERS if worker_count is None else worker_count
            )
        return _default_scheduler


def owlv2_scheduler_stats():
    with _scheduler_lock:
        return _default_scheduler.stats if _default_scheduler is not None else None
