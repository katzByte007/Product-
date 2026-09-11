"""Looping video file reader — smooth playback for demo camera streams."""
import logging
import threading
import time

import cv2

from config import VIDEO_BUFFER_SIZE, VIDEO_READ_SLEEP

logger = logging.getLogger(__name__)

_ROTATE_FLAGS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class VideoFileReader:
    """Reads MP4/WebM files in a loop with minimal buffering."""

    def __init__(self, camera_id: str, video_path: str, rotation: int = 0):
        self.camera_id = camera_id
        self.video_path = video_path
        self.rotation = int(rotation or 0) % 360
        if self.rotation not in (0, 90, 180, 270):
            self.rotation = 0
        self.frame = None
        self.frame_ts = 0.0
        self.running = False
        self.lock = threading.Lock()
        self.status = "initializing"
        self._thread = None
        self._target_interval = 1.0 / 30.0
        # Incremented each time the file rewinds — trackers use this to avoid ID inflation
        self.loop_generation = 0

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
            name=f"VideoRead-{self.camera_id}",
        )
        self._thread.start()

    def _disable_auto_orientation(self, cap):
        """Use only DB rotation — FFmpeg display-matrix tags otherwise flip cam1 left/right."""
        prop = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", 49)
        try:
            cap.set(prop, 0)
        except Exception:
            pass

    def _open_capture(self):
        cap = cv2.VideoCapture()
        self._disable_auto_orientation(cap)
        cap.open(self.video_path)
        self._disable_auto_orientation(cap)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, VIDEO_BUFFER_SIZE)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._target_interval = 1.0 / max(1.0, min(float(fps), 30.0))
        return cap

    def _apply_rotation(self, frame):
        flag = _ROTATE_FLAGS.get(self.rotation)
        if flag is None:
            return frame
        return cv2.rotate(frame, flag)

    def _read_loop(self):
        cap = None
        while self.running:
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = self._open_capture()
                if not cap.isOpened():
                    self.status = "error"
                    logger.error("Camera %s: cannot open %s", self.camera_id, self.video_path)
                    time.sleep(2.0)
                    continue
                self.status = "active"
                logger.info(
                    "Camera %s: playing %s (rotation=%s)",
                    self.camera_id,
                    self.video_path,
                    self.rotation,
                )

            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.loop_generation += 1
                continue

            frame = self._apply_rotation(frame)
            with self.lock:
                self.frame = frame
                self.frame_ts = time.time()

            elapsed = time.perf_counter() - t0
            sleep_t = max(VIDEO_READ_SLEEP, self._target_interval - elapsed)
            time.sleep(sleep_t)

        if cap is not None:
            cap.release()
        self.status = "stopped"

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
