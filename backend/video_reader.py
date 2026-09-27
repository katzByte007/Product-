"""Looping video file reader — smooth playback for demo camera streams."""
import logging
import threading
import time

import cv2

from config import CAMERA_FRAME_STALE_SEC, VIDEO_BUFFER_SIZE, VIDEO_READ_SLEEP

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
        self._stop_event = threading.Event()
        self.lock = threading.Lock()
        self.status = "initializing"
        self._thread = None
        self._target_interval = 1.0 / 30.0
        # Incremented each time the file rewinds — trackers use this to avoid ID inflation
        self.loop_generation = 0

    def start(self):
        if self.running:
            return
        self._stop_event.clear()
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
        if self.video_path.lower().startswith(("rtsp://", "rtsps://")):
            timeout_ms = 5000
            params = [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                timeout_ms,
            ]
            cap.open(self.video_path, cv2.CAP_FFMPEG, params)
        else:
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
        reconnect_delay = 1.0
        is_live_source = self.video_path.lower().startswith(("rtsp://", "rtsps://"))
        while self.running:
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                try:
                    cap = self._open_capture()
                except Exception:
                    cap = None
                    logger.exception("Camera %s: capture open failed", self.camera_id)
                if cap is None or not cap.isOpened():
                    self.status = "reconnecting" if is_live_source else "error"
                    logger.error("Camera %s: cannot open %s", self.camera_id, self.video_path)
                    if cap is not None:
                        cap.release()
                    cap = None
                    self._stop_event.wait(reconnect_delay if is_live_source else 2.0)
                    reconnect_delay = min(reconnect_delay * 2.0, 30.0)
                    continue
                self.status = "active"
                logger.info(
                    "Camera %s: playing %s (rotation=%s)",
                    self.camera_id,
                    self.video_path,
                    self.rotation,
                )

            t0 = time.perf_counter()
            try:
                ret, frame = cap.read()
            except Exception:
                logger.exception("Camera %s: capture read failed", self.camera_id)
                ret, frame = False, None
            if not ret or frame is None:
                if is_live_source:
                    logger.warning("Camera %s: RTSP read failed; reconnecting", self.camera_id)
                    cap.release()
                    cap = None
                    self.status = "reconnecting"
                    self._stop_event.wait(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2.0, 30.0)
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.loop_generation += 1
                    time.sleep(max(VIDEO_READ_SLEEP, 0.01))
                continue

            reconnect_delay = 1.0
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
            if self.frame is None or self.status != "active":
                return None
            if time.time() - self.frame_ts > CAMERA_FRAME_STALE_SEC:
                return None
            return self.frame.copy()

    @property
    def health(self):
        with self.lock:
            frame_ts = self.frame_ts
        age_sec = max(0.0, time.time() - frame_ts) if frame_ts else None
        fresh = age_sec is not None and age_sec <= CAMERA_FRAME_STALE_SEC
        return {
            "status": self.status if fresh or self.status != "active" else "stale",
            "healthy": self.status == "active" and fresh,
            "frame_age_sec": round(age_sec, 3) if age_sec is not None else None,
        }

    def stop(self):
        self.running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
