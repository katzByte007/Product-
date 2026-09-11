"""Per-camera JPEG publisher — encode once, serve many snapshot clients."""
import hashlib
import logging
import threading
import time

import cv2

from config import DISPLAY_FPS, DISPLAY_JPEG_QUALITY, DISPLAY_MAX_WIDTH

logger = logging.getLogger(__name__)

_JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), DISPLAY_JPEG_QUALITY]
_publishers = {}
_pub_lock = threading.Lock()


def _build_display_frame(camera_id: str):
    from backend.display_frame import build_display_frame

    return build_display_frame(camera_id)


class StreamPublisher:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self._jpeg = None
        self._jpeg_lock = threading.Lock()
        self._running = False
        self._thread = None
        self._interval = 1.0 / max(1.0, DISPLAY_FPS)
        self._stagger_s = (int(hashlib.md5(camera_id.encode()).hexdigest()[:8], 16) % 1000) / 1000.0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name=f"StreamPub-{self.camera_id}",
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        with self._jpeg_lock:
            self._jpeg = None

    def get_jpeg(self):
        with self._jpeg_lock:
            return self._jpeg

    def _scale(self, frame):
        if DISPLAY_MAX_WIDTH <= 0:
            return frame
        h, w = frame.shape[:2]
        if w <= DISPLAY_MAX_WIDTH:
            return frame
        scale = DISPLAY_MAX_WIDTH / float(w)
        return cv2.resize(frame, (DISPLAY_MAX_WIDTH, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    def _loop(self):
        if self._stagger_s > 0:
            time.sleep(self._stagger_s)
        while self._running:
            t0 = time.perf_counter()
            try:
                frame = _build_display_frame(self.camera_id)
                if frame is not None:
                    frame = self._scale(frame)
                    ok, buf = cv2.imencode(".jpg", frame, _JPEG_PARAMS)
                    if ok:
                        with self._jpeg_lock:
                            self._jpeg = buf.tobytes()
            except Exception as e:
                logger.debug("StreamPublisher %s: %s", self.camera_id, e)
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.001, self._interval - elapsed))


def start_stream_publisher(camera_id: str):
    with _pub_lock:
        pub = _publishers.get(camera_id)
        if pub is None:
            pub = StreamPublisher(camera_id)
            _publishers[camera_id] = pub
            pub.start()


def stop_stream_publisher(camera_id: str):
    with _pub_lock:
        pub = _publishers.pop(camera_id, None)
    if pub:
        pub.stop()


def get_stream_publisher(camera_id: str):
    return _publishers.get(camera_id)
