import numpy as np
import time

import backend.video_reader as video_reader


class FakeCapture:
    def __init__(self, reader, frames):
        self.reader = reader
        self.frames = iter(frames)
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        item = next(self.frames)
        if item == "stop":
            self.reader.running = False
            return True, np.zeros((4, 4, 3), dtype=np.uint8)
        return item, None

    def release(self):
        self.released = True


def test_rtsp_read_failure_releases_and_reopens(monkeypatch):
    reader = video_reader.VideoFileReader("cam-1", "rtsp://example.invalid/live")
    first = FakeCapture(reader, ["failure"])
    second = FakeCapture(reader, ["stop"])
    captures = iter([first, second])
    waits = []

    class FakeStopEvent:
        def wait(self, delay):
            waits.append((delay, reader.status))
            return False

    reader._stop_event = FakeStopEvent()
    reader._open_capture = lambda: next(captures)
    monkeypatch.setattr(video_reader.time, "sleep", lambda _: None)
    reader.running = True

    reader._read_loop()

    assert first.released is True
    assert second.released is True
    assert waits == [(1.0, "reconnecting")]


def test_rtsp_capture_opens_with_bounded_timeouts(monkeypatch):
    calls = []

    class OpenCapture:
        def set(self, *_):
            return True

        def open(self, *args):
            calls.append(args)
            return True

        def isOpened(self):
            return True

        def get(self, _):
            return 25.0

    monkeypatch.setattr(video_reader.cv2, "VideoCapture", OpenCapture)
    reader = video_reader.VideoFileReader("cam-1", "rtsp://example.invalid/live")

    reader._open_capture()

    assert calls == [(
        "rtsp://example.invalid/live",
        video_reader.cv2.CAP_FFMPEG,
        [
            video_reader.cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            5000,
            video_reader.cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            5000,
        ],
    )]


def test_reader_health_rejects_stale_and_reconnecting_frames(monkeypatch):
    reader = video_reader.VideoFileReader("cam-1", "rtsp://example.invalid/live")
    reader.status = "active"
    reader.frame = np.zeros((4, 4, 3), dtype=np.uint8)
    reader.frame_ts = time.time()

    assert reader.health["healthy"] is True
    assert reader.get_frame() is not None

    reader.frame_ts -= video_reader.CAMERA_FRAME_STALE_SEC + 1
    assert reader.health["status"] == "stale"
    assert reader.health["healthy"] is False
    assert reader.get_frame() is None

    reader.status = "reconnecting"
    assert reader.health["status"] == "reconnecting"
    assert reader.health["healthy"] is False