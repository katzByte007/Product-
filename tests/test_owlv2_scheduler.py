import threading

import numpy as np

from backend.owlv2_scheduler import Owlv2Scheduler


class FakeReader:
    def __init__(self, frame):
        self.frame = frame
        self.copy_modes = []

    def get_frame(self, copy=True):
        self.copy_modes.append(copy)
        return self.frame


def test_scheduler_borrows_reader_frame_for_callback():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    reader = FakeReader(frame)
    received = []
    callback_called = threading.Event()

    def on_frame(value):
        received.append(value)
        callback_called.set()

    scheduler = Owlv2Scheduler(worker_count=1)
    try:
        scheduler.register(
            "cam-1",
            reader,
            on_frame,
            interval=0.01,
        )
        assert callback_called.wait(1.0)
    finally:
        scheduler.stop()

    assert False in reader.copy_modes
    assert received[0] is frame