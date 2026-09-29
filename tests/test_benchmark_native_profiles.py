import sys
import types

import config
from benchmark_native import MODES, TASKS_BY_MODE, YoloRunner


def test_single_yolo_profile_is_one_detector_per_camera():
    assert TASKS_BY_MODE["yolo-single"] == ("yolo-headcount",)


def test_single_yolo_profile_loads_only_one_model(monkeypatch):
    loaded_paths = []

    class FakeYolo:
        def __init__(self, path):
            loaded_paths.append(path)

    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeYolo
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    YoloRunner("cpu", 640, TASKS_BY_MODE["yolo-single"])

    assert loaded_paths == [config.HEADCOUNT_MODEL_PATH]


def test_multiprompt_owlv2_profile_uses_one_forward_per_camera():
    assert TASKS_BY_MODE["owlv2-multiprompt"] == ("owlv2-multiprompt",)
    assert "owlv2-multiprompt" in MODES