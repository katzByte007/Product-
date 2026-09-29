import sys
import types

import numpy as np

from backend.yolo_world import YoloWorldRunner, normalize_labels


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, index):
        return self.value[index]

    def tolist(self):
        return self.value


class FakeBox:
    xyxy = FakeTensor([[1.2, 2.1, 31.8, 42.4]])
    cls = FakeTensor([0])
    conf = FakeTensor([0.91])


def test_normalize_labels_deduplicates_without_splitting_phrases():
    assert normalize_labels(["White helmet", " white helmet ", "blue gloves"]) == (
        "White helmet",
        "blue gloves",
    )


def test_runner_sets_prompt_once_and_returns_normalized_detections(monkeypatch):
    calls = {"load": 0, "labels": [], "predict": 0}

    class FakeResult:
        boxes = [FakeBox()]
        names = {0: "white helmet"}

    class FakeModel:
        def __init__(self, path):
            calls["load"] += 1

        def set_classes(self, labels):
            calls["labels"].append(labels)

        def predict(self, source, **kwargs):
            calls["predict"] += 1
            return [FakeResult()]

    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLOWorld = FakeModel
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    runner = YoloWorldRunner(model_path="fake-world.pt", device="cpu")
    first = runner.detect(np.zeros((64, 64, 3), dtype=np.uint8), ["white helmet"])
    second = runner.detect(np.zeros((64, 64, 3), dtype=np.uint8), ["white helmet"])

    assert first == [{"box": (1, 2, 32, 42), "score": 0.91, "label": "white helmet"}]
    assert second == first
    assert calls == {"load": 1, "labels": [["white helmet"]], "predict": 2}