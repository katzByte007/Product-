import hashlib
import sys
import types

import numpy as np
import pytest

import app as flask_app
from backend.reid.preprocessing import INPUT_HEIGHT, INPUT_WIDTH, preprocess_crops
from backend.reid.validation import evaluate_labeled_pairs, select_threshold


def test_preprocessing_returns_rgb_normalized_nchw_batch():
    crop = np.zeros((64, 32, 3), dtype=np.uint8)
    crop[:, :, 0] = 255
    batch = preprocess_crops([crop])

    assert batch.shape == (1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    assert batch.dtype == np.float32
    np.testing.assert_allclose(batch[0, 0, 0, 0], (0.0 - 0.485) / 0.229, rtol=1e-5)
    np.testing.assert_allclose(batch[0, 2, 0, 0], (1.0 - 0.406) / 0.225, rtol=1e-5)


def test_threshold_selection_enforces_false_match_rate_constraint():
    results = [
        {"same": True, "similarity": 0.91},
        {"same": True, "similarity": 0.82},
        {"same": False, "similarity": 0.75},
        {"same": False, "similarity": 0.20},
    ]

    selected = select_threshold(results, max_false_match_rate=0.0)

    assert selected["threshold"] == 0.82
    assert selected["false_match_rate"] == 0.0
    assert selected["false_reject_rate"] == 0.0


def test_labeled_pair_evaluation_returns_cosine_similarity():
    class FakeProvider:
        def embed(self, images):
            vectors = []
            for image in images:
                if image[0, 0, 0] == 0:
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.0, 1.0])
            return np.asarray(vectors, dtype=np.float32)

    black = np.zeros((2, 2, 3), dtype=np.uint8)
    white = np.full((2, 2, 3), 255, dtype=np.uint8)
    results = evaluate_labeled_pairs(FakeProvider(), [
        {"crop_a": black, "crop_b": black, "same": "same", "camera_a": "a", "camera_b": "b"},
        {"crop_a": black, "crop_b": white, "same": "different", "camera_a": "a", "camera_b": "c"},
    ])

    assert [item["similarity"] for item in results] == [1.0, 0.0]
    assert [item["same"] for item in results] == [True, False]


def test_osnet_provider_requires_local_checkpoint(tmp_path):
    from backend.reid.torchreid_osnet import OSNetAINProvider

    with pytest.raises(FileNotFoundError):
        OSNetAINProvider(tmp_path / "missing.pth")


def test_osnet_provider_returns_normalized_float32_embeddings(tmp_path):
    import torch

    from backend.reid.torchreid_osnet import OSNetAINProvider

    class FeatureModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, batch):
            return torch.full((batch.shape[0], 4), 3.0, device=batch.device)

    weights = tmp_path / "model.pth"
    weights.write_bytes(b"reviewed local checkpoint placeholder")
    provider = OSNetAINProvider(weights, device="cpu", model=FeatureModel())
    output = provider.embed([np.zeros((64, 32, 3), dtype=np.uint8)] * 2)

    assert output.shape == (2, 4)
    assert output.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(output, axis=1), 1.0, atol=1e-6)


def test_reid_configuration_fails_closed_until_validated(monkeypatch):
    from backend.cross_camera_tracking import cross_camera_tracker

    monkeypatch.setattr(flask_app, "VISION_REID_ENABLED", True)
    monkeypatch.setattr(flask_app, "VISION_REID_MODEL", "osnet_ain_x1_0")
    monkeypatch.setattr(flask_app, "VISION_REID_VALIDATION_STATUS", "pending")
    monkeypatch.setattr(flask_app, "VISION_REID_VALIDATION_DATASET", "")

    status = flask_app._configure_reid_provider()

    assert status["enabled"] is False
    assert status["reason"] == "validation_not_approved"
    assert cross_camera_tracker.reid_enabled is False


def test_reid_configuration_rejects_checkpoint_hash_mismatch(monkeypatch, tmp_path):
    from backend.cross_camera_tracking import cross_camera_tracker

    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"changed after approval")
    monkeypatch.setattr(flask_app, "VISION_REID_ENABLED", True)
    monkeypatch.setattr(flask_app, "VISION_REID_MODEL", "osnet_ain_x1_0")
    monkeypatch.setattr(flask_app, "VISION_REID_VALIDATION_STATUS", "approved")
    monkeypatch.setattr(flask_app, "VISION_REID_VALIDATION_DATASET", "plant-v1")
    monkeypatch.setattr(flask_app, "VISION_REID_SIMILARITY_THRESHOLD", "0.83")
    monkeypatch.setattr(flask_app, "VISION_REID_WEIGHTS", str(checkpoint))
    monkeypatch.setattr(flask_app, "VISION_REID_WEIGHTS_SHA256", "0" * 64)

    status = flask_app._configure_reid_provider()

    assert status["enabled"] is False
    assert status["reason"] == "checkpoint_sha256_mismatch"
    assert cross_camera_tracker.reid_enabled is False


def test_reid_configuration_installs_osnet_only_after_approval(monkeypatch, tmp_path):
    from backend.cross_camera_tracking import cross_camera_tracker

    class FakeProvider:
        def __init__(self, weights_path, *, device):
            assert weights_path == str(tmp_path / "weights.pth")
            assert device == "cpu"
            self.device_name = "cpu"

        def embed(self, _crops):
            return np.ones((1, 4), dtype=np.float32)

    module = types.ModuleType("backend.reid.torchreid_osnet")
    module.OSNetAINProvider = FakeProvider
    monkeypatch.setitem(sys.modules, "backend.reid.torchreid_osnet", module)
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"approved test checkpoint")
    monkeypatch.setattr(flask_app, "VISION_REID_ENABLED", True)
    monkeypatch.setattr(flask_app, "VISION_REID_MODEL", "osnet_ain_x1_0")
    monkeypatch.setattr(flask_app, "VISION_REID_VALIDATION_STATUS", "approved")
    monkeypatch.setattr(flask_app, "VISION_REID_VALIDATION_DATASET", "plant-v1")
    monkeypatch.setattr(flask_app, "VISION_REID_SIMILARITY_THRESHOLD", "0.83")
    monkeypatch.setattr(flask_app, "VISION_REID_WEIGHTS", str(checkpoint))
    monkeypatch.setattr(
        flask_app,
        "VISION_REID_WEIGHTS_SHA256",
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(flask_app, "VISION_REID_DEVICE", "cpu")
    monkeypatch.setattr(flask_app, "VISION_REID_QUEUE_SIZE", 3)

    status = flask_app._configure_reid_provider()

    assert status["enabled"] is True
    assert status["validation_dataset"] == "plant-v1"
    assert cross_camera_tracker.reid_enabled is True
    assert cross_camera_tracker.min_embedding_similarity == 0.83
    cross_camera_tracker.set_reid_provider(None)