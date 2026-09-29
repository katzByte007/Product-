import numpy as np

from backend.beta_ai import (
    BetaOwlv2Processor,
    _build_cached_text_key,
    _motion_score,
    should_run_owlv2,
)


def test_motion_score_is_near_zero_for_static_frames():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    assert _motion_score(frame, frame) < 0.01


def test_motion_gate_skips_static_scene():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    same = frame.copy()
    assert should_run_owlv2(frame, same, threshold=0.02) is False


def test_motion_gate_runs_when_scene_changes():
    base = np.zeros((100, 120, 3), dtype=np.uint8)
    moved = base.copy()
    moved[20:60, 20:80] = 255
    assert should_run_owlv2(base, moved, threshold=0.01) is True


def test_processor_returns_roi_from_previous_frame_on_motion(monkeypatch):
    import backend.beta_ai as beta_ai

    monkeypatch.setattr(beta_ai, "OWLV2_MOTION_GATE_ENABLED", True)
    monkeypatch.setattr(beta_ai, "OWLV2_MOTION_THRESHOLD", 0.01)
    processor = BetaOwlv2Processor("cam-1", object(), ["person"])
    base = np.zeros((100, 120, 3), dtype=np.uint8)
    moved = base.copy()
    moved[20:60, 20:80] = 255

    should_infer, first_roi = processor._should_infer(base)
    assert should_infer is True
    assert first_roi is None

    should_infer, motion_roi = processor._should_infer(moved)
    assert should_infer is True
    assert motion_roi is not None
    assert motion_roi[0] < 20 and motion_roi[2] > 80


def test_cached_text_key_normalizes_labels():
    labels = ["Person", " person ", "forklift", "Person"]
    assert _build_cached_text_key(labels) == ("person", "forklift")
