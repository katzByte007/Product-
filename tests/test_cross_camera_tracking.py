import threading

import numpy as np

from backend.cross_camera_tracking import (
    CameraTransition,
    CrossCameraTracker,
    TrackObservation,
    crops_from_tracks,
    observations_from_tracks,
)


def observation(camera, track_id, timestamp, *, zone=None, embedding=None):
    return TrackObservation(
        camera_id=camera,
        local_track_id=track_id,
        timestamp=timestamp,
        frame_id=int(timestamp * 10),
        bbox=(1, 2, 30, 60),
        zone_id=zone,
        embedding=embedding,
    )


def test_topology_and_time_window_create_candidate_not_identity():
    tracker = CrossCameraTracker([
        CameraTransition("cam-a", "cam-b", 2, 30, "exit", "entry")
    ], tracklet_gap_sec=1)

    tracker.observe_camera("cam-a", "headcount", [observation("cam-a", 17, 10, zone="exit")], now=10)
    tracker.observe_camera("cam-a", "headcount", [observation("cam-a", 17, 11, zone="exit")], now=11)
    tracker.observe_camera("cam-a", "headcount", [], now=12)
    matches = tracker.observe_camera(
        "cam-b", "headcount", [observation("cam-b", 42, 17, zone="entry")], now=17
    )

    assert len(matches) == 1
    assert matches[0].status == "candidate"
    assert matches[0].source_camera_id == "cam-a"
    assert matches[0].destination_camera_id == "cam-b"
    assert matches[0].transition_seconds == 6


def test_rejects_out_of_window_and_wrong_zone():
    tracker = CrossCameraTracker([
        CameraTransition("cam-a", "cam-b", 2, 5, "exit", "entry")
    ], tracklet_gap_sec=1)

    tracker.observe_camera("cam-a", "local", [observation("cam-a", 1, 10, zone="other")], now=10)
    tracker.observe_camera("cam-a", "local", [], now=12)
    assert tracker.observe_camera(
        "cam-b", "local", [observation("cam-b", 2, 20, zone="entry")], now=20
    ) == []


def test_ambiguous_sources_are_not_associated():
    tracker = CrossCameraTracker([
        CameraTransition("cam-a", "cam-b", 1, 20)
    ], tracklet_gap_sec=1)

    tracker.observe_camera(
        "cam-a", "local", [observation("cam-a", 1, 10), observation("cam-a", 2, 10)], now=10
    )
    tracker.observe_camera("cam-a", "local", [], now=12)
    assert tracker.observe_camera(
        "cam-b", "local", [observation("cam-b", 3, 15)], now=15
    ) == []


def test_embedding_similarity_is_a_hard_gate_when_available():
    tracker = CrossCameraTracker([
        CameraTransition("cam-a", "cam-b", 1, 20)
    ], tracklet_gap_sec=1, min_embedding_similarity=0.8)
    tracker.observe_camera(
        "cam-a", "local", [observation("cam-a", 1, 10, embedding=np.array([1.0, 0.0]))], now=10
    )
    tracker.observe_camera("cam-a", "local", [], now=12)

    matches = tracker.observe_camera(
        "cam-b", "local", [observation("cam-b", 2, 15, embedding=np.array([0.0, 1.0]))], now=15
    )
    assert matches == []

def test_similarity_threshold_is_unset_by_default():
    tracker = CrossCameraTracker()

    assert tracker.min_embedding_similarity is None

def test_observations_from_tracks_preserves_local_ids_and_zone():
    observations = observations_from_tracks(
        "cam-a",
        [{"id": 17, "bbox": (1.2, 2, 30, 60), "center": (15, 30)}],
        12.5,
        42,
        zone_resolver=lambda track: "exit" if track["center"][0] > 10 else None,
    )

    assert len(observations) == 1
    assert observations[0].camera_id == "cam-a"
    assert observations[0].local_track_id == 17
    assert observations[0].bbox == (1, 2, 30, 60)
    assert observations[0].zone_id == "exit"
    assert observations[0].frame_id == 42


def test_transition_rejects_invalid_times_and_same_camera():
    import pytest

    with pytest.raises(ValueError):
        CameraTransition("cam-a", "cam-b", 5, 2)
    with pytest.raises(ValueError):
        CameraTransition("cam-a", "cam-a", 0, 2)


def test_multiple_zone_routes_between_cameras_are_supported():
    tracker = CrossCameraTracker([
        CameraTransition("cam-a", "cam-b", 1, 10, "north", "entry-north"),
        CameraTransition("cam-a", "cam-b", 1, 10, "south", "entry-south"),
    ], tracklet_gap_sec=1)
    tracker.observe_camera("cam-a", "local", [observation("cam-a", 1, 10, zone="south")], now=10)
    tracker.observe_camera("cam-a", "local", [], now=12)

    matches = tracker.observe_camera(
        "cam-b", "local", [observation("cam-b", 2, 15, zone="entry-south")], now=15
    )

    assert len(matches) == 1
    assert matches[0].transition_seconds == 5


def test_source_reset_invalidates_closed_tracklets():
    tracker = CrossCameraTracker([
        CameraTransition("cam-a", "cam-b", 1, 20)
    ], tracklet_gap_sec=1)
    tracker.observe_camera("cam-a", "local", [observation("cam-a", 1, 10)], now=10)
    tracker.observe_camera("cam-a", "local", [], now=12)
    tracker.reset_source("cam-a", "local")

    assert tracker.observe_camera(
        "cam-b", "local", [observation("cam-b", 2, 15)], now=15
    ) == []


def test_different_local_tracker_namespaces_are_not_compared():
    tracker = CrossCameraTracker([
        CameraTransition("cam-a", "cam-b", 1, 20)
    ], tracklet_gap_sec=1)
    tracker.observe_camera("cam-a", "headcount", [observation("cam-a", 1, 10)], now=10)
    tracker.observe_camera("cam-a", "headcount", [], now=12)

    assert tracker.observe_camera(
        "cam-b", "entryexit", [observation("cam-b", 1, 15)], now=15
    ) == []


def test_bounded_reid_worker_embeds_sampled_crops_before_association():
    class FakeProvider:
        def __init__(self):
            self.calls = 0
            self.ready = threading.Event()

        def embed(self, crops):
            self.calls += len(crops)
            self.ready.set()
            return np.array([[1.0, 0.0]], dtype=np.float32)

    provider = FakeProvider()
    tracker = CrossCameraTracker(
        [CameraTransition("cam-a", "cam-b", 2, 30)],
        tracklet_gap_sec=1,
        reid_provider=provider,
        reid_queue_size=2,
    )
    start = 1_800_000_000.0
    crop = np.ones((64, 32, 3), dtype=np.uint8)
    tracker.observe_camera(
        "cam-a", "headcount", [observation("cam-a", 1, start)], now=start,
        crops={1: crop},
    )
    assert provider.ready.wait(2)
    tracker.observe_camera("cam-a", "headcount", [], now=start + 2)
    provider.ready.clear()
    tracker.observe_camera(
        "cam-b", "headcount", [observation("cam-b", 2, start + 7)], now=start + 7,
        crops={2: crop},
    )
    assert provider.ready.wait(2)
    tracker.observe_camera("cam-b", "headcount", [], now=start + 9)

    assert len(tracker.candidates()) == 1
    assert tracker.reid_stats()["completed"] == 2
    tracker.set_reid_provider(None)


def test_crop_extractor_clips_boxes_and_skips_low_quality_crops():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crops = crops_from_tracks(frame, [
        {"id": 1, "bbox": (-10, -5, 40, 80)},
        {"id": 2, "bbox": (10, 10, 30, 50)},
    ])

    assert set(crops) == {1}
    assert crops[1].shape == (80, 40, 3)


def test_reid_queue_drops_overflow_without_blocking_submit():
    from backend.cross_camera_tracking import BoundedReIDWorker

    class BlockingProvider:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def embed(self, crops):
            self.started.set()
            self.release.wait(2)
            return np.array([[1.0, 0.0]], dtype=np.float32)

    provider = BlockingProvider()
    worker = BoundedReIDWorker(provider, lambda _key, _vector: None, max_queue_size=1)
    assert worker.submit("first", np.zeros((64, 32, 3), dtype=np.uint8))
    assert provider.started.wait(2)
    assert worker.submit("queued", np.zeros((64, 32, 3), dtype=np.uint8))
    assert not worker.submit("overflow", np.zeros((64, 32, 3), dtype=np.uint8))
    assert worker.stats()["dropped"] == 1
    provider.release.set()
    worker.stop()