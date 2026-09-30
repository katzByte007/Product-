"""In-process cross-camera candidate association over local person tracks."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import queue
from typing import Protocol
import threading
import time
import uuid

import numpy as np


@dataclass(frozen=True)
class TrackObservation:
    camera_id: str
    local_track_id: str | int
    timestamp: float
    frame_id: int
    bbox: tuple[int, int, int, int]
    class_id: str = "person"
    confidence: float = 1.0
    zone_id: str | None = None
    direction: str | None = None
    embedding: np.ndarray | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class CameraTransition:
    source_camera_id: str
    destination_camera_id: str
    min_seconds: float
    max_seconds: float
    source_zone_id: str | None = None
    destination_zone_id: str | None = None
    enabled: bool = True

    def __post_init__(self):
        if not self.source_camera_id or not self.destination_camera_id:
            raise ValueError("transition camera IDs are required")
        if self.source_camera_id == self.destination_camera_id:
            raise ValueError("a camera transition must connect different cameras")
        if (
            not math.isfinite(self.min_seconds)
            or not math.isfinite(self.max_seconds)
            or self.min_seconds < 0
            or self.max_seconds < self.min_seconds
        ):
            raise ValueError("transition time bounds must satisfy 0 <= min <= max")


class ReIDProvider(Protocol):
    def embed(self, crops: list[np.ndarray]) -> np.ndarray: ...


class BoundedReIDWorker:
    """Single-provider worker with bounded, best-effort crop admission."""

    def __init__(self, provider: ReIDProvider, on_embedding, max_queue_size=64):
        self.provider = provider
        self.on_embedding = on_embedding
        self._queue = queue.Queue(maxsize=max(1, int(max_queue_size)))
        self._stopped = threading.Event()
        self.submitted = 0
        self.completed = 0
        self.dropped = 0
        self.errors = 0
        self._thread = threading.Thread(target=self._run, name="CrossCameraReID", daemon=True)
        self._thread.start()

    def submit(self, track_key, crop):
        if self._stopped.is_set():
            return False
        try:
            self._queue.put_nowait((track_key, crop))
        except queue.Full:
            self.dropped += 1
            return False
        self.submitted += 1
        return True

    def stats(self):
        return {
            "queue_depth": self._queue.qsize(),
            "submitted": self.submitted,
            "completed": self.completed,
            "dropped": self.dropped,
            "errors": self.errors,
        }

    def stop(self, timeout=2.0):
        self._stopped.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=max(0.0, float(timeout)))

    def _run(self):
        while not self._stopped.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            track_key, crop = item
            try:
                vectors = np.asarray(self.provider.embed([crop]), dtype=np.float32)
                if vectors.ndim == 1:
                    vector = vectors
                elif vectors.ndim == 2 and len(vectors) == 1:
                    vector = vectors[0]
                else:
                    raise ValueError("ReID provider must return one embedding per crop")
                vector = vector.reshape(-1)
                if not vector.size or not np.all(np.isfinite(vector)):
                    raise ValueError("ReID provider returned an invalid embedding")
                norm = float(np.linalg.norm(vector))
                if norm == 0:
                    raise ValueError("ReID provider returned a zero embedding")
                self.on_embedding(track_key, vector / norm)
                self.completed += 1
            except Exception:
                self.errors += 1
            finally:
                self._queue.task_done()


def observations_from_tracks(camera_id, tracked, timestamp, frame_id, zone_resolver=None):
    """Convert local person tracks into the downstream cross-camera contract."""
    observations = []
    for track in tracked:
        bbox = tuple(int(value) for value in track["bbox"])
        observations.append(TrackObservation(
            camera_id=str(camera_id),
            local_track_id=track["id"],
            timestamp=float(timestamp),
            frame_id=int(frame_id),
            bbox=bbox,
            class_id=str(track.get("class_id", "person")),
            confidence=float(track.get("confidence", 1.0)),
            zone_id=zone_resolver(track) if zone_resolver else None,
        ))
    return observations


def crops_from_tracks(frame, tracked, *, min_width=32, min_height=64, max_crops=16):
    """Return bounded, copied person crops that are large enough for Re-ID."""
    if frame is None or frame.ndim < 2:
        return {}
    frame_height, frame_width = frame.shape[:2]
    crops = {}
    for track in tracked:
        x1, y1, x2, y2 = (int(value) for value in track["bbox"])
        x1, x2 = max(0, x1), min(frame_width, x2)
        y1, y2 = max(0, y1), min(frame_height, y2)
        if x2 - x1 < min_width or y2 - y1 < min_height:
            continue
        crop = frame[y1:y2, x1:x2]
        if crop.size:
            crops[track["id"]] = crop.copy()
        if len(crops) >= max(1, int(max_crops)):
            break
    return crops


@dataclass
class Tracklet:
    tracklet_id: str
    camera_id: str
    source: str
    local_track_id: str
    class_id: str
    start_time: float
    end_time: float
    observations: deque = field(default_factory=lambda: deque(maxlen=24))
    embedding_samples: deque = field(default_factory=lambda: deque(maxlen=8))
    closed: bool = False
    eligible: bool = True

    @property
    def entry_zone(self):
        return next((item.zone_id for item in self.observations if item.zone_id), None)

    @property
    def exit_zone(self):
        return next((item.zone_id for item in reversed(self.observations) if item.zone_id), None)

    @property
    def embedding(self):
        if not self.embedding_samples:
            return None
        vectors = np.asarray(self.embedding_samples, dtype=np.float32)
        mean = vectors.mean(axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm else None


@dataclass(frozen=True)
class AssociationCandidate:
    candidate_global_id: str
    source_tracklet_id: str
    destination_tracklet_id: str
    source_camera_id: str
    destination_camera_id: str
    transition_seconds: float
    confidence: float
    status: str = "candidate"
    source_local_track_id: str = ""
    destination_local_track_id: str = ""
    source_tracker: str = ""
    destination_tracker: str = ""
    class_id: str = "person"
    source_start_time: float = 0.0
    source_end_time: float = 0.0
    destination_start_time: float = 0.0
    destination_end_time: float = 0.0
    source_entry_zone: str | None = None
    source_exit_zone: str | None = None
    destination_entry_zone: str | None = None
    destination_exit_zone: str | None = None


class CrossCameraTracker:
    """Tracklet association service; topology-only matches remain unconfirmed."""

    def __init__(
        self,
        transitions: list[CameraTransition] | None = None,
        *,
        tracklet_gap_sec: float = 2.0,
        identity_timeout_sec: float = 45.0,
        max_closed_tracklets: int = 2000,
        max_candidates: int = 2000,
        min_embedding_similarity: float | None = None,
        ambiguity_margin: float = 0.08,
        reid_provider: ReIDProvider | None = None,
        reid_queue_size: int = 64,
        reid_sample_interval_sec: float = 1.0,
    ):
        self._lock = threading.RLock()
        self._transitions = {}
        self.set_transitions(transitions or [])
        self.tracklet_gap_sec = max(0.0, float(tracklet_gap_sec))
        self.identity_timeout_sec = max(0.0, float(identity_timeout_sec))
        self.min_embedding_similarity = None
        self.ambiguity_margin = max(0.0, float(ambiguity_margin))
        self.reid_provider = None
        self.reid_sample_interval_sec = max(0.0, float(reid_sample_interval_sec))
        self._last_reid_sample = {}
        self._reid_worker = None
        self._reid_generation = 0
        self._active: dict[tuple[str, str, str], Tracklet] = {}
        self._closed: deque[Tracklet] = deque(maxlen=max(1, int(max_closed_tracklets)))
        self._candidates: dict[str, AssociationCandidate] = {}
        self._candidate_order: deque[str] = deque()
        self.max_candidates = max(1, int(max_candidates))
        self._matched_sources: set[str] = set()
        self._matched_destinations: set[str] = set()
        if min_embedding_similarity is not None:
            self.set_min_embedding_similarity(min_embedding_similarity)
        if reid_provider is not None:
            self.set_reid_provider(reid_provider, max_queue_size=reid_queue_size)

    def set_reid_provider(self, provider: ReIDProvider | None, *, max_queue_size=64):
        with self._lock:
            previous = self._reid_worker
            self._reid_worker = None
            self.reid_provider = None
            self._reid_generation += 1
            self._last_reid_sample.clear()
            for tracklet in (*self._active.values(), *self._closed):
                tracklet.embedding_samples.clear()
        if previous is not None:
            previous.stop()
        if provider is not None:
            with self._lock:
                generation = self._reid_generation
                self.reid_provider = provider
                self._reid_worker = BoundedReIDWorker(
                    provider,
                    lambda track_key, vector: self._attach_embedding(track_key, vector, generation),
                    max_queue_size,
                )

    def reid_stats(self):
        worker = self._reid_worker
        if worker is None:
            return {"enabled": False, "queue_depth": 0, "submitted": 0, "completed": 0, "dropped": 0, "errors": 0}
        return {"enabled": True, **worker.stats()}

    @property
    def reid_enabled(self):
        return self._reid_worker is not None

    def set_transitions(self, transitions: list[CameraTransition]):
        with self._lock:
            configured = {}
            for item in transitions:
                if item.enabled:
                    configured.setdefault(
                        (item.source_camera_id, item.destination_camera_id), []
                    ).append(item)
            self._transitions = configured

    def set_min_embedding_similarity(self, value: float | None):
        if value is not None:
            value = float(value)
            if not math.isfinite(value) or value < -1.0 or value > 1.0:
                raise ValueError("embedding similarity threshold must be between -1 and 1")
        with self._lock:
            self.min_embedding_similarity = value

    def observe_camera(
        self,
        camera_id: str,
        source: str,
        observations: list[TrackObservation],
        *,
        now: float | None = None,
        crops: dict[str | int, np.ndarray] | None = None,
    ) -> list[AssociationCandidate]:
        """Update a processor's local tracks and return newly formed candidates."""
        timestamp = float(now if now is not None else time.time())
        camera_id = str(camera_id)
        source = str(source)
        emitted = []
        with self._lock:
            current_keys = {
                (camera_id, source, str(item.local_track_id))
                for item in observations if item.camera_id == camera_id
            }
            self._finalize_stale(timestamp, current_keys)
            seen = set()
            for observation in observations:
                if observation.camera_id != camera_id:
                    continue
                local_id = str(observation.local_track_id)
                key = (camera_id, source, local_id)
                seen.add(key)
                tracklet = self._active.get(key)
                is_new = tracklet is None or tracklet.class_id != observation.class_id
                if is_new:
                    if tracklet is not None:
                        self._close(tracklet)
                    tracklet = Tracklet(
                        tracklet_id=uuid.uuid4().hex,
                        camera_id=camera_id,
                        source=source,
                        local_track_id=local_id,
                        class_id=observation.class_id,
                        start_time=observation.timestamp,
                        end_time=observation.timestamp,
                    )
                    self._active[key] = tracklet
                tracklet.end_time = max(tracklet.end_time, observation.timestamp)
                tracklet.observations.append(observation)
                if observation.embedding is not None:
                    vector = np.asarray(observation.embedding, dtype=np.float32).reshape(-1)
                    if vector.size and np.all(np.isfinite(vector)):
                        tracklet.embedding_samples.append(vector)
                if crops and observation.local_track_id in crops and self._reid_worker:
                    sample_key = (self._reid_generation, tracklet.tracklet_id)
                    last_sample = self._last_reid_sample.get(sample_key, float("-inf"))
                    if observation.timestamp - last_sample >= self.reid_sample_interval_sec:
                        crop = crops[observation.local_track_id]
                        if crop is not None and crop.size:
                            if self._reid_worker.submit(sample_key, crop.copy()):
                                self._last_reid_sample[sample_key] = observation.timestamp
                if is_new:
                    if self._reid_worker is None:
                        emitted.extend(self._associate_new_destination(tracklet))

            for key, tracklet in tuple(self._active.items()):
                if key[:2] == (camera_id, source) and key not in seen:
                    if timestamp - tracklet.end_time >= self.tracklet_gap_sec:
                        self._close(tracklet)
            return emitted

    def reset_source(self, camera_id: str, source: str):
        """Close source tracklets without using clip-rewind data for matching."""
        with self._lock:
            camera_id = str(camera_id)
            source = str(source)
            for key, tracklet in tuple(self._active.items()):
                if key[:2] == (camera_id, source):
                    tracklet.eligible = False
                    self._close(tracklet)
            for tracklet in self._closed:
                if tracklet.camera_id == camera_id and tracklet.source == source:
                    tracklet.eligible = False

    def candidates(self) -> list[AssociationCandidate]:
        with self._lock:
            return list(self._candidates.values())

    def get_candidate(self, candidate_id: str) -> AssociationCandidate | None:
        with self._lock:
            return self._candidates.get(candidate_id)

    def confirm_candidate(self, candidate_id: str) -> AssociationCandidate | None:
        from dataclasses import replace

        with self._lock:
            candidate = self._candidates.get(candidate_id)
            if candidate is None or candidate.status != "candidate":
                return None
            confirmed = replace(candidate, status="confirmed")
            self._candidates[candidate_id] = confirmed
            return confirmed

    def _finalize_stale(self, timestamp, protected_keys=()):
        for key, tracklet in tuple(self._active.items()):
            if key not in protected_keys and timestamp - tracklet.end_time >= self.tracklet_gap_sec:
                self._close(tracklet)
        cutoff = timestamp - self.identity_timeout_sec
        while self._closed and self._closed[0].end_time < cutoff:
            self._closed.popleft()

    def _close(self, tracklet):
        key = (tracklet.camera_id, tracklet.source, tracklet.local_track_id)
        if self._active.get(key) is not tracklet:
            return
        del self._active[key]
        tracklet.closed = True
        self._closed.append(tracklet)
        if self._reid_worker is not None:
            self._associate_new_destination(tracklet)

    def _attach_embedding(self, track_key, embedding, generation):
        with self._lock:
            if generation != self._reid_generation:
                return
            _, tracklet_id = track_key
            tracklet = next((
                item for item in self._active.values() if item.tracklet_id == tracklet_id
            ), None)
            if tracklet is None:
                tracklet = next((item for item in reversed(self._closed) if item.tracklet_id == tracklet_id), None)
            if tracklet is None:
                return
            tracklet.embedding_samples.append(np.asarray(embedding, dtype=np.float32))
            if self._reid_worker is not None:
                for destination in reversed(self._closed):
                    if destination.eligible and destination.tracklet_id not in self._matched_destinations:
                        self._associate_new_destination(destination)

    def _transition_score(self, source, destination, transition):
        elapsed = destination.start_time - source.end_time
        if elapsed < transition.min_seconds or elapsed > transition.max_seconds:
            return None
        if source.class_id != destination.class_id:
            return None
        if transition.source_zone_id and source.exit_zone != transition.source_zone_id:
            return None
        if transition.destination_zone_id and destination.entry_zone != transition.destination_zone_id:
            return None
        span = transition.max_seconds - transition.min_seconds
        if span <= 0:
            temporal = 1.0
        else:
            midpoint = transition.min_seconds + span / 2.0
            temporal = max(0.0, 1.0 - abs(elapsed - midpoint) / (span / 2.0))
        appearance = None
        source_embedding = source.embedding
        destination_embedding = destination.embedding
        if source_embedding is not None and destination_embedding is not None:
            if source_embedding.shape != destination_embedding.shape:
                return None
            appearance = float(np.dot(source_embedding, destination_embedding))
            if (
                self.min_embedding_similarity is not None
                and appearance < self.min_embedding_similarity
            ):
                return None
        elif self.reid_provider is not None:
            return None
        candidate_score = appearance if appearance is not None else temporal
        return elapsed, candidate_score

    def _associate_new_destination(self, destination):
        matches = []
        for source in self._closed:
            transitions = self._transitions.get((source.camera_id, destination.camera_id), ())
            if source.source != destination.source or not source.eligible or not transitions:
                continue
            if source.tracklet_id in self._matched_sources or destination.tracklet_id in self._matched_destinations:
                continue
            scored_routes = [
                self._transition_score(source, destination, transition)
                for transition in transitions
            ]
            valid_routes = [scored for scored in scored_routes if scored is not None]
            if valid_routes:
                elapsed, confidence = max(valid_routes, key=lambda scored: scored[1])
                matches.append((source, elapsed, confidence))
        matches.sort(key=lambda item: item[2], reverse=True)
        if not matches:
            return []
        if len(matches) > 1 and matches[0][2] - matches[1][2] < self.ambiguity_margin:
            return []
        source, elapsed, confidence = matches[0]
        candidate = AssociationCandidate(
            candidate_global_id=f"candidate-{uuid.uuid4().hex[:12]}",
            source_tracklet_id=source.tracklet_id,
            destination_tracklet_id=destination.tracklet_id,
            source_camera_id=source.camera_id,
            destination_camera_id=destination.camera_id,
            transition_seconds=elapsed,
            confidence=round(confidence, 4),
            source_local_track_id=source.local_track_id,
            destination_local_track_id=destination.local_track_id,
            source_tracker=source.source,
            destination_tracker=destination.source,
            class_id=source.class_id,
            source_start_time=source.start_time,
            source_end_time=source.end_time,
            destination_start_time=destination.start_time,
            destination_end_time=destination.end_time,
            source_entry_zone=source.entry_zone,
            source_exit_zone=source.exit_zone,
            destination_entry_zone=destination.entry_zone,
            destination_exit_zone=destination.exit_zone,
        )
        self._candidates[candidate.candidate_global_id] = candidate
        self._candidate_order.append(candidate.candidate_global_id)
        self._matched_sources.add(source.tracklet_id)
        self._matched_destinations.add(destination.tracklet_id)
        while len(self._candidate_order) > self.max_candidates:
            expired_id = self._candidate_order.popleft()
            expired = self._candidates.pop(expired_id, None)
            if expired:
                self._matched_sources.discard(expired.source_tracklet_id)
                self._matched_destinations.discard(expired.destination_tracklet_id)
        return [candidate]


cross_camera_tracker = CrossCameraTracker()


def configure_reid_provider(
    provider: ReIDProvider | None,
    *,
    max_queue_size=64,
    min_similarity: float | None = None,
):
    """Install an approved provider; no model or weights are selected by default."""
    cross_camera_tracker.set_reid_provider(provider, max_queue_size=max_queue_size)
    cross_camera_tracker.set_min_embedding_similarity(min_similarity)