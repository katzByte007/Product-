"""Common request and result contracts for local and process workers."""
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Optional, Sequence


@dataclass(slots=True)
class InferenceRequest:
    request_id: str
    camera_id: str
    model_key: str
    frame: Any
    labels: tuple[str, ...] = ()
    submitted_at: float = field(default_factory=monotonic)

    @property
    def batch_key(self) -> tuple[str, tuple[str, ...]]:
        """Requests with the same model and labels can share one forward pass."""
        return self.model_key, self.labels


@dataclass(slots=True)
class InferenceResult:
    request_id: str
    camera_id: str
    model_key: str
    detections: Any = None
    error: Optional[str] = None
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def latency_ms(self) -> float:
        if not self.started_at or not self.finished_at:
            return 0.0
        return (self.finished_at - self.started_at) * 1000.0


def group_compatible_requests(requests: Sequence[InferenceRequest], max_batch: int):
    """Group requests by model and labels while preserving input order."""
    groups: dict[tuple[str, tuple[str, ...]], list[InferenceRequest]] = {}
    for request in requests:
        groups.setdefault(request.batch_key, []).append(request)
    batches = []
    for group in groups.values():
        for offset in range(0, len(group), max(1, int(max_batch))):
            batches.append(group[offset : offset + max(1, int(max_batch))])
    return batches
