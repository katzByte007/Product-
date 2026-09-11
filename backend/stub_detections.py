"""Lightweight demo processors for ANPR and FR (no heavy ML deps)."""


class DummyDetectionProcessor:
    """Marks a detection module active in UI without running inference."""

    def __init__(self, camera_id: str, module: str, config=None):
        self.camera_id = camera_id
        self.module = module
        self.config = config or {}
        self.stats = {"status": "demo", "module": module, "camera_id": camera_id}
        self._running = False

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def get_annotated_frame(self):
        return None

    def get_last_detections(self):
        return []

    def reload_embeddings(self):
        pass
