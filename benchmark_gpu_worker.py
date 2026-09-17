"""Smoke test the bounded GPU worker process without loading a model."""
from backend.gpu_worker import GpuWorkerConfig, GpuWorkerProcess


if __name__ == "__main__":
    worker = GpuWorkerProcess(
        GpuWorkerConfig(gpu_id="cpu", queue_size=4, model_key="noop")
    )
    worker.start()
    request_id = worker.submit("camera-1", [[1]])
    assert request_id
    worker.stop()
    print("gpu_worker_lifecycle_ok", worker.stats)
