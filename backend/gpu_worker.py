"""Bounded GPU worker process primitives for enterprise deployment.

The Flask process remains the control plane. A deployment supervisor can start
one worker per GPU and route camera/model requests to its bounded input queue.
"""
import multiprocessing as mp
import os
import queue
import time
import uuid
from dataclasses import dataclass

from backend.inference_protocol import InferenceRequest, InferenceResult, group_compatible_requests


@dataclass(frozen=True)
class GpuWorkerConfig:
    gpu_id: str = "0"
    queue_size: int = 128
    batch_size: int = 4
    model_path: str = ""
    model_key: str = "yolo"


class GpuWorkerProcess:
    """A restartable, bounded process endpoint for batched model inference."""

    def __init__(self, config: GpuWorkerConfig):
        self.config = config
        self.requests = mp.Queue(maxsize=max(1, config.queue_size))
        self.results = mp.Queue(maxsize=max(1, config.queue_size))
        self.process = None
        self.dropped = mp.Value("L", 0)

    def start(self):
        if self.process is not None and self.process.is_alive():
            return
        self.process = mp.Process(
            target=_worker_main,
            args=(self.config, self.requests, self.results, self.dropped),
            name=f"GPUWorker-{self.config.gpu_id}",
            daemon=True,
        )
        self.process.start()

    def submit(self, camera_id, frame, labels=()):
        request = InferenceRequest(
            request_id=uuid.uuid4().hex,
            camera_id=str(camera_id),
            model_key=self.config.model_key,
            frame=frame,
            labels=tuple(labels),
        )
        try:
            self.requests.put_nowait(request)
            return request.request_id
        except queue.Full:
            with self.dropped.get_lock():
                self.dropped.value += 1
            return None

    def stop(self, timeout=3.0):
        if self.process is None:
            return
        try:
            self.requests.put_nowait(None)
        except queue.Full:
            pass
        self.process.join(timeout=timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)
        self.process = None

    @property
    def stats(self):
        return {
            "gpu_id": self.config.gpu_id,
            "alive": bool(self.process and self.process.is_alive()),
            "dropped": self.dropped.value,
        }


def _worker_main(config, requests, results, dropped):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    model = _load_model(config)
    while True:
        try:
            first = requests.get(timeout=0.25)
        except queue.Empty:
            continue
        if first is None:
            return
        if model is None:
            _put_result(results, InferenceResult(
                request_id=first.request_id,
                camera_id=first.camera_id,
                model_key=first.model_key,
                error="model unavailable",
                started_at=time.monotonic(),
                finished_at=time.monotonic(),
            ))
            continue
        batch = [first]
        while len(batch) < config.batch_size:
            try:
                batch.append(requests.get_nowait())
            except queue.Empty:
                break
        for compatible in group_compatible_requests(batch, config.batch_size):
            started = time.monotonic()
            try:
                detections = _infer(model, compatible)
                for request, result in zip(compatible, detections):
                    _put_result(results, InferenceResult(
                        request_id=request.request_id,
                        camera_id=request.camera_id,
                        model_key=request.model_key,
                        detections=result,
                        started_at=started,
                        finished_at=time.monotonic(),
                    ))
            except Exception as exc:
                for request in compatible:
                    _put_result(results, InferenceResult(
                        request_id=request.request_id,
                        camera_id=request.camera_id,
                        model_key=request.model_key,
                        error=str(exc)[:240],
                        started_at=started,
                        finished_at=time.monotonic(),
                    ))


def _load_model(config):
    if config.model_key == "yolo" and config.model_path:
        from ultralytics import YOLO
        return YOLO(config.model_path)
    if config.model_key == "owlv2" and config.model_path:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        processor = Owlv2Processor.from_pretrained(
            config.model_path, local_files_only=True
        )
        model = Owlv2ForObjectDetection.from_pretrained(
            config.model_path, local_files_only=True
        )
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return processor, model.to(device).eval(), device
    return None


def _infer(model, requests):
    if model is None:
        return [None] * len(requests)
    if isinstance(model, tuple):
        processor, model_instance, device = model
        from PIL import Image
        import torch
        labels = list(requests[0].labels)
        images = [
            Image.fromarray(request.frame[:, :, ::-1])
            for request in requests
        ]
        inputs = processor(
            text=[labels] * len(images),
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(device) if hasattr(value, "to") else value
                  for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model_instance(**inputs)
        return [outputs] * len(requests)
    output = model([request.frame for request in requests], verbose=False)
    return list(output)


def _put_result(results, result):
    try:
        results.put_nowait(result)
    except Exception:
        pass
