"""OSNet-AIN x1.0 feature provider backed by Torchreid and a local checkpoint."""
from __future__ import annotations

import os

import numpy as np

from backend.reid.preprocessing import preprocess_crops


MODEL_NAME = "osnet_ain_x1_0"


def resolve_device(torch, requested="auto"):
    requested = (requested or "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)
        if mps():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Re-ID but is unavailable")
    if requested == "mps":
        mps = getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)
        if not mps():
            raise RuntimeError("MPS was requested for Re-ID but is unavailable")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("Re-ID device must be auto, cpu, cuda, or mps")
    return torch.device(requested)


class OSNetAINProvider:
    """Embed BGR person crops into normalized float32 OSNet-AIN features.

    The checkpoint must be a locally obtained person-ReID checkpoint for
    `osnet_ain_x1_0`; ImageNet classification weights are not accepted as a
    substitute. Checkpoint files are loaded with pickle support and must be
    trusted artifacts.
    """

    def __init__(self, weights_path, *, device="auto", model=None, torch_module=None):
        weights_path = os.path.abspath(os.path.expanduser(os.fspath(weights_path)))
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"OSNet-AIN Re-ID checkpoint not found: {weights_path}")

        if torch_module is None:
            try:
                import torch as torch_module
            except ImportError as exc:
                raise RuntimeError("PyTorch is required for OSNet-AIN Re-ID") from exc
        self.torch = torch_module
        resolved_device = resolve_device(torch_module, device)
        self.device = resolved_device
        self.device_name = str(resolved_device)

        if model is None:
            try:
                import torchreid
            except ImportError as exc:
                raise RuntimeError(
                    "Torchreid is required; install requirements-reid.txt"
                ) from exc
            model = torchreid.models.build_model(
                name=MODEL_NAME,
                num_classes=1000,
                loss="softmax",
                pretrained=False,
            )
            self._load_reid_weights(model, weights_path)

        self.model = model.to(resolved_device)
        self.model.eval()

    def _load_reid_weights(self, model, weights_path):
        checkpoint = self.torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint)))
        else:
            state = checkpoint
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        if not isinstance(state, dict):
            raise ValueError("Re-ID checkpoint does not contain a state dictionary")

        model_state = model.state_dict()
        compatible = {}
        for name, tensor in state.items():
            if not hasattr(tensor, "shape"):
                continue
            key = name
            while key.startswith("module."):
                key = key[len("module."):]
            if key.startswith(("classifier.", "fc.")):
                continue
            if key in model_state and tuple(tensor.shape) == tuple(model_state[key].shape):
                compatible[key] = tensor

        feature_keys = [
            key for key in model_state
            if not key.startswith(("classifier.", "fc."))
        ]
        coverage = len(compatible) / max(1, len(feature_keys))
        if coverage < 0.95:
            raise ValueError(
                f"Checkpoint is incompatible with {MODEL_NAME}: feature-weight coverage {coverage:.1%}"
            )
        model.load_state_dict(compatible, strict=False)

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, 0), dtype=np.float32)
        inputs = self.torch.from_numpy(preprocess_crops(crops)).to(self.device)
        with self.torch.inference_mode():
            features = self.model(inputs)
        if isinstance(features, (tuple, list)):
            features = features[-1]
        features = features.reshape(features.shape[0], -1)
        features = self.torch.nn.functional.normalize(features.float(), p=2, dim=1)
        vectors = features.detach().cpu().numpy().astype(np.float32, copy=False)
        if vectors.shape[0] != len(crops) or not np.all(np.isfinite(vectors)):
            raise ValueError("OSNet-AIN produced invalid embeddings")
        return vectors