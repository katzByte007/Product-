"""Torchreid-compatible OSNet person crop preprocessing."""
import cv2
import numpy as np


INPUT_HEIGHT = 256
INPUT_WIDTH = 128
MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def preprocess_crops(crops):
    """Convert BGR uint8 crops to normalized RGB NCHW float32 arrays."""
    if not crops:
        return np.empty((0, 3, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)

    batch = np.empty((len(crops), 3, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)
    for index, crop in enumerate(crops):
        if crop is None or not isinstance(crop, np.ndarray) or crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError("each Re-ID crop must be a three-channel image array")
        if crop.size == 0:
            raise ValueError("Re-ID crops must not be empty")
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - MEAN) / STD
        batch[index] = normalized.transpose(2, 0, 1)
    return batch