"""Labeled-pair evaluation and threshold selection for plant Re-ID data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os

import cv2
import numpy as np


def evaluate_labeled_pairs(provider, pairs):
    """Return cosine scores for rows with crop_a, crop_b, and same fields."""
    results = []
    for pair in pairs:
        image_a = pair["crop_a"]
        image_b = pair["crop_b"]
        if isinstance(image_a, (str, os.PathLike)):
            image_a = cv2.imread(os.fspath(image_a), cv2.IMREAD_COLOR)
        if isinstance(image_b, (str, os.PathLike)):
            image_b = cv2.imread(os.fspath(image_b), cv2.IMREAD_COLOR)
        if image_a is None or image_b is None:
            raise ValueError("could not load one of the labeled pair crops")
        embeddings = np.asarray(provider.embed([image_a, image_b]), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != 2:
            raise ValueError("provider must return one embedding per crop")
        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(norms == 0) or not np.all(np.isfinite(embeddings)):
            raise ValueError("provider returned invalid embeddings")
        embeddings /= norms[:, None]
        label = pair["same"]
        if isinstance(label, str):
            normalized_label = label.strip().lower()
            if normalized_label not in {"1", "true", "yes", "same", "0", "false", "no", "different"}:
                raise ValueError(f"invalid same/different label: {label}")
            label = normalized_label in {"1", "true", "yes", "same"}
        results.append({
            "same": bool(label),
            "similarity": float(np.dot(embeddings[0], embeddings[1])),
            "camera_a": pair.get("camera_a"),
            "camera_b": pair.get("camera_b"),
        })
    if not results or not any(row["same"] for row in results) or all(row["same"] for row in results):
        raise ValueError("validation pairs must include both same-person and different-person examples")
    return results


def select_threshold(results, *, max_false_match_rate):
    """Choose the lowest threshold meeting the requested false-match rate."""
    max_false_match_rate = float(max_false_match_rate)
    if not 0.0 <= max_false_match_rate <= 1.0:
        raise ValueError("max_false_match_rate must be between 0 and 1")
    scores = np.asarray([row["similarity"] for row in results], dtype=np.float64)
    labels = np.asarray([row["same"] for row in results], dtype=bool)
    negative_count = int((~labels).sum())
    positive_count = int(labels.sum())
    if not negative_count or not positive_count:
        raise ValueError("threshold selection requires both positive and negative pairs")

    candidates = np.unique(scores)
    candidates = np.concatenate((candidates, [np.nextafter(candidates.max(), np.inf)]))
    operating_points = []
    for threshold in candidates:
        predicted = scores >= threshold
        false_matches = int(np.logical_and(predicted, ~labels).sum())
        true_matches = int(np.logical_and(predicted, labels).sum())
        false_match_rate = false_matches / negative_count
        if false_match_rate <= max_false_match_rate:
            recall = true_matches / positive_count
            precision = true_matches / max(1, true_matches + false_matches)
            operating_points.append((recall, precision, float(threshold), false_match_rate))

    recall, precision, threshold, false_match_rate = max(
        operating_points, key=lambda point: (point[0], point[1], point[2])
    )
    return {
        "threshold": threshold,
        "max_false_match_rate": max_false_match_rate,
        "false_match_rate": false_match_rate,
        "false_reject_rate": 1.0 - recall,
        "precision": precision,
        "recall": recall,
        "positive_pairs": positive_count,
        "negative_pairs": negative_count,
    }


def _read_pairs(path):
    root = os.path.dirname(os.path.abspath(path))
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"crop_a", "crop_b", "same"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("CSV must contain crop_a,crop_b,same columns")
    for row in rows:
        row["crop_a"] = os.path.join(root, row["crop_a"])
        row["crop_b"] = os.path.join(root, row["crop_b"])
    return rows


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs_csv")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-false-match-rate", type=float, required=True)
    parser.add_argument("--dataset-id", required=True)
    args = parser.parse_args()

    from backend.reid.torchreid_osnet import OSNetAINProvider

    provider = OSNetAINProvider(args.weights, device=args.device)
    pairs = _read_pairs(args.pairs_csv)
    results = evaluate_labeled_pairs(provider, pairs)
    report = select_threshold(results, max_false_match_rate=args.max_false_match_rate)
    report["model"] = "osnet_ain_x1_0"
    report["dataset_id"] = args.dataset_id
    report["pairs_csv"] = os.path.abspath(args.pairs_csv)
    report["pairs_csv_sha256"] = _sha256_file(args.pairs_csv)
    report["weights_sha256"] = _sha256_file(args.weights)
    report["camera_pairs"] = sorted({
        f"{row['camera_a']}->{row['camera_b']}"
        for row in results if row.get("camera_a") and row.get("camera_b")
    })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()