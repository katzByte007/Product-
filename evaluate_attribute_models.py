"""Evaluate OWLv2 and YOLO-World on labeled prompt-detection images."""
import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2


def box_iou(first, second):
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(1e-9, first_area + second_area - intersection)


def score_image(annotations, detections, iou_threshold, prompts=()):
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    labels = {str(label).casefold() for label in prompts}
    labels.update(item["label"].casefold() for item in annotations)
    labels.update(item["label"].casefold() for item in detections)
    for label in labels:
        truth = [item for item in annotations if item["label"].casefold() == label]
        predicted = [item for item in detections if item["label"].casefold() == label]
        predicted.sort(key=lambda item: item["score"], reverse=True)
        matched = set()
        for detection in predicted:
            candidates = [
                (box_iou(detection["box"], target["box"]), index)
                for index, target in enumerate(truth)
                if index not in matched
            ]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_iou >= iou_threshold:
                counts[label]["tp"] += 1
                matched.add(best_index)
            else:
                counts[label]["fp"] += 1
        counts[label]["fn"] += len(truth) - len(matched)
    return counts


def read_dataset(path):
    dataset_path = Path(path).resolve()
    records = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            image_path = Path(record["image"])
            if not image_path.is_absolute():
                image_path = dataset_path.parent / image_path
            prompts = record.get("prompts") or []
            annotations = record.get("annotations") or []
            if not isinstance(prompts, list) or not prompts or not all(
                isinstance(prompt, str) and prompt.strip() for prompt in prompts
            ):
                raise ValueError(f"line {line_number}: prompts must be a non-empty list of strings")
            if not isinstance(annotations, list):
                raise ValueError(f"line {line_number}: annotations must be a list")
            for annotation in annotations:
                box = annotation.get("box", [])
                if (
                    not annotation.get("label")
                    or len(box) != 4
                    or not all(isinstance(value, (int, float)) for value in box)
                ):
                    raise ValueError(f"line {line_number}: annotations need label and [x1, y1, x2, y2] box")
            records.append({"image": image_path, "prompts": prompts, "annotations": annotations})
    if not records:
        raise ValueError("dataset contains no labeled images")
    return records


class Owlv2Predictor:
    def __init__(self, width):
        from backend import beta_ai

        self.beta = beta_ai
        self.processor, self.model = beta_ai._get_owlv2()
        if self.processor is None or self.model is None:
            raise RuntimeError("OWLv2 failed to initialize")
        self.device = str(next(self.model.parameters()).device)
        self.width = width

    def predict(self, frame, prompts, confidence):
        beta = self.beta
        started = time.perf_counter()
        height, width = frame.shape[:2]
        resized = frame
        if width > self.width:
            scale = self.width / float(width)
            resized = cv2.resize(
                frame,
                (self.width, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        resized_height, resized_width = resized.shape[:2]
        image = beta.Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        labels = beta._get_cached_text_labels(prompts)
        text = [labels]
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with beta.torch.inference_mode():
            outputs = self.model(**inputs)
        detections = beta._owlv2_decode_boxes(
            self.processor,
            outputs,
            inputs,
            resized_height,
            resized_width,
            confidence,
            text,
        )
        scale_x = width / float(resized_width)
        scale_y = height / float(resized_height)
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            detection["box"] = (
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y,
            )
        return detections, (time.perf_counter() - started) * 1000.0


class YoloWorldPredictor:
    def __init__(self, image_size):
        from backend.yolo_world import get_yolo_world_runner

        self.runner = get_yolo_world_runner()
        self.image_size = image_size

    def predict(self, frame, prompts, confidence):
        started = time.perf_counter()
        detections = self.runner.detect(frame, prompts, confidence, self.image_size)
        return detections, (time.perf_counter() - started) * 1000.0


def evaluate(model_name, predictor, records, confidence, iou_threshold):
    total_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    latencies = []
    warmup_record = records[0]
    warmup_frame = cv2.imread(str(warmup_record["image"]))
    if warmup_frame is None:
        raise RuntimeError(f"could not read image: {warmup_record['image']}")
    predictor.predict(warmup_frame, warmup_record["prompts"], confidence)
    for record in records:
        frame = cv2.imread(str(record["image"]))
        if frame is None:
            raise RuntimeError(f"could not read image: {record['image']}")
        detections, elapsed_ms = predictor.predict(frame, record["prompts"], confidence)
        latencies.append(elapsed_ms)
        image_counts = score_image(
            record["annotations"], detections, iou_threshold, record["prompts"]
        )
        for label, counts in image_counts.items():
            for key in ("tp", "fp", "fn"):
                total_counts[label][key] += counts[key]

    total = {key: sum(counts[key] for counts in total_counts.values()) for key in ("tp", "fp", "fn")}
    per_label = {}
    for label, counts in sorted(total_counts.items()):
        per_label[label] = _metrics(counts)
    return {
        "model": model_name,
        "images": len(records),
        "warmup_images": 1,
        "confidence": confidence,
        "iou_threshold": iou_threshold,
        "overall": _metrics(total),
        "per_label": per_label,
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": _percentile(latencies, 0.95),
    }


def _metrics(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
    }


def _percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main():
    parser = argparse.ArgumentParser(description="Evaluate attribute detection accuracy and latency")
    parser.add_argument("dataset", help="JSONL annotations with image, prompts, and boxes")
    parser.add_argument("--model", choices=("owlv2", "yolo-world", "both"), default="both")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--owl-width", type=int, default=640)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    os.environ["VISION_VLM_DEVICE"] = args.device
    records = read_dataset(args.dataset)
    selected = ("owlv2", "yolo-world") if args.model == "both" else (args.model,)
    results = []
    for model_name in selected:
        predictor = (
            Owlv2Predictor(args.owl_width)
            if model_name == "owlv2"
            else YoloWorldPredictor(args.imgsz)
        )
        results.append(evaluate(model_name, predictor, records, args.confidence, args.iou_threshold))
    report = {"dataset": str(Path(args.dataset).resolve()), "results": results}
    serialized = json.dumps(report, indent=2)
    if args.output == "-":
        print(serialized)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()