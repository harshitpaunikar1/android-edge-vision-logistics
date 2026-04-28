"""
TensorFlow Lite inference pipeline for on-device object detection in logistics.
Handles model loading, image preprocessing, NMS post-processing, and benchmarking.
"""
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None
        logger.warning("TFLite runtime not available. Install tflite_runtime or tensorflow.")


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x, y, w, h normalized
    timestamp: float


class NMSProcessor:
    """Non-maximum suppression for filtering overlapping detections."""

    def __init__(self, iou_threshold: float = 0.45, score_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    def compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = w1 * h1 + w2 * h2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def apply_nms(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return []
        detections = [d for d in detections if d.confidence >= self.score_threshold]
        detections.sort(key=lambda d: d.confidence, reverse=True)
        kept = []
        while detections:
            best = detections.pop(0)
            kept.append(best)
            detections = [
                d for d in detections
                if self.compute_iou(best.bbox, d.bbox) < self.iou_threshold
            ]
        return kept


class TFLiteInferencePipeline:
    """
    End-to-end TFLite inference pipeline for object detection on Android/edge devices.
    Supports INT8, FP16, and FP32 models with configurable NMS post-processing.
    """

    def __init__(self, model_path: str, label_path: str,
                 input_size: Tuple[int, int] = (320, 320), num_threads: int = 4):
        self.model_path = model_path
        self.label_path = label_path
        self.input_size = input_size
        self.num_threads = num_threads
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.labels: List[str] = []
        self.nms = NMSProcessor()

    def load_model(self) -> bool:
        """Load TFLite interpreter and allocate tensors."""
        if tflite is None:
            logger.error("TFLite runtime not available")
            return False
        if not os.path.exists(self.model_path):
            logger.error("Model file not found: %s", self.model_path)
            return False
        self.interpreter = tflite.Interpreter(
            model_path=self.model_path,
            num_threads=self.num_threads
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.labels = self.load_labels()
        logger.info("Model loaded: %s | Input: %s", self.model_path, self.input_details[0]["shape"])
        return True

    def load_labels(self) -> List[str]:
        if not os.path.exists(self.label_path):
            return []
        with open(self.label_path) as f:
            return [line.strip() for line in f.readlines()]

    def preprocess_image(self, image_array: np.ndarray) -> np.ndarray:
        """Resize and normalize image for model input."""
        import cv2
        resized = cv2.resize(image_array, self.input_size)
        input_dtype = self.input_details[0]["dtype"] if self.input_details else np.float32
        if input_dtype == np.uint8:
            return np.expand_dims(resized.astype(np.uint8), axis=0)
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    def run_inference(self, image_array: np.ndarray) -> List[np.ndarray]:
        """Run TFLite inference and return raw output tensors."""
        if self.interpreter is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        preprocessed = self.preprocess_image(image_array)
        self.interpreter.set_tensor(self.input_details[0]["index"], preprocessed)
        self.interpreter.invoke()
        return [self.interpreter.get_tensor(od["index"]) for od in self.output_details]

    def postprocess(self, outputs: List[np.ndarray], original_shape: Tuple[int, int]) -> List[Detection]:
        """Decode boxes, filter by score, and apply NMS."""
        if not outputs:
            return []
        boxes = outputs[0][0] if len(outputs) > 0 else np.array([])
        scores = outputs[2][0] if len(outputs) > 2 else np.array([])
        classes = outputs[1][0].astype(int) if len(outputs) > 1 else np.array([])
        h, w = original_shape
        detections = []
        for i, score in enumerate(scores):
            if score < self.nms.score_threshold:
                continue
            if i >= len(boxes):
                break
            y1, x1, y2, x2 = boxes[i]
            bbox = (x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h)
            label = self.labels[classes[i]] if classes[i] < len(self.labels) else str(classes[i])
            detections.append(Detection(label=label, confidence=float(score), bbox=bbox, timestamp=time.time()))
        return self.nms.apply_nms(detections)

    def benchmark(self, image_array: np.ndarray, n_runs: int = 50) -> Dict:
        """Run n_runs inferences and return latency statistics."""
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.run_inference(image_array)
            latencies.append((time.perf_counter() - t0) * 1000)
        return {
            "mean_ms": round(float(np.mean(latencies)), 2),
            "p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "fps": round(1000.0 / float(np.mean(latencies)), 1),
            "n_runs": n_runs,
        }

    def quantize_summary(self, model_path: str) -> Dict:
        """Infer quantization type from filename and file size."""
        size_mb = os.path.getsize(model_path) / 1e6 if os.path.exists(model_path) else 0.0
        name = os.path.basename(model_path).lower()
        if "int8" in name:
            q_type = "INT8"
        elif "fp16" in name or "float16" in name:
            q_type = "FP16"
        else:
            q_type = "FP32"
        return {"model_path": model_path, "model_size_mb": round(size_mb, 2), "quantization_type": q_type}


if __name__ == "__main__":
    pipeline = TFLiteInferencePipeline(
        model_path="model_int8.tflite",
        label_path="labels.txt",
        input_size=(320, 320),
        num_threads=4,
    )
    dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    print("Quantize summary:", pipeline.quantize_summary("model_int8.tflite"))
    print("Pipeline ready. Call load_model() then run_inference(frame).")
