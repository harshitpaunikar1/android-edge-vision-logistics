"""
TFLite quantization pipeline for logistics object detection models.
Converts float32 SavedModels to float16 and INT8 TFLite format with calibration.
"""
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not installed. Quantization pipeline will not run.")


@dataclass
class QuantizationResult:
    source_model: str
    output_path: str
    quantization_type: str
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    conversion_time_s: float


class CalibrationDataset:
    """
    Provides representative frames for INT8 post-training quantization calibration.
    Wraps a list of numpy arrays or a generator function.
    """

    def __init__(self, frames: Optional[List[np.ndarray]] = None,
                 input_size: Tuple[int, int] = (320, 320),
                 num_samples: int = 100):
        self.frames = frames
        self.input_size = input_size
        self.num_samples = num_samples

    def _synthetic_frames(self) -> Iterator[List[np.ndarray]]:
        """Generate random frames when no real frames are provided."""
        for _ in range(self.num_samples):
            frame = np.random.randint(0, 255, (*self.input_size, 3), dtype=np.uint8)
            normalized = frame.astype(np.float32) / 255.0
            yield [np.expand_dims(normalized, axis=0)]

    def as_representative_dataset(self) -> Callable:
        """Return a callable compatible with TFLite converter representative_dataset."""
        if self.frames is not None:
            frames = self.frames

            def _gen():
                for f in frames:
                    resized = f
                    if f.shape[:2] != self.input_size:
                        try:
                            import cv2
                            resized = cv2.resize(f, self.input_size)
                        except ImportError:
                            resized = f
                    normalized = resized.astype(np.float32) / 255.0
                    yield [np.expand_dims(normalized, axis=0)]

            return _gen
        return self._synthetic_frames


class QuantizationPipeline:
    """
    Converts a TensorFlow SavedModel to TFLite with float32, float16, or INT8 quantization.
    Records compression ratios and conversion times for benchmarking.
    """

    def __init__(self, saved_model_dir: str, output_dir: str = "quantized_models"):
        self.saved_model_dir = saved_model_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_size_mb(self, path: str) -> float:
        if os.path.isfile(path):
            return os.path.getsize(path) / 1e6
        total = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, files in os.walk(path)
            for f in files
        )
        return total / 1e6

    def convert_float32(self) -> QuantizationResult:
        """Export to FP32 TFLite with no quantization."""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available.")
        t0 = time.perf_counter()
        converter = tf.lite.TFLiteConverter.from_saved_model(self.saved_model_dir)
        tflite_model = converter.convert()
        out_path = os.path.join(self.output_dir, "model_fp32.tflite")
        with open(out_path, "wb") as f:
            f.write(tflite_model)
        elapsed = time.perf_counter() - t0
        orig_mb = self._get_size_mb(self.saved_model_dir)
        quant_mb = self._get_size_mb(out_path)
        return QuantizationResult(
            source_model=self.saved_model_dir,
            output_path=out_path,
            quantization_type="FP32",
            original_size_mb=round(orig_mb, 2),
            quantized_size_mb=round(quant_mb, 2),
            compression_ratio=round(orig_mb / quant_mb, 2) if quant_mb > 0 else 0,
            conversion_time_s=round(elapsed, 2),
        )

    def convert_float16(self) -> QuantizationResult:
        """Export to FP16 TFLite using default float16 optimization."""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available.")
        t0 = time.perf_counter()
        converter = tf.lite.TFLiteConverter.from_saved_model(self.saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        out_path = os.path.join(self.output_dir, "model_fp16.tflite")
        with open(out_path, "wb") as f:
            f.write(tflite_model)
        elapsed = time.perf_counter() - t0
        orig_mb = self._get_size_mb(self.saved_model_dir)
        quant_mb = self._get_size_mb(out_path)
        return QuantizationResult(
            source_model=self.saved_model_dir,
            output_path=out_path,
            quantization_type="FP16",
            original_size_mb=round(orig_mb, 2),
            quantized_size_mb=round(quant_mb, 2),
            compression_ratio=round(orig_mb / quant_mb, 2) if quant_mb > 0 else 0,
            conversion_time_s=round(elapsed, 2),
        )

    def convert_int8(self, calibration_dataset: Optional[CalibrationDataset] = None) -> QuantizationResult:
        """
        Export to INT8 TFLite using full integer post-training quantization.
        Requires a representative dataset for calibration.
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available.")
        if calibration_dataset is None:
            calibration_dataset = CalibrationDataset()
        t0 = time.perf_counter()
        converter = tf.lite.TFLiteConverter.from_saved_model(self.saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = calibration_dataset.as_representative_dataset()
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32
        tflite_model = converter.convert()
        out_path = os.path.join(self.output_dir, "model_int8.tflite")
        with open(out_path, "wb") as f:
            f.write(tflite_model)
        elapsed = time.perf_counter() - t0
        orig_mb = self._get_size_mb(self.saved_model_dir)
        quant_mb = self._get_size_mb(out_path)
        return QuantizationResult(
            source_model=self.saved_model_dir,
            output_path=out_path,
            quantization_type="INT8",
            original_size_mb=round(orig_mb, 2),
            quantized_size_mb=round(quant_mb, 2),
            compression_ratio=round(orig_mb / quant_mb, 2) if quant_mb > 0 else 0,
            conversion_time_s=round(elapsed, 2),
        )

    def run_all(self, calibration_frames: Optional[List[np.ndarray]] = None) -> List[QuantizationResult]:
        """Run FP32, FP16, and INT8 conversions and return all results."""
        results = []
        for fn in [self.convert_float32, self.convert_float16]:
            try:
                results.append(fn())
                logger.info("Converted: %s", results[-1].quantization_type)
            except Exception as exc:
                logger.error("Conversion failed: %s", exc)
        try:
            cal = CalibrationDataset(frames=calibration_frames)
            results.append(self.convert_int8(cal))
            logger.info("INT8 conversion done.")
        except Exception as exc:
            logger.error("INT8 conversion failed: %s", exc)
        return results

    def print_summary(self, results: List[QuantizationResult]) -> None:
        header = f"{'Type':<8} {'Orig MB':<10} {'Quant MB':<10} {'Ratio':<8} {'Time s':<8} Output"
        print(header)
        print("-" * len(header))
        for r in results:
            print(f"{r.quantization_type:<8} {r.original_size_mb:<10} {r.quantized_size_mb:<10} "
                  f"{r.compression_ratio:<8} {r.conversion_time_s:<8} {r.output_path}")


if __name__ == "__main__":
    print("QuantizationPipeline loaded.")
    print("Usage: pipeline = QuantizationPipeline('path/to/saved_model', 'output_dir')")
    print("       results = pipeline.run_all()")
    print("       pipeline.print_summary(results)")
    dummy_frames = [np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8) for _ in range(10)]
    cal = CalibrationDataset(frames=dummy_frames)
    gen_fn = cal.as_representative_dataset()
    batch = next(gen_fn())
    print(f"Calibration batch shape: {batch[0].shape}, dtype: {batch[0].dtype}")
