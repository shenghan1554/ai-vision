"""Ultralytics YOLOv8 detection engine with switchable PT/ONNX models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_MODEL_EXTENSIONS = {".pt", ".onnx"}


@dataclass(slots=True)
class Detection:
    """One object detection in original-image pixel coordinates."""

    class_id: int
    name: str
    confidence: float
    box: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "name": self.name,
            "confidence": round(self.confidence, 6),
            "bbox_xyxy": [round(float(value), 3) for value in self.box],
        }


@dataclass(slots=True)
class DetectionResult:
    """Detections, an annotated BGR image, and end-to-end prediction timing."""

    detections: list[Detection]
    annotated: np.ndarray
    inference_ms: float
    names: dict[int, str]


def load_names(names_path: str | Path | None) -> dict[int, str] | None:
    """Load optional class names, one UTF-8 name per line."""
    if names_path is None or not str(names_path).strip():
        return None
    path = Path(names_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"类别文件不存在：{path}")
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not names:
        raise ValueError(f"类别文件为空：{path}")
    return dict(enumerate(names))


def _normalise_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def _resolve_device(device: str) -> str:
    value = str(device).strip().lower()
    if not value or value == "auto":
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    if value == "gpu":
        return "0"
    return value


class YOLOv8Engine:
    """Load any compatible YOLOv8 detection model without code changes.

    ``.pt`` models run through PyTorch and ``.onnx`` models through ONNX
    Runtime. Ultralytics supplies preprocessing, NMS, and model metadata.
    """

    def __init__(
        self,
        model_path: str | Path,
        names_path: str | Path | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "auto",
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"模型文件不存在：{self.model_path}")
        suffix = self.model_path.suffix.lower()
        if suffix not in SUPPORTED_MODEL_EXTENSIONS:
            raise ValueError("模型格式仅支持 .pt 或 .onnx，请重新选择模型文件。")
        if self.model_path.stat().st_size == 0:
            raise ValueError(f"模型文件为空：{self.model_path}")
        if not 0.0 <= float(conf) <= 1.0:
            raise ValueError("置信度阈值必须在 0 到 1 之间。")
        if not 0.0 <= float(iou) <= 1.0:
            raise ValueError("IoU 阈值必须在 0 到 1 之间。")
        if int(imgsz) < 32:
            raise ValueError("推理尺寸必须大于或等于 32。")

        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = _resolve_device(device)
        self.format = suffix[1:].upper()
        self._names_override = load_names(names_path)
        self._names = dict(self._names_override or {})

        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path.resolve()), task="detect")
        except Exception as exc:
            raise RuntimeError(
                f"无法加载 {self.format} 模型：{self.model_path}\n"
                "请确认这是兼容的 YOLOv8 目标检测模型，且文件完整。"
                f"\n原始错误：{exc}"
            ) from exc

        if not self._names:
            self._names = _normalise_names(getattr(self._model, "names", {}))

    @property
    def names(self) -> dict[int, str]:
        return dict(self._names)

    @property
    def model_description(self) -> str:
        class_info = f"，{len(self._names)} 个类别" if self._names else ""
        return (
            f"{self.model_path.name}（{self.format}，设备 {self.device}{class_info}）"
        )

    def warmup(self) -> None:
        """Run one dummy inference so the first real call is not slow.

        ONNX Runtime's first inference (graph optimisation / memory arena) can
        take seconds; doing it while the UI shows "loading model" keeps the
        photo capture itself fast.
        """
        try:
            import numpy as _np

            dummy = _np.zeros((self.imgsz, self.imgsz, 3), dtype=_np.uint8)
            self.predict(dummy)
        except Exception:
            # Warm-up is an optimisation only; never fail model loading.
            pass

    def predict(self, image_bgr: np.ndarray) -> DetectionResult:
        """Run one BGR image through the currently selected model."""
        if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3:
            raise ValueError("输入图像必须是有效的 BGR 图像数组。")
        if image_bgr.shape[0] < 1 or image_bgr.shape[1] < 1:
            raise ValueError("输入图像尺寸无效。")

        started = time.perf_counter()
        results = self._model.predict(
            source=image_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            task="detect",
            verbose=False,
        )
        inference_ms = (time.perf_counter() - started) * 1000.0
        if not results:
            return DetectionResult([], image_bgr.copy(), inference_ms, self.names)

        result = results[0]
        result_names = _normalise_names(getattr(result, "names", {}))
        names = self._names_override or result_names or self._names
        if names:
            self._names = names
            # result.plot() uses this mapping, so custom label files also affect
            # the saved annotated image rather than only the JSON output.
            result.names = names

        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            coordinates = boxes.xyxy.detach().cpu().numpy()
            class_ids = boxes.cls.detach().cpu().numpy().astype(int)
            confidences = boxes.conf.detach().cpu().numpy()
            for coords, class_id, confidence in zip(
                coordinates, class_ids, confidences
            ):
                detections.append(
                    Detection(
                        class_id=int(class_id),
                        name=names.get(int(class_id), f"class_{int(class_id)}"),
                        confidence=float(confidence),
                        box=tuple(float(value) for value in coords[:4]),
                    )
                )

        annotated = result.plot(boxes=True, labels=True, conf=True)
        return DetectionResult(detections, annotated, inference_ms, dict(names))
